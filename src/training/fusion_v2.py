"""Fusion V2 subject-level training and evaluation."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.model.encoders import SequenceBinaryClassifier
from src.training.evaluate import (
    compute_metrics,
    save_calibration_summary,
    save_confusion_matrix_csv,
    save_curves,
    write_error_review,
)


@dataclass
class FusionV2TrainConfig:
    batch_size: int = 8
    hidden_dim: int = 128
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    patience: int = 3
    device: str = "cuda"
    freeze_warm_start_epochs: int = 3
    modality_dropout: float = 0.15
    fused_binary_weight: float = 1.0
    acoustic_aux_weight: float = 0.30
    visual_aux_weight: float = 0.15
    teacher_kl_weight: float = 0.20
    phq_regression_weight: float = 0.25
    phq_severity_weight: float = 0.10
    gate_entropy_weight: float = 0.02


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_to_device(item: Any, device: torch.device):
    if isinstance(item, dict):
        return {key: _move_to_device(value, device) for key, value in item.items()}
    if isinstance(item, list):
        return [_move_to_device(value, device) for value in item]
    if torch.is_tensor(item):
        return item.to(device)
    return item


def _binary_kl_from_logits(student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    student_probs = torch.sigmoid(student_logits).clamp(1e-6, 1.0 - 1e-6)
    teacher_probs = teacher_probs.clamp(1e-6, 1.0 - 1e-6)
    kl = teacher_probs * (teacher_probs.log() - student_probs.log())
    kl += (1.0 - teacher_probs) * ((1.0 - teacher_probs).log() - (1.0 - student_probs).log())
    return kl.mean()


def _masked_mse(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return predictions.new_tensor(0.0)
    diff = (predictions[mask] - targets[mask]) ** 2
    return diff.mean()


def _masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return logits.new_tensor(0.0)
    return nn.functional.cross_entropy(logits[mask], targets[mask])


class WindowTeacherWrapper(nn.Module):
    def __init__(
        self,
        modality_key: str,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        checkpoint_path: Path,
    ):
        super().__init__()
        self.modality_key = modality_key
        self.model = SequenceBinaryClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        self.model.load_state_dict(state_dict)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        values = batch["modalities"][self.modality_key]
        batch_size, num_windows, window_size, dim = values.shape
        flat_values = values.reshape(batch_size * num_windows, window_size, dim)
        logits = self.model(flat_values)
        probabilities = torch.sigmoid(logits).reshape(batch_size, num_windows)
        window_mask = batch["window_mask"].to(probabilities.dtype)
        return (probabilities * window_mask).sum(dim=1) / window_mask.sum(dim=1).clamp_min(1.0)


def load_teacher_wrappers(teacher_specs: Optional[list[dict[str, Any]]], device: torch.device) -> tuple[dict[str, WindowTeacherWrapper], dict[str, Any]]:
    wrappers: dict[str, WindowTeacherWrapper] = {}
    metadata = {"available": {}, "paths": {}}
    if not teacher_specs:
        return wrappers, metadata

    for spec in teacher_specs:
        checkpoint_path = Path(spec["checkpoint_path"])
        metadata["paths"][spec["name"]] = str(checkpoint_path)
        if not checkpoint_path.exists():
            metadata["available"][spec["name"]] = False
            continue
        wrapper = WindowTeacherWrapper(
            modality_key=spec["modality_key"],
            input_dim=spec["input_dim"],
            hidden_dim=spec["hidden_dim"],
            num_layers=spec["num_layers"],
            dropout=spec["dropout"],
            checkpoint_path=checkpoint_path,
        )
        wrapper.to(device)
        wrappers[spec["name"]] = wrapper
        metadata["available"][spec["name"]] = True
    return wrappers, metadata


def _build_optimizer(model: nn.Module, config: FusionV2TrainConfig):
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def _compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    teachers: dict[str, WindowTeacherWrapper],
    config: FusionV2TrainConfig,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    labels = batch["label_binary"].float()
    label_masks = batch["label_masks"]
    logits = outputs["mixture_logit"]
    fused_binary = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    acoustic_aux = nn.functional.binary_cross_entropy_with_logits(outputs["acoustic_logit"], labels)
    visual_aux = nn.functional.binary_cross_entropy_with_logits(outputs["visual_logit"], labels)

    teacher_outputs = {}
    teacher_loss = logits.new_tensor(0.0)
    if teachers:
        if "acoustic_teacher" in teachers:
            teacher_outputs["acoustic_teacher"] = teachers["acoustic_teacher"](batch)
            teacher_loss = teacher_loss + _binary_kl_from_logits(outputs["acoustic_logit"], teacher_outputs["acoustic_teacher"])
        if "visual_teacher" in teachers:
            teacher_outputs["visual_teacher"] = teachers["visual_teacher"](batch)
            teacher_loss = teacher_loss + _binary_kl_from_logits(outputs["visual_logit"], teacher_outputs["visual_teacher"])

    phq_reg = _masked_mse(outputs["phq_score"], batch["label_phq"], label_masks["phq"])
    phq_severity = _masked_cross_entropy(outputs["phq_severity_logits"], batch["label_phq_severity"], label_masks["phq_severity"])
    entropy = -(outputs["gate_weights"] * outputs["gate_weights"].clamp_min(1e-6).log()).sum(dim=-1).mean()
    gate_entropy_regularization = -entropy

    total = config.fused_binary_weight * fused_binary
    total = total + config.acoustic_aux_weight * acoustic_aux
    total = total + config.visual_aux_weight * visual_aux
    total = total + config.teacher_kl_weight * teacher_loss
    total = total + config.phq_regression_weight * phq_reg
    total = total + config.phq_severity_weight * phq_severity
    total = total + config.gate_entropy_weight * gate_entropy_regularization

    components = {
        "fused_binary_loss": float(fused_binary.item()),
        "acoustic_aux_binary_loss": float(acoustic_aux.item()),
        "visual_aux_binary_loss": float(visual_aux.item()),
        "teacher_kl_loss": float(teacher_loss.item()),
        "phq_regression_loss": float(phq_reg.item()),
        "phq_severity_ce_loss": float(phq_severity.item()),
        "gate_entropy_regularization": float(gate_entropy_regularization.item()),
    }
    return total, components, teacher_outputs


@torch.no_grad()
def evaluate_fusion_v2_model(
    model: nn.Module,
    loader,
    device: torch.device,
    teachers: Optional[dict[str, WindowTeacherWrapper]] = None,
) -> dict[str, Any]:
    model.eval()
    rows = []
    teacher_mae: dict[str, list[float]] = {}

    for raw_batch in loader:
        batch = _move_to_device(raw_batch, device)
        outputs = model(batch)
        probabilities = torch.sigmoid(outputs["mixture_logit"]).cpu().numpy()
        acoustic_probs = torch.sigmoid(outputs["acoustic_logit"]).cpu().numpy()
        visual_probs = torch.sigmoid(outputs["visual_logit"]).cpu().numpy()
        fused_probs = torch.sigmoid(outputs["fused_logit"]).cpu().numpy()
        gate_weights = outputs["gate_weights"].cpu().numpy()
        quality = batch["quality_features"].cpu().numpy()
        teacher_probs = {}
        if teachers:
            for name, teacher in teachers.items():
                teacher_probs[name] = teacher(batch).cpu().numpy()

        for row_index, subject_id in enumerate(raw_batch["subject_id"]):
            if torch.is_tensor(raw_batch["window_mask"]):
                num_windows = int(raw_batch["window_mask"][row_index].sum().item())
            else:
                num_windows = 0
            row = {
                "subject_id": str(subject_id),
                "label": int(raw_batch["label_binary"][row_index]),
                "num_windows": num_windows,
                "probability": float(probabilities[row_index]),
                "prediction": int(probabilities[row_index] >= 0.5),
                "acoustic_probability": float(acoustic_probs[row_index]),
                "visual_probability": float(visual_probs[row_index]),
                "fused_probability": float(fused_probs[row_index]),
                "quality_score": float(quality[row_index][0]),
                "gate_acoustic": float(gate_weights[row_index][0]),
                "gate_visual": float(gate_weights[row_index][1]),
                "gate_fused": float(gate_weights[row_index][2]),
            }
            for teacher_name, teacher_values in teacher_probs.items():
                teacher_value = float(teacher_values[row_index])
                row[f"{teacher_name}_probability"] = teacher_value
                teacher_mae.setdefault(teacher_name, []).append(abs(row["probability"] - teacher_value))
            rows.append(row)

    frame = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    metrics = compute_metrics(frame)

    gate_stats = {
        "gate_acoustic_mean": float(frame["gate_acoustic"].mean()),
        "gate_visual_mean": float(frame["gate_visual"].mean()),
        "gate_fused_mean": float(frame["gate_fused"].mean()),
        "gate_acoustic_std": float(frame["gate_acoustic"].std(ddof=0)),
        "gate_visual_std": float(frame["gate_visual"].std(ddof=0)),
        "gate_fused_std": float(frame["gate_fused"].std(ddof=0)),
    }

    if len(frame) >= 3:
        bins = pd.qcut(frame["quality_score"], q=min(3, len(frame)), duplicates="drop")
        slice_rows = []
        for bucket, group in frame.groupby(bins, observed=False):
            if len(group) == 0:
                continue
            slice_metrics = compute_metrics(group)
            slice_rows.append(
                {
                    "quality_bucket": str(bucket),
                    "num_subjects": int(len(group)),
                    "macro_f1": slice_metrics["macro_f1"],
                    "binary_f1": slice_metrics["binary_f1"],
                    "recall": slice_metrics["recall"],
                    "precision": slice_metrics["precision"],
                    "auroc": slice_metrics["auroc"],
                    "pr_auc": slice_metrics["pr_auc"],
                }
            )
        quality_slice_metrics = pd.DataFrame(slice_rows)
    else:
        quality_slice_metrics = pd.DataFrame(
            [
                {
                    "quality_bucket": "all",
                    "num_subjects": int(len(frame)),
                    "macro_f1": metrics["macro_f1"],
                    "binary_f1": metrics["binary_f1"],
                    "recall": metrics["recall"],
                    "precision": metrics["precision"],
                    "auroc": metrics["auroc"],
                    "pr_auc": metrics["pr_auc"],
                }
            ]
        )

    teacher_alignment = {}
    for teacher_name, errors in teacher_mae.items():
        teacher_alignment[teacher_name] = {
            "mae": float(np.mean(errors)),
            "num_subjects": int(len(errors)),
        }

    return {
        "subject_predictions": frame,
        "metrics": metrics,
        "gate_stats": gate_stats,
        "quality_slice_metrics": quality_slice_metrics,
        "teacher_alignment": teacher_alignment,
    }


def persist_fusion_v2_seed_artifacts(seed_dir: Path, seed: int, result: dict):
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "history.json").write_text(json.dumps(result["history"], indent=2), encoding="utf-8")
    (seed_dir / "seed_summary.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "best_dev_score": result["best_dev_score"],
                "teacher_metadata": result.get("teacher_metadata", {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save({"seed": seed, "state_dict": result["state_dict"]}, seed_dir / "checkpoint.pt")

    for split_name in ["dev", "test"]:
        payload = result.get(f"{split_name}_results")
        if payload is None:
            continue
        predictions = payload["subject_predictions"]
        predictions.to_csv(seed_dir / f"{split_name}_subject_predictions.csv", index=False)
        (seed_dir / f"{split_name}_metrics.json").write_text(json.dumps(payload["metrics"], indent=2), encoding="utf-8")
        (seed_dir / f"{split_name}_gate_stats.json").write_text(json.dumps(payload["gate_stats"], indent=2), encoding="utf-8")
        payload["quality_slice_metrics"].to_csv(seed_dir / f"{split_name}_quality_slice_metrics.csv", index=False)
        (seed_dir / f"{split_name}_teacher_alignment.json").write_text(
            json.dumps(payload["teacher_alignment"], indent=2),
            encoding="utf-8",
        )
        save_confusion_matrix_csv(predictions, seed_dir, split_name)
        save_calibration_summary(predictions, seed_dir, split_name)
        save_curves(predictions, seed_dir, split_name)
        write_error_review(predictions, seed_dir / f"{split_name}_error_review.md")


def train_fusion_v2_one_seed(
    model: nn.Module,
    train_loader,
    dev_loader,
    test_loader,
    config: FusionV2TrainConfig,
    teacher_specs: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    model.to(device)
    teachers, teacher_metadata = load_teacher_wrappers(teacher_specs, device)
    optimizer = _build_optimizer(model, config)

    best_state = copy.deepcopy(model.state_dict())
    best_dev_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_items = 0
        component_totals: dict[str, float] = {}

        for raw_batch in train_loader:
            batch = _move_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, components, _ = _compute_loss(outputs, batch, teachers, config)
            loss.backward()
            optimizer.step()

            batch_size = batch["label_binary"].shape[0]
            running_loss += float(loss.item()) * batch_size
            running_items += batch_size
            for key, value in components.items():
                component_totals[key] = component_totals.get(key, 0.0) + value * batch_size

        dev_results = evaluate_fusion_v2_model(model, dev_loader, device, teachers)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": running_loss / max(running_items, 1),
            "dev_macro_f1": dev_results["metrics"]["macro_f1"],
            "dev_binary_f1": dev_results["metrics"]["binary_f1"],
        }
        for key, total in component_totals.items():
            epoch_metrics[key] = total / max(running_items, 1)
        history.append(epoch_metrics)

        if dev_results["metrics"]["macro_f1"] > best_dev_score:
            best_dev_score = dev_results["metrics"]["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    model.load_state_dict(best_state)
    dev_results = evaluate_fusion_v2_model(model, dev_loader, device, teachers)
    test_results = evaluate_fusion_v2_model(model, test_loader, device, teachers) if test_loader is not None else None

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_dev_score": best_dev_score,
        "dev_results": dev_results,
        "test_results": test_results,
        "state_dict": copy.deepcopy(model.state_dict()),
        "teacher_metadata": teacher_metadata,
    }
