"""Vision V3 subject-level training and evaluation."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.training.evaluate import (
    compute_metrics,
    save_calibration_summary,
    save_confusion_matrix_csv,
    save_curves,
    write_error_review,
)
from src.training.fusion_v2 import (
    WindowTeacherWrapper,
    _binary_kl_from_logits,
    _move_to_device,
    load_teacher_wrappers,
)


@dataclass
class VisionV3TrainConfig:
    batch_size: int = 8
    hidden_dim: int = 128
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    patience: int = 3
    device: str = "cuda"
    modality_dropout: float = 0.10
    mixture_binary_weight: float = 1.0
    visual_aux_weight: float = 0.35
    acoustic_aux_weight: float = 0.10
    teacher_kl_weight: float = 0.10
    gate_entropy_weight: float = 0.01


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_optimizer(model: nn.Module, config: VisionV3TrainConfig):
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def _compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    teachers: dict[str, WindowTeacherWrapper],
    config: VisionV3TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = batch["label_binary"].float()
    mixture_binary = nn.functional.binary_cross_entropy_with_logits(outputs["mixture_logit"], labels)
    visual_aux = nn.functional.binary_cross_entropy_with_logits(outputs["visual_logit"], labels)

    acoustic_aux = outputs["mixture_logit"].new_tensor(0.0)
    if outputs.get("has_acoustic", False):
        acoustic_aux = nn.functional.binary_cross_entropy_with_logits(outputs["acoustic_logit"], labels)

    teacher_loss = outputs["mixture_logit"].new_tensor(0.0)
    if teachers:
        if "visual_teacher" in teachers:
            teacher_loss = teacher_loss + _binary_kl_from_logits(outputs["visual_logit"], teachers["visual_teacher"](batch))
        if outputs.get("has_acoustic", False) and "acoustic_teacher" in teachers:
            teacher_loss = teacher_loss + _binary_kl_from_logits(outputs["acoustic_logit"], teachers["acoustic_teacher"](batch))

    entropy = -(outputs["gate_weights"] * outputs["gate_weights"].clamp_min(1e-6).log()).sum(dim=-1).mean()
    gate_entropy_regularization = -entropy

    total = config.mixture_binary_weight * mixture_binary
    total = total + config.visual_aux_weight * visual_aux
    total = total + config.acoustic_aux_weight * acoustic_aux
    total = total + config.teacher_kl_weight * teacher_loss
    total = total + config.gate_entropy_weight * gate_entropy_regularization

    components = {
        "mixture_binary_loss": float(mixture_binary.item()),
        "visual_aux_binary_loss": float(visual_aux.item()),
        "acoustic_aux_binary_loss": float(acoustic_aux.item()),
        "teacher_kl_loss": float(teacher_loss.item()),
        "gate_entropy_regularization": float(gate_entropy_regularization.item()),
    }
    return total, components


@torch.no_grad()
def evaluate_vision_v3_model(
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
        visual_probs = torch.sigmoid(outputs["visual_logit"]).cpu().numpy()
        fused_probs = torch.sigmoid(outputs["fused_logit"]).cpu().numpy()
        acoustic_probs = torch.sigmoid(outputs["acoustic_logit"]).cpu().numpy()
        gate_weights = outputs["gate_weights"].cpu().numpy()
        quality = batch["quality_features"].cpu().numpy()

        teacher_probs = {}
        if teachers:
            for name, teacher in teachers.items():
                teacher_probs[name] = teacher(batch).cpu().numpy()

        for row_index, subject_id in enumerate(raw_batch["subject_id"]):
            num_windows = int(raw_batch["window_mask"][row_index].sum().item()) if torch.is_tensor(raw_batch["window_mask"]) else 0
            row = {
                "subject_id": str(subject_id),
                "label": int(raw_batch["label_binary"][row_index]),
                "num_windows": num_windows,
                "probability": float(probabilities[row_index]),
                "prediction": int(probabilities[row_index] >= 0.5),
                "visual_probability": float(visual_probs[row_index]),
                "fused_probability": float(fused_probs[row_index]),
                "acoustic_probability": float(acoustic_probs[row_index]),
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

    teacher_alignment = {
        teacher_name: {"mae": float(np.mean(errors)), "num_subjects": int(len(errors))}
        for teacher_name, errors in teacher_mae.items()
    }
    return {
        "subject_predictions": frame,
        "metrics": metrics,
        "gate_stats": gate_stats,
        "quality_slice_metrics": quality_slice_metrics,
        "teacher_alignment": teacher_alignment,
    }


def persist_vision_v3_seed_artifacts(seed_dir: Path, seed: int, result: dict):
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


def train_vision_v3_one_seed(
    model: nn.Module,
    train_loader,
    dev_loader,
    test_loader,
    config: VisionV3TrainConfig,
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
            loss, components = _compute_loss(outputs, batch, teachers, config)
            loss.backward()
            optimizer.step()

            batch_size = batch["label_binary"].shape[0]
            running_loss += float(loss.item()) * batch_size
            running_items += batch_size
            for key, value in components.items():
                component_totals[key] = component_totals.get(key, 0.0) + value * batch_size

        dev_results = evaluate_vision_v3_model(model, dev_loader, device, teachers)
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
    dev_results = evaluate_vision_v3_model(model, dev_loader, device, teachers)
    test_results = evaluate_vision_v3_model(model, test_loader, device, teachers) if test_loader is not None else None
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_dev_score": best_dev_score,
        "dev_results": dev_results,
        "test_results": test_results,
        "state_dict": copy.deepcopy(model.state_dict()),
        "teacher_metadata": teacher_metadata,
    }
