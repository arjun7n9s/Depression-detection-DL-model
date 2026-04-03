"""Reusable baseline training loop."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.training.evaluate import (
    build_subject_predictions,
    compute_metrics,
    save_calibration_summary,
    save_confusion_matrix_csv,
    save_curves,
    write_error_review,
)


@dataclass
class TrainConfig:
    batch_size: int = 64
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 3
    patience: int = 2
    device: str = "cuda"
    aggregation_methods: tuple[str, ...] = ("mean", "topk", "attention")
    loss_name: str = "bce"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    use_pos_weight: bool = False


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        alpha_t = torch.where(targets > 0.5, self.alpha, 1.0 - self.alpha)
        focal = alpha_t * torch.pow(1.0 - pt, self.gamma) * bce
        return focal.mean()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _extract_inputs(batch: dict, modality: str) -> torch.Tensor:
    if "inputs" in batch:
        return batch["inputs"]
    return batch[modality]


def _build_criterion(config: TrainConfig, train_loader, device: torch.device) -> nn.Module:
    if config.loss_name == "focal":
        return BinaryFocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)

    kwargs = {}
    if config.use_pos_weight:
        labels = train_loader.dataset.get_labels()
        positives = sum(labels)
        negatives = max(len(labels) - positives, 1)
        pos_weight = negatives / max(positives, 1)
        kwargs["pos_weight"] = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(**kwargs)


@torch.no_grad()
def collect_window_predictions(model, loader, modality: str, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        inputs = _extract_inputs(batch, modality).to(device)
        logits = model(inputs)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        labels = batch["label_binary"].cpu().numpy()
        subject_ids = batch["subject_id"]
        for subject_id, label, probability in zip(subject_ids, labels, probabilities, strict=False):
            rows.append(
                {
                    "subject_id": str(subject_id),
                    "label": int(label),
                    "window_probability": float(probability),
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_model(model, loader, modality: str, device: torch.device, methods: tuple[str, ...]) -> dict:
    window_predictions = collect_window_predictions(model, loader, modality, device)
    results = {}
    for method in methods:
        subject_predictions = build_subject_predictions(
            window_predictions["subject_id"],
            window_predictions["label"],
            window_predictions["window_probability"],
            method,
        )
        results[method] = {
            "subject_predictions": subject_predictions,
            "metrics": compute_metrics(subject_predictions),
        }
    return results


def train_one_seed(
    model: nn.Module,
    train_loader,
    dev_loader,
    test_loader,
    modality: str,
    config: TrainConfig,
):
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = _build_criterion(config, train_loader, device)

    best_state = copy.deepcopy(model.state_dict())
    best_dev_score = float("-inf")
    best_epoch = 0
    best_method = config.aggregation_methods[0]
    epochs_without_improvement = 0

    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_items = 0
        for batch in train_loader:
            inputs = _extract_inputs(batch, modality).to(device)
            labels = batch["label_binary"].float().to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = len(labels)
            running_loss += float(loss.item()) * batch_size
            running_items += batch_size

        train_loss = running_loss / max(running_items, 1)
        dev_results = evaluate_model(model, dev_loader, modality, device, config.aggregation_methods)
        method_scores = {method: payload["metrics"]["macro_f1"] for method, payload in dev_results.items()}
        epoch_best_method = max(method_scores, key=method_scores.get)
        epoch_best_score = method_scores[epoch_best_method]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_macro_f1": epoch_best_score,
                "best_method": epoch_best_method,
            }
        )

        if epoch_best_score > best_dev_score:
            best_dev_score = epoch_best_score
            best_method = epoch_best_method
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    model.load_state_dict(best_state)
    dev_results = evaluate_model(model, dev_loader, modality, device, config.aggregation_methods)
    test_results = None
    if test_loader is not None:
        test_results = evaluate_model(model, test_loader, modality, device, config.aggregation_methods)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_method": best_method,
        "best_dev_score": best_dev_score,
        "dev_results": dev_results,
        "test_results": test_results,
        "state_dict": copy.deepcopy(model.state_dict()),
    }


def _write_split_artifacts(seed_dir: Path, split_name: str, payload: dict):
    predictions = payload["subject_predictions"]
    metrics = payload["metrics"]
    predictions.to_csv(seed_dir / f"{split_name}_subject_predictions.csv", index=False)
    (seed_dir / f"{split_name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix_csv(predictions, seed_dir, split_name)
    save_calibration_summary(predictions, seed_dir, split_name)
    save_curves(predictions, seed_dir, split_name)


def persist_seed_artifacts(seed_dir: Path, seed: int, result: dict, selected_method: str):
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "history.json").write_text(
        json.dumps(result["history"], indent=2),
        encoding="utf-8",
    )
    (seed_dir / "seed_summary.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "best_method": result["best_method"],
                "best_dev_score": result["best_dev_score"],
                "selected_method": selected_method,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_split_artifacts(seed_dir, "dev", result["dev_results"][selected_method])
    if result["test_results"] is not None:
        _write_split_artifacts(seed_dir, "test", result["test_results"][selected_method])
        write_error_review(result["test_results"][selected_method]["subject_predictions"], seed_dir / "error_review.md")
    else:
        write_error_review(result["dev_results"][selected_method]["subject_predictions"], seed_dir / "error_review.md")
