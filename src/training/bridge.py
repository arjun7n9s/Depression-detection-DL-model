"""Training CLI for the D-Vlog bridge projection model."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.data.bridge_dataset import make_bridge_dataloaders
from src.model.bridge import BridgeProjectionModel
from src.paths import DVLOG_BRIDGE_ROOT, PROJECT_ROOT, repo_relative


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "bridge" / "dvlog_bridge_v1"


@dataclass
class BridgeTrainConfig:
    bridge_root: str = str(DVLOG_BRIDGE_ROOT)
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    seed: int = 7
    batch_size: int = 32
    window_size: int = 30
    stride: int = 15
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 12
    patience: int = 3
    device: str = "cuda"
    include_acoustic_target: bool = True
    acoustic_loss_weight: float = 0.15
    cosine_loss_weight: float = 0.10
    max_subjects: int | None = None
    max_windows: int | None = None


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_config(path: Path) -> BridgeTrainConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BridgeTrainConfig(**payload)


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    squared = (prediction - target) ** 2 * weights
    denom = weights.sum().clamp_min(1.0) * prediction.shape[-1]
    return squared.sum() / denom


def _masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    absolute = (prediction - target).abs() * weights
    denom = weights.sum().clamp_min(1.0) * prediction.shape[-1]
    return absolute.sum() / denom


def _masked_cosine(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat_mask = mask.reshape(-1)
    if not torch.any(flat_mask):
        return prediction.new_tensor(0.0)
    pred_flat = prediction.reshape(-1, prediction.shape[-1])[flat_mask]
    target_flat = target.reshape(-1, target.shape[-1])[flat_mask]
    cosine = nn.functional.cosine_similarity(pred_flat, target_flat, dim=-1)
    return cosine.mean()


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    config: BridgeTrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    visual_mse = _masked_mse(outputs["visual_pred"], batch["visual_target"], batch["window_mask"])
    visual_cosine = _masked_cosine(outputs["visual_pred"], batch["visual_target"], batch["window_mask"])
    total = visual_mse + config.cosine_loss_weight * (1.0 - visual_cosine)

    acoustic_mse = outputs["visual_pred"].new_tensor(0.0)
    if config.include_acoustic_target and "acoustic_pred" in outputs and "acoustic_target" in batch:
        acoustic_mse = _masked_mse(outputs["acoustic_pred"], batch["acoustic_target"], batch["window_mask"])
        total = total + config.acoustic_loss_weight * acoustic_mse

    return total, {
        "visual_mse": float(visual_mse.item()),
        "visual_cosine": float(visual_cosine.item()),
        "acoustic_mse": float(acoustic_mse.item()),
    }


@torch.no_grad()
def evaluate_bridge(
    model: BridgeProjectionModel,
    loader,
    device: torch.device,
    include_acoustic_target: bool,
) -> dict[str, Any]:
    model.eval()
    visual_mse_values = []
    visual_mae_values = []
    visual_cosine_values = []
    acoustic_mse_values = []
    acoustic_mae_values = []
    batch_count = 0

    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        outputs = model(batch["student_input"])
        visual_mse_values.append(float(_masked_mse(outputs["visual_pred"], batch["visual_target"], batch["window_mask"]).item()))
        visual_mae_values.append(float(_masked_mae(outputs["visual_pred"], batch["visual_target"], batch["window_mask"]).item()))
        visual_cosine_values.append(float(_masked_cosine(outputs["visual_pred"], batch["visual_target"], batch["window_mask"]).item()))
        if include_acoustic_target and "acoustic_pred" in outputs and "acoustic_target" in batch:
            acoustic_mse_values.append(float(_masked_mse(outputs["acoustic_pred"], batch["acoustic_target"], batch["window_mask"]).item()))
            acoustic_mae_values.append(float(_masked_mae(outputs["acoustic_pred"], batch["acoustic_target"], batch["window_mask"]).item()))
        batch_count += 1

    metrics = {
        "visual_mse": float(np.mean(visual_mse_values)) if visual_mse_values else math.inf,
        "visual_mae": float(np.mean(visual_mae_values)) if visual_mae_values else math.inf,
        "visual_cosine": float(np.mean(visual_cosine_values)) if visual_cosine_values else 0.0,
        "num_batches": batch_count,
    }
    if include_acoustic_target and acoustic_mse_values:
        metrics["acoustic_mse"] = float(np.mean(acoustic_mse_values))
        metrics["acoustic_mae"] = float(np.mean(acoustic_mae_values))
    return metrics


def train_bridge(config: BridgeTrainConfig) -> dict[str, Any]:
    _set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")

    datasets, loaders = make_bridge_dataloaders(
        bridge_root=Path(config.bridge_root),
        window_size=config.window_size,
        stride=config.stride,
        batch_size=config.batch_size,
        include_acoustic_target=config.include_acoustic_target,
        max_subjects=config.max_subjects,
        max_windows=config.max_windows,
    )

    train_dataset = datasets["train"]
    model = BridgeProjectionModel(
        input_dim=train_dataset.input_dim,
        visual_dim=train_dataset.visual_dim,
        acoustic_dim=train_dataset.acoustic_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        include_acoustic_target=config.include_acoustic_target,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_valid_score = float("inf")
    best_epoch = 0
    history = []
    patience_remaining = config.patience

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        train_visual_mse = []
        for raw_batch in loaders["train"]:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["student_input"])
            loss, components = _compute_loss(outputs, batch, config)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_visual_mse.append(components["visual_mse"])

        valid_metrics = evaluate_bridge(model, loaders["valid"], device, config.include_acoustic_target)
        test_metrics = evaluate_bridge(model, loaders["test"], device, config.include_acoustic_target)
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else math.inf,
            "train_visual_mse": float(np.mean(train_visual_mse)) if train_visual_mse else math.inf,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        }
        history.append(epoch_record)

        valid_score = valid_metrics["visual_mse"]
        if valid_score < best_valid_score:
            best_valid_score = valid_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_remaining = config.patience
        else:
            patience_remaining -= 1
            if patience_remaining <= 0:
                break

    model.load_state_dict(best_state)
    final_valid_metrics = evaluate_bridge(model, loaders["valid"], device, config.include_acoustic_target)
    final_test_metrics = evaluate_bridge(model, loaders["test"], device, config.include_acoustic_target)

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "input_dim": train_dataset.input_dim,
        "visual_dim": train_dataset.visual_dim,
        "acoustic_dim": train_dataset.acoustic_dim,
        "norm_stats": {
            key: {
                "mean": value["mean"].tolist(),
                "std": value["std"].tolist(),
            }
            for key, value in train_dataset.norm_stats.items()
        },
    }
    torch.save(checkpoint_payload, output_root / "checkpoint.pt")
    (output_root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "best_epoch": best_epoch,
        "best_valid_score": best_valid_score,
        "valid_metrics": final_valid_metrics,
        "test_metrics": final_test_metrics,
        "dataset": {
            split: {
                "subjects": len(dataset.subjects),
                "windows": len(dataset),
            }
            for split, dataset in datasets.items()
        },
        "config": asdict(config),
        "artifacts": {
            "checkpoint": repo_relative(output_root / "checkpoint.pt"),
            "history": repo_relative(output_root / "history.json"),
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the D-Vlog bridge projection model.")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    config = _load_config(args.config)
    summary = train_bridge(config)
    print("=" * 60)
    print("D-VLOG BRIDGE TRAINING")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
