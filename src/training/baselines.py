"""Run unimodal baseline experiments for D-Vlog and E-DAIC."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.data.dvlog_dataset import create_dvlog_dataloaders
from src.data.edaic_dataset import create_edaic_dataloaders
from src.model.encoders import SequenceBinaryClassifier
from src.paths import RESULTS_ROOT
from src.training.trainer import TrainConfig, persist_seed_artifacts, set_seed, train_one_seed


BASELINE_INPUT_DIMS = {
    ("dvlog", "acoustic"): 25,
    ("dvlog", "visual"): 136,
    ("edaic", "acoustic"): 23,
    ("edaic", "visual"): 49,
}


def build_loaders(dataset_name: str, modality: str, batch_size: int, window_size: int, stride: int, min_valid_ratio: float):
    if dataset_name == "dvlog":
        loaders = create_dvlog_dataloaders(
            batch_size=batch_size,
            window_size=window_size,
            stride=stride,
            num_workers=0,
            balanced_sampling=True,
        )
        return {"train": loaders["train"], "dev": loaders["valid"], "test": loaders["test"]}
    if dataset_name == "edaic":
        return create_edaic_dataloaders(
            modality=modality,
            batch_size=batch_size,
            window_size=window_size,
            stride=stride,
            num_workers=0,
            balanced_sampling=True,
            min_valid_ratio=min_valid_ratio,
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")


def run_baseline(dataset_name: str, modality: str, config: TrainConfig, seeds: list[int], window_size: int, stride: int, min_valid_ratio: float):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / "baselines" / f"{dataset_name}_{modality}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = asdict(config) | {
        "dataset": dataset_name,
        "modality": modality,
        "window_size": window_size,
        "stride": stride,
        "min_valid_ratio": min_valid_ratio,
        "seeds": seeds,
        "input_dim": BASELINE_INPUT_DIMS[(dataset_name, modality)],
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    seed_results = []
    for seed in seeds:
        set_seed(seed)
        loaders = build_loaders(dataset_name, modality, config.batch_size, window_size, stride, min_valid_ratio)
        model = SequenceBinaryClassifier(
            input_dim=BASELINE_INPUT_DIMS[(dataset_name, modality)],
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        result = train_one_seed(model, loaders["train"], loaders["dev"], loaders["test"], modality, config)
        seed_results.append({"seed": seed, **result})

    aggregator_scores = {method: [] for method in config.aggregation_methods}
    for payload in seed_results:
        for method in config.aggregation_methods:
            aggregator_scores[method].append(payload["dev_results"][method]["metrics"]["macro_f1"])

    selected_method = max(
        aggregator_scores,
        key=lambda method: float(np.mean(aggregator_scores[method])),
    )

    summary = {
        "dataset": dataset_name,
        "modality": modality,
        "selected_aggregation": selected_method,
        "dev_macro_f1_mean": float(np.mean(aggregator_scores[selected_method])),
        "dev_macro_f1_std": float(np.std(aggregator_scores[selected_method])),
        "seeds": [],
    }

    test_macro_scores = []
    test_weighted_scores = []
    for payload in seed_results:
        seed_dir = run_dir / f"seed_{payload['seed']}"
        persist_seed_artifacts(seed_dir, payload["seed"], payload, selected_method)
        test_metrics = payload["test_results"][selected_method]["metrics"]
        dev_metrics = payload["dev_results"][selected_method]["metrics"]
        summary["seeds"].append(
            {
                "seed": payload["seed"],
                "best_epoch": payload["best_epoch"],
                "best_dev_method": payload["best_method"],
                "dev_metrics": dev_metrics,
                "test_metrics": test_metrics,
            }
        )
        test_macro_scores.append(test_metrics["macro_f1"])
        test_weighted_scores.append(test_metrics["weighted_f1"])

    summary["test_macro_f1_mean"] = float(np.mean(test_macro_scores))
    summary["test_macro_f1_std"] = float(np.std(test_macro_scores))
    summary["test_weighted_f1_mean"] = float(np.mean(test_weighted_scores))
    summary["test_weighted_f1_std"] = float(np.std(test_weighted_scores))

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir, summary


def main():
    parser = argparse.ArgumentParser(description="Run unimodal baseline experiments.")
    parser.add_argument("--dataset", choices=["dvlog", "edaic", "all"], default="all")
    parser.add_argument("--modality", choices=["acoustic", "visual", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--min-valid-ratio", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = TrainConfig(
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
    )

    jobs = []
    datasets = ["dvlog", "edaic"] if args.dataset == "all" else [args.dataset]
    modalities = ["acoustic", "visual"] if args.modality == "all" else [args.modality]
    for dataset_name in datasets:
        for modality in modalities:
            jobs.append((dataset_name, modality))

    for dataset_name, modality in jobs:
        print(f"\n=== Running baseline: {dataset_name} / {modality} ===")
        run_dir, summary = run_baseline(
            dataset_name=dataset_name,
            modality=modality,
            config=config,
            seeds=args.seeds,
            window_size=args.window_size,
            stride=args.stride,
            min_valid_ratio=args.min_valid_ratio,
        )
        print(f"Saved to: {run_dir}")
        print(
            f"Selected aggregation: {summary['selected_aggregation']} | "
            f"Dev macro F1 {summary['dev_macro_f1_mean']:.4f} +/- {summary['dev_macro_f1_std']:.4f} | "
            f"Test macro F1 {summary['test_macro_f1_mean']:.4f} +/- {summary['test_macro_f1_std']:.4f}"
        )


if __name__ == "__main__":
    main()
