"""Evaluation helpers for baseline experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.model.aggregation import aggregate_probabilities


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    ece = 0.0
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        mask = (y_prob >= left) & (y_prob < right if i < bins - 1 else y_prob <= right)
        if not np.any(mask):
            continue
        bin_acc = np.mean(y_true[mask])
        bin_conf = np.mean(y_prob[mask])
        ece += abs(bin_acc - bin_conf) * (np.sum(mask) / total)
    return float(ece)


def calibration_summary(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> dict:
    edges = np.linspace(0.0, 1.0, bins + 1)
    summary = {
        "num_examples": int(len(y_true)),
        "bins": bins,
        "ece": expected_calibration_error(y_true, y_prob, bins=bins),
        "brier_score": float(np.mean((y_prob - y_true) ** 2)),
        "bin_stats": [],
    }
    for i in range(bins):
        left, right = float(edges[i]), float(edges[i + 1])
        mask = (y_prob >= left) & (y_prob < right if i < bins - 1 else y_prob <= right)
        if not np.any(mask):
            summary["bin_stats"].append(
                {
                    "bin_index": i,
                    "left": left,
                    "right": right,
                    "count": 0,
                    "mean_confidence": None,
                    "mean_accuracy": None,
                }
            )
            continue
        summary["bin_stats"].append(
            {
                "bin_index": i,
                "left": left,
                "right": right,
                "count": int(np.sum(mask)),
                "mean_confidence": float(np.mean(y_prob[mask])),
                "mean_accuracy": float(np.mean(y_true[mask])),
            }
        )
    return summary


def build_subject_predictions(
    subject_ids: Iterable[str],
    labels: Iterable[int],
    probabilities: Iterable[float],
    method: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "subject_id": list(subject_ids),
            "label": list(labels),
            "window_probability": list(probabilities),
        }
    )
    grouped = []
    for subject_id, group in frame.groupby("subject_id", sort=True):
        grouped.append(
            {
                "subject_id": subject_id,
                "label": int(group["label"].iloc[0]),
                "num_windows": int(len(group)),
                "probability": aggregate_probabilities(group["window_probability"].values, method=method),
            }
        )
    result = pd.DataFrame(grouped).sort_values("subject_id").reset_index(drop=True)
    result["prediction"] = (result["probability"] >= 0.5).astype(int)
    return result


def compute_metrics(subject_predictions: pd.DataFrame) -> dict:
    y_true = subject_predictions["label"].to_numpy(dtype=int)
    y_prob = subject_predictions["probability"].to_numpy(dtype=float)
    y_pred = subject_predictions["prediction"].to_numpy(dtype=int)

    metrics = {
        "num_subjects": int(len(subject_predictions)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "ece": expected_calibration_error(y_true, y_prob),
        "brier_score": float(np.mean((y_prob - y_true) ** 2)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["auroc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def save_curves(subject_predictions: pd.DataFrame, output_dir: Path, prefix: str):
    y_true = subject_predictions["label"].to_numpy(dtype=int)
    y_prob = subject_predictions["probability"].to_numpy(dtype=float)
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    roc_path = output_dir / f"{prefix}_roc.png"
    pr_path = output_dir / f"{prefix}_pr.png"

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} ROC")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{prefix} PR")
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()


def save_confusion_matrix_csv(subject_predictions: pd.DataFrame, output_dir: Path, prefix: str):
    y_true = subject_predictions["label"].to_numpy(dtype=int)
    y_pred = subject_predictions["prediction"].to_numpy(dtype=int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    frame = pd.DataFrame(matrix, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    frame.to_csv(output_dir / f"{prefix}_confusion_matrix.csv")


def save_calibration_summary(subject_predictions: pd.DataFrame, output_dir: Path, prefix: str):
    y_true = subject_predictions["label"].to_numpy(dtype=int)
    y_prob = subject_predictions["probability"].to_numpy(dtype=float)
    summary = calibration_summary(y_true, y_prob)
    (output_dir / f"{prefix}_calibration.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_error_review(subject_predictions: pd.DataFrame, output_path: Path):
    false_positives = subject_predictions[(subject_predictions["label"] == 0) & (subject_predictions["prediction"] == 1)]
    false_negatives = subject_predictions[(subject_predictions["label"] == 1) & (subject_predictions["prediction"] == 0)]

    lines = [
        "# Error Review",
        "",
        f"- False positives: {len(false_positives)}",
        f"- False negatives: {len(false_negatives)}",
        f"- Subjects with <= 2 windows: {int(np.sum(subject_predictions['num_windows'] <= 2))}",
        "",
        "## Highest-confidence false positives",
    ]
    for _, row in false_positives.sort_values("probability", ascending=False).head(10).iterrows():
        lines.append(f"- {row['subject_id']}: p={row['probability']:.4f}, windows={int(row['num_windows'])}")

    lines.extend(["", "## Lowest-confidence false negatives"])
    for _, row in false_negatives.sort_values("probability", ascending=True).head(10).iterrows():
        lines.append(f"- {row['subject_id']}: p={row['probability']:.4f}, windows={int(row['num_windows'])}")

    sparse_failures = subject_predictions[
        (subject_predictions["num_windows"] <= 2) & (subject_predictions["prediction"] != subject_predictions["label"])
    ]
    lines.extend(["", "## Sparse-window failures"])
    if len(sparse_failures) == 0:
        lines.append("- None")
    else:
        for _, row in sparse_failures.sort_values("probability", ascending=False).head(10).iterrows():
            lines.append(
                f"- {row['subject_id']}: label={int(row['label'])}, pred={int(row['prediction'])}, "
                f"p={row['probability']:.4f}, windows={int(row['num_windows'])}"
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")
