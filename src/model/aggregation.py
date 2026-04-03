"""Window-to-subject aggregation helpers."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _safe_softmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def aggregate_probabilities(probabilities: Iterable[float], method: str, topk_fraction: float = 0.25) -> float:
    probs = np.asarray(list(probabilities), dtype=np.float64)
    if len(probs) == 0:
        return float("nan")
    if method == "mean":
        return float(np.mean(probs))
    if method == "topk":
        k = max(1, int(math.ceil(len(probs) * topk_fraction)))
        topk = np.sort(probs)[-k:]
        return float(np.mean(topk))
    if method == "attention":
        weights = _safe_softmax(probs)
        return float(np.sum(probs * weights))
    raise ValueError(f"Unknown aggregation method: {method}")
