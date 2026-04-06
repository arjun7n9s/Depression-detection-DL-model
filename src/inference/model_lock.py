"""Helpers for loading the repo's strategic locked model choices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.paths import PROJECT_ROOT


MODEL_LOCK_PATH = PROJECT_ROOT / "configs" / "strategic_model_lock.json"


def load_model_lock(path: Path | None = None) -> dict[str, Any]:
    lock_path = path or MODEL_LOCK_PATH
    return json.loads(lock_path.read_text(encoding="utf-8"))


def dataset_lock(dataset_name: str, path: Path | None = None) -> dict[str, Any]:
    lock = load_model_lock(path)
    try:
        return lock["datasets"][dataset_name]
    except KeyError as exc:
        raise KeyError(f"Dataset '{dataset_name}' not found in strategic model lock.") from exc


def benchmark_winner(dataset_name: str, path: Path | None = None) -> dict[str, Any]:
    return dataset_lock(dataset_name, path)["benchmark_winner"]


def preferred_direction(dataset_name: str, path: Path | None = None) -> dict[str, Any]:
    return dataset_lock(dataset_name, path)["preferred_direction"]
