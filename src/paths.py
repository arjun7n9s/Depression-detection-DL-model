"""Shared path helpers for local and public-friendly project layouts."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL_ROOT = Path("D:/DL-Datasets")


def _path_from_env(name: str) -> Path | None:
    raw = os.getenv(name)
    if not raw:
        return None
    return Path(raw).expanduser()


def _first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


EXTERNAL_DATA_ROOT = _path_from_env("MINDSENSE_EXTERNAL_DATA_ROOT") or _first_existing(
    DEFAULT_EXTERNAL_ROOT,
    PROJECT_ROOT / "artifacts",
)

PROCESSED_ROOT = _path_from_env("MINDSENSE_PROCESSED_ROOT") or _first_existing(
    EXTERNAL_DATA_ROOT / "processed",
    PROJECT_ROOT / "artifacts" / "processed",
)

DVLOG_VIDEOS_DIR = _path_from_env("MINDSENSE_DVLOG_VIDEOS_DIR") or _first_existing(
    EXTERNAL_DATA_ROOT / "dvlog_videos",
    PROJECT_ROOT / "data" / "dvlog_videos",
)

RESULTS_ROOT = PROJECT_ROOT / "results"


def repo_relative(path: Path) -> str:
    """Render paths relative to the repository when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")
