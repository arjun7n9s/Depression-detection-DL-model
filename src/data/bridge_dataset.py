"""Datasets for training a D-Vlog bridge from live-compatible visual features to teacher spaces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.paths import DVLOG_BRIDGE_ROOT


STUDENT_MODALITY_KEYS = ["body_pose", "hand_pose", "gaze_blink", "face_affect_embed"]
TEACHER_TARGET_KEYS = ["visual_teacher", "acoustic_teacher"]


@dataclass
class BridgeDatasetConfig:
    bridge_root: Path = DVLOG_BRIDGE_ROOT
    fold: str = "train"
    window_size: int = 30
    stride: int = 15
    max_subjects: int | None = None
    max_windows: int | None = None
    include_acoustic_target: bool = True
    norm_stats: dict[str, dict[str, np.ndarray]] | None = None


def _window_ranges(seq_len: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    if seq_len <= window_size:
        return [(0, seq_len)]
    ranges = [(start, window_size) for start in range(0, seq_len - window_size + 1, stride)]
    final_start = seq_len - window_size
    if not ranges or ranges[-1][0] != final_start:
        ranges.append((final_start, window_size))
    return ranges


def _safe_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


class BridgeWindowDataset(Dataset):
    def __init__(self, config: BridgeDatasetConfig):
        self.config = config
        self.config.bridge_root = Path(self.config.bridge_root)
        self.include_acoustic_target = config.include_acoustic_target
        self.subjects: list[dict[str, Any]] = []
        self.samples: list[dict[str, Any]] = []
        self._load_subjects()
        if not self.subjects:
            raise RuntimeError(
                f"No bridge subjects found for fold={config.fold} under {config.bridge_root}. "
                "Run the bridge extractor first."
            )
        self.norm_stats = config.norm_stats or self._compute_norm_stats()
        self.input_dim = int(sum(self.subjects[0]["student_modalities"][key].shape[1] + 1 for key in STUDENT_MODALITY_KEYS))
        self.visual_dim = int(self.subjects[0]["teacher_targets"]["visual_teacher"].shape[1])
        self.acoustic_dim = int(self.subjects[0]["teacher_targets"]["acoustic_teacher"].shape[1])
        self._build_samples()

    def _iter_subject_dirs(self) -> list[Path]:
        subject_dirs = [path for path in self.config.bridge_root.iterdir() if path.is_dir() and path.name.isdigit()]
        return sorted(subject_dirs, key=lambda path: int(path.name))

    def _load_subjects(self):
        loaded = 0
        for subject_dir in self._iter_subject_dirs():
            metadata_path = subject_dir / "metadata.json"
            student_path = subject_dir / "student_modalities.npz"
            teacher_path = subject_dir / "teacher_targets.npz"
            if not metadata_path.exists() or not student_path.exists() or not teacher_path.exists():
                continue

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("fold") != self.config.fold:
                continue

            with np.load(student_path) as student_raw:
                student_modalities = {
                    key: student_raw[key].astype(np.float32) for key in STUDENT_MODALITY_KEYS
                }
                student_masks = {
                    key: student_raw[f"{key}_mask"].astype(bool) for key in STUDENT_MODALITY_KEYS
                }
            with np.load(teacher_path) as teacher_raw:
                teacher_targets = {
                    "visual_teacher": teacher_raw["visual_teacher"].astype(np.float32),
                    "acoustic_teacher": teacher_raw["acoustic_teacher"].astype(np.float32),
                }

            lengths = [len(values) for values in student_modalities.values()] + [len(values) for values in teacher_targets.values()]
            total_len = min(lengths) if lengths else 0
            if total_len <= 0:
                continue

            student_modalities = {key: values[:total_len] for key, values in student_modalities.items()}
            student_masks = {key: values[:total_len] for key, values in student_masks.items()}
            teacher_targets = {key: values[:total_len] for key, values in teacher_targets.items()}

            self.subjects.append(
                {
                    "subject_id": metadata["subject_id"],
                    "fold": metadata["fold"],
                    "student_modalities": student_modalities,
                    "student_masks": student_masks,
                    "teacher_targets": teacher_targets,
                    "seq_len": total_len,
                }
            )
            loaded += 1
            if self.config.max_subjects is not None and loaded >= self.config.max_subjects:
                break

    def _compute_norm_stats(self) -> dict[str, dict[str, np.ndarray]]:
        stats: dict[str, dict[str, np.ndarray]] = {}
        for key in STUDENT_MODALITY_KEYS:
            values = []
            for subject in self.subjects:
                mask = subject["student_masks"][key]
                valid = subject["student_modalities"][key][mask]
                if len(valid):
                    values.append(valid)
            if values:
                mean, std = _safe_stats(np.concatenate(values, axis=0))
                stats[key] = {"mean": mean, "std": std}

        visual_values = [subject["teacher_targets"]["visual_teacher"] for subject in self.subjects]
        visual_mean, visual_std = _safe_stats(np.concatenate(visual_values, axis=0))
        stats["visual_teacher"] = {"mean": visual_mean, "std": visual_std}

        if self.include_acoustic_target:
            acoustic_values = [subject["teacher_targets"]["acoustic_teacher"] for subject in self.subjects]
            acoustic_mean, acoustic_std = _safe_stats(np.concatenate(acoustic_values, axis=0))
            stats["acoustic_teacher"] = {"mean": acoustic_mean, "std": acoustic_std}
        return stats

    def _build_samples(self):
        samples = []
        for subject in self.subjects:
            ranges = _window_ranges(subject["seq_len"], self.config.window_size, self.config.stride)
            for start, win_len in ranges:
                samples.append(
                    {
                        "subject_id": subject["subject_id"],
                        "subject": subject,
                        "start": start,
                        "win_len": win_len,
                    }
                )
        if self.config.max_windows is not None:
            samples = samples[: self.config.max_windows]
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        subject = sample["subject"]
        start = sample["start"]
        win_len = sample["win_len"]
        pad = self.config.window_size - win_len

        modality_tensors = []
        for key in STUDENT_MODALITY_KEYS:
            values = subject["student_modalities"][key][start : start + win_len].copy()
            mask = subject["student_masks"][key][start : start + win_len].copy().astype(np.float32)
            if key in self.norm_stats:
                values = (values - self.norm_stats[key]["mean"]) / self.norm_stats[key]["std"]
            values = values * mask[:, None]
            combined = np.concatenate([values, mask[:, None]], axis=1)
            if pad > 0:
                combined = np.pad(combined, ((0, pad), (0, 0)), mode="constant")
            modality_tensors.append(combined.astype(np.float32))

        student_input = np.concatenate(modality_tensors, axis=1)
        window_mask = np.zeros((self.config.window_size,), dtype=bool)
        window_mask[:win_len] = True

        visual_target = subject["teacher_targets"]["visual_teacher"][start : start + win_len].copy()
        visual_target = (visual_target - self.norm_stats["visual_teacher"]["mean"]) / self.norm_stats["visual_teacher"]["std"]
        if pad > 0:
            visual_target = np.pad(visual_target, ((0, pad), (0, 0)), mode="constant")

        item = {
            "student_input": torch.from_numpy(student_input),
            "window_mask": torch.from_numpy(window_mask),
            "visual_target": torch.from_numpy(visual_target.astype(np.float32)),
            "subject_id": sample["subject_id"],
        }

        if self.include_acoustic_target:
            acoustic_target = subject["teacher_targets"]["acoustic_teacher"][start : start + win_len].copy()
            acoustic_target = (acoustic_target - self.norm_stats["acoustic_teacher"]["mean"]) / self.norm_stats["acoustic_teacher"]["std"]
            if pad > 0:
                acoustic_target = np.pad(acoustic_target, ((0, pad), (0, 0)), mode="constant")
            item["acoustic_target"] = torch.from_numpy(acoustic_target.astype(np.float32))
        return item


def make_bridge_datasets(
    bridge_root: Path,
    window_size: int,
    stride: int,
    include_acoustic_target: bool = True,
    max_subjects: int | None = None,
    max_windows: int | None = None,
) -> dict[str, BridgeWindowDataset]:
    train_config = BridgeDatasetConfig(
        bridge_root=bridge_root,
        fold="train",
        window_size=window_size,
        stride=stride,
        include_acoustic_target=include_acoustic_target,
        max_subjects=max_subjects,
        max_windows=max_windows,
    )
    train_dataset = BridgeWindowDataset(train_config)
    shared_stats = train_dataset.norm_stats

    valid_dataset = BridgeWindowDataset(
        BridgeDatasetConfig(
            bridge_root=bridge_root,
            fold="valid",
            window_size=window_size,
            stride=stride,
            include_acoustic_target=include_acoustic_target,
            max_subjects=max_subjects,
            max_windows=max_windows,
            norm_stats=shared_stats,
        )
    )
    test_dataset = BridgeWindowDataset(
        BridgeDatasetConfig(
            bridge_root=bridge_root,
            fold="test",
            window_size=window_size,
            stride=stride,
            include_acoustic_target=include_acoustic_target,
            max_subjects=max_subjects,
            max_windows=max_windows,
            norm_stats=shared_stats,
        )
    )
    return {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}


def make_bridge_dataloaders(
    bridge_root: Path,
    window_size: int,
    stride: int,
    batch_size: int,
    include_acoustic_target: bool = True,
    max_subjects: int | None = None,
    max_windows: int | None = None,
) -> tuple[dict[str, BridgeWindowDataset], dict[str, DataLoader]]:
    datasets = make_bridge_datasets(
        bridge_root=bridge_root,
        window_size=window_size,
        stride=stride,
        include_acoustic_target=include_acoustic_target,
        max_subjects=max_subjects,
        max_windows=max_windows,
    )
    loaders = {
        split: DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
        for split, dataset in datasets.items()
    }
    return datasets, loaders
