"""Vision V3 subject-level datasets."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.dataset_audit import DVLOG_FEATURES_DIR, DVLOG_LABELS_CSV
from src.paths import PROCESSED_ROOT


DVLOG_VISION_V3_DIR = PROCESSED_ROOT / "dvlog_video_v3"

VISION_V3_BUNDLES: dict[str, list[str]] = {
    "existing_visual_only": ["visual"],
    "visual_pose": ["visual", "body_pose"],
    "visual_pose_hands_gaze": ["visual", "body_pose", "hand_pose", "gaze_blink"],
    "visual_full_aux_audio": ["visual", "face_affect_embed", "body_pose", "hand_pose", "gaze_blink", "acoustic"],
}

VISION_V3_MODALITY_GROUPS = {
    "visual": ["visual", "face_affect_embed", "body_pose", "hand_pose", "gaze_blink"],
    "acoustic": ["acoustic"],
}


def _window_ranges(seq_len: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    if seq_len <= window_size:
        return [(0, seq_len)]
    ranges = [(start, window_size) for start in range(0, seq_len - window_size + 1, stride)]
    final_start = seq_len - window_size
    if not ranges or ranges[-1][0] != final_start:
        ranges.append((final_start, window_size))
    return ranges


def _safe_scalar_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


class VisionV3SubjectDataset(Dataset):
    def __init__(
        self,
        fold: str,
        bundle_name: str,
        window_size: int,
        stride: int,
        min_valid_ratio: float = 0.5,
        norm_stats: Optional[dict[str, dict[str, np.ndarray]]] = None,
        max_seq_len: int = 3600,
    ):
        if bundle_name not in VISION_V3_BUNDLES:
            raise ValueError(f"Unsupported Vision V3 bundle: {bundle_name}")
        self.fold = fold
        self.bundle_name = bundle_name
        self.modalities = list(VISION_V3_BUNDLES[bundle_name])
        self.window_size = window_size
        self.stride = stride
        self.min_valid_ratio = min_valid_ratio
        self.max_seq_len = max_seq_len
        self.subjects: list[dict] = []
        self._load_subjects()
        if not self.subjects:
            raise RuntimeError(
                f"No Vision V3 subjects loaded for fold={fold}, bundle={bundle_name}. "
                "Run the D-Vlog video extractor first for bundles that require raw-video features."
            )
        self.norm_stats = norm_stats or self._compute_norm_stats()
        self.modality_dims = {
            key: int(self.subjects[0]["modalities"][key].shape[1]) for key in self.modalities
        }

    def _load_subjects(self):
        with open(DVLOG_LABELS_CSV, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader if row["fold"] == self.fold]

        for row in rows:
            subject_id = str(int(row["index"]))
            subject_dir = DVLOG_FEATURES_DIR / subject_id
            extracted_dir = DVLOG_VISION_V3_DIR / subject_id

            modality_arrays: dict[str, np.ndarray] = {}
            modality_masks: dict[str, np.ndarray] = {}
            lengths = []

            if "visual" in self.modalities:
                visual_path = subject_dir / f"{subject_id}_visual.npy"
                if not visual_path.exists():
                    continue
                visual = np.load(str(visual_path)).astype(np.float32)[: self.max_seq_len]
                modality_arrays["visual"] = visual
                modality_masks["visual"] = np.ones((len(visual),), dtype=bool)
                lengths.append(len(visual))

            if "acoustic" in self.modalities:
                acoustic_path = subject_dir / f"{subject_id}_acoustic.npy"
                if not acoustic_path.exists():
                    continue
                acoustic = np.load(str(acoustic_path)).astype(np.float32)[: self.max_seq_len]
                modality_arrays["acoustic"] = acoustic
                modality_masks["acoustic"] = np.ones((len(acoustic),), dtype=bool)
                lengths.append(len(acoustic))

            extracted_specs = {
                "face_affect_embed": ("face_affect_embed.npy", "face_affect_embed_mask.npy"),
                "body_pose": ("body_pose.npy", "body_pose_mask.npy"),
                "hand_pose": ("hand_pose.npy", "hand_pose_mask.npy"),
                "gaze_blink": ("gaze_blink.npy", "gaze_blink_mask.npy"),
            }
            for modality, (array_name, mask_name) in extracted_specs.items():
                if modality not in self.modalities:
                    continue
                array_path = extracted_dir / array_name
                mask_path = extracted_dir / mask_name
                if not array_path.exists() or not mask_path.exists():
                    continue
                values = np.load(str(array_path)).astype(np.float32)[: self.max_seq_len]
                mask = np.load(str(mask_path)).astype(bool)[: self.max_seq_len]
                modality_arrays[modality] = values
                modality_masks[modality] = mask
                lengths.append(len(values))

            if set(modality_arrays) != set(self.modalities) or not lengths:
                continue

            total_len = min(lengths)
            if total_len <= 0:
                continue

            modality_arrays = {key: value[:total_len] for key, value in modality_arrays.items()}
            modality_masks = {key: value[:total_len] for key, value in modality_masks.items()}
            shared_valid = np.zeros((total_len,), dtype=bool)
            for values in modality_masks.values():
                shared_valid |= values

            self.subjects.append(
                {
                    "subject_id": subject_id,
                    "label_binary": 1 if row["label"] == "depression" else 0,
                    "modalities": modality_arrays,
                    "modality_masks": modality_masks,
                    "shared_valid": shared_valid,
                }
            )

    def _compute_norm_stats(self) -> dict[str, dict[str, np.ndarray]]:
        stats = {}
        for modality in self.modalities:
            values = []
            for subject in self.subjects:
                valid = subject["modalities"][modality][subject["modality_masks"][modality]]
                if len(valid):
                    values.append(valid)
            if not values:
                continue
            stacked = np.concatenate(values, axis=0)
            mean, std = _safe_scalar_stats(stacked)
            stats[modality] = {"mean": mean, "std": std}
        return stats

    def __len__(self) -> int:
        return len(self.subjects)

    def _window_quality_vector(self, subject: dict, start: int, win_len: int, num_windows: int) -> np.ndarray:
        shared_window = subject["shared_valid"][start : start + win_len]
        padding_ratio = float(self.window_size - win_len) / float(self.window_size)
        quality = [
            float(np.mean(shared_window)) if len(shared_window) else 0.0,
            padding_ratio,
            float(start / max(1, len(subject["shared_valid"]) - 1)),
            float(np.log1p(num_windows)),
        ]
        for modality in self.modalities:
            values = subject["modalities"][modality][start : start + win_len]
            mask = subject["modality_masks"][modality][start : start + win_len]
            valid_values = values[mask]
            if len(valid_values) == 0:
                quality.extend([0.0, 0.0, 0.0])
            else:
                timestep_nonzero = np.any(np.abs(valid_values) > 0, axis=1)
                quality.extend(
                    [
                        float(np.mean(timestep_nonzero)),
                        float(np.mean(np.abs(valid_values))),
                        float(np.mean(np.std(valid_values, axis=0, dtype=np.float64))),
                    ]
                )
        return np.asarray(quality, dtype=np.float32)

    def __getitem__(self, idx: int) -> dict:
        subject = self.subjects[idx]
        seq_len = len(subject["shared_valid"])
        ranges = _window_ranges(seq_len, self.window_size, self.stride)
        windows = []
        for window_index, (start, win_len) in enumerate(ranges):
            valid_ratio = float(np.mean(subject["shared_valid"][start : start + win_len]))
            if valid_ratio >= self.min_valid_ratio:
                windows.append((window_index, start, win_len))
        if not windows:
            windows = [(0, 0, min(seq_len, self.window_size))]

        packed_modalities: dict[str, list[np.ndarray]] = {key: [] for key in self.modalities}
        packed_masks: dict[str, list[np.ndarray]] = {key: [] for key in self.modalities}
        window_quality = []

        for _, start, win_len in windows:
            for modality in self.modalities:
                values = subject["modalities"][modality][start : start + win_len].copy()
                mask = subject["modality_masks"][modality][start : start + win_len].copy().astype(bool)
                if modality in self.norm_stats:
                    values = (values - self.norm_stats[modality]["mean"]) / self.norm_stats[modality]["std"]
                if win_len < self.window_size:
                    pad = self.window_size - win_len
                    values = np.pad(values, ((0, pad), (0, 0)), mode="constant")
                    mask = np.pad(mask, (0, pad), mode="constant", constant_values=False)
                packed_modalities[modality].append(values.astype(np.float32))
                packed_masks[modality].append(mask.astype(bool))
            window_quality.append(self._window_quality_vector(subject, start, win_len, len(windows)))

        quality_arr = np.stack(window_quality, axis=0)
        return {
            "modalities": {key: torch.from_numpy(np.stack(values, axis=0)) for key, values in packed_modalities.items()},
            "modality_masks": {key: torch.from_numpy(np.stack(values, axis=0)).bool() for key, values in packed_masks.items()},
            "window_quality_features": torch.from_numpy(quality_arr),
            "quality_features": torch.from_numpy(quality_arr.mean(axis=0)),
            "window_mask": torch.ones((len(windows),), dtype=torch.bool),
            "label_binary": torch.tensor(subject["label_binary"], dtype=torch.long),
            "label_masks": {"binary": torch.tensor(True, dtype=torch.bool)},
            "subject_id": subject["subject_id"],
            "dataset_source": "dvlog",
            "fold": self.fold,
            "window_idx": torch.arange(len(windows), dtype=torch.long),
        }

    def get_labels(self) -> list[int]:
        return [int(subject["label_binary"]) for subject in self.subjects]


def _collate_subject_batch(samples: list[dict]) -> dict:
    max_windows = max(int(sample["window_mask"].shape[0]) for sample in samples)
    modality_keys = list(samples[0]["modalities"].keys())
    quality_dim = int(samples[0]["window_quality_features"].shape[-1])

    batch_modalities = {}
    batch_masks = {}
    for key in modality_keys:
        tensors = []
        masks = []
        for sample in samples:
            values = sample["modalities"][key]
            mask = sample["modality_masks"][key]
            pad_windows = max_windows - values.shape[0]
            if pad_windows > 0:
                values = torch.cat(
                    [values, torch.zeros((pad_windows, values.shape[1], values.shape[2]), dtype=values.dtype)],
                    dim=0,
                )
                mask = torch.cat([mask, torch.zeros((pad_windows, mask.shape[1]), dtype=torch.bool)], dim=0)
            tensors.append(values)
            masks.append(mask)
        batch_modalities[key] = torch.stack(tensors, dim=0)
        batch_masks[key] = torch.stack(masks, dim=0)

    window_quality = []
    window_masks = []
    for sample in samples:
        values = sample["window_quality_features"]
        mask = sample["window_mask"]
        pad_windows = max_windows - values.shape[0]
        if pad_windows > 0:
            values = torch.cat([values, torch.zeros((pad_windows, quality_dim), dtype=values.dtype)], dim=0)
            mask = torch.cat([mask, torch.zeros((pad_windows,), dtype=torch.bool)], dim=0)
        window_quality.append(values)
        window_masks.append(mask)

    return {
        "modalities": batch_modalities,
        "modality_masks": batch_masks,
        "window_quality_features": torch.stack(window_quality, dim=0),
        "quality_features": torch.stack([sample["quality_features"] for sample in samples], dim=0),
        "window_mask": torch.stack(window_masks, dim=0),
        "label_binary": torch.stack([sample["label_binary"] for sample in samples], dim=0),
        "label_masks": {"binary": torch.stack([sample["label_masks"]["binary"] for sample in samples], dim=0)},
        "subject_id": [sample["subject_id"] for sample in samples],
        "dataset_source": [sample["dataset_source"] for sample in samples],
        "fold": [sample["fold"] for sample in samples],
        "window_idx": [sample["window_idx"] for sample in samples],
    }


def create_vision_v3_dataloaders(
    bundle_name: str,
    batch_size: int,
    window_size: int,
    stride: int,
    num_workers: int = 0,
    balanced_sampling: bool = True,
    min_valid_ratio: float = 0.5,
) -> dict[str, DataLoader]:
    train_ds = VisionV3SubjectDataset(
        fold="train",
        bundle_name=bundle_name,
        window_size=window_size,
        stride=stride,
        min_valid_ratio=min_valid_ratio,
    )
    norm_stats = train_ds.norm_stats
    dev_ds = VisionV3SubjectDataset(
        fold="valid",
        bundle_name=bundle_name,
        window_size=window_size,
        stride=stride,
        min_valid_ratio=min_valid_ratio,
        norm_stats=norm_stats,
    )
    test_ds = VisionV3SubjectDataset(
        fold="test",
        bundle_name=bundle_name,
        window_size=window_size,
        stride=stride,
        min_valid_ratio=min_valid_ratio,
        norm_stats=norm_stats,
    )

    sampler = None
    shuffle = True
    if balanced_sampling:
        labels = train_ds.get_labels()
        counts = [max(labels.count(0), 1), max(labels.count(1), 1)]
        weights = [1.0 / counts[label] for label in labels]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False

    quality_dim = int(train_ds[0]["quality_features"].shape[0])
    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=_collate_subject_batch,
        ),
        "dev": DataLoader(
            dev_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_collate_subject_batch,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_collate_subject_batch,
        ),
        "metadata": {
            "modality_dims": train_ds.modality_dims,
            "modalities": train_ds.modalities,
            "modality_groups": {
                "visual": [key for key in train_ds.modalities if key in VISION_V3_MODALITY_GROUPS["visual"]],
                "acoustic": [key for key in train_ds.modalities if key in VISION_V3_MODALITY_GROUPS["acoustic"]],
            },
            "bundle_name": bundle_name,
            "quality_dim": quality_dim,
        },
    }
