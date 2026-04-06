"""Subject-level multimodal datasets for Fusion V2."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.dataset_audit import DVLOG_FEATURES_DIR, DVLOG_LABELS_CSV, EDAIC_DETAILED_LABELS
from src.data.edaic_dataset import _phq_total, _resample_to_seconds
from src.paths import PROCESSED_ROOT


EDAIC_PROCESSED_DIR = PROCESSED_ROOT / "edaic"

FUSION_V2_BUNDLES: dict[str, dict[str, list[str]]] = {
    "dvlog": {
        "current_av": ["acoustic", "visual"],
    },
    "edaic": {
        "current_av": ["pose_gaze_aus", "acoustic"],
        "rich_av": ["pose_gaze_aus", "acoustic", "mfcc", "cnn_resnet"],
    },
}

FUSION_V2_MODALITY_GROUPS: dict[str, dict[str, list[str]]] = {
    "dvlog": {
        "acoustic": ["acoustic"],
        "visual": ["visual"],
    },
    "edaic": {
        "acoustic": ["acoustic", "mfcc"],
        "visual": ["pose_gaze_aus", "cnn_resnet"],
    },
}


def _severity_label(phq_total: float) -> int:
    if math.isnan(phq_total):
        return -1
    if phq_total <= 4:
        return 0
    if phq_total <= 9:
        return 1
    if phq_total <= 14:
        return 2
    if phq_total <= 19:
        return 3
    return 4


def _safe_scalar_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def _window_ranges(seq_len: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    if seq_len <= window_size:
        return [(0, seq_len)]
    ranges = [(start, window_size) for start in range(0, seq_len - window_size + 1, stride)]
    final_start = seq_len - window_size
    if not ranges or ranges[-1][0] != final_start:
        ranges.append((final_start, window_size))
    return ranges


def _resample_scalar(values: np.ndarray, timestamps: np.ndarray, total_secs: int) -> np.ndarray:
    if len(values) == 0 or len(timestamps) == 0 or total_secs <= 0:
        return np.zeros((0,), dtype=np.float32)
    bucketed = [[] for _ in range(total_secs)]
    for value, ts in zip(values, timestamps, strict=False):
        sec = int(np.floor(float(ts)))
        if 0 <= sec < total_secs:
            bucketed[sec].append(float(value))
    output = np.zeros((total_secs,), dtype=np.float32)
    for sec, items in enumerate(bucketed):
        if items:
            output[sec] = float(np.mean(items))
    return output


def _align_modalities(
    features: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    extras: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    lengths = [len(values) for values in features.values()]
    if not lengths:
        raise ValueError("No modality features available to align.")
    total_secs = min(lengths)
    aligned_features = {key: values[:total_secs] for key, values in features.items()}
    aligned_masks = {key: values[:total_secs].astype(bool) for key, values in masks.items()}
    aligned_extras = {}
    for key, values in extras.items():
        aligned_extras[key] = values[:total_secs]
    shared_valid = np.ones((total_secs,), dtype=bool)
    for values in aligned_masks.values():
        shared_valid &= values
    return aligned_features, aligned_masks, aligned_extras, shared_valid


class FusionV2SubjectDataset(Dataset):
    """Subject-level multimodal dataset with window packing and quality features."""

    def __init__(
        self,
        dataset_name: str,
        fold: str,
        bundle_name: str,
        window_size: int,
        stride: int,
        min_valid_ratio: float = 0.5,
        normalization_source: str = "train",
        norm_stats: Optional[dict[str, dict[str, np.ndarray]]] = None,
        max_seq_len: int = 3600,
    ):
        if dataset_name not in FUSION_V2_BUNDLES:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        if bundle_name not in FUSION_V2_BUNDLES[dataset_name]:
            raise ValueError(f"Unsupported bundle '{bundle_name}' for dataset '{dataset_name}'.")
        self.dataset_name = dataset_name
        self.fold = fold
        self.bundle_name = bundle_name
        self.modalities = list(FUSION_V2_BUNDLES[dataset_name][bundle_name])
        self.window_size = window_size
        self.stride = stride
        self.min_valid_ratio = min_valid_ratio
        self.normalization_source = normalization_source
        self.max_seq_len = max_seq_len
        self.subjects: list[dict] = []

        if dataset_name == "dvlog":
            self._load_dvlog_subjects()
        else:
            self._load_edaic_subjects()

        if not self.subjects:
            raise RuntimeError(
                f"No subjects loaded for dataset={dataset_name} fold={fold} bundle={bundle_name}. "
                "If this is E-DAIC rich_av, run the extractor to materialize mfcc/cnn features first."
            )

        self.norm_stats = norm_stats or self._compute_norm_stats()
        self.modality_dims = {
            key: int(self.subjects[0]["modalities"][key].shape[1]) for key in self.modalities
        }

    def _load_dvlog_subjects(self):
        with open(DVLOG_LABELS_CSV, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader if row["fold"] == self.fold]

        for row in rows:
            subject_id = str(int(row["index"]))
            subject_dir = DVLOG_FEATURES_DIR / subject_id
            acoustic_path = subject_dir / f"{subject_id}_acoustic.npy"
            visual_path = subject_dir / f"{subject_id}_visual.npy"
            if not acoustic_path.exists() or not visual_path.exists():
                continue

            acoustic = np.load(str(acoustic_path)).astype(np.float32)[: self.max_seq_len]
            visual = np.load(str(visual_path)).astype(np.float32)[: self.max_seq_len]
            total_len = min(len(acoustic), len(visual))
            if total_len <= 0:
                continue

            modalities = {
                "acoustic": acoustic[:total_len],
                "visual": visual[:total_len],
            }
            modality_masks = {key: np.ones((total_len,), dtype=bool) for key in modalities}
            shared_valid = np.ones((total_len,), dtype=bool)
            label_binary = 1 if row["label"] == "depression" else 0
            self.subjects.append(
                {
                    "subject_id": subject_id,
                    "label_binary": label_binary,
                    "label_phq": float("nan"),
                    "label_phq_severity": -1,
                    "modalities": modalities,
                    "modality_masks": modality_masks,
                    "extra_quality": {},
                    "shared_valid": shared_valid,
                }
            )

    def _load_edaic_subjects(self):
        with open(EDAIC_DETAILED_LABELS, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader if row["split"] == self.fold]

        for row in rows:
            subject_id = f"{row['Participant']}_P"
            subject_dir = EDAIC_PROCESSED_DIR / subject_id
            metadata_path = subject_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

            features: dict[str, np.ndarray] = {}
            masks: dict[str, np.ndarray] = {}
            extras: dict[str, np.ndarray] = {}

            visual_path = subject_dir / "visual.npy"
            visual_ts_path = subject_dir / "visual_timestamps.npy"
            confidence_path = subject_dir / "confidence.npy"
            success_path = subject_dir / "success.npy"
            acoustic_path = subject_dir / "acoustic.npy"
            acoustic_ts_path = subject_dir / "acoustic_timestamps.npy"
            mfcc_path = subject_dir / "mfcc.npy"
            mfcc_ts_path = subject_dir / "mfcc_timestamps.npy"
            cnn_path = subject_dir / "cnn_resnet.npy"
            cnn_ts_path = subject_dir / "cnn_resnet_timestamps.npy"

            visual_timestamps = None
            if "pose_gaze_aus" in self.modalities:
                if not all(path.exists() for path in [visual_path, visual_ts_path, confidence_path, success_path]):
                    continue
                visual_raw = np.load(str(visual_path)).astype(np.float32)
                visual_ts = np.load(str(visual_ts_path)).astype(np.float32)
                confidence = np.load(str(confidence_path)).astype(np.float32)
                success = np.load(str(success_path)).astype(np.float32)
                visual_timestamps = visual_ts
                visual_features, visual_valid = _resample_to_seconds(
                    visual_raw,
                    visual_ts,
                    quality=confidence,
                    success=success,
                    quality_threshold=0.5,
                )
                features["pose_gaze_aus"] = visual_features
                masks["pose_gaze_aus"] = visual_valid
                extras["pose_confidence"] = _resample_scalar(confidence, visual_ts, len(visual_features))
                extras["pose_success"] = _resample_scalar(success, visual_ts, len(visual_features))

            if "acoustic" in self.modalities:
                if not all(path.exists() for path in [acoustic_path, acoustic_ts_path]):
                    continue
                acoustic_raw = np.load(str(acoustic_path)).astype(np.float32)
                acoustic_ts = np.load(str(acoustic_ts_path)).astype(np.float32)
                acoustic_features, acoustic_valid = _resample_to_seconds(acoustic_raw, acoustic_ts)
                features["acoustic"] = acoustic_features
                masks["acoustic"] = acoustic_valid

            if "mfcc" in self.modalities:
                if not all(path.exists() for path in [mfcc_path, mfcc_ts_path]):
                    continue
                mfcc_raw = np.load(str(mfcc_path)).astype(np.float32)
                mfcc_ts = np.load(str(mfcc_ts_path)).astype(np.float32)
                mfcc_features, mfcc_valid = _resample_to_seconds(mfcc_raw, mfcc_ts)
                features["mfcc"] = mfcc_features
                masks["mfcc"] = mfcc_valid

            if "cnn_resnet" in self.modalities:
                if not all(path.exists() for path in [cnn_path, cnn_ts_path]):
                    continue
                cnn_raw = np.load(str(cnn_path)).astype(np.float32)
                cnn_ts = np.load(str(cnn_ts_path)).astype(np.float32)
                cnn_features, cnn_valid = _resample_to_seconds(cnn_raw, cnn_ts)
                features["cnn_resnet"] = cnn_features
                masks["cnn_resnet"] = cnn_valid

            if set(features) != set(self.modalities):
                continue

            aligned_features, aligned_masks, aligned_extras, shared_valid = _align_modalities(features, masks, extras)
            if len(shared_valid) == 0 or np.sum(shared_valid) == 0:
                continue

            valid_indices = np.flatnonzero(shared_valid)
            start = int(valid_indices[0])
            end = int(valid_indices[-1]) + 1
            aligned_features = {key: values[start:end] for key, values in aligned_features.items()}
            aligned_masks = {key: values[start:end] for key, values in aligned_masks.items()}
            aligned_extras = {key: values[start:end] for key, values in aligned_extras.items()}
            shared_valid = shared_valid[start:end]

            label_phq = float(_phq_total(row))
            self.subjects.append(
                {
                    "subject_id": subject_id,
                    "label_binary": int(row["Depression_label"]),
                    "label_phq": label_phq,
                    "label_phq_severity": _severity_label(label_phq),
                    "modalities": aligned_features,
                    "modality_masks": aligned_masks,
                    "extra_quality": aligned_extras,
                    "shared_valid": shared_valid.astype(bool),
                    "metadata": metadata,
                }
            )

    def _compute_norm_stats(self) -> dict[str, dict[str, np.ndarray]]:
        stats = {}
        for modality in self.modalities:
            values = []
            for subject in self.subjects:
                modality_values = subject["modalities"][modality]
                modality_mask = subject["modality_masks"][modality]
                valid = modality_values[modality_mask]
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

    def _window_quality_vector(
        self,
        subject: dict,
        start: int,
        win_len: int,
        window_index: int,
        num_windows: int,
    ) -> np.ndarray:
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

        if self.dataset_name == "edaic":
            confidence = subject["extra_quality"].get("pose_confidence")
            success = subject["extra_quality"].get("pose_success")
            if confidence is not None and success is not None:
                conf_window = confidence[start : start + win_len]
                success_window = success[start : start + win_len]
                quality.extend(
                    [
                        float(np.mean(conf_window)) if len(conf_window) else 0.0,
                        float(np.mean(conf_window < 0.5)) if len(conf_window) else 0.0,
                        float(np.mean(success_window > 0.5)) if len(success_window) else 0.0,
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

        for window_index, start, win_len in windows:
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

            window_quality.append(self._window_quality_vector(subject, start, win_len, window_index, len(windows)))

        window_quality_arr = np.stack(window_quality, axis=0)
        return {
            "modalities": {
                key: torch.from_numpy(np.stack(values, axis=0))
                for key, values in packed_modalities.items()
            },
            "modality_masks": {
                key: torch.from_numpy(np.stack(values, axis=0)).bool()
                for key, values in packed_masks.items()
            },
            "window_quality_features": torch.from_numpy(window_quality_arr),
            "quality_features": torch.from_numpy(window_quality_arr.mean(axis=0)),
            "window_mask": torch.ones((len(windows),), dtype=torch.bool),
            "label_binary": torch.tensor(subject["label_binary"], dtype=torch.long),
            "label_phq": torch.tensor(subject["label_phq"], dtype=torch.float32),
            "label_phq_severity": torch.tensor(subject["label_phq_severity"], dtype=torch.long),
            "label_masks": {
                "binary": torch.tensor(True, dtype=torch.bool),
                "phq": torch.tensor(not math.isnan(subject["label_phq"]), dtype=torch.bool),
                "phq_severity": torch.tensor(subject["label_phq_severity"] >= 0, dtype=torch.bool),
            },
            "subject_id": subject["subject_id"],
            "dataset_source": self.dataset_name,
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
                mask = torch.cat(
                    [mask, torch.zeros((pad_windows, mask.shape[1]), dtype=torch.bool)],
                    dim=0,
                )
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
        "label_phq": torch.stack([sample["label_phq"] for sample in samples], dim=0),
        "label_phq_severity": torch.stack([sample["label_phq_severity"] for sample in samples], dim=0),
        "label_masks": {
            "binary": torch.stack([sample["label_masks"]["binary"] for sample in samples], dim=0),
            "phq": torch.stack([sample["label_masks"]["phq"] for sample in samples], dim=0),
            "phq_severity": torch.stack([sample["label_masks"]["phq_severity"] for sample in samples], dim=0),
        },
        "subject_id": [sample["subject_id"] for sample in samples],
        "dataset_source": [sample["dataset_source"] for sample in samples],
        "fold": [sample["fold"] for sample in samples],
        "window_idx": [sample["window_idx"] for sample in samples],
    }


def create_fusion_v2_dataloaders(
    dataset_name: str,
    bundle_name: str,
    batch_size: int,
    window_size: int,
    stride: int,
    num_workers: int = 0,
    balanced_sampling: bool = True,
    min_valid_ratio: float = 0.5,
) -> dict[str, DataLoader]:
    fold_names = {"dvlog": {"dev": "valid"}, "edaic": {"dev": "dev"}}
    train_ds = FusionV2SubjectDataset(
        dataset_name=dataset_name,
        fold="train",
        bundle_name=bundle_name,
        window_size=window_size,
        stride=stride,
        min_valid_ratio=min_valid_ratio,
    )
    norm_stats = train_ds.norm_stats
    dev_ds = FusionV2SubjectDataset(
        dataset_name=dataset_name,
        fold=fold_names[dataset_name]["dev"],
        bundle_name=bundle_name,
        window_size=window_size,
        stride=stride,
        min_valid_ratio=min_valid_ratio,
        norm_stats=norm_stats,
    )
    test_ds = FusionV2SubjectDataset(
        dataset_name=dataset_name,
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
            "modality_groups": FUSION_V2_MODALITY_GROUPS[dataset_name],
            "bundle_name": bundle_name,
            "quality_dim": int(train_ds[0]["quality_features"].shape[0]),
        },
    }
