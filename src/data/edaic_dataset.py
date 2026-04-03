"""
E-DAIC PyTorch dataset for unimodal and later multimodal training.

This loader consumes SSD-backed extracted arrays from `edaic_extractor.py`,
resamples them to 1 Hz, applies quality filtering, and creates configurable
windows. The default use in this milestone is unimodal baselines.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.paths import PROCESSED_ROOT, PROJECT_ROOT

EDAIC_LABELS = PROJECT_ROOT / "wwwedaic" / "labels" / "detailed_lables.csv"
EDAIC_PROCESSED_DIR = PROCESSED_ROOT / "edaic"
EDAIC_STATS_DIR = PROCESSED_ROOT / "edaic_stats"
EXTRACTION_REPORT = EDAIC_PROCESSED_DIR / "extraction_report.json"


def _phq_total(row: dict[str, str]) -> int:
    cols = [
        "PHQ8_1_NoInterest",
        "PHQ8_2_Depressed",
        "PHQ8_3_Sleep",
        "PHQ8_4_Tired",
        "PHQ8_5_Appetite",
        "PHQ8_6_Failure",
        "PHQ8_7_Concentration",
        "PHQ8_8_Psychomotor",
    ]
    return sum(int(row[col]) for col in cols)


def _resample_to_seconds(
    features: np.ndarray,
    timestamps: np.ndarray,
    quality: Optional[np.ndarray] = None,
    success: Optional[np.ndarray] = None,
    quality_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate irregular/high-rate sequences into 1 Hz bins.

    Returns:
      features_1hz: (T_sec, D)
      valid_mask: (T_sec,)
    """
    if len(features) == 0 or len(timestamps) == 0:
        return np.zeros((0, features.shape[1] if features.ndim == 2 else 0), dtype=np.float32), np.zeros((0,), dtype=bool)

    total_secs = int(np.floor(float(np.max(timestamps)))) + 1
    dim = int(features.shape[1])
    second_features = np.zeros((total_secs, dim), dtype=np.float32)
    valid_mask = np.zeros((total_secs,), dtype=bool)

    bucketed = [[] for _ in range(total_secs)]
    q_bucketed = [[] for _ in range(total_secs)] if quality is not None else None
    s_bucketed = [[] for _ in range(total_secs)] if success is not None else None

    for idx, ts in enumerate(timestamps):
        sec = int(np.floor(float(ts)))
        if 0 <= sec < total_secs:
            bucketed[sec].append(features[idx])
            if q_bucketed is not None:
                q_bucketed[sec].append(float(quality[idx]))
            if s_bucketed is not None:
                s_bucketed[sec].append(float(success[idx]))

    for sec in range(total_secs):
        if not bucketed[sec]:
            continue
        second_features[sec] = np.mean(np.asarray(bucketed[sec], dtype=np.float32), axis=0)
        valid = True
        if q_bucketed is not None and q_bucketed[sec]:
            valid = valid and float(np.mean(q_bucketed[sec])) >= quality_threshold
        if s_bucketed is not None and s_bucketed[sec]:
            valid = valid and float(np.mean(s_bucketed[sec])) > 0.5
        valid_mask[sec] = valid

    return second_features, valid_mask


class EDAICDataset(Dataset):
    """
    E-DAIC dataset with 1 Hz resampling and quality-aware window generation.

    Supported modalities:
      - "visual"
      - "acoustic"
      - "both"
    """

    def __init__(
        self,
        fold: str,
        modality: str,
        window_size: int = 30,
        stride: int = 15,
        min_window_frames: int = 15,
        min_valid_ratio: float = 0.5,
        quality_threshold: float = 0.5,
        normalize: bool = True,
        norm_stats: Optional[dict[str, np.ndarray]] = None,
        max_subjects: Optional[int] = None,
    ):
        super().__init__()
        if modality not in {"visual", "acoustic", "both"}:
            raise ValueError(f"Unsupported modality: {modality}")

        self.fold = fold
        self.modality = modality
        self.window_size = window_size
        self.stride = stride
        self.min_window_frames = min_window_frames
        self.min_valid_ratio = min_valid_ratio
        self.quality_threshold = quality_threshold
        self.normalize = normalize
        self.max_subjects = max_subjects

        self.subjects: list[dict] = []
        self.windows: list[tuple[int, int, int, float]] = []

        self._load_subjects()
        if self.max_subjects is not None:
            self.subjects = self.subjects[: self.max_subjects]

        self._prepare_subject_sequences()
        self.norm_stats = norm_stats
        if self.normalize and self.norm_stats is None:
            self.norm_stats = self._load_or_compute_norm_stats()
        self._build_window_index()

    def _load_subjects(self):
        extraction_failures = {}
        if EXTRACTION_REPORT.exists():
            report = json.loads(EXTRACTION_REPORT.read_text(encoding="utf-8"))
            for item in report.get("errors", []):
                extraction_failures[item["participant"]] = item

        with open(EDAIC_LABELS, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row["split"] != self.fold:
                    continue
                pid = row["Participant"]
                pid_dir = EDAIC_PROCESSED_DIR / f"{pid}_P"
                metadata_path = pid_dir / "metadata.json"
                metadata = {}
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

                subject = {
                    "subject_id": f"{pid}_P",
                    "label_binary": int(row["Depression_label"]),
                    "label_phq": float(_phq_total(row)),
                    "gender": row["gender"],
                    "age": int(row["age"]) if row["age"] else -1,
                    "visual_path": pid_dir / "visual.npy",
                    "visual_timestamps_path": pid_dir / "visual_timestamps.npy",
                    "confidence_path": pid_dir / "confidence.npy",
                    "success_path": pid_dir / "success.npy",
                    "acoustic_path": pid_dir / "acoustic.npy",
                    "acoustic_timestamps_path": pid_dir / "acoustic_timestamps.npy",
                    "metadata": metadata,
                    "extraction_issue": extraction_failures.get(f"{pid}_P"),
                }

                if self.modality == "visual" and not subject["visual_path"].exists():
                    continue
                if self.modality == "acoustic" and not subject["acoustic_path"].exists():
                    continue
                if self.modality == "both" and (
                    not subject["visual_path"].exists() or not subject["acoustic_path"].exists()
                ):
                    continue

                self.subjects.append(subject)

    def _prepare_subject_sequences(self):
        prepared_subjects = []
        for subject in self.subjects:
            prepared = dict(subject)

            visual_features = visual_valid = None
            acoustic_features = acoustic_valid = None

            if subject["visual_path"].exists():
                visual_raw = np.load(str(subject["visual_path"])).astype(np.float32)
                visual_ts = np.load(str(subject["visual_timestamps_path"])).astype(np.float32)
                confidence = np.load(str(subject["confidence_path"])).astype(np.float32)
                success = np.load(str(subject["success_path"])).astype(np.float32)
                visual_features, visual_valid = _resample_to_seconds(
                    visual_raw,
                    visual_ts,
                    quality=confidence,
                    success=success,
                    quality_threshold=self.quality_threshold,
                )

            if subject["acoustic_path"].exists():
                acoustic_raw = np.load(str(subject["acoustic_path"])).astype(np.float32)
                acoustic_ts = np.load(str(subject["acoustic_timestamps_path"])).astype(np.float32)
                acoustic_features, acoustic_valid = _resample_to_seconds(acoustic_raw, acoustic_ts)

            if self.modality == "visual":
                if visual_features is None:
                    continue
                features = visual_features
                valid_mask = visual_valid
            elif self.modality == "acoustic":
                if acoustic_features is None:
                    continue
                features = acoustic_features
                valid_mask = acoustic_valid
            else:
                if visual_features is None or acoustic_features is None:
                    continue
                total_secs = min(len(visual_features), len(acoustic_features))
                features = {
                    "visual": visual_features[:total_secs],
                    "acoustic": acoustic_features[:total_secs],
                }
                valid_mask = visual_valid[:total_secs] & acoustic_valid[:total_secs]

            if len(valid_mask) == 0:
                continue

            valid_indices = np.flatnonzero(valid_mask)
            if len(valid_indices) == 0:
                continue

            start = int(valid_indices[0])
            end = int(valid_indices[-1]) + 1
            if self.modality == "both":
                features = {
                    "visual": features["visual"][start:end],
                    "acoustic": features["acoustic"][start:end],
                }
            else:
                features = features[start:end]
            valid_mask = valid_mask[start:end]

            if len(valid_mask) < self.min_window_frames:
                continue

            prepared["features"] = features
            prepared["valid_mask"] = valid_mask.astype(np.float32)
            prepared_subjects.append(prepared)

        self.subjects = prepared_subjects

    def _stats_path(self) -> Path:
        EDAIC_STATS_DIR.mkdir(parents=True, exist_ok=True)
        return EDAIC_STATS_DIR / f"norm_stats_{self.modality}.npz"

    def _load_or_compute_norm_stats(self) -> dict[str, np.ndarray]:
        stats_path = self._stats_path()
        if self.fold != "train" and stats_path.exists():
            stats = np.load(str(stats_path))
            return {key: stats[key].astype(np.float32) for key in stats.files}

        if self.fold != "train":
            raise ValueError("Non-train E-DAIC fold requires precomputed train normalization stats.")

        if self.modality == "both":
            sums = {
                "visual_sum": None,
                "visual_sq_sum": None,
                "visual_count": 0,
                "acoustic_sum": None,
                "acoustic_sq_sum": None,
                "acoustic_count": 0,
            }
            for subject in self.subjects:
                valid = subject["valid_mask"] > 0
                visual = subject["features"]["visual"][valid]
                acoustic = subject["features"]["acoustic"][valid]
                if sums["visual_sum"] is None:
                    sums["visual_sum"] = np.zeros(visual.shape[1], dtype=np.float64)
                    sums["visual_sq_sum"] = np.zeros(visual.shape[1], dtype=np.float64)
                    sums["acoustic_sum"] = np.zeros(acoustic.shape[1], dtype=np.float64)
                    sums["acoustic_sq_sum"] = np.zeros(acoustic.shape[1], dtype=np.float64)
                sums["visual_sum"] += visual.sum(axis=0)
                sums["visual_sq_sum"] += (visual ** 2).sum(axis=0)
                sums["visual_count"] += len(visual)
                sums["acoustic_sum"] += acoustic.sum(axis=0)
                sums["acoustic_sq_sum"] += (acoustic ** 2).sum(axis=0)
                sums["acoustic_count"] += len(acoustic)

            stats = {
                "visual_mean": (sums["visual_sum"] / max(sums["visual_count"], 1)).astype(np.float32),
                "visual_std": np.sqrt(
                    np.maximum(
                        sums["visual_sq_sum"] / max(sums["visual_count"], 1)
                        - (sums["visual_sum"] / max(sums["visual_count"], 1)) ** 2,
                        1e-8,
                    )
                ).astype(np.float32),
                "acoustic_mean": (sums["acoustic_sum"] / max(sums["acoustic_count"], 1)).astype(np.float32),
                "acoustic_std": np.sqrt(
                    np.maximum(
                        sums["acoustic_sq_sum"] / max(sums["acoustic_count"], 1)
                        - (sums["acoustic_sum"] / max(sums["acoustic_count"], 1)) ** 2,
                        1e-8,
                    )
                ).astype(np.float32),
            }
        else:
            running_sum = None
            running_sq = None
            running_count = 0
            for subject in self.subjects:
                valid = subject["valid_mask"] > 0
                values = subject["features"][valid]
                if running_sum is None:
                    running_sum = np.zeros(values.shape[1], dtype=np.float64)
                    running_sq = np.zeros(values.shape[1], dtype=np.float64)
                running_sum += values.sum(axis=0)
                running_sq += (values ** 2).sum(axis=0)
                running_count += len(values)
            mean = (running_sum / max(running_count, 1)).astype(np.float32)
            std = np.sqrt(np.maximum(running_sq / max(running_count, 1) - (running_sum / max(running_count, 1)) ** 2, 1e-8)).astype(np.float32)
            stats = {"mean": mean, "std": std}

        np.savez(self._stats_path(), **stats)
        return stats

    def _build_window_index(self):
        for subject_idx, subject in enumerate(self.subjects):
            seq_len = len(subject["valid_mask"])
            if seq_len <= self.window_size:
                valid_ratio = float(np.mean(subject["valid_mask"]))
                if valid_ratio >= self.min_valid_ratio:
                    self.windows.append((subject_idx, 0, seq_len, valid_ratio))
                continue

            seen = set()
            for start in range(0, seq_len - self.window_size + 1, self.stride):
                valid_ratio = float(np.mean(subject["valid_mask"][start : start + self.window_size]))
                if valid_ratio >= self.min_valid_ratio:
                    self.windows.append((subject_idx, start, self.window_size, valid_ratio))
                    seen.add(start)

            final_start = seq_len - self.window_size
            if final_start not in seen:
                valid_ratio = float(np.mean(subject["valid_mask"][final_start : final_start + self.window_size]))
                if valid_ratio >= self.min_valid_ratio:
                    self.windows.append((subject_idx, final_start, self.window_size, valid_ratio))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        subject_idx, start, win_len, valid_ratio = self.windows[idx]
        subject = self.subjects[subject_idx]

        if self.modality == "both":
            visual = subject["features"]["visual"][start : start + win_len].copy()
            acoustic = subject["features"]["acoustic"][start : start + win_len].copy()
            if self.normalize and self.norm_stats is not None:
                visual = (visual - self.norm_stats["visual_mean"]) / self.norm_stats["visual_std"]
                acoustic = (acoustic - self.norm_stats["acoustic_mean"]) / self.norm_stats["acoustic_std"]
            if len(visual) < self.window_size:
                pad = self.window_size - len(visual)
                visual = np.pad(visual, ((0, pad), (0, 0)), mode="constant")
                acoustic = np.pad(acoustic, ((0, pad), (0, 0)), mode="constant")
            return {
                "visual": torch.from_numpy(visual.astype(np.float32)),
                "acoustic": torch.from_numpy(acoustic.astype(np.float32)),
                "label_binary": subject["label_binary"],
                "label_phq": subject["label_phq"],
                "subject_id": subject["subject_id"],
                "window_idx": idx,
                "dataset_source": "edaic",
                "fold": self.fold,
                "quality_score": valid_ratio,
            }

        values = subject["features"][start : start + win_len].copy()
        if self.normalize and self.norm_stats is not None:
            values = (values - self.norm_stats["mean"]) / self.norm_stats["std"]
        if len(values) < self.window_size:
            pad = self.window_size - len(values)
            values = np.pad(values, ((0, pad), (0, 0)), mode="constant")
        return {
            "inputs": torch.from_numpy(values.astype(np.float32)),
            "label_binary": subject["label_binary"],
            "label_phq": subject["label_phq"],
            "subject_id": subject["subject_id"],
            "window_idx": idx,
            "dataset_source": "edaic",
            "fold": self.fold,
            "quality_score": valid_ratio,
            "modality": self.modality,
        }

    def get_labels(self) -> list[int]:
        return [self.subjects[self.windows[i][0]]["label_binary"] for i in range(len(self.windows))]

    def get_subject_ids(self) -> list[str]:
        return [subject["subject_id"] for subject in self.subjects]


def create_edaic_dataloaders(
    modality: str,
    batch_size: int = 16,
    window_size: int = 30,
    stride: int = 15,
    num_workers: int = 0,
    balanced_sampling: bool = True,
    min_valid_ratio: float = 0.5,
) -> dict[str, DataLoader]:
    train_ds = EDAICDataset(
        fold="train",
        modality=modality,
        window_size=window_size,
        stride=stride,
        min_valid_ratio=min_valid_ratio,
    )
    train_stats = train_ds.norm_stats
    dev_ds = EDAICDataset(
        fold="dev",
        modality=modality,
        window_size=window_size,
        stride=stride,
        min_valid_ratio=min_valid_ratio,
        norm_stats=train_stats,
    )
    test_ds = EDAICDataset(
        fold="test",
        modality=modality,
        window_size=window_size,
        stride=stride,
        min_valid_ratio=min_valid_ratio,
        norm_stats=train_stats,
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
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True),
        "dev": DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }


def verify_dataset():
    print("=" * 60)
    print("E-DAIC DATASET VERIFICATION")
    print("=" * 60)
    for modality in ["visual", "acoustic"]:
        print(f"\n--- {modality.upper()} ---")
        for fold in ["train", "dev", "test"]:
            ds = EDAICDataset(fold=fold, modality=modality, window_size=30, stride=15)
            print(f"{fold}: subjects={len(ds.subjects)} windows={len(ds.windows)}")
            if len(ds) > 0:
                sample = ds[0]
                print(f"  sample_shape={tuple(sample['inputs'].shape)} dtype={sample['inputs'].dtype}")
                print(f"  label={sample['label_binary']} subject={sample['subject_id']} quality={sample['quality_score']:.2f}")

    train_visual = set(EDAICDataset("train", "visual").get_subject_ids())
    dev_visual = set(EDAICDataset("dev", "visual").get_subject_ids())
    test_visual = set(EDAICDataset("test", "visual").get_subject_ids())
    print("\n--- SUBJECT INTEGRITY ---")
    print(f"train/dev overlap: {len(train_visual & dev_visual)}")
    print(f"train/test overlap: {len(train_visual & test_visual)}")
    print(f"dev/test overlap: {len(dev_visual & test_visual)}")


if __name__ == "__main__":
    verify_dataset()
