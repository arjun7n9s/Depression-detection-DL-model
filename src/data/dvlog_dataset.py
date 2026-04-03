"""
D-Vlog PyTorch Dataset — Step 2 of the Depression Detection Pipeline.

Loads pre-extracted .npy acoustic/visual features for D-Vlog subjects.
Handles:
  - float64 → float32 casting
  - Acoustic/visual length alignment via min-truncation
  - Windowing with configurable window_size and stride
  - Train-fold-only normalization stats computation
  - Weighted sampling support for class imbalance

Usage:
    from src.data.dvlog_dataset import DVlogDataset, create_dvlog_dataloaders
"""

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.data.dataset_audit import DVLOG_FEATURES_DIR, DVLOG_LABELS_CSV, DVLOG_NORM_STATS
from src.paths import PROCESSED_ROOT


DVLOG_STATS_DIR = PROCESSED_ROOT / "dvlog_stats"
DVLOG_TRAIN_ONLY_STATS = DVLOG_STATS_DIR / "train_only_norm_stats.npz"


def load_dvlog_normalization_stats(source: str) -> dict[str, np.ndarray]:
    if source == "provided":
        stats = np.load(str(DVLOG_NORM_STATS))
        return {
            "acoustic_mean": stats["acoustic_mean"].astype(np.float32),
            "acoustic_std": stats["acoustic_std"].astype(np.float32),
            "visual_mean": stats["visual_mean"].astype(np.float32),
            "visual_std": stats["visual_std"].astype(np.float32),
        }

    if source != "train":
        raise ValueError(f"Unsupported normalization source: {source}")

    if DVLOG_TRAIN_ONLY_STATS.exists():
        stats = np.load(str(DVLOG_TRAIN_ONLY_STATS))
        return {
            "acoustic_mean": stats["acoustic_mean"].astype(np.float32),
            "acoustic_std": stats["acoustic_std"].astype(np.float32),
            "visual_mean": stats["visual_mean"].astype(np.float32),
            "visual_std": stats["visual_std"].astype(np.float32),
        }

    return {}


class DVlogDataset(Dataset):
    """
    PyTorch Dataset for D-Vlog pre-extracted features.

    Each item returns a dict with:
        - visual: (window_size, 136) float32 tensor
        - acoustic: (window_size, 25) float32 tensor
        - label_binary: int (0 or 1)
        - subject_id: str
        - window_idx: int
        - dataset_source: "dvlog"
    """

    def __init__(
        self,
        fold: str,
        window_size: int = 30,
        stride: int = 15,
        normalize: bool = True,
        norm_stats: Optional[dict] = None,
        normalization_source: str = "train",
        max_seq_len: int = 3600,
        min_window_frames: int = 15,
    ):
        """
        Args:
            fold: "train", "valid", or "test"
            window_size: Number of timesteps per window (seconds at 1fps)
            stride: Window stride in timesteps
            normalize: Whether to normalize features
            norm_stats: Pre-computed {acoustic_mean, acoustic_std, visual_mean, visual_std}
                        If None and normalize=True, will compute from train fold.
            normalization_source: "train" for train-fold-only stats, "provided" for shipped stats.
            max_seq_len: Cap sequences at this length (seconds)
            min_window_frames: Minimum valid frames to keep a window
        """
        super().__init__()
        self.fold = fold
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.normalization_source = normalization_source
        self.max_seq_len = max_seq_len
        self.min_window_frames = min_window_frames

        # Load labels
        self.subjects = []
        self._load_labels()

        # Compute or load normalization stats
        self.norm_stats = norm_stats
        if self.normalize and self.norm_stats is None:
            if self.normalization_source == "provided":
                self.norm_stats = load_dvlog_normalization_stats("provided")
            else:
                cached_stats = load_dvlog_normalization_stats("train")
                self.norm_stats = cached_stats if cached_stats else self._compute_norm_stats()

        # Build window index: list of (subject_idx, window_start)
        self.windows = []
        self._build_window_index()

    def _load_labels(self):
        """Load subject metadata from labels.csv and filter by fold."""
        with open(DVLOG_LABELS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["fold"] != self.fold:
                    continue
                sid = int(row["index"])
                acoustic_path = DVLOG_FEATURES_DIR / str(sid) / f"{sid}_acoustic.npy"
                visual_path = DVLOG_FEATURES_DIR / str(sid) / f"{sid}_visual.npy"

                if not acoustic_path.exists() or not visual_path.exists():
                    continue

                self.subjects.append({
                    "subject_id": sid,
                    "label": 1 if row["label"] == "depression" else 0,
                    "gender": row["gender"],
                    "duration": float(row["duration"]),
                    "acoustic_path": acoustic_path,
                    "visual_path": visual_path,
                })

    def _build_window_index(self):
        """Create (subject_idx, window_start) pairs for all valid windows."""
        for subj_idx, subj in enumerate(self.subjects):
            # Get sequence length without loading full array
            acoustic = np.load(str(subj["acoustic_path"]), mmap_mode="r")
            visual = np.load(str(subj["visual_path"]), mmap_mode="r")
            aligned_len = min(len(acoustic), len(visual))
            aligned_len = min(aligned_len, self.max_seq_len)

            if aligned_len < self.min_window_frames:
                continue

            # Generate window start positions
            if aligned_len <= self.window_size:
                # Sequence shorter than window — use single padded window
                self.windows.append((subj_idx, 0, aligned_len))
            else:
                for start in range(0, aligned_len - self.window_size + 1, self.stride):
                    self.windows.append((subj_idx, start, self.window_size))

                # Handle remainder if last window doesn't reach the end
                last_start = self.windows[-1][1] if self.windows else 0
                if last_start + self.window_size < aligned_len:
                    # Add a final window anchored at the end
                    final_start = aligned_len - self.window_size
                    if final_start != last_start:
                        self.windows.append((subj_idx, final_start, self.window_size))

    def _compute_norm_stats(self) -> dict:
        """Compute normalization stats from the train fold only."""
        if self.fold != "train":
            raise ValueError("Non-train fold requires pre-computed norm_stats.")

        # Online computation: running mean/var (Welford's algorithm)
        n_acoustic = 0
        sum_acoustic = None
        sum_sq_acoustic = None

        n_visual = 0
        sum_visual = None
        sum_sq_visual = None

        for subj in self.subjects:
            acoustic = np.load(str(subj["acoustic_path"])).astype(np.float32)
            visual = np.load(str(subj["visual_path"])).astype(np.float32)

            # Truncate to aligned length
            aligned_len = min(len(acoustic), len(visual), self.max_seq_len)
            acoustic = acoustic[:aligned_len]
            visual = visual[:aligned_len]

            if sum_acoustic is None:
                sum_acoustic = np.zeros(acoustic.shape[1], dtype=np.float64)
                sum_sq_acoustic = np.zeros(acoustic.shape[1], dtype=np.float64)
                sum_visual = np.zeros(visual.shape[1], dtype=np.float64)
                sum_sq_visual = np.zeros(visual.shape[1], dtype=np.float64)

            sum_acoustic += acoustic.sum(axis=0).astype(np.float64)
            sum_sq_acoustic += (acoustic ** 2).sum(axis=0).astype(np.float64)
            n_acoustic += len(acoustic)

            sum_visual += visual.sum(axis=0).astype(np.float64)
            sum_sq_visual += (visual ** 2).sum(axis=0).astype(np.float64)
            n_visual += len(visual)

        acoustic_mean = (sum_acoustic / n_acoustic).astype(np.float32)
        acoustic_var = (sum_sq_acoustic / n_acoustic - acoustic_mean.astype(np.float64) ** 2)
        acoustic_std = np.sqrt(np.maximum(acoustic_var, 1e-8)).astype(np.float32)

        visual_mean = (sum_visual / n_visual).astype(np.float32)
        visual_var = (sum_sq_visual / n_visual - visual_mean.astype(np.float64) ** 2)
        visual_std = np.sqrt(np.maximum(visual_var, 1e-8)).astype(np.float32)

        stats = {
            "acoustic_mean": acoustic_mean,
            "acoustic_std": acoustic_std,
            "visual_mean": visual_mean,
            "visual_std": visual_std,
        }
        DVLOG_STATS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(DVLOG_TRAIN_ONLY_STATS, **stats)
        return stats

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        subj_idx, win_start, win_len = self.windows[idx]
        subj = self.subjects[subj_idx]

        # Load features and cast to float32
        acoustic = np.load(str(subj["acoustic_path"])).astype(np.float32)
        visual = np.load(str(subj["visual_path"])).astype(np.float32)

        # Align lengths
        aligned_len = min(len(acoustic), len(visual), self.max_seq_len)
        acoustic = acoustic[:aligned_len]
        visual = visual[:aligned_len]

        # Extract window
        acoustic_win = acoustic[win_start:win_start + win_len]
        visual_win = visual[win_start:win_start + win_len]

        # Normalize
        if self.normalize and self.norm_stats is not None:
            acoustic_win = (acoustic_win - self.norm_stats["acoustic_mean"]) / self.norm_stats["acoustic_std"]
            visual_win = (visual_win - self.norm_stats["visual_mean"]) / self.norm_stats["visual_std"]

        # Pad if window is shorter than window_size
        if len(acoustic_win) < self.window_size:
            pad_len = self.window_size - len(acoustic_win)
            acoustic_win = np.pad(acoustic_win, ((0, pad_len), (0, 0)), mode="constant")
            visual_win = np.pad(visual_win, ((0, pad_len), (0, 0)), mode="constant")

        return {
            "visual": torch.from_numpy(visual_win),
            "acoustic": torch.from_numpy(acoustic_win),
            "label_binary": subj["label"],
            "label_phq": float("nan"),  # D-Vlog has no PHQ scores
            "subject_id": str(subj["subject_id"]),
            "window_idx": idx,
            "dataset_source": "dvlog",
            "fold": self.fold,
        }

    def get_labels(self) -> list[int]:
        """Return label for each window (for weighted sampling)."""
        return [self.subjects[self.windows[i][0]]["label"] for i in range(len(self.windows))]

    def get_subject_ids(self) -> list[str]:
        """Return unique subject IDs in this dataset."""
        return [str(s["subject_id"]) for s in self.subjects]


def create_dvlog_dataloaders(
    batch_size: int = 16,
    window_size: int = 30,
    stride: int = 15,
    num_workers: int = 4,
    balanced_sampling: bool = True,
    normalization_source: str = "train",
) -> dict[str, DataLoader]:
    """
    Create train/valid/test DataLoaders for D-Vlog.

    Returns dict with keys "train", "valid", "test".
    """
    # Build train set first to compute normalization stats
    initial_stats = None
    if normalization_source == "provided":
        initial_stats = load_dvlog_normalization_stats("provided")

    train_ds = DVlogDataset(
        fold="train",
        window_size=window_size,
        stride=stride,
        norm_stats=initial_stats,
        normalization_source=normalization_source,
    )
    train_norm_stats = train_ds.norm_stats

    valid_ds = DVlogDataset(
        fold="valid", window_size=window_size, stride=stride,
        norm_stats=train_norm_stats,
        normalization_source=normalization_source,
    )
    test_ds = DVlogDataset(
        fold="test", window_size=window_size, stride=stride,
        norm_stats=train_norm_stats,
        normalization_source=normalization_source,
    )

    # Weighted sampler for class balance in training
    train_sampler = None
    shuffle_train = True
    if balanced_sampling:
        labels = train_ds.get_labels()
        class_counts = [0, 0]
        for lab in labels:
            class_counts[lab] += 1

        # Weight inversely proportional to class frequency
        weights = [1.0 / class_counts[lab] for lab in labels]
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        shuffle_train = False  # Sampler handles ordering

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "valid": DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return loaders


def verify_dataset():
    """Quick verification that the dataset loads correctly."""
    print("=" * 60)
    print("D-VLOG DATASET VERIFICATION")
    print("=" * 60)

    for fold in ["train", "valid", "test"]:
        print(f"\n--- {fold.upper()} ---")
        ds = DVlogDataset(
            fold=fold,
            window_size=30,
            stride=15,
            normalize=(fold == "train"),
            norm_stats=None if fold == "train" else None,  # Will auto-compute/load
        )
        print(f"  Subjects: {len(ds.subjects)}")
        print(f"  Windows:  {len(ds.windows)}")

        # Label distribution
        labels = ds.get_labels()
        n_dep = sum(labels)
        n_norm = len(labels) - n_dep
        print(f"  Labels (window-level): depression={n_dep}, normal={n_norm}")

        # Subject-level label distribution
        subj_labels = [s["label"] for s in ds.subjects]
        s_dep = sum(subj_labels)
        s_norm = len(subj_labels) - s_dep
        print(f"  Labels (subject-level): depression={s_dep}, normal={s_norm}")

        if len(ds) > 0:
            sample = ds[0]
            print(f"  Sample visual shape:   {sample['visual'].shape}")
            print(f"  Sample acoustic shape: {sample['acoustic'].shape}")
            print(f"  Sample visual dtype:   {sample['visual'].dtype}")
            print(f"  Sample label:          {sample['label_binary']}")
            print(f"  Sample subject_id:     {sample['subject_id']}")

            # Check for NaN in normalized data
            if torch.isnan(sample["visual"]).any():
                print(f"  ⚠️ WARNING: NaN in visual features!")
            if torch.isnan(sample["acoustic"]).any():
                print(f"  ⚠️ WARNING: NaN in acoustic features!")

        if fold == "train":
            # Save norm stats for valid/test
            _train_norm = ds.norm_stats

    # Subject overlap check
    train_ds = DVlogDataset(fold="train", window_size=30, stride=15, normalize=False)
    valid_ds = DVlogDataset(fold="valid", window_size=30, stride=15, normalize=False)
    test_ds = DVlogDataset(fold="test", window_size=30, stride=15, normalize=False)

    train_sids = set(train_ds.get_subject_ids())
    valid_sids = set(valid_ds.get_subject_ids())
    test_sids = set(test_ds.get_subject_ids())

    tv_overlap = train_sids & valid_sids
    tt_overlap = train_sids & test_sids
    vt_overlap = valid_sids & test_sids

    print(f"\n--- SUBJECT INTEGRITY ---")
    print(f"  Train & Valid overlap:  {len(tv_overlap)} {'PASS' if len(tv_overlap) == 0 else 'FAIL'}")
    print(f"  Train & Test overlap:   {len(tt_overlap)} {'PASS' if len(tt_overlap) == 0 else 'FAIL'}")
    print(f"  Valid & Test overlap:   {len(vt_overlap)} {'PASS' if len(vt_overlap) == 0 else 'FAIL'}")
    print(f"  Total unique subjects: {len(train_sids | valid_sids | test_sids)}")


if __name__ == "__main__":
    verify_dataset()
