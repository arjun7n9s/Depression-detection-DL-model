"""Prepare paired bridge features for D-Vlog raw-video and offline teacher spaces."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.dataset_audit import DVLOG_FEATURES_DIR, DVLOG_LABELS_CSV
from src.inference.model_lock import benchmark_winner
from src.paths import DVLOG_BRIDGE_ROOT, PROCESSED_ROOT


DVLOG_VISION_V3_ROOT = PROCESSED_ROOT / "dvlog_video_v3"


@dataclass
class BridgePrepConfig:
    output_root: Path = DVLOG_BRIDGE_ROOT
    overwrite: bool = False
    max_subjects: int | None = None


def _load_label_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with open(DVLOG_LABELS_CSV, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = str(int(row["index"]))
            rows[subject_id] = {
                "label": row["label"],
                "fold": row["fold"],
                "duration": float(row["duration"]),
                "gender": row["gender"],
            }
    return rows


def _bridge_subject_is_complete(subject_dir: Path) -> bool:
    required = [
        subject_dir / "student_modalities.npz",
        subject_dir / "teacher_targets.npz",
        subject_dir / "metadata.json",
    ]
    return all(path.exists() for path in required)


def _aligned_length(arrays: list[np.ndarray]) -> int:
    return min(len(values) for values in arrays if len(values) > 0)


def _load_optional(subject_dir: Path, name: str, fallback_dim: int) -> tuple[np.ndarray, np.ndarray]:
    array_path = subject_dir / f"{name}.npy"
    mask_path = subject_dir / f"{name}_mask.npy"
    if not array_path.exists():
        return np.zeros((0, fallback_dim), dtype=np.float32), np.zeros((0,), dtype=bool)
    values = np.load(str(array_path)).astype(np.float32)
    if mask_path.exists():
        mask = np.load(str(mask_path)).astype(bool)
    else:
        mask = np.ones((len(values),), dtype=bool)
    return values, mask


def _pair_subject(subject_id: str, label_rows: dict[str, dict[str, Any]], config: BridgePrepConfig) -> dict[str, Any] | None:
    source_dir = DVLOG_FEATURES_DIR / subject_id
    video_dir = DVLOG_VISION_V3_ROOT / subject_id
    if not source_dir.exists() or not video_dir.exists():
        return None

    output_dir = config.output_root / subject_id
    if output_dir.exists() and _bridge_subject_is_complete(output_dir) and not config.overwrite:
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        return {"subject_id": subject_id, "status": "skipped", "metadata": metadata}

    acoustic = np.load(str(source_dir / f"{subject_id}_acoustic.npy")).astype(np.float32)
    visual = np.load(str(source_dir / f"{subject_id}_visual.npy")).astype(np.float32)
    body_pose, body_mask = _load_optional(video_dir, "body_pose", 132)
    hand_pose, hand_mask = _load_optional(video_dir, "hand_pose", 126)
    gaze_blink, gaze_mask = _load_optional(video_dir, "gaze_blink", 8)
    face_affect, face_mask = _load_optional(video_dir, "face_affect_embed", 48)
    timestamps = np.load(str(video_dir / "frame_timestamps.npy")).astype(np.float32)

    arrays = [acoustic, visual, body_pose, hand_pose, gaze_blink]
    if len(face_affect):
        arrays.append(face_affect)
    total_len = _aligned_length(arrays)
    if total_len <= 0:
        return None

    acoustic = acoustic[:total_len]
    visual = visual[:total_len]
    body_pose = body_pose[:total_len]
    hand_pose = hand_pose[:total_len]
    gaze_blink = gaze_blink[:total_len]
    timestamps = timestamps[:total_len]
    body_mask = body_mask[:total_len]
    hand_mask = hand_mask[:total_len]
    gaze_mask = gaze_mask[:total_len]
    if len(face_affect):
        face_affect = face_affect[:total_len]
        face_mask = face_mask[:total_len]
    else:
        face_affect = np.zeros((total_len, 0), dtype=np.float32)
        face_mask = np.zeros((total_len,), dtype=bool)

    student_modalities = {
        "body_pose": body_pose,
        "hand_pose": hand_pose,
        "gaze_blink": gaze_blink,
        "face_affect_embed": face_affect,
        "frame_timestamps": timestamps,
        "body_pose_mask": body_mask,
        "hand_pose_mask": hand_mask,
        "gaze_blink_mask": gaze_mask,
        "face_affect_embed_mask": face_mask,
    }
    teacher_targets = {
        "visual_teacher": visual,
        "acoustic_teacher": acoustic,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "student_modalities.npz", **student_modalities)
    np.savez_compressed(output_dir / "teacher_targets.npz", **teacher_targets)

    label_row = label_rows[subject_id]
    metadata = {
        "subject_id": subject_id,
        "fold": label_row["fold"],
        "label": label_row["label"],
        "gender": label_row["gender"],
        "duration": label_row["duration"],
        "aligned_length": int(total_len),
        "student_modalities": {
            "body_pose_dim": int(body_pose.shape[1]),
            "hand_pose_dim": int(hand_pose.shape[1]),
            "gaze_blink_dim": int(gaze_blink.shape[1]),
            "face_affect_dim": int(face_affect.shape[1]),
        },
        "teacher_targets": {
            "visual_dim": int(visual.shape[1]),
            "acoustic_dim": int(acoustic.shape[1]),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"subject_id": subject_id, "status": "complete", "metadata": metadata}


def run_bridge_prep(config: BridgePrepConfig) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    labels = _load_label_rows()
    strategic_winner = benchmark_winner("dvlog")
    subject_ids = sorted([path.name for path in DVLOG_VISION_V3_ROOT.iterdir() if path.is_dir() and path.name.isdigit()], key=int)
    if config.max_subjects is not None:
        subject_ids = subject_ids[: config.max_subjects]

    report = {
        "stage": "dvlog_bridge_prep",
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "source_lock": strategic_winner,
        "processed": [],
        "summary": {"complete": 0, "skipped": 0, "failed": 0, "missing": 0},
    }

    for index, subject_id in enumerate(subject_ids, start=1):
        try:
            result = _pair_subject(subject_id, labels, config)
            if result is None:
                report["summary"]["missing"] += 1
                report["processed"].append({"subject_id": subject_id, "status": "missing"})
            else:
                report["summary"][result["status"]] += 1
                report["processed"].append(result)
        except Exception as exc:
            report["summary"]["failed"] += 1
            report["processed"].append({"subject_id": subject_id, "status": "failed", "error": str(exc)})

        if index % 50 == 0 or index == len(subject_ids):
            summary = report["summary"]
            print(
                f"[{index}/{len(subject_ids)}] complete={summary['complete']} "
                f"skipped={summary['skipped']} missing={summary['missing']} failed={summary['failed']}"
            )

    report_path = config.output_root / "bridge_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare paired bridge features for D-Vlog.")
    parser.add_argument("--output-root", type=Path, default=DVLOG_BRIDGE_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-subjects", type=int, default=None)
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    config = BridgePrepConfig(output_root=args.output_root, overwrite=args.overwrite, max_subjects=args.max_subjects)
    report = run_bridge_prep(config)
    print("=" * 60)
    print("D-VLOG BRIDGE PREP")
    print("=" * 60)
    print(f"Output root: {config.output_root}")
    print(f"Summary: {report['summary']}")
    print(f"Report: {config.output_root / 'bridge_report.json'}")


if __name__ == "__main__":
    main()
