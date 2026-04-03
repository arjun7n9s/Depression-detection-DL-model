"""
Manifest Generator — Step 1b of the Depression Detection Pipeline.

Generates manifest.jsonl where each line = one subject-level record.
Windowing happens at DataLoader time, not at manifest time — this keeps the
manifest agnostic to window size/stride hyperparameters.

For D-Vlog: reads pre-extracted .npy features and labels.csv.
For E-DAIC: records extracted, partial, and unextracted subject state after
edaic_extractor runs.

Usage:
    python -m src.data.manifest_generator
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data.dataset_audit import (
    DVLOG_FEATURES_DIR,
    DVLOG_LABELS_CSV,
    DVLOG_VIDEO_LINKS_CSV,
    DVLOG_VIDEOS_DIR,
    EDAIC_DETAILED_LABELS,
    EDAIC_ARCHIVES_DIR,
)
from src.paths import PROCESSED_ROOT


MANIFEST_PATH = PROCESSED_ROOT / "manifest.jsonl"
EDAIC_EXTRACTION_REPORT = PROCESSED_ROOT / "edaic" / "extraction_report.json"


def generate_dvlog_manifest() -> list[dict]:
    """Generate manifest entries for all D-Vlog subjects."""
    entries = []

    # Load labels
    labels = {}
    with open(DVLOG_LABELS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["index"])
            labels[sid] = {
                "label": row["label"],
                "duration": float(row["duration"]),
                "gender": row["gender"],
                "fold": row["fold"],
            }

    # Load video key mapping (for raw video cross-reference)
    video_keys = {}
    if DVLOG_VIDEO_LINKS_CSV.exists():
        with open(DVLOG_VIDEO_LINKS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_keys[int(row["video_id"])] = row["key"]

    # Check which videos are downloaded
    downloaded_keys = set()
    if DVLOG_VIDEOS_DIR.exists():
        downloaded_keys = {f.stem for f in DVLOG_VIDEOS_DIR.glob("*.mp4")}

    # Scan all subject directories
    subject_dirs = sorted(
        [d for d in DVLOG_FEATURES_DIR.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name),
    )

    for sdir in subject_dirs:
        sid = int(sdir.name)
        if sid not in labels:
            continue

        acoustic_path = sdir / f"{sid}_acoustic.npy"
        visual_path = sdir / f"{sid}_visual.npy"

        if not acoustic_path.exists() or not visual_path.exists():
            continue

        # Get sequence lengths without loading full arrays
        acoustic = np.load(str(acoustic_path), mmap_mode="r")
        visual = np.load(str(visual_path), mmap_mode="r")
        a_len, a_dim = acoustic.shape
        v_len, v_dim = visual.shape
        aligned_len = min(a_len, v_len)

        label_info = labels[sid]
        label_binary = 1 if label_info["label"] == "depression" else 0

        # Video availability
        video_key = video_keys.get(sid, None)
        has_video = video_key in downloaded_keys if video_key else False

        quality_flags = []
        if a_len != v_len:
            quality_flags.append("length_mismatch")
        if aligned_len < 30:
            quality_flags.append("very_short")
        if aligned_len > 3600:
            quality_flags.append("very_long")

        entry = {
            "subject_id": str(sid),
            "dataset_source": "dvlog",
            "label_binary": label_binary,
            "label_phq": None,  # D-Vlog has no PHQ scores
            "fold": label_info["fold"],
            "gender": label_info["gender"],
            "duration_seconds": label_info["duration"],
            "acoustic_path": str(acoustic_path),
            "visual_path": str(visual_path),
            "acoustic_len": int(a_len),
            "visual_len": int(v_len),
            "aligned_len": int(aligned_len),
            "acoustic_dim": int(a_dim),
            "visual_dim": int(v_dim),
            "has_raw_video": has_video,
            "video_key": video_key,
            "quality_flags": quality_flags,
        }
        entries.append(entry)

    return entries


def generate_edaic_manifest() -> list[dict]:
    """
    Generate manifest entries for E-DAIC subjects.

    This reads from the processed directory when present and keeps extraction
    failures explicit instead of silently dropping them from the manifest.
    """
    processed_edaic = PROCESSED_ROOT / "edaic"

    # Load labels
    if not EDAIC_DETAILED_LABELS.exists():
        print("  E-DAIC labels not found, skipping.")
        return []

    labels = {}
    with open(EDAIC_DETAILED_LABELS, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            pid = int(row["Participant"])
            phq_subscores = []
            for i in range(1, 9):
                col_candidates = [c for c in fieldnames if c.startswith(f"PHQ8_{i}_")]
                if col_candidates:
                    try:
                        phq_subscores.append(int(row[col_candidates[0]]))
                    except (ValueError, KeyError):
                        phq_subscores.append(0)
            phq_total = sum(phq_subscores)

            labels[pid] = {
                "phq_total": phq_total,
                "depression_label": int(row.get("Depression_label", 0)),
                "split": row.get("split", "unknown"),
                "gender": row.get("gender", "unknown"),
                "age": int(row.get("age", -1)) if row.get("age", "").strip() else -1,
            }

    extraction_issues = {}
    if EDAIC_EXTRACTION_REPORT.exists():
        report = json.loads(EDAIC_EXTRACTION_REPORT.read_text(encoding="utf-8"))
        for item in report.get("errors", []):
            extraction_issues[item["participant"]] = item

    entries = []
    for pid, info in sorted(labels.items()):
        archive_path = EDAIC_ARCHIVES_DIR / f"{pid}_P.tar.gz"
        pid_dir = processed_edaic / f"{pid}_P"

        visual_path = pid_dir / "visual.npy"
        visual_ts_path = pid_dir / "visual_timestamps.npy"
        acoustic_path = pid_dir / "acoustic.npy"
        acoustic_ts_path = pid_dir / "acoustic_timestamps.npy"
        metadata_path = pid_dir / "metadata.json"

        session_type = "ai" if pid >= 492 else "woz"
        quality_flags = []
        extraction_status = "not_extracted"
        extraction_error = None
        extracted = False
        metadata = {}

        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            extraction_status = metadata.get("status", "not_extracted")
            extracted = extraction_status == "success"
            if extraction_status == "partial":
                quality_flags.append("partial_extraction")
            for warning in metadata.get("warnings", []):
                quality_flags.append(f"extract_warning:{warning}")

        if f"{pid}_P" in extraction_issues:
            extraction_error = extraction_issues[f"{pid}_P"].get("error")
            quality_flags.append("corrupt_archive")

        v_len = v_dim = a_len = a_dim = aligned_len = None
        if visual_path.exists():
            visual = np.load(str(visual_path), mmap_mode="r")
            v_len, v_dim = visual.shape
        else:
            quality_flags.append("missing_visual")

        if acoustic_path.exists():
            acoustic = np.load(str(acoustic_path), mmap_mode="r")
            a_len, a_dim = acoustic.shape
        else:
            quality_flags.append("missing_acoustic")

        if v_len is not None and a_len is not None:
            aligned_len = min(v_len, a_len)
            if a_len != v_len:
                quality_flags.append("length_mismatch")
        elif v_len is not None:
            aligned_len = v_len
        elif a_len is not None:
            aligned_len = a_len
        else:
            quality_flags.append("not_extracted")

        if aligned_len is not None and aligned_len < 30:
            quality_flags.append("very_short")

        entry = {
            "subject_id": f"{pid}_P",
            "dataset_source": "edaic",
            "label_binary": info["depression_label"],
            "label_phq": info["phq_total"],
            "fold": info["split"],
            "gender": info["gender"],
            "age": info["age"],
            "session_type": session_type,
            "archive_path": str(archive_path) if archive_path.exists() else None,
            "acoustic_path": str(acoustic_path) if acoustic_path.exists() else None,
            "acoustic_timestamps_path": str(acoustic_ts_path) if acoustic_ts_path.exists() else None,
            "visual_path": str(visual_path) if visual_path.exists() else None,
            "visual_timestamps_path": str(visual_ts_path) if visual_ts_path.exists() else None,
            "acoustic_len": int(a_len) if a_len is not None else None,
            "visual_len": int(v_len) if v_len is not None else None,
            "aligned_len": int(aligned_len) if aligned_len is not None else None,
            "acoustic_dim": int(a_dim) if a_dim is not None else None,
            "visual_dim": int(v_dim) if v_dim is not None else None,
            "quality_flags": sorted(set(quality_flags)),
            "extracted": extracted,
            "extraction_status": extraction_status,
            "extraction_error": extraction_error,
        }
        entries.append(entry)

    return entries


def main():
    print("Generating manifest...")
    print(f"Output: {MANIFEST_PATH}")

    # Ensure output directory exists
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    # Generate D-Vlog entries
    print("\n[1/2] Generating D-Vlog manifest entries...")
    dvlog_entries = generate_dvlog_manifest()
    print(f"  Generated {len(dvlog_entries)} D-Vlog entries")

    # Summarize D-Vlog
    dvlog_folds = {}
    for e in dvlog_entries:
        fold = e["fold"]
        label = e["label_binary"]
        key = f"{fold}_dep{label}"
        dvlog_folds[key] = dvlog_folds.get(key, 0) + 1
    print(f"  Fold x Label: {dvlog_folds}")

    # Generate E-DAIC entries
    print("\n[2/2] Generating E-DAIC manifest entries...")
    edaic_entries = generate_edaic_manifest()
    print(f"  Generated {len(edaic_entries)} E-DAIC entries")

    edaic_folds = {}
    for e in edaic_entries:
        fold = e["fold"]
        label = e["label_binary"]
        key = f"{fold}_dep{label}"
        edaic_folds[key] = edaic_folds.get(key, 0) + 1
    print(f"  Fold x Label: {edaic_folds}")

    # Write manifest
    all_entries = dvlog_entries + edaic_entries
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, default=str) + "\n")

    print(f"\nTotal manifest entries: {len(all_entries)}")
    print(f"Manifest written to: {MANIFEST_PATH}")

    # Quick integrity check
    dvlog_subjects = {e["subject_id"] for e in dvlog_entries}
    edaic_subjects = {e["subject_id"] for e in edaic_entries}
    overlap = dvlog_subjects & edaic_subjects
    if overlap:
        print(f"\nWARNING: {len(overlap)} subjects appear in both datasets: {overlap}")
    else:
        print("\nOK: No subject ID overlap between datasets")


if __name__ == "__main__":
    main()
