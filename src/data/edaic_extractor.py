"""
E-DAIC feature extractor.

Extracts OpenFace visual features and OpenSMILE eGeMAPS acoustic features from
participant archives into SSD-backed NumPy artifacts. The extractor is
idempotent, salvages readable members from partially corrupted archives, and
produces an explicit extraction report describing complete, partial, and failed
subjects.

Output per subject directory:
  - visual.npy                (T_v, 49) float32      [if visual recovered]
  - visual_timestamps.npy     (T_v,) float32         [if visual recovered]
  - confidence.npy            (T_v,) float32         [if visual recovered]
  - success.npy               (T_v,) int32           [if visual recovered]
  - acoustic.npy              (T_a, 23) float32      [if acoustic recovered]
  - acoustic_timestamps.npy   (T_a,) float32         [if acoustic recovered]
  - metadata.json
"""

from __future__ import annotations

import csv
import io
import json
import os
import tarfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data.dataset_audit import EDAIC_ARCHIVES_DIR


PROCESSED_EDAIC_DIR = Path(r"D:\DL-Datasets\processed\edaic")
REPORT_PATH = PROCESSED_EDAIC_DIR / "extraction_report.json"
EXTRACTOR_VERSION = "2.0"

OPENFACE_POSE_COLS = ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
OPENFACE_GAZE_COLS = [
    "gaze_0_x",
    "gaze_0_y",
    "gaze_0_z",
    "gaze_1_x",
    "gaze_1_y",
    "gaze_1_z",
    "gaze_angle_x",
    "gaze_angle_y",
]
OPENFACE_AU_R_COLS = [
    "AU01_r",
    "AU02_r",
    "AU04_r",
    "AU05_r",
    "AU06_r",
    "AU07_r",
    "AU09_r",
    "AU10_r",
    "AU12_r",
    "AU14_r",
    "AU15_r",
    "AU17_r",
    "AU20_r",
    "AU23_r",
    "AU25_r",
    "AU26_r",
    "AU45_r",
]
OPENFACE_AU_C_COLS = [
    "AU01_c",
    "AU02_c",
    "AU04_c",
    "AU05_c",
    "AU06_c",
    "AU07_c",
    "AU09_c",
    "AU10_c",
    "AU12_c",
    "AU14_c",
    "AU15_c",
    "AU17_c",
    "AU20_c",
    "AU23_c",
    "AU25_c",
    "AU26_c",
    "AU28_c",
    "AU45_c",
]
OPENFACE_FEATURE_COLS = OPENFACE_POSE_COLS + OPENFACE_GAZE_COLS + OPENFACE_AU_R_COLS + OPENFACE_AU_C_COLS
OPENFACE_REQUIRED_COLS = {"timestamp", "confidence", "success", *OPENFACE_FEATURE_COLS}
EGEMAPS_METADATA_COLS = {"name", "frameTime"}
COMPLETE_FILES = [
    "visual.npy",
    "visual_timestamps.npy",
    "confidence.npy",
    "success.npy",
    "acoustic.npy",
    "acoustic_timestamps.npy",
    "metadata.json",
]


def _subject_dir_complete(pid_dir: Path) -> bool:
    metadata_path = pid_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    if not all((pid_dir / name).exists() for name in COMPLETE_FILES):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return metadata.get("extractor_version") == EXTRACTOR_VERSION and metadata.get("status") == "success"


def _read_target_members(archive_path: Path) -> tuple[str | None, str | None, list[str]]:
    """
    Read the two feature CSV payloads directly from the archive.

    For truncated archives, this still recovers whichever members appear before
    the corruption point.
    """
    pid = archive_path.stem.replace(".tar", "").split("_")[0]
    openface_name = f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv"
    egemaps_name = f"{pid}_OpenSMILE2.3.0_egemaps.csv"

    openface_content = None
    egemaps_content = None
    warnings: list[str] = []

    with tarfile.open(str(archive_path), "r:gz") as tar:
        try:
            for member in tar:
                basename = os.path.basename(member.name)
                if basename not in {openface_name, egemaps_name}:
                    continue
                try:
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        warnings.append(f"{basename}: extractfile returned None")
                        continue
                    content = extracted.read().decode("utf-8", errors="replace")
                except (tarfile.TarError, EOFError, OSError) as exc:
                    warnings.append(f"{basename}: {exc}")
                    continue

                if basename == openface_name:
                    openface_content = content
                elif basename == egemaps_name:
                    egemaps_content = content

                if openface_content is not None and egemaps_content is not None:
                    break
        except (tarfile.TarError, EOFError, OSError) as exc:
            warnings.append(str(exc))

    return openface_content, egemaps_content, warnings


def extract_openface_features(csv_content: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return visual features, timestamps, confidence, success."""
    reader = csv.DictReader(io.StringIO(csv_content))
    fieldnames = {field.strip() for field in (reader.fieldnames or [])}
    missing = sorted(OPENFACE_REQUIRED_COLS - fieldnames)
    if missing:
        raise ValueError(f"Missing OpenFace columns: {missing}")

    features = []
    timestamps = []
    confidence = []
    success = []

    for raw_row in reader:
        row = {key.strip(): value.strip() for key, value in raw_row.items() if key is not None and value is not None}
        try:
            features.append([float(row[col]) for col in OPENFACE_FEATURE_COLS])
            timestamps.append(float(row["timestamp"]))
            confidence.append(float(row["confidence"]))
            success.append(int(float(row["success"])))
        except (KeyError, ValueError):
            continue

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(timestamps, dtype=np.float32),
        np.asarray(confidence, dtype=np.float32),
        np.asarray(success, dtype=np.int32),
    )


def extract_egemaps_features(csv_content: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return acoustic features, timestamps, and feature column names."""
    reader = csv.DictReader(io.StringIO(csv_content), delimiter=";")
    fieldnames = [field.strip() for field in (reader.fieldnames or [])]
    feature_cols = [field for field in fieldnames if field not in EGEMAPS_METADATA_COLS]
    if "frameTime" not in fieldnames:
        raise ValueError("Missing eGeMAPS frameTime column")

    rows = []
    timestamps = []
    for raw_row in reader:
        row = {key.strip(): value.strip() for key, value in raw_row.items() if key is not None and value is not None}
        try:
            rows.append([float(row[col]) for col in feature_cols])
            timestamps.append(float(row["frameTime"]))
        except (KeyError, ValueError):
            continue

    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(timestamps, dtype=np.float32),
        feature_cols,
    )


def process_archive(archive_path: Path, output_dir: Path) -> dict:
    pid_dir = output_dir / archive_path.stem.replace(".tar", "")
    pid_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "participant": archive_path.stem.replace(".tar", ""),
        "archive": archive_path.name,
        "status": "failed",
        "warnings": [],
    }

    if _subject_dir_complete(pid_dir):
        result["status"] = "skipped"
        result["warnings"].append("already_complete")
        return result

    openface_content = None
    egemaps_content = None
    try:
        openface_content, egemaps_content, warnings = _read_target_members(archive_path)
        result["warnings"].extend(warnings)
    except (tarfile.TarError, EOFError, OSError) as exc:
        result["warnings"].append(str(exc))

    metadata = {
        "participant": result["participant"],
        "archive": archive_path.name,
        "extractor_version": EXTRACTOR_VERSION,
        "extracted_at": datetime.now().isoformat(),
        "warnings": list(result["warnings"]),
        "visual_available": False,
        "acoustic_available": False,
    }

    if openface_content is not None:
        try:
            visual, visual_timestamps, confidence, success = extract_openface_features(openface_content)
            np.save(str(pid_dir / "visual.npy"), visual)
            np.save(str(pid_dir / "visual_timestamps.npy"), visual_timestamps)
            np.save(str(pid_dir / "confidence.npy"), confidence)
            np.save(str(pid_dir / "success.npy"), success)
            metadata.update(
                {
                    "visual_available": True,
                    "visual_shape": list(visual.shape),
                    "visual_timestamp_shape": list(visual_timestamps.shape),
                    "confidence_mean": float(np.mean(confidence)) if len(confidence) else None,
                    "low_confidence_ratio": float(np.mean(confidence < 0.5)) if len(confidence) else None,
                    "success_ratio": float(np.mean(success > 0)) if len(success) else None,
                    "visual_columns": OPENFACE_FEATURE_COLS,
                }
            )
        except ValueError as exc:
            result["warnings"].append(f"visual_parse_error: {exc}")

    if egemaps_content is not None:
        try:
            acoustic, acoustic_timestamps, acoustic_feature_cols = extract_egemaps_features(egemaps_content)
            np.save(str(pid_dir / "acoustic.npy"), acoustic)
            np.save(str(pid_dir / "acoustic_timestamps.npy"), acoustic_timestamps)
            metadata.update(
                {
                    "acoustic_available": True,
                    "acoustic_shape": list(acoustic.shape),
                    "acoustic_timestamp_shape": list(acoustic_timestamps.shape),
                    "acoustic_columns": acoustic_feature_cols,
                }
            )
        except ValueError as exc:
            result["warnings"].append(f"acoustic_parse_error: {exc}")

    if metadata["visual_available"] and metadata["acoustic_available"]:
        result["status"] = "success"
    elif metadata["visual_available"] or metadata["acoustic_available"]:
        result["status"] = "partial"
        result["warnings"].append("partial_subject_recovered")
    else:
        result["warnings"].append("no_required_features_recovered")

    metadata["status"] = result["status"]
    metadata["warnings"] = list(dict.fromkeys(result["warnings"]))
    (pid_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    result.update(
        {
            "status": metadata["status"],
            "visual_available": metadata["visual_available"],
            "acoustic_available": metadata["acoustic_available"],
            "warnings": metadata["warnings"],
        }
    )
    return result


def _print_summary(processed_dir: Path):
    complete = 0
    partial = 0
    failed = 0
    visual_dims = set()
    acoustic_dims = set()
    visual_lengths = []
    acoustic_lengths = []

    for pid_dir in sorted(processed_dir.iterdir()):
        if not pid_dir.is_dir():
            continue
        metadata_path = pid_dir / "metadata.json"
        if not metadata_path.exists():
            failed += 1
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        status = metadata.get("status")
        if status == "success":
            complete += 1
        elif status == "partial":
            partial += 1
        elif status != "skipped":
            failed += 1

        if metadata.get("visual_available"):
            visual_dims.add(metadata["visual_shape"][1])
            visual_lengths.append(metadata["visual_shape"][0])
        if metadata.get("acoustic_available"):
            acoustic_dims.add(metadata["acoustic_shape"][1])
            acoustic_lengths.append(metadata["acoustic_shape"][0])

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Complete subjects: {complete}")
    print(f"Partial subjects:  {partial}")
    print(f"Failed subjects:   {failed}")
    if visual_lengths:
        print(f"Visual dims: {sorted(visual_dims)} | seq len mean={np.mean(visual_lengths):.0f}")
    if acoustic_lengths:
        print(f"Acoustic dims: {sorted(acoustic_dims)} | seq len mean={np.mean(acoustic_lengths):.0f}")


def main():
    print("=" * 60)
    print("E-DAIC FEATURE EXTRACTOR")
    print("=" * 60)

    PROCESSED_EDAIC_DIR.mkdir(parents=True, exist_ok=True)
    archives = sorted(EDAIC_ARCHIVES_DIR.glob("*.tar.gz"))
    print(f"Found {len(archives)} archives")
    print(f"Output directory: {PROCESSED_EDAIC_DIR}")

    complete_subjects = {path.name for path in PROCESSED_EDAIC_DIR.iterdir() if path.is_dir() and _subject_dir_complete(path)}
    remaining = [archive for archive in archives if archive.stem.replace(".tar", "") not in complete_subjects]
    print(f"Already complete: {len(complete_subjects)}")
    print(f"To process: {len(remaining)}")

    start_time = time.time()
    results = []
    for index, archive_path in enumerate(remaining, start=1):
        result = process_archive(archive_path, PROCESSED_EDAIC_DIR)
        results.append(result)
        if index % 25 == 0 or index == len(remaining):
            elapsed = max(time.time() - start_time, 1e-6)
            rate = index / elapsed
            eta = (len(remaining) - index) / rate if rate > 0 else 0
            partial_count = sum(1 for item in results if item["status"] == "partial")
            failed_count = sum(1 for item in results if item["status"] == "failed")
            print(
                f"[{index}/{len(remaining)}] {rate:.2f} archives/sec | "
                f"ETA {eta:.0f}s | partial {partial_count} | failed {failed_count}"
            )

    refreshed_subjects = [path for path in PROCESSED_EDAIC_DIR.iterdir() if path.is_dir()]
    report = {
        "timestamp": datetime.now().isoformat(),
        "extractor_version": EXTRACTOR_VERSION,
        "total_archives": len(archives),
        "already_complete": len(complete_subjects),
        "processed_this_run": len(results),
        "successful": sum(1 for item in results if item["status"] == "success"),
        "partial": sum(1 for item in results if item["status"] == "partial"),
        "failed": sum(1 for item in results if item["status"] == "failed"),
        "skipped": sum(1 for item in results if item["status"] == "skipped"),
        "complete_subject_dirs": len([path for path in refreshed_subjects if _subject_dir_complete(path)]),
        "results": results,
        "errors": [item for item in results if item["status"] in {"partial", "failed"}],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nExtraction report saved to: {REPORT_PATH}")
    _print_summary(PROCESSED_EDAIC_DIR)


if __name__ == "__main__":
    main()
