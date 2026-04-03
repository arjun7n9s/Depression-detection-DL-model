"""
Dataset Audit — Step 1 of the Depression Detection Pipeline.

Performs comprehensive data quality validation for both D-Vlog and E-DAIC datasets.
Outputs audit_report.json (machine-readable) and prints a human-readable summary.

Usage:
    python -m src.data.dataset_audit
"""

import json
import csv
import os
import sys
import tarfile
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np

from src.paths import DVLOG_VIDEOS_DIR, PROJECT_ROOT


# ---------------------------------------------------------------------------
# Path Configuration — derived from actual filesystem layout
# ---------------------------------------------------------------------------

DVLOG_FEATURES_DIR = PROJECT_ROOT / "data" / "D-Vlog dataset" / "dvlog-dataset"
DVLOG_LABELS_CSV = DVLOG_FEATURES_DIR / "labels.csv"
DVLOG_NORM_STATS = DVLOG_FEATURES_DIR / "normalization_stats.npz"

EDAIC_ROOT = PROJECT_ROOT / "wwwedaic"
EDAIC_ARCHIVES_DIR = EDAIC_ROOT / "data"
EDAIC_LABELS_DIR = EDAIC_ROOT / "labels"
EDAIC_DETAILED_LABELS = EDAIC_LABELS_DIR / "detailed_lables.csv"  # sic: original typo

DVLOG_VIDEO_LINKS_CSV = PROJECT_ROOT / "data" / "D-Vlog dataset" / "dvlog-video-links-2026.csv"

# Output
AUDIT_OUTPUT_DIR = PROJECT_ROOT / "data"
AUDIT_REPORT_PATH = AUDIT_OUTPUT_DIR / "audit_report.json"


# ---------------------------------------------------------------------------
# D-Vlog Audit
# ---------------------------------------------------------------------------

def audit_dvlog() -> dict:
    """Audit the D-Vlog pre-extracted feature dataset."""
    report = {
        "dataset": "dvlog",
        "features_dir": str(DVLOG_FEATURES_DIR),
        "labels_csv": str(DVLOG_LABELS_CSV),
        "timestamp": datetime.now().isoformat(),
    }

    # --- Load labels ---
    if not DVLOG_LABELS_CSV.exists():
        report["error"] = f"Labels file not found: {DVLOG_LABELS_CSV}"
        return report

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

    report["total_labeled_subjects"] = len(labels)
    report["folds"] = dict(Counter(v["fold"] for v in labels.values()))
    report["labels"] = dict(Counter(v["label"] for v in labels.values()))
    report["genders"] = dict(Counter(v["gender"] for v in labels.values()))

    # Per-fold label distribution
    fold_label_dist = {}
    for v in labels.values():
        key = f"{v['fold']}_{v['label']}"
        fold_label_dist[key] = fold_label_dist.get(key, 0) + 1
    report["fold_label_distribution"] = fold_label_dist

    # --- Scan feature files ---
    subjects_with_features = set()
    subject_dirs = [d for d in DVLOG_FEATURES_DIR.iterdir() if d.is_dir() and d.name.isdigit()]

    feature_stats = []
    missing_acoustic = []
    missing_visual = []
    length_mismatches = []
    corrupt_files = []
    dtype_issues = []

    for sdir in sorted(subject_dirs, key=lambda x: int(x.name)):
        sid = int(sdir.name)
        acoustic_path = sdir / f"{sid}_acoustic.npy"
        visual_path = sdir / f"{sid}_visual.npy"

        has_acoustic = acoustic_path.exists()
        has_visual = visual_path.exists()

        if not has_acoustic:
            missing_acoustic.append(sid)
        if not has_visual:
            missing_visual.append(sid)

        if not has_acoustic or not has_visual:
            continue

        subjects_with_features.add(sid)

        # Load and validate
        try:
            acoustic = np.load(str(acoustic_path))
            visual = np.load(str(visual_path))
        except Exception as e:
            corrupt_files.append({"subject_id": sid, "error": str(e)})
            continue

        a_len, a_dim = acoustic.shape
        v_len, v_dim = visual.shape

        # Check dtype
        if acoustic.dtype != np.float64:
            dtype_issues.append({"subject_id": sid, "file": "acoustic", "dtype": str(acoustic.dtype)})
        if visual.dtype != np.float64:
            dtype_issues.append({"subject_id": sid, "file": "visual", "dtype": str(visual.dtype)})

        # Check dimension consistency
        if a_dim != 25:
            dtype_issues.append({"subject_id": sid, "file": "acoustic", "expected_dim": 25, "actual_dim": a_dim})
        if v_dim != 136:
            dtype_issues.append({"subject_id": sid, "file": "visual", "expected_dim": 136, "actual_dim": v_dim})

        # Length mismatch
        if a_len != v_len:
            length_mismatches.append({
                "subject_id": sid,
                "acoustic_len": int(a_len),
                "visual_len": int(v_len),
                "diff": abs(int(a_len) - int(v_len)),
            })

        # Check for NaN/Inf
        a_nan = bool(np.any(np.isnan(acoustic)))
        a_inf = bool(np.any(np.isinf(acoustic)))
        v_nan = bool(np.any(np.isnan(visual)))
        v_inf = bool(np.any(np.isinf(visual)))

        aligned_len = min(a_len, v_len)
        feature_stats.append({
            "subject_id": sid,
            "acoustic_len": int(a_len),
            "visual_len": int(v_len),
            "aligned_len": int(aligned_len),
            "acoustic_dim": int(a_dim),
            "visual_dim": int(v_dim),
            "acoustic_has_nan": a_nan,
            "acoustic_has_inf": a_inf,
            "visual_has_nan": v_nan,
            "visual_has_inf": v_inf,
            "label": labels.get(sid, {}).get("label", "unknown"),
            "fold": labels.get(sid, {}).get("fold", "unknown"),
        })

    # --- Summary statistics ---
    aligned_lengths = [s["aligned_len"] for s in feature_stats]
    acoustic_lengths = [s["acoustic_len"] for s in feature_stats]
    visual_lengths = [s["visual_len"] for s in feature_stats]

    report["subjects_with_features"] = len(subjects_with_features)
    report["missing_acoustic"] = missing_acoustic
    report["missing_visual"] = missing_visual
    report["corrupt_files"] = corrupt_files
    report["dtype_issues"] = dtype_issues
    report["length_mismatches"] = length_mismatches
    report["num_length_mismatches"] = len(length_mismatches)

    if aligned_lengths:
        report["sequence_length_stats"] = {
            "aligned": {
                "min": int(np.min(aligned_lengths)),
                "max": int(np.max(aligned_lengths)),
                "mean": float(np.mean(aligned_lengths)),
                "median": float(np.median(aligned_lengths)),
                "std": float(np.std(aligned_lengths)),
            },
            "acoustic": {
                "min": int(np.min(acoustic_lengths)),
                "max": int(np.max(acoustic_lengths)),
                "mean": float(np.mean(acoustic_lengths)),
                "median": float(np.median(acoustic_lengths)),
            },
            "visual": {
                "min": int(np.min(visual_lengths)),
                "max": int(np.max(visual_lengths)),
                "mean": float(np.mean(visual_lengths)),
                "median": float(np.median(visual_lengths)),
            },
        }

    # NaN/Inf summary
    nan_subjects = [s["subject_id"] for s in feature_stats if s["acoustic_has_nan"] or s["visual_has_nan"]]
    inf_subjects = [s["subject_id"] for s in feature_stats if s["acoustic_has_inf"] or s["visual_has_inf"]]
    report["subjects_with_nan"] = nan_subjects
    report["subjects_with_inf"] = inf_subjects

    # Subjects in labels but missing features
    labeled_ids = set(labels.keys())
    feature_ids = subjects_with_features
    report["labeled_but_no_features"] = sorted(labeled_ids - feature_ids)
    report["features_but_no_label"] = sorted(feature_ids - labeled_ids)

    # --- Normalization stats check ---
    if DVLOG_NORM_STATS.exists():
        try:
            stats = np.load(str(DVLOG_NORM_STATS))
            report["normalization_stats"] = {
                "files_in_npz": list(stats.files),
                "shapes": {k: list(stats[k].shape) for k in stats.files},
                "dtypes": {k: str(stats[k].dtype) for k in stats.files},
            }
        except Exception as e:
            report["normalization_stats_error"] = str(e)
    else:
        report["normalization_stats"] = "NOT FOUND"

    # --- Video availability cross-reference ---
    if DVLOG_VIDEOS_DIR.exists() and DVLOG_VIDEO_LINKS_CSV.exists():
        video_files = {f.stem for f in DVLOG_VIDEOS_DIR.glob("*.mp4")}
        video_links = {}
        with open(DVLOG_VIDEO_LINKS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_links[int(row["video_id"])] = row["key"]

        videos_available = {sid for sid, key in video_links.items() if key in video_files}
        report["raw_video_coverage"] = {
            "total_linked": len(video_links),
            "videos_downloaded": len(video_files),
            "subjects_with_video": len(videos_available),
            "subjects_without_video": len(set(video_links.keys()) - videos_available),
        }

    report["per_subject_details"] = feature_stats

    return report


# ---------------------------------------------------------------------------
# E-DAIC Audit
# ---------------------------------------------------------------------------

def audit_edaic() -> dict:
    """Audit the E-DAIC dataset (archives + labels)."""
    report = {
        "dataset": "edaic",
        "archives_dir": str(EDAIC_ARCHIVES_DIR),
        "labels_dir": str(EDAIC_LABELS_DIR),
        "timestamp": datetime.now().isoformat(),
    }

    # --- Load labels ---
    if not EDAIC_DETAILED_LABELS.exists():
        report["error"] = f"Labels file not found: {EDAIC_DETAILED_LABELS}"
        return report

    labels = {}
    with open(EDAIC_DETAILED_LABELS, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        report["label_columns"] = fieldnames

        for row in reader:
            pid = int(row["Participant"])
            phq_subscores = []
            for i in range(1, 9):
                col_candidates = [c for c in fieldnames if c.startswith(f"PHQ8_{i}_")]
                if col_candidates:
                    try:
                        phq_subscores.append(int(row[col_candidates[0]]))
                    except (ValueError, KeyError):
                        phq_subscores.append(None)

            phq_total = sum(s for s in phq_subscores if s is not None)
            depression_label = int(row.get("Depression_label", -1))
            split = row.get("split", "unknown")
            gender = row.get("gender", "unknown")

            try:
                age = int(row.get("age", -1))
            except ValueError:
                age = -1

            labels[pid] = {
                "phq_subscores": phq_subscores,
                "phq_total": phq_total,
                "depression_label": depression_label,
                "split": split,
                "gender": gender,
                "age": age,
            }

    report["total_labeled_subjects"] = len(labels)
    report["splits"] = dict(Counter(v["split"] for v in labels.values()))
    report["depression_labels"] = dict(Counter(v["depression_label"] for v in labels.values()))
    report["genders"] = dict(Counter(v["gender"] for v in labels.values()))

    # Age distribution
    ages = [v["age"] for v in labels.values() if v["age"] > 0]
    if ages:
        report["age_stats"] = {
            "min": min(ages),
            "max": max(ages),
            "mean": float(np.mean(ages)),
            "median": float(np.median(ages)),
        }

    # PHQ-8 distribution
    phq_scores = [v["phq_total"] for v in labels.values()]
    report["phq_stats"] = {
        "min": int(np.min(phq_scores)),
        "max": int(np.max(phq_scores)),
        "mean": float(np.mean(phq_scores)),
        "median": float(np.median(phq_scores)),
        "std": float(np.std(phq_scores)),
    }

    # PHQ severity bands
    severity_bands = {"0-4_minimal": 0, "5-9_mild": 0, "10-14_moderate": 0,
                      "15-19_mod_severe": 0, "20-27_severe": 0}
    for s in phq_scores:
        if s <= 4:
            severity_bands["0-4_minimal"] += 1
        elif s <= 9:
            severity_bands["5-9_mild"] += 1
        elif s <= 14:
            severity_bands["10-14_moderate"] += 1
        elif s <= 19:
            severity_bands["15-19_mod_severe"] += 1
        else:
            severity_bands["20-27_severe"] += 1
    report["phq_severity_bands"] = severity_bands

    # Per-split depression rates
    split_dep = {}
    for v in labels.values():
        key = v["split"]
        if key not in split_dep:
            split_dep[key] = {"total": 0, "depressed": 0}
        split_dep[key]["total"] += 1
        if v["depression_label"] == 1:
            split_dep[key]["depressed"] += 1
    for k in split_dep:
        t = split_dep[k]["total"]
        d = split_dep[k]["depressed"]
        split_dep[k]["rate"] = round(d / t, 3) if t > 0 else 0
    report["split_depression_rates"] = split_dep

    # WoZ vs AI session detection (sessions >= 600 are AI-controlled)
    woz_sessions = {pid for pid in labels if pid < 492}
    ai_sessions = {pid for pid in labels if pid >= 492}
    report["session_types"] = {
        "woz_range": f"<492 ({len(woz_sessions)} sessions)",
        "ai_range": f">=492 ({len(ai_sessions)} sessions)",
        "note": "Approximate boundary; test set is sessions 600-718 (AI-controlled)",
    }

    # --- Scan archives ---
    if not EDAIC_ARCHIVES_DIR.exists():
        report["archives_error"] = f"Archives directory not found: {EDAIC_ARCHIVES_DIR}"
        return report

    archives = sorted(EDAIC_ARCHIVES_DIR.glob("*.tar.gz"))
    archive_pids = []
    for a in archives:
        # Extract PID from filename like "300_P.tar.gz"
        stem = a.stem.replace(".tar", "")  # "300_P"
        try:
            pid = int(stem.split("_")[0])
            archive_pids.append(pid)
        except ValueError:
            pass

    report["total_archives"] = len(archives)
    report["archive_pid_range"] = [min(archive_pids), max(archive_pids)] if archive_pids else []

    # Cross-reference labels vs archives
    labeled_pids = set(labels.keys())
    archive_pid_set = set(archive_pids)
    report["labeled_but_no_archive"] = sorted(labeled_pids - archive_pid_set)
    report["archive_but_no_label"] = sorted(archive_pid_set - labeled_pids)

    # --- Spot-check archive contents (first 3 archives) ---
    spot_checks = []
    for archive_path in archives[:3]:
        try:
            with tarfile.open(str(archive_path), "r:gz") as tar:
                members = tar.getnames()
                file_types = {}
                for m in members:
                    basename = os.path.basename(m)
                    if basename:
                        file_types[basename] = True
                spot_checks.append({
                    "archive": archive_path.name,
                    "num_files": len(members),
                    "files": sorted(os.path.basename(m) for m in members if os.path.basename(m)),
                })
        except Exception as e:
            spot_checks.append({"archive": archive_path.name, "error": str(e)})

    report["archive_spot_checks"] = spot_checks

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_dvlog_summary(report: dict):
    """Print human-readable D-Vlog audit summary."""
    print("\n" + "=" * 70)
    print("D-VLOG DATASET AUDIT")
    print("=" * 70)

    print(f"\nLabeled subjects: {report.get('total_labeled_subjects', 'N/A')}")
    print(f"Subjects with features: {report.get('subjects_with_features', 'N/A')}")

    folds = report.get("folds", {})
    print(f"\nFolds: {folds}")

    labels = report.get("labels", {})
    print(f"Labels: {labels}")

    print(f"\nFold x Label distribution: {report.get('fold_label_distribution', {})}")

    stats = report.get("sequence_length_stats", {})
    if stats:
        al = stats.get("aligned", {})
        print(f"\nSequence lengths (aligned): min={al.get('min')}, max={al.get('max')}, "
              f"mean={al.get('mean', 0):.1f}, median={al.get('median', 0):.1f}, std={al.get('std', 0):.1f}")

    mismatches = report.get("num_length_mismatches", 0)
    print(f"\nLength mismatches (acoustic vs visual): {mismatches}")
    if mismatches > 0:
        for m in report.get("length_mismatches", [])[:5]:
            print(f"  sid={m['subject_id']}: acoustic={m['acoustic_len']}, visual={m['visual_len']}, diff={m['diff']}")
        if mismatches > 5:
            print(f"  ... and {mismatches - 5} more")

    nan_subs = report.get("subjects_with_nan", [])
    inf_subs = report.get("subjects_with_inf", [])
    print(f"\nSubjects with NaN: {len(nan_subs)} {nan_subs[:10] if nan_subs else ''}")
    print(f"Subjects with Inf: {len(inf_subs)} {inf_subs[:10] if inf_subs else ''}")

    print(f"\nLabeled but no features: {report.get('labeled_but_no_features', [])}")
    print(f"Features but no label: {report.get('features_but_no_label', [])}")

    norm = report.get("normalization_stats", {})
    if isinstance(norm, dict):
        print(f"\nNormalization stats: {norm.get('files_in_npz', 'N/A')}")
        for k, v in norm.get("shapes", {}).items():
            print(f"  {k}: shape={v}, dtype={norm.get('dtypes', {}).get(k)}")

    video = report.get("raw_video_coverage", {})
    if video:
        print(f"\nRaw video coverage: {video.get('subjects_with_video')}/{video.get('total_linked')} "
              f"({video.get('videos_downloaded')} MP4s on disk)")

    corrupt = report.get("corrupt_files", [])
    if corrupt:
        print(f"\n⚠️ CORRUPT FILES: {len(corrupt)}")
        for c in corrupt:
            print(f"  sid={c['subject_id']}: {c['error']}")

    dtype = report.get("dtype_issues", [])
    if dtype:
        # Only show unexpected dtype issues (float64 is expected)
        unexpected = [d for d in dtype if "dtype" in d and d["dtype"] != "float64"]
        if unexpected:
            print(f"\n⚠️ UNEXPECTED DTYPE: {len(unexpected)}")
            for d in unexpected[:5]:
                print(f"  sid={d['subject_id']}: {d}")


def print_edaic_summary(report: dict):
    """Print human-readable E-DAIC audit summary."""
    print("\n" + "=" * 70)
    print("E-DAIC DATASET AUDIT")
    print("=" * 70)

    print(f"\nLabeled subjects: {report.get('total_labeled_subjects', 'N/A')}")
    print(f"Total archives: {report.get('total_archives', 'N/A')}")

    splits = report.get("splits", {})
    print(f"\nSplits: {splits}")

    dep = report.get("depression_labels", {})
    print(f"Depression labels: {dep}")

    print(f"Genders: {report.get('genders', {})}")

    age = report.get("age_stats", {})
    if age:
        print(f"Age: min={age.get('min')}, max={age.get('max')}, "
              f"mean={age.get('mean', 0):.1f}, median={age.get('median', 0):.1f}")

    phq = report.get("phq_stats", {})
    if phq:
        print(f"\nPHQ-8: min={phq.get('min')}, max={phq.get('max')}, "
              f"mean={phq.get('mean', 0):.1f}, median={phq.get('median', 0):.1f}, "
              f"std={phq.get('std', 0):.1f}")

    bands = report.get("phq_severity_bands", {})
    if bands:
        print(f"Severity bands: {bands}")

    rates = report.get("split_depression_rates", {})
    if rates:
        print(f"\nPer-split depression rates:")
        for split, info in rates.items():
            print(f"  {split}: {info['depressed']}/{info['total']} ({info['rate']:.1%})")

    sessions = report.get("session_types", {})
    if sessions:
        print(f"\nSession types: WoZ={sessions.get('woz_range')}, AI={sessions.get('ai_range')}")

    labeled_no_arc = report.get("labeled_but_no_archive", [])
    arc_no_label = report.get("archive_but_no_label", [])
    print(f"\nLabeled but no archive: {len(labeled_no_arc)} subjects {labeled_no_arc[:10] if labeled_no_arc else ''}")
    print(f"Archive but no label: {len(arc_no_label)} subjects {arc_no_label[:10] if arc_no_label else ''}")

    spot = report.get("archive_spot_checks", [])
    if spot:
        print(f"\nArchive spot checks:")
        for s in spot:
            if "error" in s:
                print(f"  {s['archive']}: ERROR - {s['error']}")
            else:
                print(f"  {s['archive']}: {s['num_files']} files")
                for fname in s.get("files", []):
                    print(f"    - {fname}")


def main():
    print("Starting dataset audit...")
    print(f"Project root: {PROJECT_ROOT}")

    # Run D-Vlog audit
    print("\n[1/2] Auditing D-Vlog dataset...")
    dvlog_report = audit_dvlog()
    print_dvlog_summary(dvlog_report)

    # Run E-DAIC audit
    print("\n[2/2] Auditing E-DAIC dataset...")
    edaic_report = audit_edaic()
    print_edaic_summary(edaic_report)

    # Save combined report
    combined_report = {
        "audit_timestamp": datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "dvlog": dvlog_report,
        "edaic": edaic_report,
    }

    # Remove per-subject details from saved report to keep it manageable
    # (but they were used for aggregate stats above)
    dvlog_report.pop("per_subject_details", None)

    AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"Audit report saved to: {AUDIT_REPORT_PATH}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
