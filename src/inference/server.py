"""Prototype Flask server for locked-model status, bridge readiness, and live feature extraction."""

from __future__ import annotations

import argparse
import base64
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from src.inference.feature_extractor import LiveExtractorConfig, LiveFeatureExtractor
from src.inference.model_lock import benchmark_winner, load_model_lock, preferred_direction
from src.paths import DVLOG_BRIDGE_ROOT, PROJECT_ROOT, PROCESSED_ROOT, repo_relative


DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"
DVLOG_VIDEO_ROOT = PROCESSED_ROOT / "dvlog_video_v3"
VISION_V3_SHOWDOWN_SUMMARY = (
    PROJECT_ROOT / "results" / "benchmark_quality" / "vision_v3_dvlog_showdown" / "final" / "benchmark_summary.csv"
)
EDAIC_FUSION_V1_SUMMARY = (
    PROJECT_ROOT / "results" / "benchmark_quality" / "fusion_v1_locked" / "final" / "benchmark_summary.csv"
)


@dataclass
class InferenceServerConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    face_embedding_backend: str = "color_histogram"
    bridge_root: Path = DVLOG_BRIDGE_ROOT


def _safe_json_load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    import csv

    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bridge_status(bridge_root: Path) -> dict[str, Any]:
    report = _safe_json_load(bridge_root / "bridge_report.json")
    subject_dirs = [path for path in bridge_root.iterdir()] if bridge_root.exists() else []
    completed_subjects = 0
    for path in subject_dirs:
        if not path.is_dir() or not path.name.isdigit():
            continue
        required = [
            path / "student_modalities.npz",
            path / "teacher_targets.npz",
            path / "metadata.json",
        ]
        if all(item.exists() for item in required):
            completed_subjects += 1

    status = {
        "bridge_root": str(bridge_root),
        "bridge_root_repo_relative": repo_relative(bridge_root),
        "paired_subject_dirs": completed_subjects,
        "report_exists": report is not None,
        "report": report,
    }
    if report is None:
        status["summary"] = {
            "complete": completed_subjects,
            "skipped": 0,
            "missing": 0,
            "failed": 0,
        }
    else:
        status["summary"] = report.get("summary", {})
    return status


def _benchmark_snapshot() -> dict[str, Any]:
    dvlog_rows = _safe_csv_rows(VISION_V3_SHOWDOWN_SUMMARY)
    edaic_rows = _safe_csv_rows(EDAIC_FUSION_V1_SUMMARY)

    def _find_row(rows: list[dict[str, str]], track: str) -> dict[str, str] | None:
        for row in rows:
            if row.get("track") == track:
                return row
        return None

    dvlog_row = _find_row(dvlog_rows, "dvlog_vision_v3")
    edaic_row = _find_row(edaic_rows, "edaic_bimodal")
    return {
        "dvlog": {
            "locked_winner": benchmark_winner("dvlog"),
            "preferred_direction": preferred_direction("dvlog"),
            "summary_path": repo_relative(VISION_V3_SHOWDOWN_SUMMARY),
            "metrics": dvlog_row,
        },
        "edaic": {
            "locked_winner": benchmark_winner("edaic"),
            "preferred_direction": preferred_direction("edaic"),
            "summary_path": repo_relative(EDAIC_FUSION_V1_SUMMARY),
            "metrics": edaic_row,
        },
    }


def _decode_image(payload: dict[str, Any]) -> np.ndarray:
    image_path = payload.get("image_path")
    if image_path:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image_path: {image_path}")
        return frame

    image_b64 = payload.get("image_base64")
    if not image_b64:
        raise ValueError("Provide either image_path or image_base64.")
    raw = base64.b64decode(image_b64)
    array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image_base64 into an image.")
    return frame


def _prototype_status(extraction: dict[str, Any]) -> dict[str, Any]:
    quality = extraction["quality"]
    face_valid_ratio = float(quality["face_valid_ratio"])
    if face_valid_ratio >= 1.0:
        readiness = "tracking_ready"
        message = "Visual tracking is healthy enough for prototype feature extraction."
    else:
        readiness = "tracking_weak"
        message = "Feature extraction worked, but face tracking is weak. Do not trust downstream risk estimates yet."
    return {
        "readiness": readiness,
        "message": message,
        "prediction_available": False,
        "why": "The live bridge model is not trained yet, so the server intentionally exposes features and quality only.",
    }


def create_app(config: InferenceServerConfig | None = None) -> Flask:
    config = config or InferenceServerConfig()
    extractor = LiveFeatureExtractor(LiveExtractorConfig(face_embedding_backend=config.face_embedding_backend))

    app = Flask(__name__, static_folder=str(DASHBOARD_DIR), static_url_path="/dashboard")

    @app.get("/")
    def dashboard_index():
        return send_from_directory(DASHBOARD_DIR, "index.html")

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "service": "mindsense-inference-prototype",
                "locked_datasets": ["dvlog", "edaic"],
                "prediction_mode": "features_only_until_bridge_trained",
            }
        )

    @app.get("/api/model-lock")
    def model_lock_status():
        return jsonify(load_model_lock())

    @app.get("/api/benchmarks")
    def benchmark_status():
        return jsonify(_benchmark_snapshot())

    @app.get("/api/bridge-status")
    def bridge_status():
        return jsonify(_bridge_status(config.bridge_root))

    @app.post("/api/extract-frame")
    def extract_frame():
        payload = request.get_json(silent=True) or {}
        frame = _decode_image(payload)
        result = extractor.extract_frame(frame)
        return jsonify(
            {
                "quality": result["quality"],
                "prototype": _prototype_status(result),
                "modalities": {
                    key: {
                        "dim": int(values.shape[0]),
                        "mean_abs": float(np.mean(np.abs(values))),
                        "std": float(np.std(values)),
                    }
                    for key, values in result["modalities"].items()
                },
            }
        )

    @app.post("/api/reset-session")
    def reset_session():
        extractor.reset()
        return jsonify({"status": "reset"})

    return app


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MindSense prototype inference server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--face-embedding-backend", default="color_histogram")
    parser.add_argument("--bridge-root", type=Path, default=DVLOG_BRIDGE_ROOT)
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    config = InferenceServerConfig(
        host=args.host,
        port=args.port,
        debug=args.debug,
        face_embedding_backend=args.face_embedding_backend,
        bridge_root=args.bridge_root,
    )
    app = create_app(config)
    print("=" * 60)
    print("MINDSENSE INFERENCE SERVER")
    print("=" * 60)
    print(f"Config: {json.dumps({**asdict(config), 'bridge_root': str(config.bridge_root)}, indent=2)}")
    app.run(host=config.host, port=config.port, debug=config.debug)


if __name__ == "__main__":
    main()
