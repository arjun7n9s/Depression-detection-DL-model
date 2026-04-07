"""Prototype Flask server with server-side live tracking, MJPEG streaming, and bridged inference."""

from __future__ import annotations

import argparse
import atexit
import base64
import copy
import csv
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.exceptions import BadRequest

from src.inference.camera import CameraConfig, ThreadedCamera, frame_diagnostics, probe_camera_sources
from src.inference.feature_extractor import LiveExtractorConfig, LiveFeatureExtractor
from src.inference.live_runtime import LiveDvlogRuntime, LiveDvlogRuntimeConfig
from src.inference.model_lock import benchmark_winner, load_model_lock, preferred_direction
from src.inference.overlay_renderer import (
    draw_behavioral_annotations,
    draw_face_mesh,
    draw_no_detection,
    draw_pose_skeleton,
    draw_status_panel,
)
from src.inference.reference_behavior_extractor import ReferenceBehaviorExtractor
from src.inference.reference_body_analyzer import ReferenceBodyAnalyzer
from src.inference.reference_face_analyzer import ReferenceFaceAnalyzer
from src.paths import DVLOG_BRIDGE_ROOT, PROJECT_ROOT, PROCESSED_ROOT, repo_relative


DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"
DVLOG_VIDEO_ROOT = PROCESSED_ROOT / "dvlog_video_v3"
VISION_V3_SHOWDOWN_SUMMARY = (
    PROJECT_ROOT / "results" / "benchmark_quality" / "vision_v3_dvlog_showdown" / "final" / "benchmark_summary.csv"
)
EDAIC_FUSION_V1_SUMMARY = (
    PROJECT_ROOT / "results" / "benchmark_quality" / "fusion_v1_locked" / "final" / "benchmark_summary.csv"
)
LOGGER = logging.getLogger("mindsense.inference")


@dataclass
class InferenceServerConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    face_embedding_backend: str = "color_histogram"
    bridge_root: Path = DVLOG_BRIDGE_ROOT
    runtime_device: str = "cpu"
    live_sampling_hz: float = 1.0
    camera_source: int | str = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    camera_backend: str = "auto"
    processing_fps: float = 24.0
    mjpeg_quality: int = 82
    auto_start_camera: bool = False


def _safe_json_load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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
    return {
        "bridge_root": str(bridge_root),
        "bridge_root_repo_relative": repo_relative(bridge_root),
        "paired_subject_dirs": completed_subjects,
        "report_exists": report is not None,
        "report": report,
        "summary": report.get("summary", {}) if report is not None else {
            "complete": completed_subjects,
            "skipped": 0,
            "missing": 0,
            "failed": 0,
        },
    }


def _benchmark_snapshot() -> dict[str, Any]:
    dvlog_rows = _safe_csv_rows(VISION_V3_SHOWDOWN_SUMMARY)
    edaic_rows = _safe_csv_rows(EDAIC_FUSION_V1_SUMMARY)

    def _find_row(rows: list[dict[str, str]], track: str) -> dict[str, str] | None:
        for row in rows:
            if row.get("track") == track:
                return row
        return None

    return {
        "dvlog": {
            "locked_winner": benchmark_winner("dvlog"),
            "preferred_direction": preferred_direction("dvlog"),
            "summary_path": repo_relative(VISION_V3_SHOWDOWN_SUMMARY),
            "metrics": _find_row(dvlog_rows, "dvlog_vision_v3"),
        },
        "edaic": {
            "locked_winner": benchmark_winner("edaic"),
            "preferred_direction": preferred_direction("edaic"),
            "summary_path": repo_relative(EDAIC_FUSION_V1_SUMMARY),
            "metrics": _find_row(edaic_rows, "edaic_bimodal"),
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


def _prototype_status(extraction: dict[str, Any], runtime_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    quality = extraction["quality"]
    face_valid_ratio = float(quality["face_valid_ratio"])
    inference_available = bool(runtime_payload and runtime_payload.get("inference", {}).get("available"))
    bridge_available = bool(runtime_payload and runtime_payload.get("bridge", {}).get("available"))
    if face_valid_ratio >= 1.0:
        readiness = "tracking_ready"
        message = "Visual tracking is healthy enough for prototype feature extraction."
    else:
        readiness = "tracking_weak"
        message = "Feature extraction worked, but face tracking is weak. Do not trust downstream risk estimates yet."
    if bridge_available and inference_available:
        readiness = "live_inference_ready"
        message = "Bridge projection and Vision V3 live inference are both active."
    elif bridge_available:
        readiness = "bridge_ready"
        message = "Bridge projection is active, but live Vision V3 inference is not available yet."
    return {
        "readiness": readiness,
        "message": message,
        "prediction_available": inference_available,
        "why": "Live prediction is only exposed after the bridge and locked Vision V3 path are both available.",
    }


def _idle_runtime_payload() -> dict[str, Any]:
    return {
        "buffered_frames": 0,
        "bridge": {"available": False, "reason": "waiting_for_context"},
        "inference": {"available": False, "reason": "waiting_for_context"},
        "feature_activity": 0.0,
        "artifacts": {},
    }


class LiveTrackingServer:
    def __init__(self, config: InferenceServerConfig):
        self.config = config
        self.extractor = LiveFeatureExtractor(LiveExtractorConfig(face_embedding_backend=config.face_embedding_backend))
        self.runtime = LiveDvlogRuntime(LiveDvlogRuntimeConfig(device=config.runtime_device))
        self.face_analyzer = ReferenceFaceAnalyzer()
        self.body_analyzer = ReferenceBodyAnalyzer()
        self.behavior_extractor = ReferenceBehaviorExtractor()
        self.camera: ThreadedCamera | None = None
        self.processing_thread: threading.Thread | None = None
        self.running = False
        self.lock = threading.Lock()
        self.last_model_sample_ms: int | None = None
        self.last_runtime_payload = _idle_runtime_payload()
        self.frame_seq = 0
        self.fps_frames = 0
        self.fps_start = time.perf_counter()
        self.processing_fps = 0.0
        self.blank_frame_count = 0
        self.last_camera_error: str | None = None
        self.latest_frame_jpeg = self._encode_frame(self._placeholder_frame("Camera idle"))
        self.latest_state = self._default_state("Idle. Start camera to begin live tracking.")

    @staticmethod
    def _build_overlay(face_result: dict, body_result: dict) -> dict[str, Any]:
        face_landmarks = face_result.get("landmarks_raw") if face_result.get("detected") else None
        pose_landmarks = body_result.get("landmarks_raw") if body_result.get("detected") else None

        face_landmarks_list = []
        if face_landmarks is not None and len(face_landmarks):
            face_landmarks_list = [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                for p in np.asarray(face_landmarks, dtype=np.float32).reshape(-1, 3)
            ]

        pose_landmarks_list = []
        if pose_landmarks is not None and len(pose_landmarks):
            pose_landmarks_list = [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "visibility": 1.0}
                for p in np.asarray(pose_landmarks, dtype=np.float32).reshape(-1, 3)
            ]

        from src.inference.feature_extractor import (
            FACE_OVAL,
            LEFT_BROW,
            LEFT_EYE,
            LEFT_IRIS,
            MOUTH_OUTER,
            POSE_CONNECTIONS,
            POSE_KEYPOINTS,
            RIGHT_BROW,
            RIGHT_EYE,
            RIGHT_IRIS,
        )

        def pick(points, indices):
            return [points[idx] for idx in indices if 0 <= idx < len(points)]

        return {
            "face_detected": bool(face_result.get("detected", False)),
            "pose_detected": bool(body_result.get("detected", False)),
            "face_landmarks": face_landmarks_list,
            "face_regions": {
                "oval": pick(face_landmarks_list, FACE_OVAL),
                "left_eye": pick(face_landmarks_list, LEFT_EYE),
                "right_eye": pick(face_landmarks_list, RIGHT_EYE),
                "left_iris": pick(face_landmarks_list, LEFT_IRIS),
                "right_iris": pick(face_landmarks_list, RIGHT_IRIS),
                "mouth": pick(face_landmarks_list, MOUTH_OUTER),
                "left_brow": pick(face_landmarks_list, LEFT_BROW),
                "right_brow": pick(face_landmarks_list, RIGHT_BROW),
            },
            "pose_landmarks": pose_landmarks_list,
            "pose_connections": POSE_CONNECTIONS,
            "pose_keypoints": POSE_KEYPOINTS,
        }

    def _default_state(self, message: str) -> dict[str, Any]:
        recommended_cadence_ms = int(round(1000.0 / max(self.config.live_sampling_hz, 1e-6)))
        return {
            "camera_running": False,
            "camera_backend": None,
            "camera_config": {
                "source": self.config.camera_source,
                "backend": self.config.camera_backend,
                "width": self.config.camera_width,
                "height": self.config.camera_height,
                "fps": self.config.camera_fps,
            },
            "quality": {
                "face_detected": False,
                "face_valid_ratio": 0.0,
                "bbox": [0, 0, 0, 0],
                "overlay": {
                    "face_detected": False,
                    "pose_detected": False,
                    "face_landmarks": [],
                    "face_regions": {},
                    "pose_landmarks": [],
                    "pose_connections": [],
                    "pose_keypoints": [],
                },
                "live_signals": {
                    "smile": 0.0,
                    "brow": 0.0,
                    "eye_openness": 0.0,
                    "mouth_open": 0.0,
                    "face_movement": 0.0,
                    "pose_movement": 0.0,
                    "visual": 0.0,
                    "gaze": 0.0,
                    "affect": 0.0,
                    "body": 0.0,
                    "hands": 0.0,
                },
                "modality_dims": {},
                "display_metrics": {},
                "indicators": [],
            },
            "prototype": {
                "readiness": "idle",
                "message": message,
                "prediction_available": False,
                "why": "Start the server camera loop to begin tracking.",
            },
            "buffered_frames": 0,
            "feature_activity": 0.0,
            "bridge": {"available": False, "reason": "camera_idle"},
            "inference": {"available": False, "reason": "camera_idle"},
            "modalities": {},
            "sync": {
                "frame_seq": 0,
                "processing_ms": 0.0,
                "inter_request_ms": None,
                "recommended_cadence_ms": recommended_cadence_ms,
                "model_sampled": False,
                "capture_fps": 0.0,
            },
            "diagnostics": {
                "blank_frame_count": int(self.blank_frame_count),
                "last_camera_error": self.last_camera_error,
                "frame": frame_diagnostics(frame=None),
            },
        }

    def _placeholder_frame(self, message: str) -> np.ndarray:
        frame = np.zeros((self.config.camera_height, self.config.camera_width, 3), dtype=np.uint8)
        frame[:] = (9, 14, 24)
        cv2.putText(frame, "MindSense Live Tracking", (42, 74), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (226, 232, 240), 2, cv2.LINE_AA)
        cv2.putText(frame, message, (42, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (148, 163, 184), 2, cv2.LINE_AA)
        cv2.putText(frame, "Server-side capture + mesh overlay pipeline", (42, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (94, 234, 212), 1, cv2.LINE_AA)
        return frame

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.mjpeg_quality)],
        )
        return buffer.tobytes() if success else b""

    def start(self) -> tuple[bool, str]:
        with self.lock:
            if self.running:
                return True, "camera_already_running"
            self.latest_state = self._default_state("Booting live camera and tracker.")
        try:
            self.camera = ThreadedCamera(
                CameraConfig(
                    source=self.config.camera_source,
                    width=self.config.camera_width,
                    height=self.config.camera_height,
                    fps=self.config.camera_fps,
                    backend=self.config.camera_backend,
                )
            ).start()
        except Exception as exc:
            self.last_camera_error = str(exc)
            LOGGER.exception("Camera start failed")
            with self.lock:
                self.latest_state = self._default_state(f"Camera start failed: {exc}")
                self.latest_frame_jpeg = self._encode_frame(self._placeholder_frame(f"Camera start failed: {exc}"))
            return False, str(exc)

        self.running = True
        self.last_camera_error = None
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True, name="mindsense-tracker")
        self.processing_thread.start()
        LOGGER.info("Camera started", extra={"source": self.config.camera_source, "backend": self.config.camera_backend})
        return True, "camera_started"

    def stop(self):
        self.running = False
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
        with self.lock:
            self.latest_state["camera_running"] = False
            self.latest_state["prototype"]["message"] = "Camera stopped."
            self.latest_frame_jpeg = self._encode_frame(self._placeholder_frame("Camera stopped"))
        LOGGER.info("Camera stopped")

    def close(self):
        self.stop()
        try:
            self.face_analyzer.release()
        except Exception:
            pass
        try:
            self.body_analyzer.release()
        except Exception:
            pass

    def reset(self):
        self.extractor.reset()
        self.runtime.reset()
        self.last_model_sample_ms = None
        self.last_runtime_payload = _idle_runtime_payload()
        self.blank_frame_count = 0

    def reconfigure_camera(
        self,
        *,
        source: int | str | None = None,
        backend: str | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ) -> dict[str, Any]:
        was_running = self.running
        if was_running:
            self.stop()
        if source is not None:
            self.config.camera_source = source
        if backend is not None:
            self.config.camera_backend = backend
        if width is not None:
            self.config.camera_width = width
        if height is not None:
            self.config.camera_height = height
        if fps is not None:
            self.config.camera_fps = fps
        if was_running:
            ok, message = self.start()
            return {"restarted": True, "ok": ok, "message": message}
        return {"restarted": False, "ok": True, "message": "camera_reconfigured"}

    def _maybe_sample_model(self, extraction: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        recommended_cadence_ms = int(round(1000.0 / max(self.config.live_sampling_hz, 1e-6)))
        now_ms = int(time.time() * 1000)
        should_sample = self.last_model_sample_ms is None or (now_ms - int(self.last_model_sample_ms)) >= recommended_cadence_ms
        if should_sample:
            self.last_runtime_payload = self.runtime.step(extraction, append_frame=True)
            self.last_model_sample_ms = now_ms
        return self.last_runtime_payload, should_sample

    def _build_state(
        self,
        extraction: dict[str, Any],
        runtime_payload: dict[str, Any],
        *,
        processing_ms: float,
        sampled_model: bool,
        blank_camera: bool = False,
    ) -> dict[str, Any]:
        bridge_payload = {
            key: value
            for key, value in runtime_payload["bridge"].items()
            if key not in {"visual_window", "acoustic_window"}
        }
        modalities = {
            key: {
                "dim": int(values.shape[0]),
                "mean_abs": float(np.mean(np.abs(values))),
                "std": float(np.std(values)),
            }
            for key, values in extraction["modalities"].items()
        }
        recommended_cadence_ms = int(round(1000.0 / max(self.config.live_sampling_hz, 1e-6)))
        prototype = _prototype_status(extraction, runtime_payload)
        if blank_camera:
            prototype = {
                "readiness": "camera_blank",
                "message": "Camera opened, but frames are blank. Try a different backend/source or close other camera apps.",
                "prediction_available": False,
                "why": "OpenCV is receiving black frames, so live tracking cannot lock onto the subject.",
            }

        return {
            "camera_running": True,
            "camera_backend": self.camera.backend_name if self.camera is not None else None,
            "camera_config": {
                "source": self.config.camera_source,
                "backend": self.config.camera_backend,
                "width": self.config.camera_width,
                "height": self.config.camera_height,
                "fps": self.config.camera_fps,
            },
            "quality": extraction["quality"],
            "prototype": prototype,
            "buffered_frames": runtime_payload["buffered_frames"],
            "feature_activity": runtime_payload.get("feature_activity", 0.0),
            "bridge": bridge_payload,
            "inference": runtime_payload["inference"],
            "modalities": modalities,
            "sync": {
                "frame_seq": self.frame_seq,
                "processing_ms": processing_ms,
                "inter_request_ms": None,
                "recommended_cadence_ms": recommended_cadence_ms,
                "model_sampled": bool(sampled_model),
                "capture_fps": round(self.processing_fps, 2),
            },
            "diagnostics": {
                "blank_frame_count": int(self.blank_frame_count),
                "last_camera_error": self.last_camera_error,
                "frame": frame_diagnostics(frame=None),
            },
        }

    def _annotate_frame(self, frame: np.ndarray, state: dict[str, Any]) -> np.ndarray:
        output = frame.copy()
        overlay = state["quality"].get("overlay", {})
        if overlay.get("face_detected"):
            draw_face_mesh(output, overlay)
        if overlay.get("pose_detected"):
            draw_pose_skeleton(output, overlay)
        if not overlay.get("face_detected") and not overlay.get("pose_detected"):
            draw_no_detection(output)
        draw_behavioral_annotations(output, state["quality"].get("indicators", []))

        inference = state.get("inference", {})
        metrics = {
            "Face": "locked" if state["quality"].get("face_detected") else "searching",
            "Bridge": "ready" if state["bridge"].get("available") else "warming",
            "Band": inference.get("risk_band", "waiting"),
            "Prob": f"{float(inference.get('probability', 0.0)):.2f}" if inference.get("available") else "--",
        }
        draw_status_panel(output, metrics, state["sync"].get("capture_fps", 0.0))
        return output

    def _processing_loop(self):
        target_delay = 1.0 / max(self.config.processing_fps, 1.0)
        while self.running:
            loop_started = time.perf_counter()
            if self.camera is None:
                time.sleep(0.05)
                continue
            ok, frame = self.camera.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            is_blank = float(frame.std()) < 1.0 and float(frame.mean()) < 1.0
            if is_blank:
                self.blank_frame_count += 1
            else:
                self.blank_frame_count = 0

            face_result = self.face_analyzer.analyze(frame)
            body_result = self.body_analyzer.analyze(frame)
            behavior_payload = self.behavior_extractor.extract(face_result, body_result)

            extraction = self.extractor.extract_frame(frame)
            extraction["quality"]["overlay"] = self._build_overlay(face_result, body_result)
            extraction["quality"]["face_detected"] = bool(face_result.get("detected", False))
            extraction["quality"]["face_valid_ratio"] = 1.0 if face_result.get("detected", False) else 0.0
            extraction["quality"]["live_signals"] = {
                "smile": float(behavior_payload["behavior_signals"]["smile"] / 100.0),
                "brow": float(behavior_payload["behavior_signals"]["brow"] / 100.0),
                "eye_openness": float(behavior_payload["behavior_signals"]["eye_openness"] / 100.0),
                "mouth_open": float(behavior_payload["behavior_signals"]["mouth_open"] / 100.0),
                "face_movement": float(behavior_payload["behavior_signals"]["expressiveness"] / 100.0),
                "pose_movement": float(behavior_payload["behavior_signals"]["movement"] / 100.0),
                "visual": float(max(behavior_payload["behavior_signals"]["expressiveness"], behavior_payload["behavior_signals"]["smile"]) / 100.0),
                "gaze": float(behavior_payload["behavior_signals"]["gaze"] / 100.0),
                "affect": float(behavior_payload["behavior_signals"]["affect"] / 100.0),
                "body": float(behavior_payload["behavior_signals"]["posture"] / 100.0),
                "hands": float(behavior_payload["behavior_signals"]["gesture_activity"] / 100.0),
            }
            extraction["quality"]["display_metrics"] = behavior_payload["display_metrics"]
            extraction["quality"]["indicators"] = behavior_payload["indicators"]
            runtime_payload, sampled_model = self._maybe_sample_model(extraction)

            self.fps_frames += 1
            elapsed = time.perf_counter() - self.fps_start
            if elapsed >= 1.0:
                self.processing_fps = self.fps_frames / elapsed
                self.fps_frames = 0
                self.fps_start = time.perf_counter()

            self.frame_seq += 1
            processing_ms = round((time.perf_counter() - loop_started) * 1000.0, 2)
            state = self._build_state(
                extraction,
                runtime_payload,
                processing_ms=processing_ms,
                sampled_model=sampled_model,
                blank_camera=self.blank_frame_count >= 3,
            )
            state["diagnostics"]["frame"] = frame_diagnostics(frame)
            annotated = self._annotate_frame(frame, state)
            encoded = self._encode_frame(annotated)

            with self.lock:
                self.latest_state = state
                self.latest_frame_jpeg = encoded

            remaining = target_delay - (time.perf_counter() - loop_started)
            if remaining > 0:
                time.sleep(remaining)

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.latest_state)

    def get_frame(self) -> bytes:
        with self.lock:
            return self.latest_frame_jpeg


def create_app(config: InferenceServerConfig | None = None) -> Flask:
    config = config or InferenceServerConfig()
    single_frame_extractor = LiveFeatureExtractor(LiveExtractorConfig(face_embedding_backend=config.face_embedding_backend))
    single_frame_runtime = LiveDvlogRuntime(LiveDvlogRuntimeConfig(device=config.runtime_device))
    live_tracker = LiveTrackingServer(config)

    app = Flask(__name__, static_folder=str(DASHBOARD_DIR), static_url_path="/dashboard")
    app.config["JSON_SORT_KEYS"] = False

    @app.get("/")
    def dashboard_index():
        return send_from_directory(DASHBOARD_DIR, "index.html")

    @app.get("/favicon.ico")
    def favicon():
        return send_from_directory(DASHBOARD_DIR, "favicon.svg", mimetype="image/svg+xml")

    @app.get("/health")
    def health():
        state = live_tracker.get_state()
        return jsonify(
            {
                "status": "ok",
                "service": "mindsense-inference-prototype",
                "locked_datasets": ["dvlog", "edaic"],
                "prediction_mode": "bridged_live_inference_when_context_ready",
                "live_sampling_hz": config.live_sampling_hz,
                "recommended_cadence_ms": int(round(1000.0 / max(config.live_sampling_hz, 1e-6))),
                "camera_running": state.get("camera_running", False),
                "camera_config": state.get("camera_config", {}),
                "camera_backend": state.get("camera_backend"),
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

    @app.get("/api/live-state")
    def live_state():
        return jsonify(live_tracker.get_state())

    @app.get("/api/status")
    def legacy_status():
        return jsonify(live_tracker.get_state())

    @app.get("/api/runtime-info")
    def runtime_info():
        return jsonify(
            {
                "server": {
                    "host": config.host,
                    "port": config.port,
                    "debug": config.debug,
                },
                "camera": {
                    "source": config.camera_source,
                    "backend": config.camera_backend,
                    "width": config.camera_width,
                    "height": config.camera_height,
                    "fps": config.camera_fps,
                    "processing_fps": config.processing_fps,
                },
                "runtime": {
                    "live_sampling_hz": config.live_sampling_hz,
                    "mjpeg_quality": config.mjpeg_quality,
                    "runtime_device": config.runtime_device,
                },
            }
        )

    @app.get("/api/camera/probe")
    def camera_probe():
        max_sources = request.args.get("max_sources", default=3, type=int)
        sources = list(range(max(1, min(max_sources, 6))))
        results = probe_camera_sources(
            sources=sources,
            backends=["auto", "dshow", "msmf"],
            width=config.camera_width,
            height=config.camera_height,
            fps=config.camera_fps,
        )
        return jsonify({"results": results})

    @app.get("/api/camera/config")
    def camera_config():
        return jsonify(
            {
                "source": config.camera_source,
                "backend": config.camera_backend,
                "width": config.camera_width,
                "height": config.camera_height,
                "fps": config.camera_fps,
                "processing_fps": config.processing_fps,
                "camera_running": live_tracker.get_state().get("camera_running", False),
            }
        )

    @app.post("/api/camera/config")
    def update_camera_config():
        payload = request.get_json(silent=True) or {}
        source = payload.get("source", config.camera_source)
        backend = payload.get("backend", config.camera_backend)
        width = int(payload.get("width", config.camera_width))
        height = int(payload.get("height", config.camera_height))
        fps = int(payload.get("fps", config.camera_fps))
        try:
            source = int(str(source))
        except ValueError:
            source = str(source)
        if backend not in {"auto", "any", "dshow", "msmf", "ffmpeg", "gstreamer"}:
            raise BadRequest("Unsupported camera backend.")
        result = live_tracker.reconfigure_camera(
            source=source,
            backend=backend,
            width=width,
            height=height,
            fps=fps,
        )
        return jsonify(
            {
                "status": "ok",
                "result": result,
                "state": live_tracker.get_state(),
            }
        )

    @app.post("/api/camera/start")
    def camera_start():
        ok, message = live_tracker.start()
        return jsonify({"status": "ok" if ok else "error", "message": message, "state": live_tracker.get_state()})

    @app.post("/api/camera/stop")
    def camera_stop():
        live_tracker.stop()
        return jsonify({"status": "ok", "message": "camera_stopped", "state": live_tracker.get_state()})

    @app.get("/video_feed")
    def video_feed():
        def generate():
            while True:
                frame = live_tracker.get_frame()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(1.0 / 30.0)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/extract-frame")
    def extract_frame():
        payload = request.get_json(silent=True) or {}
        started = time.perf_counter()
        frame = _decode_image(payload)
        extraction = single_frame_extractor.extract_frame(frame)
        runtime_payload = single_frame_runtime.step(extraction, append_frame=True)
        processing_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return jsonify(
            {
                "quality": extraction["quality"],
                "prototype": _prototype_status(extraction, runtime_payload),
                "buffered_frames": runtime_payload["buffered_frames"],
                "feature_activity": runtime_payload.get("feature_activity", 0.0),
                "bridge": {
                    key: value
                    for key, value in runtime_payload["bridge"].items()
                    if key not in {"visual_window", "acoustic_window"}
                },
                "inference": runtime_payload["inference"],
                "modalities": {
                    key: {
                        "dim": int(values.shape[0]),
                        "mean_abs": float(np.mean(np.abs(values))),
                        "std": float(np.std(values)),
                    }
                    for key, values in extraction["modalities"].items()
                },
                "sync": {
                    "processing_ms": processing_ms,
                    "recommended_cadence_ms": int(round(1000.0 / max(config.live_sampling_hz, 1e-6))),
                    "model_sampled": True,
                },
            }
        )

    @app.post("/api/reset-session")
    def reset_session():
        live_tracker.reset()
        return jsonify({"status": "reset"})

    app.extensions["mindsense_live_tracker"] = live_tracker
    atexit.register(live_tracker.close)
    if config.auto_start_camera:
        live_tracker.start()
    return app


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MindSense prototype inference server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--face-embedding-backend", default="color_histogram")
    parser.add_argument("--bridge-root", type=Path, default=DVLOG_BRIDGE_ROOT)
    parser.add_argument("--runtime-device", default="cpu")
    parser.add_argument("--live-sampling-hz", type=float, default=1.0)
    parser.add_argument("--camera-source", default=0)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-backend", default="auto")
    parser.add_argument("--processing-fps", type=float, default=24.0)
    parser.add_argument("--mjpeg-quality", type=int, default=82)
    parser.add_argument("--auto-start-camera", action="store_true")
    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    parser = build_argparser()
    args = parser.parse_args()
    camera_source: int | str = args.camera_source
    try:
        camera_source = int(str(args.camera_source))
    except ValueError:
        camera_source = str(args.camera_source)
    config = InferenceServerConfig(
        host=args.host,
        port=args.port,
        debug=args.debug,
        face_embedding_backend=args.face_embedding_backend,
        bridge_root=args.bridge_root,
        runtime_device=args.runtime_device,
        live_sampling_hz=args.live_sampling_hz,
        camera_source=camera_source,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        camera_backend=args.camera_backend,
        processing_fps=args.processing_fps,
        mjpeg_quality=args.mjpeg_quality,
        auto_start_camera=args.auto_start_camera,
    )
    app = create_app(config)
    print("=" * 60)
    print("MINDSENSE INFERENCE SERVER")
    print("=" * 60)
    print(
        f"Config: {json.dumps({**asdict(config), 'bridge_root': str(config.bridge_root)}, indent=2)}"
    )
    try:
        from waitress import serve

        LOGGER.info("Serving with waitress")
        serve(app, host=config.host, port=config.port, threads=8)
    except Exception:
        LOGGER.info("Serving with Flask development server fallback")
        app.run(host=config.host, port=config.port, debug=config.debug, threaded=True)


if __name__ == "__main__":
    main()
