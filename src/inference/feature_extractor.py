"""Live-compatible visual feature extraction and landmark overlays for the locked D-Vlog Vision V3 path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.data.dvlog_video_extractor import (
    BODY_DIM,
    COLOR_HIST_DIM,
    GAZE_BLINK_DIM,
    HAND_DIM,
    _build_face_detector,
    _build_face_embedder,
    _detect_face_bbox,
    _extract_body_proxy,
    _extract_gaze_blink_proxy,
    _extract_hand_proxy,
)
from src.paths import PROJECT_ROOT


LIVE_VISUAL_DIM = 136
TASKS_ROOT = PROJECT_ROOT.parent / "DeepLearning" / "models" / "tasks"
FACE_TASK_PATH = TASKS_ROOT / "face_landmarker.task"
POSE_TASK_PATH = TASKS_ROOT / "pose_landmarker_lite.task"

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]
POSE_KEYPOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


@dataclass
class LiveExtractorConfig:
    face_embedding_backend: str = "color_histogram"


@dataclass
class LiveExtractorState:
    previous_gray: np.ndarray | None = None
    previous_face_bbox: tuple[int, int, int, int] | None = None
    previous_face_center: tuple[float, float] | None = None
    previous_pose_points: dict[int, tuple[float, float]] | None = None


def _expand_visual_proxy(bbox: tuple[int, int, int, int], frame_shape: tuple[int, int, int], gray: np.ndarray) -> np.ndarray:
    height, width = frame_shape[:2]
    x0, y0, x1, y1 = bbox
    center_x = (x0 + x1) * 0.5 / max(width, 1)
    center_y = (y0 + y1) * 0.5 / max(height, 1)
    box_w = max(1, x1 - x0) / max(width, 1)
    box_h = max(1, y1 - y0) / max(height, 1)
    face_crop = gray[y0:y1, x0:x1]
    if face_crop.size == 0:
        base = np.zeros((8,), dtype=np.float32)
    else:
        grid = cv2.resize(face_crop, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        row_profile = grid.mean(axis=1)
        col_profile = grid.mean(axis=0)
        base = np.concatenate(
            [
                grid.flatten(),
                row_profile,
                col_profile,
                np.asarray([center_x, center_y, box_w, box_h, float(face_crop.mean() / 255.0), float(face_crop.std() / 255.0)], dtype=np.float32),
            ],
            axis=0,
        )
    repeats = int(np.ceil(LIVE_VISUAL_DIM / base.size)) if base.size else 1
    return np.tile(base, repeats)[:LIVE_VISUAL_DIM].astype(np.float32)


def _serialize_landmarks(landmarks) -> list[dict[str, float]]:
    return [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in landmarks]


def _serialize_pose_landmarks(landmarks) -> list[dict[str, float]]:
    return [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z), "visibility": float(getattr(lm, "visibility", 1.0))} for lm in landmarks]


def _pick_points(landmarks: list[dict[str, float]], indices: list[int]) -> list[dict[str, float]]:
    result = []
    for index in indices:
        if 0 <= index < len(landmarks):
            result.append(landmarks[index])
    return result


def _blendshape_score(categories, name: str) -> float:
    for category in categories:
        if category.category_name == name:
            return float(category.score)
    return 0.0


class LiveFeatureExtractor:
    def __init__(self, config: LiveExtractorConfig | None = None):
        self.config = config or LiveExtractorConfig()
        self.face_detector = _build_face_detector()
        self.face_embedder = _build_face_embedder(self.config.face_embedding_backend)
        self.state = LiveExtractorState()
        self.face_landmarker = self._build_face_landmarker()
        self.pose_landmarker = self._build_pose_landmarker()

    def _build_face_landmarker(self):
        if not FACE_TASK_PATH.exists():
            return None
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(FACE_TASK_PATH)),
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def _build_pose_landmarker(self):
        if not POSE_TASK_PATH.exists():
            return None
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(POSE_TASK_PATH)),
            num_poses=1,
            min_pose_detection_confidence=0.35,
            min_pose_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def reset(self):
        self.state = LiveExtractorState()

    def _detect_backend_landmarks(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_points = []
        face_blendshapes = []
        if self.face_landmarker is not None:
            face_result = self.face_landmarker.detect(mp_image)
            if face_result.face_landmarks:
                face_points = _serialize_landmarks(face_result.face_landmarks[0])
            if face_result.face_blendshapes:
                face_blendshapes = face_result.face_blendshapes[0]

        pose_points = []
        if self.pose_landmarker is not None:
            pose_result = self.pose_landmarker.detect(mp_image)
            if pose_result.pose_landmarks:
                pose_points = _serialize_pose_landmarks(pose_result.pose_landmarks[0])

        return {
            "face_landmarks": face_points,
            "face_blendshapes": face_blendshapes,
            "pose_landmarks": pose_points,
        }

    def _live_signals(self, face_landmarks: list[dict[str, float]], face_blendshapes, pose_landmarks: list[dict[str, float]]) -> dict[str, float]:
        face_present = len(face_landmarks) > 0
        pose_present = len(pose_landmarks) > 0

        smile = brow = eye_openness = mouth_open = 0.0
        if face_blendshapes:
            smile = 0.5 * (_blendshape_score(face_blendshapes, "mouthSmileLeft") + _blendshape_score(face_blendshapes, "mouthSmileRight"))
            brow = max(
                _blendshape_score(face_blendshapes, "browInnerUp"),
                _blendshape_score(face_blendshapes, "browOuterUpLeft"),
                _blendshape_score(face_blendshapes, "browOuterUpRight"),
                _blendshape_score(face_blendshapes, "browDownLeft"),
                _blendshape_score(face_blendshapes, "browDownRight"),
            )
            eye_openness = 1.0 - 0.5 * (_blendshape_score(face_blendshapes, "eyeBlinkLeft") + _blendshape_score(face_blendshapes, "eyeBlinkRight"))
            mouth_open = max(_blendshape_score(face_blendshapes, "jawOpen"), _blendshape_score(face_blendshapes, "mouthOpen"))

        face_movement = 0.0
        if face_present and len(face_landmarks) > 1:
            centroid = (float(face_landmarks[1]["x"]), float(face_landmarks[1]["y"]))
            if self.state.previous_face_center is not None:
                dx = centroid[0] - self.state.previous_face_center[0]
                dy = centroid[1] - self.state.previous_face_center[1]
                face_movement = float(np.sqrt(dx * dx + dy * dy))
            self.state.previous_face_center = centroid
        else:
            self.state.previous_face_center = None

        pose_movement = 0.0
        if pose_present:
            current_points = {idx: (float(pose_landmarks[idx]["x"]), float(pose_landmarks[idx]["y"])) for idx in [11, 12, 15, 16] if idx < len(pose_landmarks)}
            if self.state.previous_pose_points:
                deltas = []
                for idx, point in current_points.items():
                    previous = self.state.previous_pose_points.get(idx)
                    if previous is None:
                        continue
                    deltas.append(float(np.sqrt((point[0] - previous[0]) ** 2 + (point[1] - previous[1]) ** 2)))
                if deltas:
                    pose_movement = float(np.mean(deltas))
            self.state.previous_pose_points = current_points
        else:
            self.state.previous_pose_points = None

        visual_signal = clamp = lambda v: max(0.0, min(1.0, v))
        visual = clamp(0.55 + 0.45 * max(0.0, 1.0 - face_movement * 18)) if face_present else 0.0
        gaze = clamp(0.75 * eye_openness + 0.25 * max(0.0, 1.0 - face_movement * 18)) if face_present else 0.0
        affect = clamp(max(smile, brow * 0.9, mouth_open * 0.8)) if face_present else 0.0
        body = clamp(0.35 + pose_movement * 20) if pose_present else 0.0
        hands = clamp(pose_movement * 22) if pose_present else 0.0

        return {
            "face_present": float(face_present),
            "pose_present": float(pose_present),
            "smile": smile,
            "brow": brow,
            "eye_openness": eye_openness,
            "mouth_open": mouth_open,
            "face_movement": face_movement,
            "pose_movement": pose_movement,
            "visual": visual,
            "gaze": gaze,
            "affect": affect,
            "body": body,
            "hands": hands,
        }

    def extract_frame(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        bbox, fallback_detected_face = _detect_face_bbox(frame_bgr, self.face_detector, self.state.previous_face_bbox)
        self.state.previous_face_bbox = bbox

        landmark_payload = self._detect_backend_landmarks(frame_bgr)
        face_landmarks = landmark_payload["face_landmarks"]
        pose_landmarks = landmark_payload["pose_landmarks"]
        face_detected = len(face_landmarks) > 0 or bool(fallback_detected_face)
        pose_detected = len(pose_landmarks) > 0

        body_pose = _extract_body_proxy(frame_bgr, gray, self.state.previous_gray)
        hand_pose = _extract_hand_proxy(gray, self.state.previous_gray)
        gaze_blink = _extract_gaze_blink_proxy(frame_bgr, gray, self.state.previous_gray, bbox, face_detected)
        visual = _expand_visual_proxy(bbox, frame_bgr.shape, gray)
        face_affect = self.face_embedder(frame_bgr, bbox) if self.face_embedder is not None else np.zeros((COLOR_HIST_DIM,), dtype=np.float32)

        signals = self._live_signals(face_landmarks, landmark_payload["face_blendshapes"], pose_landmarks)
        self.state.previous_gray = gray

        overlay = {
            "face_detected": bool(len(face_landmarks) > 0),
            "pose_detected": bool(pose_detected),
            "face_landmarks": face_landmarks,
            "face_regions": {
                "oval": _pick_points(face_landmarks, FACE_OVAL),
                "left_eye": _pick_points(face_landmarks, LEFT_EYE),
                "right_eye": _pick_points(face_landmarks, RIGHT_EYE),
                "left_iris": _pick_points(face_landmarks, LEFT_IRIS),
                "right_iris": _pick_points(face_landmarks, RIGHT_IRIS),
                "mouth": _pick_points(face_landmarks, MOUTH_OUTER),
                "left_brow": _pick_points(face_landmarks, LEFT_BROW),
                "right_brow": _pick_points(face_landmarks, RIGHT_BROW),
            },
            "pose_landmarks": pose_landmarks,
            "pose_connections": POSE_CONNECTIONS,
            "pose_keypoints": POSE_KEYPOINTS,
        }

        quality = {
            "face_detected": bool(face_detected),
            "face_valid_ratio": 1.0 if len(face_landmarks) > 0 else (0.5 if fallback_detected_face else 0.0),
            "bbox": tuple(int(value) for value in bbox),
            "overlay": overlay,
            "live_signals": signals,
            "modality_dims": {
                "visual": LIVE_VISUAL_DIM,
                "body_pose": BODY_DIM,
                "hand_pose": HAND_DIM,
                "gaze_blink": GAZE_BLINK_DIM,
                "face_affect_embed": int(face_affect.shape[0]),
            },
        }
        return {
            "modalities": {
                "visual": visual,
                "body_pose": body_pose,
                "hand_pose": hand_pose,
                "gaze_blink": gaze_blink,
                "face_affect_embed": face_affect.astype(np.float32),
            },
            "modality_masks": {
                "visual": bool(len(face_landmarks) > 0),
                "body_pose": bool(pose_detected),
                "hand_pose": bool(pose_detected),
                "gaze_blink": bool(len(face_landmarks) > 0),
                "face_affect_embed": bool(len(face_landmarks) > 0),
            },
            "quality": quality,
        }
