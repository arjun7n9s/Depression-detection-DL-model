"""Reference-style face analyzer for smooth live webcam tracking."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.inference.reference_metrics import (
    calculate_ear,
    calculate_gaze_ratio,
    calculate_smile_ratio,
    estimate_head_pose,
    normalize_landmarks,
)
from src.inference.feature_extractor import FACE_TASK_PATH


LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14


class ReferenceFaceAnalyzer:
    def __init__(self):
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(FACE_TASK_PATH)),
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_mesh = vision.FaceLandmarker.create_from_options(options)
        self._blink_counter = 0
        self._blink_total = 0
        self._ear_below_threshold = False
        self._frame_count = 0
        self._prev_landmarks = None
        self._landmark_velocities: list[float] = []

    def analyze(self, frame: np.ndarray) -> dict:
        self._frame_count += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.face_mesh.detect(mp_image)
        if not results.face_landmarks:
            return {"detected": False}

        face_lms = results.face_landmarks[0]
        landmarks_px = normalize_landmarks(face_lms, w, h)
        landmarks_raw = np.asarray([[lm.x, lm.y, lm.z] for lm in face_lms], dtype=np.float32)

        left_eye_pts = landmarks_px[LEFT_EYE_IDX, :2]
        right_eye_pts = landmarks_px[RIGHT_EYE_IDX, :2]
        ear_left = calculate_ear(left_eye_pts)
        ear_right = calculate_ear(right_eye_pts)
        ear_avg = (ear_left + ear_right) / 2.0

        if ear_avg < 0.18:
            self._ear_below_threshold = True
            self._blink_counter += 1
        else:
            self._ear_below_threshold = False
            self._blink_counter = 0

        if self._ear_below_threshold and self._blink_counter >= 2:
            self._blink_total += 1
            self._blink_counter = 0
            self._ear_below_threshold = False

        elapsed_seconds = max(self._frame_count / 21.0, 1.0)
        blink_rate = (self._blink_total / elapsed_seconds) * 60.0

        left_iris = landmarks_px[LEFT_IRIS_CENTER, :2]
        right_iris = landmarks_px[RIGHT_IRIS_CENTER, :2]
        gaze_left = calculate_gaze_ratio(left_iris, landmarks_px[263, :2], landmarks_px[362, :2])
        gaze_right = calculate_gaze_ratio(right_iris, landmarks_px[33, :2], landmarks_px[133, :2])

        head_lms = {
            "nose": landmarks_px[NOSE_TIP, :2],
            "chin": landmarks_px[CHIN, :2],
            "left_eye": landmarks_px[LEFT_EYE_OUTER, :2],
            "right_eye": landmarks_px[RIGHT_EYE_OUTER, :2],
            "left_mouth": landmarks_px[LEFT_MOUTH, :2],
            "right_mouth": landmarks_px[RIGHT_MOUTH, :2],
        }
        pitch, yaw, roll = estimate_head_pose(head_lms, frame.shape)
        gaze_down = pitch < -15

        mouth_lms = {
            "left": landmarks_px[MOUTH_LEFT, :2],
            "right": landmarks_px[MOUTH_RIGHT, :2],
            "top": landmarks_px[MOUTH_TOP, :2],
            "bottom": landmarks_px[MOUTH_BOTTOM, :2],
        }
        smile = calculate_smile_ratio(mouth_lms)

        if self._prev_landmarks is not None:
            velocity = np.linalg.norm(landmarks_px - self._prev_landmarks, axis=1).mean()
            self._landmark_velocities.append(float(velocity))
            max_velocities = int(21 * 2)
            if len(self._landmark_velocities) > max_velocities:
                self._landmark_velocities = self._landmark_velocities[-max_velocities:]
            expressiveness = float(np.std(self._landmark_velocities))
        else:
            expressiveness = 0.0
        self._prev_landmarks = landmarks_px.copy()

        return {
            "detected": True,
            "landmarks_raw": landmarks_raw,
            "landmarks_px": landmarks_px,
            "ear_left": float(ear_left),
            "ear_right": float(ear_right),
            "ear_avg": float(ear_avg),
            "blink_rate": float(blink_rate),
            "gaze_ratio_left": float(gaze_left),
            "gaze_ratio_right": float(gaze_right),
            "gaze_down": bool(gaze_down),
            "head_pitch": float(pitch),
            "head_yaw": float(yaw),
            "head_roll": float(roll),
            "smile_ratio": float(smile),
            "expressiveness": float(expressiveness),
        }

    def release(self):
        try:
            self.face_mesh.close()
        except Exception:
            pass
