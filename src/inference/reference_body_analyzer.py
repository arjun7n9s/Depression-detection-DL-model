"""Reference-style pose analyzer for smooth live body tracking."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.inference.feature_extractor import POSE_TASK_PATH
from src.inference.reference_metrics import calculate_angle, normalize_landmarks


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_WRIST = 15
RIGHT_WRIST = 16
NOSE = 0


class ReferenceBodyAnalyzer:
    def __init__(self):
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(POSE_TASK_PATH)),
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = vision.PoseLandmarker.create_from_options(options)
        self._prev_landmarks = None
        self._movement_history: list[float] = []
        self._gesture_history: list[float] = []

    def analyze(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.pose.detect(mp_image)
        if not results.pose_landmarks:
            return {"detected": False}

        lms = results.pose_landmarks[0]
        landmarks_px = normalize_landmarks(lms, w, h)
        landmarks_raw = np.asarray([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

        mid_ear = (landmarks_px[LEFT_EAR, :2] + landmarks_px[RIGHT_EAR, :2]) / 2
        mid_shoulder = (landmarks_px[LEFT_SHOULDER, :2] + landmarks_px[RIGHT_SHOULDER, :2]) / 2
        mid_hip = (landmarks_px[LEFT_HIP, :2] + landmarks_px[RIGHT_HIP, :2]) / 2
        spine_angle = calculate_angle(mid_ear, mid_shoulder, mid_hip)
        slouch_detected = spine_angle < 155

        left_sh_y = landmarks_px[LEFT_SHOULDER, 1]
        right_sh_y = landmarks_px[RIGHT_SHOULDER, 1]
        shoulder_width = abs(landmarks_px[LEFT_SHOULDER, 0] - landmarks_px[RIGHT_SHOULDER, 0])
        shoulder_asymmetry = abs(left_sh_y - right_sh_y) / shoulder_width if shoulder_width > 0 else 0.0

        if self._prev_landmarks is not None:
            key_joints = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_WRIST, RIGHT_WRIST, NOSE]
            displacements = [np.linalg.norm(landmarks_px[j, :2] - self._prev_landmarks[j, :2]) for j in key_joints]
            movement = np.mean(displacements) / max(w, h)
            self._movement_history.append(float(movement))
            if len(self._movement_history) > 42:
                self._movement_history = self._movement_history[-42:]
            movement_freq = float(np.mean(self._movement_history))

            left_wrist_disp = np.linalg.norm(landmarks_px[LEFT_WRIST, :2] - self._prev_landmarks[LEFT_WRIST, :2]) / max(w, h)
            right_wrist_disp = np.linalg.norm(landmarks_px[RIGHT_WRIST, :2] - self._prev_landmarks[RIGHT_WRIST, :2]) / max(w, h)
            gesture = (left_wrist_disp + right_wrist_disp) / 2.0
            self._gesture_history.append(float(gesture))
            if len(self._gesture_history) > 42:
                self._gesture_history = self._gesture_history[-42:]
            gesture_activity = float(np.mean(self._gesture_history))
        else:
            movement_freq = 0.0
            gesture_activity = 0.0

        self._prev_landmarks = landmarks_px.copy()

        nose_y = landmarks_px[NOSE, 1]
        shoulder_y = mid_shoulder[1]
        torso_length = np.linalg.norm(mid_shoulder - mid_hip)
        forward_lean = abs(nose_y - shoulder_y) / torso_length if torso_length > 0 else 0.0
        forward_lean = float(np.clip(forward_lean, 0.0, 1.0))
        stillness = float(1.0 - np.clip(movement_freq / 0.01, 0.0, 1.0))

        return {
            "detected": True,
            "landmarks_raw": landmarks_raw,
            "landmarks_px": landmarks_px,
            "spine_angle": float(spine_angle),
            "shoulder_asymmetry": float(shoulder_asymmetry),
            "movement_freq": float(movement_freq),
            "gesture_activity": float(gesture_activity),
            "forward_lean": float(forward_lean),
            "overall_stillness": float(stillness),
            "slouch_detected": bool(slouch_detected),
        }

    def release(self):
        try:
            self.pose.close()
        except Exception:
            pass
