"""Behavioral metric helpers adapted from the smoother reference dashboard."""

from __future__ import annotations

import math

import cv2
import numpy as np


def calculate_ear(eye_landmarks: np.ndarray) -> float:
    if len(eye_landmarks) < 6:
        return 0.25
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    if h == 0:
        return 0.25
    return float((v1 + v2) / (2.0 * h))


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def estimate_head_pose(landmarks_2d: dict[str, np.ndarray], frame_shape) -> tuple[float, float, float]:
    h, w = frame_shape[:2]
    model_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, -330.0, -65.0],
            [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0, -150.0, -125.0],
        ],
        dtype=np.float64,
    )
    try:
        image_points = np.array(
            [
                landmarks_2d["nose"],
                landmarks_2d["chin"],
                landmarks_2d["left_eye"],
                landmarks_2d["right_eye"],
                landmarks_2d["left_mouth"],
                landmarks_2d["right_mouth"],
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError):
        return 0.0, 0.0, 0.0

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = np.hstack((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    return float(euler_angles[0][0]), float(euler_angles[1][0]), float(euler_angles[2][0])


def calculate_gaze_ratio(iris_center: np.ndarray, eye_left_corner: np.ndarray, eye_right_corner: np.ndarray) -> float:
    eye_width = np.linalg.norm(np.asarray(eye_right_corner) - np.asarray(eye_left_corner))
    if eye_width < 1e-6:
        return 0.5
    iris_offset = np.linalg.norm(np.asarray(iris_center) - np.asarray(eye_left_corner))
    return float(np.clip(iris_offset / eye_width, 0.0, 1.0))


def calculate_smile_ratio(mouth_landmarks: dict[str, np.ndarray]) -> float:
    try:
        width = np.linalg.norm(np.asarray(mouth_landmarks["right"]) - np.asarray(mouth_landmarks["left"]))
        height = np.linalg.norm(np.asarray(mouth_landmarks["bottom"]) - np.asarray(mouth_landmarks["top"]))
        if height < 1e-6:
            return 2.0
        return float(width / height)
    except (KeyError, TypeError):
        return 2.0


def normalize_landmarks(landmarks, frame_width: int, frame_height: int) -> np.ndarray:
    coords = []
    for lm in landmarks:
        coords.append([lm.x * frame_width, lm.y * frame_height, lm.z * frame_width])
    return np.asarray(coords, dtype=np.float32)


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def normalize_ratio(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return clip01((value - lo) / (hi - lo))


def angle_to_unit(angle: float, lo: float, hi: float) -> float:
    return clip01((angle - lo) / (hi - lo))


def signed_angle_unit(angle: float, scale: float = 45.0) -> float:
    return float(np.clip(angle / max(scale, 1e-6), -1.0, 1.0))


def atan_degrees(y: float, x: float) -> float:
    return math.degrees(math.atan2(y, x + 1e-6))

