"""Server-side OpenCV overlay rendering for live face mesh and pose skeleton."""

from __future__ import annotations

import cv2
import numpy as np


def _to_px(points: list[dict[str, float]], width: int, height: int) -> np.ndarray | None:
    if not points:
        return None
    coords = np.asarray([[float(p["x"]) * width, float(p["y"]) * height] for p in points], dtype=np.float32)
    if len(coords) == 0:
        return None
    return coords.astype(np.int32)


def _draw_poly(
    overlay: np.ndarray,
    points: list[dict[str, float]],
    width: int,
    height: int,
    *,
    stroke: tuple[int, int, int] | None = None,
    fill: tuple[int, int, int] | None = None,
    closed: bool = True,
    line_width: int = 1,
):
    coords = _to_px(points, width, height)
    if coords is None or len(coords) < 2:
        return
    if fill is not None and len(coords) >= 3:
        cv2.fillPoly(overlay, [coords], fill, lineType=cv2.LINE_AA)
    if stroke is not None:
        cv2.polylines(overlay, [coords], closed, stroke, line_width, cv2.LINE_AA)


def _draw_points(
    overlay: np.ndarray,
    points: list[dict[str, float]],
    width: int,
    height: int,
    *,
    color: tuple[int, int, int],
    radius: int = 2,
):
    coords = _to_px(points, width, height)
    if coords is None:
        return
    for point in coords:
        cv2.circle(overlay, tuple(point), radius, color, -1, cv2.LINE_AA)


def draw_face_mesh(frame: np.ndarray, overlay_payload: dict, alpha: float = 0.78):
    regions = overlay_payload.get("face_regions", {})
    if not regions:
        return

    height, width = frame.shape[:2]
    overlay = frame.copy()
    _draw_poly(
        overlay,
        regions.get("oval", []),
        width,
        height,
        fill=(36, 52, 66),
        stroke=(34, 211, 238),
        line_width=2,
    )
    _draw_poly(overlay, regions.get("left_brow", []), width, height, stroke=(129, 140, 248), closed=False, line_width=2)
    _draw_poly(overlay, regions.get("right_brow", []), width, height, stroke=(129, 140, 248), closed=False, line_width=2)
    _draw_poly(overlay, regions.get("left_eye", []), width, height, fill=(28, 82, 65), stroke=(45, 212, 191), line_width=2)
    _draw_poly(overlay, regions.get("right_eye", []), width, height, fill=(28, 82, 65), stroke=(45, 212, 191), line_width=2)
    _draw_poly(overlay, regions.get("mouth", []), width, height, fill=(88, 28, 44), stroke=(251, 113, 133), line_width=2)
    _draw_points(overlay, regions.get("left_iris", []), width, height, color=(251, 191, 36), radius=2)
    _draw_points(overlay, regions.get("right_iris", []), width, height, color=(251, 191, 36), radius=2)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_pose_skeleton(frame: np.ndarray, overlay_payload: dict, alpha: float = 0.82):
    pose_landmarks = overlay_payload.get("pose_landmarks", [])
    if not pose_landmarks:
        return

    height, width = frame.shape[:2]
    overlay = frame.copy()
    connections = overlay_payload.get("pose_connections", [])
    for start, end in connections:
        if start >= len(pose_landmarks) or end >= len(pose_landmarks):
            continue
        p1 = pose_landmarks[start]
        p2 = pose_landmarks[end]
        if float(p1.get("visibility", 1.0)) < 0.35 or float(p2.get("visibility", 1.0)) < 0.35:
            continue
        start_px = (int(float(p1["x"]) * width), int(float(p1["y"]) * height))
        end_px = (int(float(p2["x"]) * width), int(float(p2["y"]) * height))
        cv2.line(overlay, start_px, end_px, (45, 212, 191), 2, cv2.LINE_AA)

    for index in overlay_payload.get("pose_keypoints", []):
        if index >= len(pose_landmarks):
            continue
        point = pose_landmarks[index]
        if float(point.get("visibility", 1.0)) < 0.35:
            continue
        px = (int(float(point["x"]) * width), int(float(point["y"]) * height))
        cv2.circle(overlay, px, 4, (45, 212, 191), -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_no_detection(frame: np.ndarray, message: str = "Step into frame for face and pose tracking"):
    height, width = frame.shape[:2]
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0]
    x = max(18, (width - text_size[0]) // 2)
    y = max(36, (height + text_size[1]) // 2)
    cv2.rectangle(
        frame,
        (x - 18, y - text_size[1] - 18),
        (x + text_size[0] + 18, y + 14),
        (15, 23, 42),
        -1,
    )
    cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (148, 163, 184), 2, cv2.LINE_AA)


def draw_status_panel(frame: np.ndarray, metrics: dict[str, str], fps: float):
    height, width = frame.shape[:2]
    panel_height = 42
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - panel_height), (width, height), (10, 14, 23), -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
    cv2.putText(
        frame,
        f"FPS {fps:.0f}",
        (14, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (226, 232, 240),
        1,
        cv2.LINE_AA,
    )
    x = 110
    for label, value in list(metrics.items())[:4]:
        text = f"{label}: {value}"
        cv2.putText(frame, text, (x, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)
        x += 185


def draw_behavioral_annotations(frame: np.ndarray, indicators: list[dict[str, object]]):
    y_offset = 34
    x_offset = 16
    for indicator in indicators:
        if not indicator.get("active", False):
            continue
        text = str(indicator.get("text", "Signal"))
        severity = float(indicator.get("severity", 0.5))
        if severity < 0.4:
            color = (80, 200, 120)
        elif severity < 0.7:
            color = (0, 180, 255)
        else:
            color = (0, 96, 255)
        cv2.circle(frame, (x_offset, y_offset - 5), 5, color, -1, cv2.LINE_AA)
        cv2.putText(frame, text, (x_offset + 14, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y_offset += 24
