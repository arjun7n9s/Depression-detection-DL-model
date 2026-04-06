"""Live-compatible visual feature extraction for the locked D-Vlog Vision V3 path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

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


LIVE_VISUAL_DIM = 136


@dataclass
class LiveExtractorConfig:
    face_embedding_backend: str = "color_histogram"


@dataclass
class LiveExtractorState:
    previous_gray: np.ndarray | None = None
    previous_face_bbox: tuple[int, int, int, int] | None = None


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


class LiveFeatureExtractor:
    def __init__(self, config: LiveExtractorConfig | None = None):
        self.config = config or LiveExtractorConfig()
        self.face_detector = _build_face_detector()
        self.face_embedder = _build_face_embedder(self.config.face_embedding_backend)
        self.state = LiveExtractorState()

    def reset(self):
        self.state = LiveExtractorState()

    def extract_frame(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        bbox, detected_face = _detect_face_bbox(frame_bgr, self.face_detector, self.state.previous_face_bbox)
        self.state.previous_face_bbox = bbox

        body_pose = _extract_body_proxy(frame_bgr, gray, self.state.previous_gray)
        hand_pose = _extract_hand_proxy(gray, self.state.previous_gray)
        gaze_blink = _extract_gaze_blink_proxy(frame_bgr, gray, self.state.previous_gray, bbox, detected_face)
        visual = _expand_visual_proxy(bbox, frame_bgr.shape, gray)
        face_affect = self.face_embedder(frame_bgr, bbox) if self.face_embedder is not None else np.zeros((COLOR_HIST_DIM,), dtype=np.float32)

        self.state.previous_gray = gray

        quality = {
            "face_detected": bool(detected_face),
            "face_valid_ratio": 1.0 if detected_face else 0.0,
            "bbox": tuple(int(value) for value in bbox),
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
            "quality": quality,
        }
