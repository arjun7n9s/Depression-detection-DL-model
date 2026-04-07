"""Threaded webcam capture for smooth server-side live tracking."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class CameraConfig:
    source: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: str = "auto"


class ThreadedCamera:
    """Continuously captures the latest webcam frame on a background thread."""

    def __init__(self, config: CameraConfig | None = None):
        self.config = config or CameraConfig()
        self.source = self._normalize_source(self.config.source)
        self.lock = threading.Lock()
        self.frame = None
        self.frame_updated = False
        self.running = False
        self.thread: threading.Thread | None = None
        self.cap: cv2.VideoCapture | None = None
        self.backend_name = "unopened"
        self.open_attempts: list[str] = []

    @staticmethod
    def _normalize_source(src: int | str) -> int | str:
        if isinstance(src, int):
            return src
        src_str = str(src).strip()
        try:
            return int(src_str)
        except ValueError:
            return src_str

    @staticmethod
    def _resolve_backend_candidates(name: str) -> list[tuple[str, int]]:
        lookup = {
            "auto": cv2.CAP_ANY,
            "any": cv2.CAP_ANY,
            "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
            "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
            "ffmpeg": getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY),
            "gstreamer": getattr(cv2, "CAP_GSTREAMER", cv2.CAP_ANY),
        }
        key = str(name).strip().lower()
        if key not in lookup:
            raise RuntimeError(f"Unsupported camera backend '{name}'.")
        if key in {"auto", "any"}:
            ordered = ["any", "dshow", "msmf", "ffmpeg"]
            return [(item, lookup[item]) for item in ordered if item in lookup]
        return [(key, lookup[key])]

    @staticmethod
    def _warmup_capture(cap: cv2.VideoCapture, attempts: int = 18, delay: float = 0.08) -> tuple[bool, object]:
        for _ in range(attempts):
            ret, frame = cap.read()
            if ret and frame is not None:
                # Reject blank frames that some Windows camera backends return
                # even though the device technically "opened" successfully.
                if float(frame.std()) < 1.0 and float(frame.mean()) < 1.0:
                    time.sleep(delay)
                    continue
                return ret, frame
            time.sleep(delay)
        return False, None

    def _open_capture(self):
        for backend_name, backend in self._resolve_backend_candidates(self.config.backend):
            cap = cv2.VideoCapture(self.source, backend)
            if not cap or not cap.isOpened():
                self.open_attempts.append(f"{backend_name}: open failed")
                if cap is not None:
                    cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            ret, frame = self._warmup_capture(cap)
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.frame_updated = True
                self.cap = cap
                self.backend_name = backend_name
                return

            self.open_attempts.append(f"{backend_name}: read failed")
            cap.release()

        attempted = ", ".join(self.open_attempts) if self.open_attempts else self.config.backend
        raise RuntimeError(
            f"Failed to open camera source '{self.source}'. Attempted backends: {attempted}."
        )

    def start(self):
        if self.running:
            return self
        self._open_capture()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True, name="mindsense-camera")
        self.thread.start()
        return self

    def _capture_loop(self):
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.frame_updated = True
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            self.frame_updated = False
            return True, self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        with self.lock:
            self.frame = None
            self.frame_updated = False
        self.backend_name = "stopped"


def frame_diagnostics(frame: np.ndarray | None) -> dict[str, Any]:
    if frame is None:
        return {
            "available": False,
            "blank": True,
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
            "shape": None,
        }
    return {
        "available": True,
        "blank": bool(float(frame.std()) < 1.0 and float(frame.mean()) < 1.0),
        "mean": float(frame.mean()),
        "std": float(frame.std()),
        "min": int(frame.min()),
        "max": int(frame.max()),
        "shape": [int(v) for v in frame.shape],
    }


def probe_camera_sources(
    *,
    sources: list[int | str] | None = None,
    backends: list[str] | None = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> list[dict[str, Any]]:
    """Probe common camera source/backend pairs and report which ones are viable."""

    sources = sources or [0, 1, 2]
    backends = backends or ["auto", "dshow", "msmf"]
    results: list[dict[str, Any]] = []

    for source in sources:
        for backend in backends:
            try:
                camera = ThreadedCamera(
                    CameraConfig(
                        source=source,
                        width=width,
                        height=height,
                        fps=fps,
                        backend=backend,
                    )
                ).start()
                time.sleep(0.4)
                ok, frame = camera.read()
                diag = frame_diagnostics(frame if ok else None)
                results.append(
                    {
                        "source": source,
                        "backend": backend,
                        "opened": True,
                        "backend_name": camera.backend_name,
                        "diagnostics": diag,
                    }
                )
                camera.stop()
            except Exception as exc:
                results.append(
                    {
                        "source": source,
                        "backend": backend,
                        "opened": False,
                        "error": str(exc),
                    }
                )
    return results
