"""Reference-style live metrics + indicators for dashboard tracking."""

from __future__ import annotations

import numpy as np

from src.inference.reference_metrics import clip01, normalize_ratio


class ReferenceBehaviorExtractor:
    def extract(self, face_result: dict, body_result: dict) -> dict:
        face_detected = bool(face_result.get("detected", False))
        body_detected = bool(body_result.get("detected", False))

        smile = normalize_ratio(float(face_result.get("smile_ratio", 2.0)), 1.7, 4.8) if face_detected else 0.0
        eye_openness = normalize_ratio(float(face_result.get("ear_avg", 0.2)), 0.16, 0.34) if face_detected else 0.0
        gaze_center = 0.5 * (
            float(face_result.get("gaze_ratio_left", 0.5)) + float(face_result.get("gaze_ratio_right", 0.5))
        )
        gaze = (1.0 - min(1.0, abs(gaze_center - 0.5) * 2.0)) if face_detected else 0.0
        expressiveness = clip01(float(face_result.get("expressiveness", 0.0)) / 6.0) if face_detected else 0.0
        brow = clip01(0.5 + abs(float(face_result.get("head_pitch", 0.0))) / 40.0) if face_detected else 0.0
        mouth_open = clip01(1.0 / max(float(face_result.get("smile_ratio", 2.0)), 1e-6)) if face_detected else 0.0

        posture = clip01(float(body_result.get("spine_angle", 180.0)) / 180.0) if body_detected else 0.0
        movement = clip01(float(body_result.get("movement_freq", 0.0)) / 0.01) if body_detected else 0.0
        gesture_activity = clip01(float(body_result.get("gesture_activity", 0.0)) / 0.01) if body_detected else 0.0
        stillness = clip01(float(body_result.get("overall_stillness", 0.0))) if body_detected else 0.0
        forward_lean = clip01(float(body_result.get("forward_lean", 0.0))) if body_detected else 0.0

        behavior_signals = {
            "eye_openness": eye_openness * 100.0,
            "blink_rate": float(face_result.get("blink_rate", 0.0)),
            "smile": smile * 100.0,
            "expressiveness": expressiveness * 100.0,
            "gaze_down": bool(face_result.get("gaze_down", False)),
            "head_pitch": float(face_result.get("head_pitch", 0.0)),
            "head_yaw": float(face_result.get("head_yaw", 0.0)),
            "posture": posture * 100.0,
            "movement": movement * 100.0,
            "gesture_activity": gesture_activity * 100.0,
            "stillness": stillness * 100.0,
            "forward_lean": forward_lean * 100.0,
            "face_detected": face_detected,
            "body_detected": body_detected,
            "brow": brow * 100.0,
            "mouth_open": mouth_open * 100.0,
            "gaze": gaze * 100.0,
            "affect": max(smile, expressiveness) * 100.0,
        }

        display_metrics = {
            "face_detected": face_detected,
            "body_detected": body_detected,
            "eye_openness": round(behavior_signals["eye_openness"], 1),
            "blink_rate": round(behavior_signals["blink_rate"], 1),
            "smile": round(behavior_signals["smile"], 1),
            "expressiveness": round(behavior_signals["expressiveness"], 1),
            "head_pitch": round(behavior_signals["head_pitch"], 1),
            "head_yaw": round(behavior_signals["head_yaw"], 1),
            "gaze_down": behavior_signals["gaze_down"],
            "posture": round(behavior_signals["posture"], 1),
            "movement": round(behavior_signals["movement"], 1),
            "gesture_activity": round(behavior_signals["gesture_activity"], 1),
            "stillness": round(behavior_signals["stillness"], 1),
            "slouching": bool(body_result.get("slouch_detected", False)),
        }

        indicators = []
        if behavior_signals["gaze_down"]:
            indicators.append({"text": "Reduced Eye Contact", "active": True, "severity": 0.5})
        if behavior_signals["posture"] < 60:
            indicators.append({"text": "Posture Drift", "active": True, "severity": 0.5})
        if behavior_signals["expressiveness"] < 25 and face_detected:
            indicators.append({"text": "Low Expressiveness", "active": True, "severity": 0.6})
        if behavior_signals["movement"] < 20 and body_detected:
            indicators.append({"text": "Reduced Movement", "active": True, "severity": 0.5})

        return {
            "quality_score": float(np.clip((1.0 if face_detected else 0.0) * 0.7 + (1.0 if body_detected else 0.0) * 0.3, 0.0, 1.0)),
            "behavior_signals": behavior_signals,
            "display_metrics": display_metrics,
            "indicators": indicators,
        }

