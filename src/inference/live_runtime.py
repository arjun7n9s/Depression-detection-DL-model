"""Live D-Vlog runtime that bridges webcam-compatible features into the locked Vision V3 path."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.bridge_dataset import STUDENT_MODALITY_KEYS
from src.data.vision_v3_datasets import VISION_V3_MODALITY_GROUPS, VisionV3SubjectDataset
from src.model.bridge import BridgeProjectionModel
from src.model.vision_v3 import VisionV3Model
from src.paths import PROJECT_ROOT, repo_relative


BRIDGE_CHECKPOINT = PROJECT_ROOT / "results" / "bridge" / "dvlog_bridge_v1" / "checkpoint.pt"
VISION_V3_ROOT = PROJECT_ROOT / "results" / "benchmark_quality" / "vision_v3_dvlog_showdown" / "final" / "dvlog_vision_v3" / "locked_run"


@dataclass
class LiveDvlogRuntimeConfig:
    bridge_checkpoint: Path = BRIDGE_CHECKPOINT
    vision_v3_root: Path = VISION_V3_ROOT
    device: str = "cpu"


def _safe_sigmoid(value: torch.Tensor) -> float:
    return float(torch.sigmoid(value).detach().cpu().item())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class LiveDvlogRuntime:
    def __init__(self, config: LiveDvlogRuntimeConfig | None = None):
        self.config = config or LiveDvlogRuntimeConfig()
        requested_device = self.config.device
        if requested_device != "cpu" and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        self.bridge_model: BridgeProjectionModel | None = None
        self.bridge_norm_stats: dict[str, dict[str, np.ndarray]] = {}
        self.bridge_window_size = 30

        self.vision_model: VisionV3Model | None = None
        self.vision_norm_stats: dict[str, dict[str, np.ndarray]] = {}
        self.vision_window_size = 9
        self.vision_modalities = ["visual", "face_affect_embed", "body_pose", "hand_pose", "gaze_blink", "acoustic"]

        self.buffers: dict[str, deque[np.ndarray]] = {}
        self.mask_buffers: dict[str, deque[bool]] = {}

        # --- Online calibration state ---
        self._logit_ema = 0.0
        self._logit_ema_alpha = 0.20
        self._prob_ema = 0.5
        self._prob_ema_alpha = 0.35
        self._prediction_count = 0
        self._face_quality_ema = 0.5
        self._face_quality_alpha = 0.20
        self._feature_activity_ema = 0.0

        self._load_bridge()
        self._load_vision_v3()
        self.reset()

    def reset(self):
        self.buffers = {
            "visual": deque(maxlen=max(self.bridge_window_size, self.vision_window_size)),
            "body_pose": deque(maxlen=max(self.bridge_window_size, self.vision_window_size)),
            "hand_pose": deque(maxlen=max(self.bridge_window_size, self.vision_window_size)),
            "gaze_blink": deque(maxlen=max(self.bridge_window_size, self.vision_window_size)),
            "face_affect_embed": deque(maxlen=max(self.bridge_window_size, self.vision_window_size)),
        }
        self.mask_buffers = {
            "visual": deque(maxlen=max(self.bridge_window_size, self.vision_window_size)),
            "body_pose": deque(maxlen=max(self.bridge_window_size, self.vision_window_size)),
            "hand_pose": deque(maxlen=max(self.bridge_window_size, self.vision_window_size)),
            "gaze_blink": deque(maxlen=max(self.bridge_window_size, self.vision_window_size)),
            "face_affect_embed": deque(maxlen=max(self.bridge_window_size, self.vision_window_size)),
        }
        self._logit_ema = 0.0
        self._prob_ema = 0.5
        self._prediction_count = 0
        self._face_quality_ema = 0.5
        self._feature_activity_ema = 0.0

    def _load_bridge(self):
        checkpoint_path = Path(self.config.bridge_checkpoint)
        if not checkpoint_path.exists():
            return
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.bridge_window_size = int(payload["config"]["window_size"])
        self.bridge_norm_stats = {
            key: {
                "mean": np.asarray(value["mean"], dtype=np.float32),
                "std": np.asarray(value["std"], dtype=np.float32),
            }
            for key, value in payload["norm_stats"].items()
        }
        self.bridge_model = BridgeProjectionModel(
            input_dim=int(payload["input_dim"]),
            visual_dim=int(payload["visual_dim"]),
            acoustic_dim=int(payload["acoustic_dim"]),
            hidden_dim=int(payload["config"]["hidden_dim"]),
            num_layers=int(payload["config"]["num_layers"]),
            dropout=float(payload["config"]["dropout"]),
            include_acoustic_target=bool(payload["config"]["include_acoustic_target"]),
        ).to(self.device)
        self.bridge_model.load_state_dict(payload["state_dict"])
        self.bridge_model.eval()

    def _select_best_seed(self, summary: dict[str, Any]) -> int:
        best_seed = None
        best_dev = float("-inf")
        for seed_payload in summary.get("seeds", []):
            dev_macro = float(seed_payload.get("dev_metrics", {}).get("macro_f1", float("-inf")))
            if dev_macro > best_dev:
                best_dev = dev_macro
                best_seed = int(seed_payload["seed"])
        if best_seed is None:
            raise RuntimeError("Could not select a Vision V3 seed from summary.json")
        return best_seed

    def _load_vision_v3(self):
        root = Path(self.config.vision_v3_root)
        summary_path = root / "summary.json"
        config_path = root / "config.json"
        if not summary_path.exists() or not config_path.exists():
            return

        summary = _load_json(summary_path)
        config = _load_json(config_path)
        seed = self._select_best_seed(summary)
        checkpoint_path = root / f"seed_{seed}" / "checkpoint.pt"
        if not checkpoint_path.exists():
            return

        train_dataset = VisionV3SubjectDataset(
            fold="train",
            bundle_name=config["bundle_name"],
            window_size=int(config["window_size"]),
            stride=int(config["stride"]),
            min_valid_ratio=float(config.get("min_valid_ratio", 0.5)),
        )
        self.vision_norm_stats = train_dataset.norm_stats
        self.vision_window_size = int(config["window_size"])

        modality_dims = {
            key: train_dataset.modality_dims[key]
            for key in self.vision_modalities
            if key in train_dataset.modality_dims
        }
        modality_groups = {
            "visual": [key for key in modality_dims if key in VISION_V3_MODALITY_GROUPS["visual"]],
            "acoustic": [key for key in modality_dims if key in VISION_V3_MODALITY_GROUPS["acoustic"]],
        }
        quality_dim = int(train_dataset[0]["quality_features"].shape[0])

        payload = torch.load(checkpoint_path, map_location="cpu")
        self.vision_model = VisionV3Model(
            modality_dims=modality_dims,
            modality_groups=modality_groups,
            quality_dim=quality_dim,
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config["dropout"]),
            subject_aggregation=str(config["subject_aggregation"]),
            fusion_variant=str(config["fusion_variant"]),
            fixed_prior_weights=tuple(config.get("fixed_prior_weights", [0.10, 0.60, 0.30])),
            modality_dropout=0.0,
        ).to(self.device)
        self.vision_model.load_state_dict(payload["state_dict"])
        self.vision_model.eval()

    def _append_frame(self, modalities: dict[str, np.ndarray], modality_masks: dict[str, bool]):
        for key in self.buffers:
            self.buffers[key].append(np.asarray(modalities[key], dtype=np.float32))
            self.mask_buffers[key].append(bool(modality_masks.get(key, True)))

    def _bridge_step(self) -> dict[str, Any]:
        if self.bridge_model is None:
            return {
                "available": False,
                "reason": "bridge_checkpoint_missing",
            }
        if len(self.buffers["body_pose"]) < self.bridge_window_size:
            return {
                "available": False,
                "reason": "insufficient_context",
                "required_frames": self.bridge_window_size,
                "current_frames": len(self.buffers["body_pose"]),
            }

        modality_tensors = []
        for key in STUDENT_MODALITY_KEYS:
            values = np.stack(list(self.buffers[key])[-self.bridge_window_size :], axis=0).astype(np.float32)
            mask = np.asarray(list(self.mask_buffers[key])[-self.bridge_window_size :], dtype=np.float32)
            stats = self.bridge_norm_stats[key]
            values = (values - stats["mean"]) / stats["std"]
            values = values * mask[:, None]
            modality_tensors.append(np.concatenate([values, mask[:, None]], axis=1))

        student_input = np.concatenate(modality_tensors, axis=1)
        tensor = torch.from_numpy(student_input).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.bridge_model(tensor)

        visual_pred = outputs["visual_pred"].detach().cpu().numpy()[0]
        visual_pred = visual_pred * self.bridge_norm_stats["visual_teacher"]["std"] + self.bridge_norm_stats["visual_teacher"]["mean"]
        acoustic_pred = None
        if "acoustic_pred" in outputs:
            acoustic_pred = outputs["acoustic_pred"].detach().cpu().numpy()[0]
            acoustic_pred = acoustic_pred * self.bridge_norm_stats["acoustic_teacher"]["std"] + self.bridge_norm_stats["acoustic_teacher"]["mean"]

        return {
            "available": True,
            "required_frames": self.bridge_window_size,
            "current_frames": len(self.buffers["body_pose"]),
            "visual_window": visual_pred.astype(np.float32),
            "acoustic_window": acoustic_pred.astype(np.float32) if acoustic_pred is not None else None,
            "summary": {
                "visual_mean_abs": float(np.mean(np.abs(visual_pred))),
                "visual_std": float(np.std(visual_pred)),
                "acoustic_mean_abs": float(np.mean(np.abs(acoustic_pred))) if acoustic_pred is not None else 0.0,
                "acoustic_std": float(np.std(acoustic_pred)) if acoustic_pred is not None else 0.0,
            },
        }

    def _quality_vector(
        self,
        modality_arrays: dict[str, np.ndarray],
        modality_masks: dict[str, np.ndarray],
        num_windows: int,
    ) -> np.ndarray:
        shared_valid = np.zeros((self.vision_window_size,), dtype=bool)
        for mask in modality_masks.values():
            shared_valid |= mask
        quality = [
            float(np.mean(shared_valid)) if len(shared_valid) else 0.0,
            0.0,
            0.0,
            float(np.log1p(num_windows)),
        ]
        for key in self.vision_modalities:
            values = modality_arrays[key]
            mask = modality_masks[key]
            valid_values = values[mask]
            if len(valid_values) == 0:
                quality.extend([0.0, 0.0, 0.0])
            else:
                timestep_nonzero = np.any(np.abs(valid_values) > 0, axis=1)
                quality.extend(
                    [
                        float(np.mean(timestep_nonzero)),
                        float(np.mean(np.abs(valid_values))),
                        float(np.mean(np.std(valid_values, axis=0, dtype=np.float64))),
                    ]
                )
        return np.asarray(quality, dtype=np.float32)

    def _calibrate_prediction(self, raw_logit: float, face_quality: float, feature_activity: float) -> tuple[float, dict[str, Any]]:
        """Aggressive online calibration to correct distribution shift + make predictions responsive.

        Key design: the model was trained on D-Vlog features, live webcam features are OOD.
        We correct by essentially removing the systematic model bias and anchoring the
        prediction around the quality of the behavioral tracking signal.
        """
        # --- Track face quality ---
        self._face_quality_ema = (
            self._face_quality_alpha * face_quality
            + (1 - self._face_quality_alpha) * self._face_quality_ema
        )

        # --- Track feature activity (how much features change between frames) ---
        self._feature_activity_ema = (
            0.25 * feature_activity + 0.75 * self._feature_activity_ema
        )

        # --- Track systematic logit bias via EMA ---
        if self._prediction_count == 0:
            self._logit_ema = raw_logit
        else:
            self._logit_ema = (
                self._logit_ema_alpha * raw_logit
                + (1 - self._logit_ema_alpha) * self._logit_ema
            )
        self._prediction_count += 1

        # --- Aggressive bias correction ---
        # Remove 90% of detected systematic drift after a short warmup
        warmup = min(1.0, self._prediction_count / 6.0)
        bias_correction = self._logit_ema * 0.90 * warmup
        corrected_logit = raw_logit - bias_correction

        # --- Quality-weighted blending toward neutral ---
        # Low face quality → push hard toward 50%
        # High face quality → allow model variance (but still capped)
        quality_trust = max(0.15, min(0.75, self._face_quality_ema * 0.75))
        calibrated_logit = quality_trust * corrected_logit

        # --- Feature activity modulation ---
        # Higher activity (user moving, expressions changing) → allow more deviation from 50%
        # Low activity (static face) → pull closer to neutral
        activity_boost = max(0.5, min(1.3, 0.6 + self._feature_activity_ema * 3.0))
        calibrated_logit = calibrated_logit * activity_boost

        # --- Clamp logit to prevent extreme predictions ---
        calibrated_logit = max(-1.5, min(1.5, calibrated_logit))

        # --- Convert to probability ---
        raw_prob = float(torch.sigmoid(torch.tensor(calibrated_logit)).item())

        # --- Light EMA smoothing (responsive but stable) ---
        self._prob_ema = (
            self._prob_ema_alpha * raw_prob
            + (1 - self._prob_ema_alpha) * self._prob_ema
        )

        calibration_meta = {
            "raw_logit": round(raw_logit, 4),
            "logit_ema": round(self._logit_ema, 4),
            "bias_correction": round(bias_correction, 4),
            "corrected_logit": round(corrected_logit, 4),
            "quality_trust": round(quality_trust, 4),
            "activity_boost": round(activity_boost, 4),
            "calibrated_logit": round(calibrated_logit, 4),
            "raw_probability": round(raw_prob, 4),
            "smoothed_probability": round(self._prob_ema, 4),
            "face_quality_ema": round(self._face_quality_ema, 4),
            "feature_activity_ema": round(self._feature_activity_ema, 4),
            "prediction_count": self._prediction_count,
        }
        return self._prob_ema, calibration_meta

    def _vision_step(self, bridge_payload: dict[str, Any], face_quality: float = 1.0, feature_activity: float = 0.0) -> dict[str, Any]:
        if self.vision_model is None:
            return {
                "available": False,
                "reason": "vision_checkpoint_missing",
            }
        if not bridge_payload.get("available"):
            return {
                "available": False,
                "reason": "bridge_unavailable",
            }

        visual_window = bridge_payload["visual_window"][-self.vision_window_size :]
        acoustic_window = bridge_payload["acoustic_window"][-self.vision_window_size :] if bridge_payload["acoustic_window"] is not None else None
        if len(visual_window) < self.vision_window_size:
            return {
                "available": False,
                "reason": "insufficient_context",
                "required_frames": self.vision_window_size,
                "current_frames": len(visual_window),
            }

        modality_arrays = {
            "visual": visual_window.astype(np.float32),
            "face_affect_embed": np.stack(list(self.buffers["face_affect_embed"])[-self.vision_window_size :], axis=0).astype(np.float32),
            "body_pose": np.stack(list(self.buffers["body_pose"])[-self.vision_window_size :], axis=0).astype(np.float32),
            "hand_pose": np.stack(list(self.buffers["hand_pose"])[-self.vision_window_size :], axis=0).astype(np.float32),
            "gaze_blink": np.stack(list(self.buffers["gaze_blink"])[-self.vision_window_size :], axis=0).astype(np.float32),
            "acoustic": acoustic_window.astype(np.float32) if acoustic_window is not None else np.zeros((self.vision_window_size, 25), dtype=np.float32),
        }
        modality_masks = {
            "visual": np.ones((self.vision_window_size,), dtype=bool),
            "face_affect_embed": np.asarray(list(self.mask_buffers["face_affect_embed"])[-self.vision_window_size :], dtype=bool),
            "body_pose": np.asarray(list(self.mask_buffers["body_pose"])[-self.vision_window_size :], dtype=bool),
            "hand_pose": np.asarray(list(self.mask_buffers["hand_pose"])[-self.vision_window_size :], dtype=bool),
            "gaze_blink": np.asarray(list(self.mask_buffers["gaze_blink"])[-self.vision_window_size :], dtype=bool),
            "acoustic": np.ones((self.vision_window_size,), dtype=bool),
        }

        normalized_modalities = {}
        for key, values in modality_arrays.items():
            stats = self.vision_norm_stats[key]
            normalized_modalities[key] = ((values - stats["mean"]) / stats["std"]).astype(np.float32)

        quality_vector = self._quality_vector(modality_arrays, modality_masks, num_windows=1)
        batch = {
            "modalities": {
                key: torch.from_numpy(values).unsqueeze(0).unsqueeze(0).to(self.device)
                for key, values in normalized_modalities.items()
            },
            "modality_masks": {
                key: torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
                for key, mask in modality_masks.items()
            },
            "window_quality_features": torch.from_numpy(quality_vector).unsqueeze(0).unsqueeze(0).to(self.device),
            "quality_features": torch.from_numpy(quality_vector).unsqueeze(0).to(self.device),
            "window_mask": torch.ones((1, 1), dtype=torch.bool, device=self.device),
        }

        with torch.no_grad():
            outputs = self.vision_model(batch)

        # Raw model outputs (uncalibrated)
        raw_mixture_logit = float(outputs["mixture_logit"].detach().cpu().item())
        visual_probability = _safe_sigmoid(outputs["visual_logit"])
        fused_probability = _safe_sigmoid(outputs["fused_logit"])
        acoustic_probability = _safe_sigmoid(outputs["acoustic_logit"])
        gates = outputs["gate_weights"].detach().cpu().numpy()[0]

        # Apply online calibration to correct distribution shift bias
        mixture_probability, calibration = self._calibrate_prediction(raw_mixture_logit, face_quality, feature_activity)

        risk_band = "low"
        if mixture_probability >= 0.62:
            risk_band = "elevated"
        elif mixture_probability >= 0.42:
            risk_band = "moderate"

        return {
            "available": True,
            "probability": mixture_probability,
            "prediction": int(mixture_probability >= 0.5),
            "risk_band": risk_band,
            "branch_probabilities": {
                "acoustic": acoustic_probability,
                "visual": visual_probability,
                "fused": fused_probability,
            },
            "gate_weights": {
                "acoustic": float(gates[0]),
                "visual": float(gates[1]),
                "fused": float(gates[2]),
            },
            "calibration": calibration,
            "window_size": self.vision_window_size,
        }

    def step(self, extraction: dict[str, Any], append_frame: bool = True) -> dict[str, Any]:
        modality_masks = extraction.get("modality_masks", {})
        if append_frame:
            self._append_frame(extraction["modalities"], modality_masks)
        bridge_payload = self._bridge_step()
        face_quality = float(extraction.get("quality", {}).get("face_valid_ratio", 0.0))

        # Compute feature activity (how much modalities changed vs previous frame)
        feature_activity = 0.0
        for key in ["body_pose", "gaze_blink", "face_affect_embed"]:
            buf = self.buffers.get(key)
            if buf and len(buf) >= 2:
                prev = buf[-2]
                curr = buf[-1]
                delta = float(np.mean(np.abs(curr - prev)))
                feature_activity += delta
        feature_activity = min(1.0, feature_activity)

        inference_payload = self._vision_step(
            bridge_payload, face_quality=face_quality, feature_activity=feature_activity
        )
        return {
            "buffered_frames": len(self.buffers["body_pose"]),
            "bridge": bridge_payload,
            "inference": inference_payload,
            "feature_activity": round(feature_activity, 4),
            "artifacts": {
                "bridge_checkpoint": repo_relative(Path(self.config.bridge_checkpoint)),
                "vision_root": repo_relative(Path(self.config.vision_v3_root)),
            },
        }
