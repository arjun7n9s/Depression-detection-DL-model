"""Vision-first multimodal model for Vision V3."""

from __future__ import annotations

import torch
from torch import nn

from src.model.fusion_v2 import (
    SubjectTransformerAggregator,
    TemporalGRUEncoder,
    TemporalTransformerEncoder,
    _masked_mean,
    _masked_mean_scalar,
)


class VisionV3Model(nn.Module):
    def __init__(
        self,
        modality_dims: dict[str, int],
        modality_groups: dict[str, list[str]],
        quality_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        subject_aggregation: str = "transformer",
        fusion_variant: str = "fixed_prior",
        fixed_prior_weights: tuple[float, float, float] = (0.10, 0.60, 0.30),
        modality_dropout: float = 0.10,
    ):
        super().__init__()
        self.modality_keys = list(modality_dims.keys())
        self.visual_keys = [key for key in modality_groups.get("visual", []) if key in modality_dims]
        self.acoustic_keys = [key for key in modality_groups.get("acoustic", []) if key in modality_dims]
        self.quality_dim = quality_dim
        self.hidden_dim = hidden_dim
        self.subject_aggregation = subject_aggregation
        self.fusion_variant = fusion_variant
        self.modality_dropout = modality_dropout

        encoders = {}
        for key, input_dim in modality_dims.items():
            if key in {"visual", "body_pose"}:
                encoders[key] = TemporalTransformerEncoder(input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=dropout)
            else:
                encoders[key] = TemporalGRUEncoder(input_dim, hidden_dim, num_layers=1, dropout=dropout)
        self.encoders = nn.ModuleDict(encoders)

        self.visual_subject_aggregator = SubjectTransformerAggregator(hidden_dim + quality_dim, hidden_dim, num_layers=2, num_heads=4, dropout=dropout)
        self.visual_head = nn.Linear(hidden_dim, 1)
        self.audio_head = nn.Linear(hidden_dim, 1)
        self.fused_window_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fused_subject_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 + quality_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fused_head = nn.Linear(hidden_dim, 1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim + quality_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.register_buffer("fixed_gate", torch.tensor(fixed_prior_weights, dtype=torch.float32))

    def _apply_modality_dropout(
        self,
        inputs: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.training or self.modality_dropout <= 0 or len(inputs) <= 1:
            return inputs, masks

        batch_windows = next(iter(inputs.values())).shape[0]
        keep = torch.rand(batch_windows, len(self.modality_keys), device=next(iter(inputs.values())).device) > self.modality_dropout
        if self.visual_keys:
            visual_indices = [self.modality_keys.index(key) for key in self.visual_keys]
            visual_keep = keep[:, visual_indices]
            dropped_visual = visual_keep.sum(dim=1) == 0
            if dropped_visual.any():
                keep[dropped_visual, visual_indices[0]] = True

        output_inputs = {}
        output_masks = {}
        for modality_index, key in enumerate(self.modality_keys):
            modality_keep = keep[:, modality_index].view(batch_windows, 1, 1)
            output_inputs[key] = inputs[key] * modality_keep.to(inputs[key].dtype)
            output_masks[key] = masks[key] & modality_keep.squeeze(-1)
        return output_inputs, output_masks

    def _group_mean(self, pooled: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
        available = [pooled[key] for key in keys if key in pooled]
        if not available:
            first = next(iter(pooled.values()))
            return torch.zeros_like(first)
        return torch.stack(available, dim=0).mean(dim=0)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        modalities = batch["modalities"]
        masks = batch["modality_masks"]
        window_quality = batch["window_quality_features"]
        window_mask = batch["window_mask"]
        subject_quality = batch["quality_features"]

        batch_size = window_mask.shape[0]
        num_windows = window_mask.shape[1]
        window_size = next(iter(masks.values())).shape[2]
        flat_inputs = {key: value.reshape(batch_size * num_windows, window_size, value.shape[-1]) for key, value in modalities.items()}
        flat_masks = {key: value.reshape(batch_size * num_windows, window_size) for key, value in masks.items()}

        flat_inputs, flat_masks = self._apply_modality_dropout(flat_inputs, flat_masks)

        pooled_features = {}
        for key in self.modality_keys:
            _, pooled = self.encoders[key](flat_inputs[key], flat_masks[key])
            pooled_features[key] = pooled

        visual_window_embedding = self._group_mean(pooled_features, self.visual_keys)
        audio_window_embedding = self._group_mean(pooled_features, self.acoustic_keys)
        fused_window_embedding = self.fused_window_proj(torch.cat([visual_window_embedding, audio_window_embedding], dim=-1))

        visual_windows = visual_window_embedding.reshape(batch_size, num_windows, self.hidden_dim)
        fused_windows = fused_window_embedding.reshape(batch_size, num_windows, self.hidden_dim)

        visual_subject_inputs = torch.cat([visual_windows, window_quality], dim=-1)
        if self.subject_aggregation == "mean":
            visual_subject_embedding = _masked_mean(visual_subject_inputs, window_mask, dim=1)[..., : self.hidden_dim]
        else:
            visual_subject_embedding = self.visual_subject_aggregator(visual_subject_inputs, window_mask)

        audio_subject_embedding = _masked_mean(audio_window_embedding.reshape(batch_size, num_windows, self.hidden_dim), window_mask, dim=1)
        fused_subject_embedding = self.fused_subject_proj(
            torch.cat([visual_subject_embedding, audio_subject_embedding, subject_quality], dim=-1)
        )

        visual_logit = self.visual_head(visual_subject_embedding).squeeze(-1)
        fused_logit = self.fused_head(fused_subject_embedding).squeeze(-1)
        acoustic_window_logits = self.audio_head(audio_window_embedding).squeeze(-1)
        acoustic_logit = _masked_mean_scalar(acoustic_window_logits.reshape(batch_size, num_windows), window_mask, dim=1)

        if not self.acoustic_keys:
            acoustic_logit = acoustic_logit * 0.0

        if self.fusion_variant == "learned_prior":
            gate_weights = torch.softmax(self.gate_mlp(torch.cat([visual_subject_embedding, subject_quality], dim=-1)), dim=-1)
        else:
            gate_weights = self.fixed_gate.to(visual_subject_embedding.device).unsqueeze(0).expand(batch_size, -1)
            if not self.acoustic_keys:
                gate_weights = gate_weights.clone()
                gate_weights[:, 0] = 0.0
                gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        stacked_logits = torch.stack([acoustic_logit, visual_logit, fused_logit], dim=-1)
        mixture_logit = torch.sum(gate_weights * stacked_logits, dim=-1)

        return {
            "acoustic_logit": acoustic_logit,
            "visual_logit": visual_logit,
            "fused_logit": fused_logit,
            "mixture_logit": mixture_logit,
            "gate_weights": gate_weights,
            "window_embedding": visual_subject_embedding,
            "window_logits": {
                "acoustic": acoustic_window_logits.reshape(batch_size, num_windows),
                "visual": self.visual_head(visual_window_embedding).squeeze(-1).reshape(batch_size, num_windows),
                "fused": self.fused_head(fused_window_embedding).squeeze(-1).reshape(batch_size, num_windows),
            },
            "has_acoustic": bool(self.acoustic_keys),
        }
