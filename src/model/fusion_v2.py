"""Fusion V2 heterogeneous multimodal model."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    summed = (values * weights.unsqueeze(-1)).sum(dim=dim)
    denom = weights.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return summed / denom


def _masked_mean_scalar(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    summed = (values * weights).sum(dim=dim)
    denom = weights.sum(dim=dim).clamp_min(1.0)
    return summed / denom


def _stabilize_sequence_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2:
        return mask
    safe_mask = mask.clone()
    empty_rows = safe_mask.sum(dim=1) == 0
    if empty_rows.any():
        safe_mask[empty_rows, 0] = True
    return safe_mask


def _sinusoidal_position_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TemporalGRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = _stabilize_sequence_mask(mask)
        projected = self.input_proj(self.input_norm(inputs))
        encoded, _ = self.gru(projected)
        encoded = self.dropout(encoded)
        pooled = _masked_mean(encoded, mask, dim=1)
        return encoded, pooled


class TemporalTransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = _stabilize_sequence_mask(mask)
        encoded = self.input_proj(self.input_norm(inputs))
        encoded = encoded + _sinusoidal_position_encoding(encoded.shape[1], encoded.shape[2], encoded.device).unsqueeze(0)
        encoded = self.encoder(encoded, src_key_padding_mask=~mask)
        encoded = self.dropout(encoded)
        pooled = _masked_mean(encoded, mask, dim=1)
        return encoded, pooled


class LatentFusionEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        latent_num: int = 8,
        cross_attn_heads: int = 4,
        self_attn_layers: int = 4,
        self_attn_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(latent_num, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, cross_attn_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=self_attn_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_stack = nn.TransformerEncoder(layer, num_layers=self_attn_layers)

    def forward(self, modality_sequences: list[torch.Tensor], modality_masks: list[torch.Tensor]) -> torch.Tensor:
        batch_size = modality_sequences[0].shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        for sequence, mask in zip(modality_sequences, modality_masks, strict=False):
            mask = _stabilize_sequence_mask(mask)
            update, _ = self.cross_attn(latents, sequence, sequence, key_padding_mask=~mask)
            latents = self.cross_norm(latents + update)
        latents = self.self_stack(latents)
        return latents.mean(dim=1)


class SubjectTransformerAggregator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, window_features: torch.Tensor, window_mask: torch.Tensor) -> torch.Tensor:
        window_mask = _stabilize_sequence_mask(window_mask)
        projected = self.input_proj(window_features)
        cls = self.cls_token.expand(projected.shape[0], -1, -1)
        tokens = torch.cat([cls, projected], dim=1)
        cls_mask = torch.ones((window_mask.shape[0], 1), dtype=torch.bool, device=window_mask.device)
        token_mask = torch.cat([cls_mask, window_mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~token_mask)
        return encoded[:, 0]


class FusionV2Model(nn.Module):
    def __init__(
        self,
        modality_dims: dict[str, int],
        modality_groups: dict[str, list[str]],
        quality_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        latent_num: int = 8,
        cross_attn_heads: int = 4,
        self_attn_layers: int = 4,
        self_attn_heads: int = 4,
        gate_variant: str = "quality_mixture",
        subject_aggregation: str = "transformer",
        modality_dropout: float = 0.15,
    ):
        super().__init__()
        self.modality_keys = list(modality_dims.keys())
        self.modality_groups = modality_groups
        self.quality_dim = quality_dim
        self.hidden_dim = hidden_dim
        self.gate_variant = gate_variant
        self.subject_aggregation = subject_aggregation
        self.modality_dropout = modality_dropout

        encoders = {}
        for key, input_dim in modality_dims.items():
            if key == "pose_gaze_aus":
                encoders[key] = TemporalTransformerEncoder(input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=dropout)
            else:
                encoders[key] = TemporalGRUEncoder(input_dim, hidden_dim, num_layers=1, dropout=dropout)
        self.encoders = nn.ModuleDict(encoders)

        self.fusion = LatentFusionEncoder(
            hidden_dim=hidden_dim,
            latent_num=latent_num,
            cross_attn_heads=cross_attn_heads,
            self_attn_layers=self_attn_layers,
            self_attn_heads=self_attn_heads,
            dropout=dropout,
        )
        self.per_modality_heads = nn.ModuleDict({key: nn.Linear(hidden_dim, 1) for key in self.modality_keys})
        self.acoustic_head = nn.Linear(hidden_dim, 1)
        self.visual_head = nn.Linear(hidden_dim, 1)
        self.fused_window_head = nn.Linear(hidden_dim, 1)
        self.subject_aggregator = SubjectTransformerAggregator(hidden_dim + quality_dim, hidden_dim, num_layers=2, num_heads=4, dropout=dropout)
        self.fused_subject_head = nn.Linear(hidden_dim, 1)
        self.phq_head = nn.Linear(hidden_dim, 1)
        self.phq_severity_head = nn.Linear(hidden_dim, 5)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim + quality_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def _apply_modality_dropout(
        self,
        inputs: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.training or self.modality_dropout <= 0 or len(inputs) <= 1:
            return inputs, masks

        batch_windows = next(iter(inputs.values())).shape[0]
        keep = torch.rand(batch_windows, len(self.modality_keys), device=next(iter(inputs.values())).device) > self.modality_dropout
        dropped_all = keep.sum(dim=1) == 0
        if dropped_all.any():
            keep[dropped_all, 0] = True

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

        batch_size, num_windows, window_size = window_mask.shape[0], window_mask.shape[1], next(iter(masks.values())).shape[2]
        flat_inputs = {
            key: value.reshape(batch_size * num_windows, window_size, value.shape[-1])
            for key, value in modalities.items()
        }
        flat_masks = {
            key: value.reshape(batch_size * num_windows, window_size)
            for key, value in masks.items()
        }
        flat_quality = window_quality.reshape(batch_size * num_windows, window_quality.shape[-1])

        flat_inputs, flat_masks = self._apply_modality_dropout(flat_inputs, flat_masks)

        encoded_sequences = {}
        pooled_features = {}
        branch_logits = {}
        for key in self.modality_keys:
            encoded, pooled = self.encoders[key](flat_inputs[key], flat_masks[key])
            encoded_sequences[key] = encoded
            pooled_features[key] = pooled
            branch_logits[key] = self.per_modality_heads[key](pooled).squeeze(-1)

        fused_window_embedding = self.fusion(
            [encoded_sequences[key] for key in self.modality_keys],
            [flat_masks[key] for key in self.modality_keys],
        )
        acoustic_window_embedding = self._group_mean(pooled_features, self.modality_groups.get("acoustic", []))
        visual_window_embedding = self._group_mean(pooled_features, self.modality_groups.get("visual", []))
        acoustic_window_logit = self.acoustic_head(acoustic_window_embedding).squeeze(-1)
        visual_window_logit = self.visual_head(visual_window_embedding).squeeze(-1)
        fused_window_logit = self.fused_window_head(fused_window_embedding).squeeze(-1)

        fused_windows = fused_window_embedding.reshape(batch_size, num_windows, self.hidden_dim)
        subject_inputs = torch.cat([fused_windows, window_quality], dim=-1)
        if self.subject_aggregation == "mean":
            subject_embedding = _masked_mean(subject_inputs, window_mask, dim=1)
            subject_embedding = subject_embedding[..., : self.hidden_dim]
        else:
            subject_embedding = self.subject_aggregator(subject_inputs, window_mask)

        acoustic_subject_logit = _masked_mean_scalar(
            acoustic_window_logit.reshape(batch_size, num_windows),
            window_mask,
            dim=1,
        )
        visual_subject_logit = _masked_mean_scalar(
            visual_window_logit.reshape(batch_size, num_windows),
            window_mask,
            dim=1,
        )
        fused_logit = self.fused_subject_head(subject_embedding).squeeze(-1)

        if self.gate_variant == "fused_only":
            gate_weights = torch.zeros((batch_size, 3), dtype=subject_embedding.dtype, device=subject_embedding.device)
            gate_weights[:, 2] = 1.0
        else:
            gate_weights = torch.softmax(self.gate_mlp(torch.cat([subject_embedding, subject_quality], dim=-1)), dim=-1)

        stacked_logits = torch.stack([acoustic_subject_logit, visual_subject_logit, fused_logit], dim=-1)
        mixture_logit = torch.sum(gate_weights * stacked_logits, dim=-1)

        return {
            "branch_logits": branch_logits,
            "acoustic_logit": acoustic_subject_logit,
            "visual_logit": visual_subject_logit,
            "fused_logit": fused_logit,
            "mixture_logit": mixture_logit,
            "gate_weights": gate_weights,
            "window_embedding": subject_embedding,
            "window_logits": {
                "acoustic": acoustic_window_logit.reshape(batch_size, num_windows),
                "visual": visual_window_logit.reshape(batch_size, num_windows),
                "fused": fused_window_logit.reshape(batch_size, num_windows),
            },
            "phq_score": self.phq_head(subject_embedding).squeeze(-1),
            "phq_severity_logits": self.phq_severity_head(subject_embedding),
        }
