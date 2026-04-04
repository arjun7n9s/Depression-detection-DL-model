"""Sequence encoders used for unimodal baselines and the first bimodal milestone."""

from __future__ import annotations

import torch
from torch import nn


class WindowAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(sequence).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class SequenceBinaryClassifier(nn.Module):
    """BiGRU window encoder + attention pooling + binary classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )
        self.pool = WindowAttentionPool(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mask = inputs.abs().sum(dim=-1) > 0
        encoded, _ = self.gru(inputs)
        pooled = self.pool(encoded, mask)
        pooled = self.dropout(pooled)
        return self.head(pooled).squeeze(-1)


class BimodalSequenceClassifier(nn.Module):
    """Two-stream BiGRU encoder with gated fusion for acoustic + visual windows."""

    def __init__(
        self,
        visual_input_dim: int,
        acoustic_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0

        self.visual_proj = nn.Sequential(
            nn.Linear(visual_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.acoustic_proj = nn.Sequential(
            nn.Linear(acoustic_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.visual_gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )
        self.acoustic_gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )

        fused_dim = hidden_dim * 2
        self.visual_pool = WindowAttentionPool(fused_dim)
        self.acoustic_pool = WindowAttentionPool(fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(fused_dim * 5, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        visual_inputs = inputs["visual"]
        acoustic_inputs = inputs["acoustic"]

        visual_mask = visual_inputs.abs().sum(dim=-1) > 0
        acoustic_mask = acoustic_inputs.abs().sum(dim=-1) > 0

        visual_encoded, _ = self.visual_gru(self.visual_proj(visual_inputs))
        acoustic_encoded, _ = self.acoustic_gru(self.acoustic_proj(acoustic_inputs))

        visual_repr = self.visual_pool(visual_encoded, visual_mask)
        acoustic_repr = self.acoustic_pool(acoustic_encoded, acoustic_mask)

        gate = self.gate(torch.cat([visual_repr, acoustic_repr], dim=-1))
        gated_visual = gate * visual_repr
        gated_acoustic = (1.0 - gate) * acoustic_repr

        fused = torch.cat(
            [
                gated_visual + gated_acoustic,
                torch.abs(visual_repr - acoustic_repr),
                visual_repr * acoustic_repr,
                torch.cat([gated_visual, gated_acoustic], dim=-1),
            ],
            dim=-1,
        )
        fused = self.dropout(fused)
        return self.head(fused).squeeze(-1)
