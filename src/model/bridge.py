"""Sequence bridge model that projects live-compatible visual features into teacher spaces."""

from __future__ import annotations

import torch
from torch import nn


class BridgeProjectionModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        visual_dim: int,
        acoustic_dim: int = 0,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        include_acoustic_target: bool = True,
    ):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.include_acoustic_target = include_acoustic_target
        self.encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.visual_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, visual_dim),
        )
        if include_acoustic_target:
            self.acoustic_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, acoustic_dim),
            )
        else:
            self.acoustic_head = None

    def forward(self, student_input: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded, _ = self.encoder(student_input)
        encoded = self.norm(encoded)
        outputs = {
            "visual_pred": self.visual_head(encoded),
        }
        if self.acoustic_head is not None:
            outputs["acoustic_pred"] = self.acoustic_head(encoded)
        return outputs
