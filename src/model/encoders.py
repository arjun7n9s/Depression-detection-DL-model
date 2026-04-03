"""Simple unimodal sequence encoders used for milestone baselines."""

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
