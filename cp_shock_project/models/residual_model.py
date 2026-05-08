from __future__ import annotations

import torch
from torch import nn

from cp_shock_project.graph.message_passing import mlp


class ShockResidualNetwork(nn.Module):
    """Predict the local shock residual correction."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        self.net = mlp([input_dim] + [hidden_dim] * depth + [1], dropout=dropout)

    def forward(self, X: torch.Tensor, phi: torch.Tensor, h_graph: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([X, phi, h_graph, chi], dim=-1))
