from __future__ import annotations

import torch
from torch import nn

from cp_shock_project.graph.message_passing import mlp


class BaselineCpMLP(nn.Module):
    """Pointwise MLP baseline: X -> Cp."""

    def __init__(self, input_dim: int = 9, hidden_dim: int = 128, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        sizes = [input_dim] + [hidden_dim] * depth + [1]
        self.net = mlp(sizes, dropout=dropout)

    def forward(self, X: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        cp = self.net(X)
        return {"Cp_pred": cp}
