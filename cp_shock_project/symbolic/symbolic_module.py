from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from cp_shock_project.symbolic.expression import SymbolicExpression


class SymbolicShockSensorModule(nn.Module):
    """Torch wrapper around a non-differentiable deployable symbolic sensor."""

    def __init__(self, expression: SymbolicExpression):
        super().__init__()
        self.expression = expression

    @classmethod
    def from_json(cls, path: str | Path) -> "SymbolicShockSensorModule":
        return cls(SymbolicExpression.from_json(path))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        values = self.expression.evaluate(x.detach().cpu().numpy())
        return torch.as_tensor(values, dtype=x.dtype, device=device).unsqueeze(-1)


class DummyShockSensor(nn.Module):
    """Fallback sensor for smoke tests when PySR output is unavailable."""

    def __init__(self, value: float = 0.0):
        super().__init__()
        self.value = float(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1), self.value, dtype=x.dtype, device=x.device)
