from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FourierConfig:
    num_frequencies: int = 6
    include_input: bool = True
    variables_to_encode: tuple[int, ...] = (0, 1, 2, 6, 7, 8)
    direct_variables: tuple[int, ...] = (3, 4, 5)
    log_sampling: bool = True
    scale: float = 1.0


class FourierFeatureEncoder(nn.Module):
    """Fourier features for x/y/z/Mach/AoA/pi, with normals kept direct."""

    def __init__(
        self,
        input_dim: int = 9,
        num_frequencies: int = 6,
        include_input: bool = True,
        variables_to_encode: tuple[int, ...] | list[int] = (0, 1, 2, 6, 7, 8),
        direct_variables: tuple[int, ...] | list[int] = (3, 4, 5),
        log_sampling: bool = True,
        scale: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        self.variables_to_encode = tuple(int(i) for i in variables_to_encode)
        self.direct_variables = tuple(int(i) for i in direct_variables)
        self.log_sampling = bool(log_sampling)
        self.scale = float(scale)
        if self.log_sampling:
            freq = 2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)
        else:
            freq = torch.linspace(1.0, 2.0 ** max(self.num_frequencies - 1, 0), self.num_frequencies)
        self.register_buffer("frequencies", freq * math.pi * self.scale)

    @property
    def output_dim(self) -> int:
        base = self.input_dim if self.include_input else len(self.direct_variables)
        return base + 2 * len(self.variables_to_encode) * self.num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.include_input:
            parts.append(x)
        elif self.direct_variables:
            parts.append(x[..., list(self.direct_variables)])
        q = x[..., list(self.variables_to_encode)]
        angles = q.unsqueeze(-1) * self.frequencies
        parts.append(torch.sin(angles).flatten(start_dim=-2))
        parts.append(torch.cos(angles).flatten(start_dim=-2))
        return torch.cat(parts, dim=-1)
