from __future__ import annotations

import torch
from torch import nn


def mlp(sizes: list[int], activation: type[nn.Module] = nn.SiLU, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class LocalGraphEncoder(nn.Module):
    """Lightweight message passing over fixed kNN neighborhoods."""

    def __init__(self, node_dim: int, hidden_dim: int = 128, out_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.node_dim = int(node_dim)
        self.msg = mlp([2 * node_dim + 4, hidden_dim, hidden_dim], dropout=dropout)
        self.update = mlp([node_dim + hidden_dim, hidden_dim, out_dim], dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        phi: torch.Tensor,
        neighbor_x: torch.Tensor | None = None,
        neighbor_phi: torch.Tensor | None = None,
        neighbor_distances: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_i = torch.cat([x, phi], dim=-1)
        if neighbor_x is None or neighbor_phi is None:
            zeros = torch.zeros(h_i.shape[0], self.msg[-1].out_features, device=h_i.device, dtype=h_i.dtype)
            return self.update(torch.cat([h_i, zeros], dim=-1))
        h_j = torch.cat([neighbor_x, neighbor_phi], dim=-1)
        k = h_j.shape[1]
        h_i_rep = h_i.unsqueeze(1).expand(-1, k, -1)
        rel = neighbor_x[..., :3] - x[:, None, :3]
        if neighbor_distances is None:
            dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
        else:
            dist = neighbor_distances.unsqueeze(-1)
        msg_in = torch.cat([h_i_rep, h_j, rel, dist], dim=-1)
        messages = self.msg(msg_in)
        valid = torch.isfinite(dist).to(messages.dtype)
        messages = messages * valid
        denom = valid.sum(dim=1).clamp_min(1.0)
        agg = messages.sum(dim=1) / denom
        return self.update(torch.cat([h_i, agg], dim=-1))
