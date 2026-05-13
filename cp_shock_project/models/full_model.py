from __future__ import annotations

import torch
from torch import nn

from cp_shock_project.features.fourier import FourierFeatureEncoder
from cp_shock_project.graph.message_passing import LocalGraphEncoder, mlp
from cp_shock_project.models.residual_model import ShockResidualNetwork
from cp_shock_project.models.smooth_model import SmoothCpNetwork
from cp_shock_project.symbolic.symbolic_module import DummyShockSensor


class _FourierGraphBase(nn.Module):
    def __init__(
        self,
        input_dim: int = 9,
        fourier_num_frequencies: int = 6,
        graph_hidden_dim: int = 128,
        graph_out_dim: int = 128,
        graph_rel_dim: int = 2,
        dropout: float = 0.0,
        fourier_variables_to_encode: tuple[int, ...] | list[int] | None = None,
        fourier_direct_variables: tuple[int, ...] | list[int] | None = None,
    ):
        super().__init__()
        if fourier_variables_to_encode is None:
            fourier_variables_to_encode = (0, 1, 2, 6, 7, 8) if input_dim >= 9 else (0, 1, 5, 6, 7)
        if fourier_direct_variables is None:
            fourier_direct_variables = (3, 4, 5) if input_dim >= 9 else (2, 3, 4)
        self.fourier = FourierFeatureEncoder(
            input_dim=input_dim,
            num_frequencies=fourier_num_frequencies,
            variables_to_encode=fourier_variables_to_encode,
            direct_variables=fourier_direct_variables,
        )
        node_dim = input_dim + self.fourier.output_dim
        self.graph = LocalGraphEncoder(
            node_dim=node_dim,
            hidden_dim=graph_hidden_dim,
            out_dim=graph_out_dim,
            dropout=dropout,
            rel_dim=graph_rel_dim,
        )
        self.combined_dim = input_dim + self.fourier.output_dim + graph_out_dim

    def encode(
        self,
        X: torch.Tensor,
        neighbor_X: torch.Tensor | None = None,
        neighbor_distances: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phi = self.fourier(X)
        neighbor_phi = None
        if neighbor_X is not None:
            b, k, d = neighbor_X.shape
            neighbor_phi = self.fourier(neighbor_X.reshape(b * k, d)).reshape(b, k, -1)
        h_graph = self.graph(X, phi, neighbor_X, neighbor_phi, neighbor_distances)
        return phi, h_graph


class FourierGraphCpNet(_FourierGraphBase):
    """Fourier features + local graph context, no symbolic gate."""

    def __init__(self, hidden_dim: int = 128, depth: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.head = mlp([self.combined_dim] + [hidden_dim] * depth + [1])

    def forward(self, X: torch.Tensor, neighbor_X: torch.Tensor | None = None, neighbor_distances: torch.Tensor | None = None, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        phi, h_graph = self.encode(X, neighbor_X, neighbor_distances)
        cp = self.head(torch.cat([X, phi, h_graph], dim=-1))
        return {"Cp_pred": cp, "h_graph": h_graph}


class SymbolicWeightedGraphFourierCpNet(FourierGraphCpNet):
    """Same predictor as FourierGraphCpNet; chi is consumed by the loss."""

    def forward(self, X: torch.Tensor, symbolic_chi: torch.Tensor | None = None, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        out = super().forward(X, **kwargs)
        if symbolic_chi is not None:
            out["chi"] = symbolic_chi
        return out


class SymbolicGatedGraphFourierResidualCpNet(_FourierGraphBase):
    """Main shock-aware residual model with deployable symbolic gate."""

    def __init__(
        self,
        symbolic_sensor: nn.Module | None = None,
        hidden_dim: int = 128,
        depth: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.symbolic_sensor = symbolic_sensor or DummyShockSensor(0.0)
        self.smooth = SmoothCpNetwork(self.combined_dim, hidden_dim=hidden_dim, depth=depth)
        self.residual = ShockResidualNetwork(self.combined_dim + 1, hidden_dim=hidden_dim, depth=depth)

    def compute_chi(self, X: torch.Tensor, symbolic_chi: torch.Tensor | None = None, **_: torch.Tensor) -> torch.Tensor:
        if symbolic_chi is not None:
            return symbolic_chi
        return self.symbolic_sensor(X)

    def forward(
        self,
        X: torch.Tensor,
        neighbor_X: torch.Tensor | None = None,
        neighbor_distances: torch.Tensor | None = None,
        symbolic_chi: torch.Tensor | None = None,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sensor_X = kwargs.get("X_raw", X)
        chi = self.compute_chi(sensor_X, symbolic_chi=symbolic_chi, **kwargs).clamp(0.0, 1.0)
        phi, h_graph = self.encode(X, neighbor_X, neighbor_distances)
        cp_smooth = self.smooth(X, phi, h_graph)
        delta = self.residual(X, phi, h_graph, chi)
        cp = cp_smooth + chi * delta
        return {"Cp_pred": cp, "Cp_smooth": cp_smooth, "delta_Cp": delta, "chi": chi, "h_graph": h_graph}


class OracleGatedGraphFourierResidualCpNet(SymbolicGatedGraphFourierResidualCpNet):
    """Upper-bound model using oracle_shock_score as chi."""

    def compute_chi(self, X: torch.Tensor, oracle_shock_score: torch.Tensor | None = None, **_: torch.Tensor) -> torch.Tensor:
        if oracle_shock_score is None:
            raise ValueError("OracleGatedGraphFourierResidualCpNet requires oracle_shock_score in each batch")
        return oracle_shock_score
