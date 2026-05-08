import numpy as np
import torch

from cp_shock_project.data.case_indexing import build_case_index
from cp_shock_project.features.shock_score import compute_oracle_shock_score
from cp_shock_project.graph.knn_graph import KNNGraphBuilder
from cp_shock_project.models import BaselineCpMLP, FourierGraphCpNet, SymbolicGatedGraphFourierResidualCpNet
from cp_shock_project.training.train_cp import train_from_config

from tests.conftest import make_synthetic_arrays, write_synthetic_data


def test_model_forward_shapes():
    X, Y = make_synthetic_arrays(n_cases=1, points_per_case=10)
    index = build_case_index(X)
    graph = KNNGraphBuilder(k_neighbors=3).build(X, index)
    x = torch.tensor(X[:5])
    nx = torch.tensor(X[graph.neighbor_indices[:5]])
    dist = torch.tensor(graph.neighbor_distances[:5])
    assert BaselineCpMLP(hidden_dim=16, depth=1)(x)["Cp_pred"].shape == (5, 1)
    assert FourierGraphCpNet(hidden_dim=16, depth=1, graph_hidden_dim=16, graph_out_dim=12, fourier_num_frequencies=2)(x, neighbor_X=nx, neighbor_distances=dist)["Cp_pred"].shape == (5, 1)
    out = SymbolicGatedGraphFourierResidualCpNet(hidden_dim=16, depth=1, graph_hidden_dim=16, graph_out_dim=12, fourier_num_frequencies=2)(x, neighbor_X=nx, neighbor_distances=dist)
    assert out["Cp_pred"].shape == (5, 1)
    assert set(["Cp_smooth", "delta_Cp", "chi"]).issubset(out)


def test_smoke_training_three_models(tmp_path):
    data_dir, X, Y, *_ = write_synthetic_data(tmp_path)
    index = build_case_index(X)
    graph = KNNGraphBuilder(k_neighbors=3).build(X, index)
    shock = compute_oracle_shock_score(X, Y, graph.neighbor_indices, graph.neighbor_distances, index)
    shock_path = tmp_path / "shock.npz"
    np.savez_compressed(shock_path, oracle_shock_score=shock.oracle_shock_score)
    base = {
        "seed": 1,
        "data": {"data_dir": str(data_dir), "val_fraction": 0.34, "seed": 1, "max_train_points": 60, "max_val_points": 30},
        "graph": {"k_neighbors": 3, "build_if_missing": True},
        "shock_score": {"train_path": str(shock_path)},
        "training": {"epochs": 1, "batch_size": 16, "lr": 0.001, "weight_decay": 0.0},
        "loss": {"global": 1.0, "shock": 0.1, "residual": 0.01},
    }
    configs = [
        {"model": {"name": "baseline_mlp", "kwargs": {"hidden_dim": 16, "depth": 1}}},
        {"model": {"name": "fourier_graph", "kwargs": {"hidden_dim": 16, "depth": 1, "graph_hidden_dim": 16, "graph_out_dim": 12, "fourier_num_frequencies": 2}}},
        {"model": {"name": "symbolic_gated_residual", "kwargs": {"hidden_dim": 16, "depth": 1, "graph_hidden_dim": 16, "graph_out_dim": 12, "fourier_num_frequencies": 2}}},
    ]
    for i, extra in enumerate(configs):
        cfg = {**base, **extra, "output_dir": str(tmp_path / f"run_{i}")}
        metrics = train_from_config(cfg)
        assert "best_score" in metrics
