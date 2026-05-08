import numpy as np

from cp_shock_project.data.case_indexing import build_case_index
from cp_shock_project.features.shock_score import compute_oracle_shock_score
from cp_shock_project.graph.knn_graph import KNNGraphBuilder

from tests.conftest import make_synthetic_arrays


def test_oracle_shock_score_detects_jump():
    X, Y = make_synthetic_arrays(n_cases=2, points_per_case=50)
    index = build_case_index(X)
    graph = KNNGraphBuilder(k_neighbors=4).build(X, index)
    features = compute_oracle_shock_score(X, Y, graph.neighbor_indices, graph.neighbor_distances, index)
    assert features.oracle_shock_score.shape == (X.shape[0],)
    assert float(features.oracle_shock_score.max()) <= 1.0
    assert np.any(features.shock_label)
