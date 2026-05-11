import numpy as np

from cp_shock_project.data.case_indexing import build_case_index
from cp_shock_project.graph.knn_graph import KNNGraphBuilder

from tests.conftest import make_synthetic_arrays


def test_knn_graph_does_not_mix_cases():
    X, _ = make_synthetic_arrays(n_cases=3, points_per_case=12)
    index = build_case_index(X)
    graph = KNNGraphBuilder(k_neighbors=3).build(X, index)
    assert graph.neighbor_indices.shape == (X.shape[0], 3)
    for i, neigh in enumerate(graph.neighbor_indices):
        for j in neigh[neigh >= 0]:
            assert np.allclose(X[i, 6:9], X[j, 6:9])


def test_knn_graph_supports_2d_projection():
    X, _ = make_synthetic_arrays(n_cases=1, points_per_case=12)
    index = build_case_index(X)
    graph_xy = KNNGraphBuilder(k_neighbors=2, projection="xy").build(X, index)
    graph_xz = KNNGraphBuilder(k_neighbors=2, projection="xz").build(X, index)
    assert graph_xy.coordinate_columns == (0, 1)
    assert graph_xz.coordinate_columns == (0, 2)
    assert graph_xy.projection == "xy"
    assert graph_xz.projection == "xz"
    assert graph_xy.neighbor_distances.shape == graph_xz.neighbor_distances.shape
