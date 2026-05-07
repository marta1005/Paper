from __future__ import annotations

import numpy as np

from shock_symbolic.features.knn_features import knn_indices_distances, local_contrast, local_weighted_gradient


def test_numpy_knn_local_features() -> None:
    coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [10, 0, 0]], dtype=np.float32)
    values = np.array([0.0, 1.0, 2.0, 2.0], dtype=np.float32)
    idx, dist = knn_indices_distances(coords, k_neighbors=2, max_numpy_points=10)
    assert idx.shape == (4, 2)
    grad = local_weighted_gradient(values, idx, dist)
    contrast = local_contrast(values, idx)
    assert grad[1] > 0.9
    assert contrast[1] >= 2.0
