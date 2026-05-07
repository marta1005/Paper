"""kNN-based local scattered-surface features."""

from __future__ import annotations

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)
EPS = 1.0e-8


def knn_indices_distances(
    coords: np.ndarray,
    k_neighbors: int = 16,
    batch_size: int = 4096,
    max_numpy_points: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return neighbor indices and distances for each point.

    `sklearn.NearestNeighbors` is used when available. A NumPy fallback supports
    tests and small smoke runs; for large snapshots install scikit-learn.
    """
    coords = np.asarray(coords, dtype=np.float32)
    n_points = int(coords.shape[0])
    k = min(int(k_neighbors), max(n_points - 1, 1))
    if n_points <= 1:
        return np.zeros((n_points, 0), dtype=np.int64), np.zeros((n_points, 0), dtype=np.float32)
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords, return_distance=True)
        return indices[:, 1:].astype(np.int64), distances[:, 1:].astype(np.float32)
    except ImportError:
        if n_points > max_numpy_points:
            raise RuntimeError(
                "scikit-learn is required for kNN on large snapshots. "
                f"Install scikit-learn or set max_points_per_case <= {max_numpy_points}."
            )
        LOGGER.warning("scikit-learn not installed; using NumPy kNN fallback for %s points.", n_points)
        all_indices = np.empty((n_points, k), dtype=np.int64)
        all_distances = np.empty((n_points, k), dtype=np.float32)
        for start in range(0, n_points, batch_size):
            stop = min(n_points, start + batch_size)
            diff = coords[start:stop, None, :] - coords[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            rows = np.arange(start, stop)
            dist[np.arange(stop - start), rows] = np.inf
            part = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
            part_dist = np.take_along_axis(dist, part, axis=1)
            order = np.argsort(part_dist, axis=1)
            all_indices[start:stop] = np.take_along_axis(part, order, axis=1)
            all_distances[start:stop] = np.take_along_axis(part_dist, order, axis=1).astype(np.float32)
        return all_indices, all_distances


def local_contrast(values: np.ndarray, neighbor_indices: np.ndarray) -> np.ndarray:
    """Compute max-min contrast among each point and its neighbors."""
    values = np.asarray(values, dtype=np.float32)
    if neighbor_indices.shape[1] == 0:
        return np.zeros_like(values, dtype=np.float32)
    neighbor_values = values[neighbor_indices]
    stacked_min = np.minimum(np.min(neighbor_values, axis=1), values)
    stacked_max = np.maximum(np.max(neighbor_values, axis=1), values)
    return (stacked_max - stacked_min).astype(np.float32)


def local_weighted_gradient(values: np.ndarray, neighbor_indices: np.ndarray, neighbor_distances: np.ndarray) -> np.ndarray:
    """Approximate gradient magnitude with mean local finite differences."""
    values = np.asarray(values, dtype=np.float32)
    if neighbor_indices.shape[1] == 0:
        return np.zeros_like(values, dtype=np.float32)
    diffs = np.abs(values[neighbor_indices] - values[:, None])
    return np.mean(diffs / (neighbor_distances + EPS), axis=1).astype(np.float32)


def local_streamwise_gradient(
    values: np.ndarray,
    coords: np.ndarray,
    streamwise_tangent: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
) -> np.ndarray:
    """Approximate streamwise derivative from local neighbor differences."""
    values = np.asarray(values, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    t = np.asarray(streamwise_tangent, dtype=np.float32)
    if neighbor_indices.shape[1] == 0:
        return np.zeros_like(values, dtype=np.float32)
    delta_x = coords[neighbor_indices] - coords[:, None, :]
    projected = np.sum(delta_x * t[:, None, :], axis=2)
    delta_value = values[neighbor_indices] - values[:, None]
    directional = delta_value * np.sign(projected) / (neighbor_distances + EPS)
    weights = np.abs(projected) / (neighbor_distances + EPS)
    return (np.sum(directional * weights, axis=1) / np.maximum(np.sum(weights, axis=1), EPS)).astype(np.float32)
