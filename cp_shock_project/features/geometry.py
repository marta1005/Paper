from __future__ import annotations

import numpy as np


def cf_magnitude(Y: np.ndarray) -> np.ndarray:
    """Compute skin-friction magnitude from Cfx/Cfy/Cfz."""
    return np.sqrt(np.sum(np.asarray(Y[:, 1:4], dtype=np.float64) ** 2, axis=1))


def valid_neighbor_mask(neighbor_indices: np.ndarray, neighbor_distances: np.ndarray | None = None) -> np.ndarray:
    mask = neighbor_indices >= 0
    if neighbor_distances is not None:
        mask &= np.isfinite(neighbor_distances)
    return mask
