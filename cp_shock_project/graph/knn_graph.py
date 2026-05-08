from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.neighbors import NearestNeighbors

from cp_shock_project.data.case_indexing import CaseIndex


@dataclass(frozen=True)
class KNNGraph:
    neighbor_indices: np.ndarray
    neighbor_distances: np.ndarray
    selected_indices: np.ndarray


class KNNGraphBuilder:
    """Build kNN graphs within each CFD case using 3D surface coordinates."""

    def __init__(self, k_neighbors: int = 16, algorithm: str = "auto", eps: float = 1e-12, chunk_size: int | None = None):
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        self.k_neighbors = int(k_neighbors)
        self.algorithm = algorithm
        self.eps = eps
        self.chunk_size = chunk_size

    def build(
        self,
        X: np.ndarray,
        case_index: CaseIndex,
        max_cases: int | None = None,
        max_points_per_case: int | None = None,
    ) -> KNNGraph:
        n = X.shape[0]
        neighbor_indices = np.full((n, self.k_neighbors), -1, dtype=np.int64)
        neighbor_distances = np.full((n, self.k_neighbors), np.inf, dtype=np.float32)
        selected: list[np.ndarray] = []
        case_ids: Iterable[int] = range(case_index.n_cases)
        if max_cases is not None:
            case_ids = list(case_ids)[: int(max_cases)]
        for cid in case_ids:
            idx = case_index.indices_for_case(int(cid))
            if max_points_per_case is not None:
                idx = idx[: int(max_points_per_case)]
            selected.append(idx)
            if idx.size <= 1:
                continue
            coords = np.asarray(X[idx, :3], dtype=np.float64)
            n_neighbors = min(self.k_neighbors + 1, idx.size)
            nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm=self.algorithm)
            nn.fit(coords)
            chunk = self.chunk_size or idx.size
            for start in range(0, idx.size, int(chunk)):
                end = min(start + int(chunk), idx.size)
                dist, local = nn.kneighbors(coords[start:end], return_distance=True)
                dist = dist[:, 1:]
                local = local[:, 1:]
                k_eff = local.shape[1]
                target = idx[start:end]
                neighbor_indices[target, :k_eff] = idx[local]
                neighbor_distances[target, :k_eff] = np.maximum(dist, self.eps).astype(np.float32)
        selected_indices = np.concatenate(selected).astype(np.int64) if selected else np.array([], dtype=np.int64)
        return KNNGraph(neighbor_indices=neighbor_indices, neighbor_distances=neighbor_distances, selected_indices=selected_indices)
