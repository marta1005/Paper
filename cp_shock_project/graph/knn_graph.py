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
    coordinate_columns: tuple[int, ...] = (0, 1)
    projection: str = "xy"


class KNNGraphBuilder:
    """Build kNN graphs within each CFD case using configurable surface projections."""

    PROJECTIONS = {
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
        "xyz": (0, 1, 2),
    }

    def __init__(
        self,
        k_neighbors: int = 16,
        algorithm: str = "auto",
        eps: float = 1e-12,
        chunk_size: int | None = None,
        projection: str = "xy",
        coordinate_columns: tuple[int, ...] | list[int] | None = None,
    ):
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        self.k_neighbors = int(k_neighbors)
        self.algorithm = algorithm
        self.eps = eps
        self.chunk_size = chunk_size
        self.projection = projection.lower()
        self.coordinate_columns = self._resolve_columns(self.projection, coordinate_columns)

    @classmethod
    def _resolve_columns(cls, projection: str, coordinate_columns: tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
        if coordinate_columns is not None:
            cols = tuple(int(c) for c in coordinate_columns)
        else:
            if projection not in cls.PROJECTIONS:
                valid = ", ".join(sorted(cls.PROJECTIONS))
                raise ValueError(f"Unknown projection {projection!r}. Valid options: {valid}")
            cols = cls.PROJECTIONS[projection]
        if len(cols) not in (2, 3):
            raise ValueError("coordinate_columns must contain 2 or 3 columns")
        if any(c < 0 or c > 2 for c in cols):
            raise ValueError("coordinate_columns must refer to geometry columns 0=x, 1=y, 2=z")
        return cols

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
            coords = np.asarray(X[np.ix_(idx, self.coordinate_columns)], dtype=np.float64)
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
        return KNNGraph(
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            selected_indices=selected_indices,
            coordinate_columns=self.coordinate_columns,
            projection=self.projection,
        )
