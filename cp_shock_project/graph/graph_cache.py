from __future__ import annotations

from pathlib import Path

import numpy as np

from cp_shock_project.graph.knn_graph import KNNGraph


def save_graph(graph: KNNGraph, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        neighbor_indices=graph.neighbor_indices,
        neighbor_distances=graph.neighbor_distances,
        selected_indices=graph.selected_indices,
        coordinate_columns=np.asarray(graph.coordinate_columns, dtype=np.int64),
        projection=np.asarray(graph.projection),
    )


def load_graph(path: str | Path) -> KNNGraph:
    data = np.load(path)
    return KNNGraph(
        neighbor_indices=data["neighbor_indices"],
        neighbor_distances=data["neighbor_distances"],
        selected_indices=data["selected_indices"] if "selected_indices" in data else np.arange(data["neighbor_indices"].shape[0]),
        coordinate_columns=tuple(int(c) for c in data["coordinate_columns"]) if "coordinate_columns" in data else (0, 1, 2),
        projection=str(data["projection"]) if "projection" in data else "xyz",
    )
