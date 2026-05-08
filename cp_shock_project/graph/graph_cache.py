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
    )


def load_graph(path: str | Path) -> KNNGraph:
    data = np.load(path)
    return KNNGraph(
        neighbor_indices=data["neighbor_indices"],
        neighbor_distances=data["neighbor_distances"],
        selected_indices=data["selected_indices"] if "selected_indices" in data else np.arange(data["neighbor_indices"].shape[0]),
    )
