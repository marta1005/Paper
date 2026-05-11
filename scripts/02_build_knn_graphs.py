#!/usr/bin/env python
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from cp_shock_project.data.case_indexing import build_case_index
from cp_shock_project.data.load_arrays import load_arrays
from cp_shock_project.graph.graph_cache import save_graph
from cp_shock_project.graph.knn_graph import KNNGraphBuilder
from cp_shock_project.utils.config import load_config


def build_one(X, split: str, cfg: dict) -> None:
    index = build_case_index(X)
    graph = KNNGraphBuilder(
        k_neighbors=int(cfg.get("k_neighbors", 16)),
        chunk_size=cfg.get("chunk_size"),
        projection=cfg.get("projection", "xy"),
        coordinate_columns=cfg.get("coordinate_columns"),
    ).build(
        X,
        index,
        max_cases=cfg.get("max_cases"),
        max_points_per_case=cfg.get("max_points_per_case"),
    )
    path = cfg.get(f"{split}_graph_path", f"processed/graphs/{split}_knn_graph.npz")
    save_graph(graph, path)
    print(f"Saved {split} {graph.projection} graph to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    arrays = load_arrays(cfg.get("data_dir", "data"), mmap=True)
    build_one(arrays.X_train, "train", cfg)
    build_one(arrays.X_test, "test", cfg)


if __name__ == "__main__":
    main()
