#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from cp_shock_project.data.case_indexing import build_case_index
from cp_shock_project.data.load_arrays import load_arrays
from cp_shock_project.features.shock_score import compute_oracle_shock_score
from cp_shock_project.graph.graph_cache import load_graph
from cp_shock_project.utils.config import load_config


def compute_one(X, Y, split: str, cfg: dict) -> None:
    graph = load_graph(cfg.get(f"{split}_graph_path", f"processed/graphs/{split}_knn_graph.npz"))
    features = compute_oracle_shock_score(
        X,
        Y,
        graph.neighbor_indices,
        graph.neighbor_distances,
        case_index=build_case_index(X),
        lower_percentile=float(cfg.get("lower_percentile", 50.0)),
        upper_percentile=float(cfg.get("upper_percentile", 99.5)),
        label_percentile=float(cfg.get("label_percentile", 98.5)),
        low_mach_downweight=cfg.get("low_mach_downweight"),
        low_mach_threshold=float(cfg.get("low_mach_threshold", 0.7)),
    )
    path = Path(cfg.get(f"{split}_output_path", f"processed/shock_scores/{split}_shock_scores.npz"))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **features.__dict__)
    print(f"Saved {split} shock scores to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    arrays = load_arrays(cfg.get("data_dir", "data"), mmap=True)
    compute_one(arrays.X_train, arrays.Y_train, "train", cfg)
    compute_one(arrays.X_test, arrays.Y_test, "test", cfg)


if __name__ == "__main__":
    main()
