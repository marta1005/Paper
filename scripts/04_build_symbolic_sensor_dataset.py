#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from cp_shock_project.data.case_indexing import build_case_index, load_case_index
from cp_shock_project.data.load_arrays import load_arrays
from cp_shock_project.symbolic.build_sensor_dataset import balanced_sensor_dataframe, write_score_diagnostics, write_sensor_splits
from cp_shock_project.utils.config import load_config
from cp_shock_project.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    arrays = load_arrays(cfg.get("data_dir", "data"), mmap=True)
    shock = np.load(cfg.get("train_shock_score_path", "processed/shock_scores/train_shock_scores.npz"))
    score = shock["oracle_shock_score"]
    out_dir = cfg.get("sensor_dataset_dir", "processed/symbolic_sensor")
    stats = write_score_diagnostics(
        score,
        out_dir,
        thresholds=tuple(float(v) for v in cfg.get("diagnostic_thresholds", [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9])),
    )
    print("oracle_shock_score stats:")
    print({k: stats[k] for k in ["min", "max", "mean", "median", "std"]})
    print("threshold diagnostics:")
    for row in stats["thresholds"].values():
        print(row)
    case_index_path = cfg.get("train_case_index_path", "processed/case_indices/train_indices.npz")
    if case_index_path and Path(case_index_path).exists():
        case_ids = load_case_index(case_index_path).case_ids
    else:
        case_ids = build_case_index(arrays.X_train).case_ids
    df = balanced_sensor_dataframe(
        arrays.X_train,
        score,
        case_ids=case_ids,
        max_samples=int(cfg.get("max_samples", 100000)),
        shock_threshold=float(cfg.get("shock_threshold", 0.5)),
        shock_fraction=float(cfg.get("shock_fraction", 0.5)),
        seed=int(cfg.get("seed", 42)),
    )
    save_json(df.attrs.get("sampling_info", {}), Path(out_dir) / "sensor_sampling_info.json")
    print("sampling info:", df.attrs.get("sampling_info", {}))
    train_path, val_path = write_sensor_splits(df, out_dir, cfg.get("val_fraction", 0.2), cfg.get("seed", 42))
    print(f"Saved {train_path} and {val_path}")


if __name__ == "__main__":
    main()
