#!/usr/bin/env python
from __future__ import annotations

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from cp_shock_project.data.case_indexing import build_case_index
from cp_shock_project.data.load_arrays import load_arrays
from cp_shock_project.symbolic.build_sensor_dataset import balanced_sensor_dataframe, write_sensor_splits
from cp_shock_project.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    arrays = load_arrays(cfg.get("data_dir", "data"), mmap=True)
    shock = np.load(cfg.get("train_shock_score_path", "processed/shock_scores/train_shock_scores.npz"))
    case_ids = build_case_index(arrays.X_train).case_ids
    df = balanced_sensor_dataframe(
        arrays.X_train,
        shock["oracle_shock_score"],
        case_ids=case_ids,
        max_samples=int(cfg.get("max_samples", 100000)),
        shock_threshold=float(cfg.get("shock_threshold", 0.5)),
        shock_fraction=float(cfg.get("shock_fraction", 0.5)),
        seed=int(cfg.get("seed", 42)),
    )
    train_path, val_path = write_sensor_splits(df, cfg.get("sensor_dataset_dir", "processed/symbolic_sensor"), cfg.get("val_fraction", 0.2), cfg.get("seed", 42))
    print(f"Saved {train_path} and {val_path}")


if __name__ == "__main__":
    main()
