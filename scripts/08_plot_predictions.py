#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from cp_shock_project.data.load_arrays import load_arrays
from cp_shock_project.training.train_cp import evaluate_from_config
from cp_shock_project.utils.config import load_config
from cp_shock_project.visualization.comparison import plot_improvement_map
from cp_shock_project.visualization.diagnostics import plot_prediction_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate_from_config(cfg)
    pred_path = Path(cfg.get("output_dir", "outputs/evaluation")) / "predictions.npz"
    pred = dict(np.load(pred_path))
    arrays = load_arrays(cfg.get("data", {}).get("data_dir", "data"), mmap=True)
    X = np.asarray(arrays.X_test[pred["point_id"]])
    plot_prediction_diagnostics(X, pred, Path(cfg.get("output_dir", "outputs/evaluation")) / "plots", cfg.get("plots", {}).get("views", ["xy", "xz"]))
    baseline_path = cfg.get("plots", {}).get("baseline_predictions_npz")
    if baseline_path:
        baseline = dict(np.load(baseline_path))
        for view in cfg.get("plots", {}).get("views", ["xy", "xz"]):
            plot_improvement_map(
                X,
                pred["Cp"],
                baseline["Cp_pred"],
                pred["Cp_pred"],
                Path(cfg.get("output_dir", "outputs/evaluation")) / "plots",
                view=view,
            )
    print(f"Saved plots under {Path(cfg.get('output_dir', 'outputs/evaluation')) / 'plots'}")


if __name__ == "__main__":
    main()
