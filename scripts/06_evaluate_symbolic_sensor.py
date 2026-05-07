#!/usr/bin/env python
"""Evaluate a symbolic shock sensor on complete CFD cases."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shock_symbolic.metrics.per_case import aggregate_metric_rows, per_case_metrics
from shock_symbolic.symbolic.expression_io import evaluate_expression, load_sensor_json
from shock_symbolic.utils.config import load_config
from shock_symbolic.utils.io import save_csv, save_json
from shock_symbolic.utils.logging import configure_logging
from shock_symbolic.visualization.diagnostics import save_prediction_diagnostics, save_symbolic_prediction_grid


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _select_feature_files(files: list[Path], cfg: dict) -> list[Path]:
    """Select feature files by explicit condition indices or case ids, preserving config order."""
    evaluation_cfg = cfg.get("evaluation", {})
    explicit_indices = evaluation_cfg.get("condition_indices")
    if explicit_indices:
        selected = []
        for raw_idx in explicit_indices:
            idx = int(raw_idx)
            if idx < 0 or idx >= len(files):
                raise IndexError(f"condition index {idx} is outside the available range [0, {len(files) - 1}]")
            selected.append(files[idx])
        return selected

    case_ids = evaluation_cfg.get("case_ids")
    if case_ids:
        by_id = {path.stem: path for path in files}
        missing = [str(case_id) for case_id in case_ids if str(case_id) not in by_id]
        if missing:
            raise KeyError(f"case_ids not found in features directory: {missing}")
        return [by_id[str(case_id)] for case_id in case_ids]

    max_cases = evaluation_cfg.get("max_cases")
    if max_cases is not None:
        return files[: int(max_cases)]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.get("logging", {}).get("level", "INFO"))
    sensor = load_sensor_json(cfg.get("sensor_path", Path(cfg.get("output_dir", "outputs/symbolic/pysr")) / "best_sensor.json"))
    split = cfg.get("evaluation", {}).get("split", "test")
    features_dir = Path(cfg.get("features_dir", "outputs/symbolic/features")) / split
    labels_dir = Path(cfg.get("labels_dir", "outputs/symbolic/labels")) / split
    output_dir = Path(cfg.get("evaluation", {}).get("output_dir", "outputs/symbolic/evaluation")) / split
    if bool(cfg.get("evaluation", {}).get("clean_output", False)) and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    figure_rows = []
    files = sorted(features_dir.glob("*.npz"))
    files = _select_feature_files(files, cfg)
    for idx, feature_path in enumerate(files):
        features = _load_npz(feature_path)
        labels = _load_npz(labels_dir / feature_path.name)
        scores = evaluate_expression(sensor["expression"], features, list(sensor["feature_names"]))
        threshold = float(sensor["threshold"])
        row = per_case_metrics(
            feature_path.stem,
            labels["shock_label"],
            scores,
            threshold,
            metadata={
                "Mach": float(np.asarray(features["Mach"])[0]),
                "AoA": float(np.asarray(features["AoA"])[0]),
                "pi_scaled": float(np.asarray(features["pi_scaled"])[0]),
            },
        )
        rows.append(row)
        if idx < int(cfg.get("evaluation", {}).get("plot_max_cases", 3)):
            save_prediction_diagnostics(
                output_dir / "figures" / feature_path.stem,
                features,
                labels,
                scores,
                threshold,
                max_points=int(cfg.get("evaluation", {}).get("plot_max_points", 80_000)),
            )
            figure_rows.append(
                {
                    "case_id": feature_path.stem,
                    "features": features,
                    "labels": labels,
                    "scores": scores,
                    "threshold": threshold,
                    "metrics": row,
                }
            )
    global_metrics = aggregate_metric_rows(rows)
    save_csv(output_dir / "per_case_metrics.csv", rows)
    save_json(output_dir / "global_metrics.json", global_metrics)
    save_symbolic_prediction_grid(
        output_dir / "shock_prediction_grid.png",
        figure_rows,
        max_points=int(cfg.get("evaluation", {}).get("plot_max_points", 80_000)),
        seed=int(cfg.get("seed", 42)),
    )
    print({"per_case_metrics": str(output_dir / "per_case_metrics.csv"), "global_metrics": global_metrics})


if __name__ == "__main__":
    main()
