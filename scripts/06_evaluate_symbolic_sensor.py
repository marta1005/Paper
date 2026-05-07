#!/usr/bin/env python
"""Evaluate a symbolic shock sensor on complete CFD cases."""

from __future__ import annotations

import argparse
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
from shock_symbolic.visualization.diagnostics import save_prediction_diagnostics


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


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
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    files = sorted(features_dir.glob("*.npz"))
    max_cases = cfg.get("evaluation", {}).get("max_cases")
    if max_cases is not None:
        files = files[: int(max_cases)]
    for idx, feature_path in enumerate(files):
        features = _load_npz(feature_path)
        labels = _load_npz(labels_dir / feature_path.name)
        scores = evaluate_expression(sensor["expression"], features, list(sensor["feature_names"]))
        threshold = float(sensor["threshold"])
        rows.append(
            per_case_metrics(
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
        )
        if idx < int(cfg.get("evaluation", {}).get("plot_max_cases", 3)):
            save_prediction_diagnostics(
                output_dir / "figures" / feature_path.stem,
                features,
                labels,
                scores,
                threshold,
                max_points=int(cfg.get("evaluation", {}).get("plot_max_points", 80_000)),
            )
    global_metrics = aggregate_metric_rows(rows)
    save_csv(output_dir / "per_case_metrics.csv", rows)
    save_json(output_dir / "global_metrics.json", global_metrics)
    print({"per_case_metrics": str(output_dir / "per_case_metrics.csv"), "global_metrics": global_metrics})


if __name__ == "__main__":
    main()
