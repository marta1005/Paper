#!/usr/bin/env python
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from cp_shock_project.symbolic.gplearn_sensor import train_gplearn_sensor
from cp_shock_project.symbolic.pysr_sensor import train_pysr_sensor
from cp_shock_project.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    backend = cfg.get("backend", "gplearn").lower()
    common = (
        cfg.get("train_sensor_path", "processed/symbolic_sensor/train_sensor.parquet"),
        cfg.get("val_sensor_path", "processed/symbolic_sensor/val_sensor.parquet"),
        cfg.get("sensor_output_dir", "outputs/symbolic_sensor"),
    )
    if backend == "gplearn":
        train_gplearn_sensor(*common, cfg.get("gplearn", {}))
    elif backend == "pysr":
        train_pysr_sensor(*common, cfg.get("pysr", {}))
    else:
        raise ValueError(f"Unknown symbolic sensor backend: {backend}")


if __name__ == "__main__":
    main()
