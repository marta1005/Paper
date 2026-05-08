#!/usr/bin/env python
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from cp_shock_project.symbolic.export import export_sensor
from cp_shock_project.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = export_sensor(cfg.get("best_sensor_path", "outputs/symbolic_sensor/best_sensor.json"), cfg.get("export_dir", "outputs/symbolic_sensor/export"))
    print(paths)


if __name__ == "__main__":
    main()
