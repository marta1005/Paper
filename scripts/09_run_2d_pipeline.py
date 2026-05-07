#!/usr/bin/env python
"""Run the full projected-2D symbolic sensor pipeline in order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, config: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), "--config", str(ROOT / config)]
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pysr", action="store_true", help="Run preprocessing/table/plot only; skip PySR training/evaluation/export.")
    parser.add_argument("--train-pysr", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-grid", action="store_true", help="Skip the critical Cp grid plot.")
    args = parser.parse_args()

    _run("00_inspect_arrays.py", "configs/data.yaml")
    _run("01_build_case_index.py", "configs/data.yaml")
    _run("02_compute_features.py", "configs/features.yaml")
    _run("03_generate_labels.py", "configs/labels.yaml")
    _run("04_build_symbolic_dataset.py", "configs/symbolic_dataset.yaml")
    run_pysr = args.train_pysr or not args.skip_pysr
    if run_pysr:
        _run("05_train_symbolic_sensor.py", "configs/pysr.yaml")
        _run("06_evaluate_symbolic_sensor.py", "configs/pysr.yaml")
        _run("07_export_sensor.py", "configs/pysr.yaml")
    if not args.skip_grid:
        _run("08_plot_critical_cp_grid.py", "configs/critical_cp_grid.yaml")


if __name__ == "__main__":
    main()
