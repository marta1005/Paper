#!/usr/bin/env python
"""Train a PySR symbolic shock sensor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shock_symbolic.symbolic.pysr_trainer import train_pysr_sensor
from shock_symbolic.utils.config import load_config
from shock_symbolic.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.get("logging", {}).get("level", "INFO"))
    try:
        sensor = train_pysr_sensor(
            table_path=cfg.get("table_path", "outputs/symbolic/tables/shock_symbolic_train.csv"),
            output_dir=cfg.get("output_dir", "outputs/symbolic/pysr"),
            feature_names=list(cfg["feature_names"]),
            target_name=cfg.get("target_name", "shock_score"),
            pysr_config=cfg.get("pysr", {}),
            threshold_metric=cfg.get("threshold_metric", "f1"),
        )
    except RuntimeError as exc:
        print({"error": str(exc)})
        raise SystemExit(2) from exc
    print(sensor)


if __name__ == "__main__":
    main()
