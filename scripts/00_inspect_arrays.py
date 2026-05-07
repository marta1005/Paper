#!/usr/bin/env python
"""Inspect ONERA CRM WBPN pointwise arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shock_symbolic.data.load_arrays import inspect_arrays
from shock_symbolic.utils.config import load_config
from shock_symbolic.utils.io import save_json
from shock_symbolic.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.get("logging", {}).get("level", "INFO"))
    payload = inspect_arrays(cfg.get("data_dir", "data"), sample_size=int(cfg.get("inspect", {}).get("sample_size", 250_000)))
    out = Path(cfg.get("outputs", {}).get("inspect_json", "outputs/symbolic/inspect/arrays.json"))
    save_json(out, payload)
    print({"inspect_json": str(out), "X_train_shape": payload["X_train"]["shape"], "Y_train_shape": payload["Y_train"]["shape"]})


if __name__ == "__main__":
    main()
