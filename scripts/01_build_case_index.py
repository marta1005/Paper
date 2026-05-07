#!/usr/bin/env python
"""Build train/test CFD condition index files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shock_symbolic.data.case_indexing import build_case_index
from shock_symbolic.utils.config import load_config
from shock_symbolic.utils.io import save_json
from shock_symbolic.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.get("logging", {}).get("level", "INFO"))
    index_cfg = cfg.get("case_index", {})
    result = build_case_index(
        data_dir=cfg.get("data_dir", "data"),
        output_dir=cfg.get("outputs", {}).get("case_index_dir", "outputs/symbolic/case_index"),
        atol=float(index_cfg.get("atol", 1.0e-7)),
        batch_size=int(index_cfg.get("batch_size", 1_000_000)),
        max_cases_per_split=index_cfg.get("max_cases_per_split"),
    )
    summary = {split: {"n_cases": len(rows), "first_case": rows[0] if rows else None} for split, rows in result.items()}
    out = Path(cfg.get("outputs", {}).get("case_index_dir", "outputs/symbolic/case_index")) / "case_index_summary.json"
    save_json(out, summary)
    print(summary)


if __name__ == "__main__":
    main()
