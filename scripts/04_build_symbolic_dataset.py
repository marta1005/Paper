#!/usr/bin/env python
"""Build a balanced tabular dataset for symbolic regression."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shock_symbolic.symbolic.build_table import build_symbolic_table
from shock_symbolic.utils.config import load_config
from shock_symbolic.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.get("logging", {}).get("level", "INFO"))
    split = cfg.get("split", "train")
    features_dir = Path(cfg.get("features_dir", "outputs/symbolic/features")) / split
    labels_dir = Path(cfg.get("labels_dir", "outputs/symbolic/labels")) / split
    feature_files = sorted(features_dir.glob("*.npz"))
    label_files = [labels_dir / f.name for f in feature_files]
    max_cases = cfg.get("max_cases")
    if max_cases is not None:
        feature_files = feature_files[: int(max_cases)]
        label_files = label_files[: int(max_cases)]
    out = build_symbolic_table(
        feature_files,
        label_files,
        output_base=cfg.get("output_base", "outputs/symbolic/tables/shock_symbolic_train"),
        feature_names=list(cfg["feature_names"]),
        target_name=cfg.get("target_name", "shock_score"),
        max_positive_per_case=cfg.get("max_positive_per_case"),
        negative_ratio=float(cfg.get("negative_ratio", 3.0)),
        seed=int(cfg.get("seed", 42)),
    )
    print({"table": str(out)})


if __name__ == "__main__":
    main()
