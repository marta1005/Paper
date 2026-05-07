#!/usr/bin/env python
"""Compute symbolic-sensor features per CFD condition."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shock_symbolic.data.case_indexing import load_case_index
from shock_symbolic.data.load_arrays import load_memmap_arrays, resolve_array_paths
from shock_symbolic.data.snapshots import snapshot_from_case
from shock_symbolic.features.grid2d_features import compute_snapshot_features_2d
from shock_symbolic.features.pressure_features import compute_snapshot_features
from shock_symbolic.utils.config import load_config
from shock_symbolic.utils.io import save_csv
from shock_symbolic.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.get("logging", {}).get("level", "INFO"))
    arrays = load_memmap_arrays(resolve_array_paths(cfg.get("data_dir", "data")))
    feature_cfg = cfg.get("features", {})
    case_index_dir = Path(cfg.get("case_index_dir", "outputs/symbolic/case_index"))
    output_dir = Path(cfg.get("output_dir", "outputs/symbolic/features"))
    clean_output = bool(feature_cfg.get("clean_output", False))
    splits = cfg.get("splits", ["train", "test"])
    rows = []
    for split in splits:
        cases = load_case_index(case_index_dir / f"case_index_{split}.csv")
        filters = cfg.get("case_filters", {})
        if filters.get("min_mach") is not None:
            cases = [case for case in cases if float(case["Mach"]) >= float(filters["min_mach"])]
        if filters.get("max_mach") is not None:
            cases = [case for case in cases if float(case["Mach"]) <= float(filters["max_mach"])]
        max_cases = feature_cfg.get("max_cases")
        if max_cases is not None:
            cases = cases[: int(max_cases)]
        split_out = output_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        if clean_output:
            for old_file in split_out.glob("*.npz"):
                old_file.unlink()
        x_key, y_key = f"X_{split}", f"Y_{split}"
        for idx, case in enumerate(cases):
            LOGGER.info("Computing features for %s (%s/%s)", case["case_id"], idx + 1, len(cases))
            mode = str(feature_cfg.get("mode", "grid2d")).lower()
            snapshot = snapshot_from_case(
                arrays[x_key],
                arrays[y_key],
                case,
                max_points=feature_cfg.get("max_points_per_case") if mode in {"pointwise", "knn"} else None,
                seed=int(cfg.get("seed", 42)) + idx,
            )
            if mode == "grid2d":
                features = compute_snapshot_features_2d(snapshot, feature_cfg.get("projection", {}))
            elif mode in {"pointwise", "knn"}:
                features = compute_snapshot_features(
                    snapshot,
                    k_neighbors=int(feature_cfg.get("k_neighbors", 16)),
                    batch_size=int(feature_cfg.get("batch_size", 4096)),
                    max_numpy_knn_points=int(feature_cfg.get("max_numpy_knn_points", 20_000)),
                )
            else:
                raise ValueError(f"Unsupported features.mode: {mode}")
            path = split_out / f"{case['case_id']}.npz"
            payload = dict(features)
            payload["case_id"] = np.asarray(case["case_id"])
            payload["split"] = np.asarray(split)
            np.savez_compressed(path, **payload)
            rows.append({"split": split, "case_id": case["case_id"], "path": str(path), "n_points": int(features["Cp"].shape[0]), "mode": mode})
    save_csv(output_dir / "features_summary.csv", rows)
    print({"features_dir": str(output_dir), "n_cases": len(rows)})


if __name__ == "__main__":
    main()
