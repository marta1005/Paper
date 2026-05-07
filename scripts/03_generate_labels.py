#!/usr/bin/env python
"""Generate shock/separation pseudo-labels from saved features."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shock_symbolic.labels import generate_separation_labels, generate_shock_labels
from shock_symbolic.utils.config import load_config
from shock_symbolic.utils.io import save_csv
from shock_symbolic.utils.logging import configure_logging
from shock_symbolic.visualization.scatter_plots import save_case_scatter_suite

LOGGER = logging.getLogger(__name__)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _select_feature_files(files: list[Path], cfg: dict) -> list[Path]:
    """Select feature files by explicit ids or sorted order, preserving config order."""
    case_ids = cfg.get("case_ids")
    if case_ids:
        by_id = {path.stem: path for path in files}
        missing = [str(case_id) for case_id in case_ids if str(case_id) not in by_id]
        if missing:
            raise KeyError(f"case_ids not found in features directory: {missing}")
        return [by_id[str(case_id)] for case_id in case_ids]

    condition_indices = cfg.get("condition_indices")
    if condition_indices:
        selected = []
        for raw_idx in condition_indices:
            idx = int(raw_idx)
            if idx < 0 or idx >= len(files):
                raise IndexError(f"condition index {idx} is outside the available range [0, {len(files) - 1}]")
            selected.append(files[idx])
        return selected

    max_cases = cfg.get("max_cases")
    if max_cases is not None:
        return files[: int(max_cases)]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.get("logging", {}).get("level", "INFO"))
    features_dir = Path(cfg.get("features_dir", "outputs/symbolic/features"))
    output_dir = Path(cfg.get("output_dir", "outputs/symbolic/labels"))
    rows = []
    clean_output = bool(cfg.get("clean_output", False))
    viz_cfg = cfg.get("visualization", {})
    plot_max_cases = int(viz_cfg.get("max_plot_cases", viz_cfg.get("max_cases", 0)))
    for split in cfg.get("splits", ["train", "test"]):
        files = sorted((features_dir / split).glob("*.npz"))
        files = _select_feature_files(files, cfg)
        split_out = output_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        if clean_output:
            for old_file in split_out.glob("*.npz"):
                old_file.unlink()
        for idx, feature_path in enumerate(files):
            LOGGER.info("Generating labels for %s", feature_path.stem)
            features = _load_npz(feature_path)
            shock = generate_shock_labels(features, cfg.get("shock", {}))
            sep = generate_separation_labels(features, cfg.get("separation", {}))
            payload = {
                "point_id": features["point_id"],
                "case_id": np.asarray(feature_path.stem),
                **shock,
                **sep,
            }
            path = split_out / f"{feature_path.stem}.npz"
            np.savez_compressed(path, **payload)
            rows.append(
                {
                    "split": split,
                    "case_id": feature_path.stem,
                    "path": str(path),
                    "shock_points": int(np.sum(shock["shock_label"])),
                    "sep_points": int(np.sum(sep["sep_label"])),
                    "shock_threshold": float(shock["shock_grad_threshold"]),
                }
            )
            if idx < plot_max_cases:
                save_case_scatter_suite(
                    Path(viz_cfg.get("output_dir", "outputs/symbolic/figures/labels")) / split / feature_path.stem,
                    features,
                    labels=payload,
                    max_points=int(viz_cfg.get("max_points", 80_000)),
                )
    save_csv(output_dir / "labels_summary.csv", rows)
    print({"labels_dir": str(output_dir), "n_cases": len(rows)})


if __name__ == "__main__":
    main()
