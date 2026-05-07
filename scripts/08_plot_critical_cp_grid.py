#!/usr/bin/env python
"""Plot a 2D grid of critical Cp cases, optionally comparing predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shock_symbolic.data.case_indexing import build_case_index_for_split, load_case_index
from shock_symbolic.data.load_arrays import load_memmap_arrays, resolve_array_paths
from shock_symbolic.data.snapshots import snapshot_from_case
from shock_symbolic.utils.config import load_config
from shock_symbolic.utils.io import save_json
from shock_symbolic.utils.logging import configure_logging
from shock_symbolic.visualization.cp_comparison import load_cp_prediction, save_critical_cp_grid_2d


def _ensure_case_index(cfg: dict, split: str, x_array):
    case_index_dir = Path(cfg.get("case_index_dir", "outputs/symbolic/case_index"))
    path = case_index_dir / f"case_index_{split}.csv"
    if path.exists():
        return load_case_index(path)
    case_index_dir.mkdir(parents=True, exist_ok=True)
    cases = build_case_index_for_split(
        x_array,
        split,
        atol=float(cfg.get("case_index", {}).get("atol", 1.0e-7)),
        batch_size=int(cfg.get("case_index", {}).get("batch_size", 1_000_000)),
    )
    return cases


def _select_cases(cases: list[dict], cfg: dict) -> list[dict]:
    explicit = cfg.get("condition_indices")
    if explicit:
        selected = []
        for raw_idx in explicit:
            idx = int(raw_idx)
            if idx < 0 or idx >= len(cases):
                raise IndexError(f"condition index {idx} is outside the available range [0, {len(cases) - 1}]")
            selected.append(cases[idx])
        return selected
    case_ids = cfg.get("case_ids")
    if case_ids:
        by_id = {str(case["case_id"]): case for case in cases}
        missing = [str(case_id) for case_id in case_ids if str(case_id) not in by_id]
        if missing:
            raise KeyError(f"case_ids not found in case index: {missing}")
        return [by_id[str(case_id)] for case_id in case_ids]

    filters = cfg.get("critical_selection", {})
    filtered = list(cases)
    if filters.get("min_mach") is not None:
        filtered = [case for case in filtered if float(case["Mach"]) >= float(filters["min_mach"])]
    if filters.get("max_mach") is not None:
        filtered = [case for case in filtered if float(case["Mach"]) <= float(filters["max_mach"])]
    if filters.get("min_abs_aoa") is not None:
        filtered = [case for case in filtered if abs(float(case["AoA"])) >= float(filters["min_abs_aoa"])]
    sort_by = str(filters.get("sort_by", "mach_abs_aoa"))
    if sort_by == "mach_abs_aoa":
        filtered.sort(key=lambda case: (float(case["Mach"]), abs(float(case["AoA"]))), reverse=True)
    elif sort_by == "abs_aoa":
        filtered.sort(key=lambda case: abs(float(case["AoA"])), reverse=True)
    elif sort_by == "mach":
        filtered.sort(key=lambda case: float(case["Mach"]), reverse=True)
    elif sort_by == "index":
        pass
    else:
        raise ValueError(f"Unsupported sort_by: {sort_by}")
    return filtered[: int(filters.get("top_k", cfg.get("top_k", 6)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.get("logging", {}).get("level", "INFO"))

    split = str(cfg.get("split", "test"))
    arrays = load_memmap_arrays(resolve_array_paths(cfg.get("data_dir", "data")))
    x_array = arrays[f"X_{split}"]
    y_array = arrays[f"Y_{split}"]
    all_cases = _ensure_case_index(cfg, split, x_array)
    cases = _select_cases(all_cases, cfg)
    if not cases:
        raise RuntimeError("No cases selected for critical Cp grid.")

    snapshots = []
    predictions = []
    pred_cfg = cfg.get("prediction", {})
    for row, case in enumerate(cases):
        snapshot = snapshot_from_case(
            x_array,
            y_array,
            case,
            max_points=None,
            seed=int(cfg.get("seed", 42)) + row,
        )
        snapshots.append(snapshot)
        predictions.append(
            load_cp_prediction(
                pred_cfg.get("path"),
                case_start=int(case["start"]),
                case_stop=int(case["stop"]),
                n_case_points=int(case["n_points"]),
                case_id=str(case["case_id"]),
                key=pred_cfg.get("key"),
                column=int(pred_cfg.get("column", 0)),
            )
        )

    output_path = Path(cfg.get("output_path", "outputs/symbolic/figures/critical_cp_grid.png"))
    payload = save_critical_cp_grid_2d(
        cases,
        snapshots,
        output_path,
        cp_predictions=predictions,
        mask_config=cfg.get("plot", {}),
        max_points_per_case=cfg.get("plot", {}).get("max_points_per_case", 80_000),
        seed=int(cfg.get("seed", 42)),
        robust_percentiles=tuple(cfg.get("plot", {}).get("robust_percentiles", [1.0, 99.0])),
        point_size=float(cfg.get("plot", {}).get("point_size", 0.9)),
        title=str(cfg.get("title", "Critical ONERA CRM WBPN Cp cases")),
    )
    summary_path = output_path.with_suffix(".json")
    save_json(summary_path, payload)
    print({"grid": str(output_path), "summary": str(summary_path), "n_cases": len(cases)})


if __name__ == "__main__":
    main()
