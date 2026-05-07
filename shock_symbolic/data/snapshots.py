"""Snapshot reconstruction from pointwise ONERA arrays."""

from __future__ import annotations

from typing import Any

import numpy as np


def snapshot_from_case(
    x_array: np.ndarray,
    y_array: np.ndarray,
    case: dict[str, Any],
    max_points: int | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Return one CFD condition snapshot, optionally subsampled."""
    start = int(case["start"])
    stop = int(case["stop"])
    n_points = stop - start
    if max_points is not None and n_points > max_points:
        rng = np.random.default_rng(seed)
        local_idx = np.sort(rng.choice(n_points, size=int(max_points), replace=False))
    else:
        local_idx = np.arange(n_points, dtype=np.int64)
    global_idx = start + local_idx
    x_block = np.asarray(x_array[global_idx], dtype=np.float32)
    y_block = np.asarray(y_array[global_idx], dtype=np.float32)
    return {
        "case_id": np.asarray(case["case_id"]),
        "point_id": global_idx.astype(np.int64),
        "x": x_block[:, 0],
        "y": x_block[:, 1],
        "z": x_block[:, 2],
        "nx": x_block[:, 3],
        "ny": x_block[:, 4],
        "nz": x_block[:, 5],
        "Mach": x_block[:, 6],
        "AoA": x_block[:, 7],
        "pi_scaled": x_block[:, 8],
        "Cp": y_block[:, 0],
        "Cfx": y_block[:, 1],
        "Cfy": y_block[:, 2],
        "Cfz": y_block[:, 3],
    }


def snapshot_conditions(snapshot: dict[str, np.ndarray]) -> dict[str, float]:
    """Return representative flow-condition scalars for a snapshot."""
    return {
        "Mach": float(np.asarray(snapshot["Mach"])[0]),
        "AoA": float(np.asarray(snapshot["AoA"])[0]),
        "pi_scaled": float(np.asarray(snapshot["pi_scaled"])[0]),
    }
