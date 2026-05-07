"""Projected 2D feature engineering for fast symbolic shock sensing."""

from __future__ import annotations

from typing import Any

import numpy as np

from shock_symbolic.features.cf_features import compute_cf_directional_features

EPS = 1.0e-8

GRID_FEATURE_NAMES = [
    "Cp",
    "Cfx",
    "Cfy",
    "Cfz",
    "Cf_mag",
    "Cf_parallel",
    "Cf_perp",
    "Cf_angle_stream",
    "grad_Cp_mag",
    "grad_Cp_streamwise",
    "local_Cp_contrast",
    "grad_Cf_mag",
    "local_Cf_contrast",
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "Mach",
    "AoA",
    "pi_scaled",
]


def _surface_mask(snapshot: dict[str, np.ndarray], config: dict[str, Any]) -> np.ndarray:
    surface = str(config.get("surface", "upper")).lower()
    nz = np.asarray(snapshot["nz"], dtype=np.float32)
    threshold = float(config.get("normal_z_threshold", 0.0))
    if surface == "upper":
        return nz >= threshold
    if surface == "lower":
        return nz <= -threshold
    if surface == "all":
        return np.ones_like(nz, dtype=bool)
    raise ValueError("projection.surface must be one of: upper, lower, all")


def _cell_ids(x: np.ndarray, y: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, x_edges.size - 2).astype(np.int64)
    y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, y_edges.size - 2).astype(np.int64)
    n_x = x_edges.size - 1
    return x_bin, y_bin, (y_bin * n_x + x_bin).astype(np.int64)


def _mean_grid(values: np.ndarray, cell_id: np.ndarray, n_cells: int, shape: tuple[int, int]) -> np.ndarray:
    sums = np.bincount(cell_id, weights=np.asarray(values, dtype=np.float64), minlength=n_cells)
    counts = np.bincount(cell_id, minlength=n_cells)
    grid = np.full(n_cells, np.nan, dtype=np.float32)
    valid = counts > 0
    grid[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    return grid.reshape(shape)


def _first_point_id(point_id: np.ndarray, cell_id: np.ndarray, n_cells: int, shape: tuple[int, int]) -> np.ndarray:
    order = np.argsort(cell_id, kind="stable")
    sorted_cells = cell_id[order]
    first = np.unique(sorted_cells, return_index=True)[1]
    out = np.full(n_cells, -1, dtype=np.int64)
    out[sorted_cells[first]] = np.asarray(point_id, dtype=np.int64)[order[first]]
    return out.reshape(shape)


def _fill_invalid(grid: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.float32).copy()
    if np.any(valid):
        fill = float(np.nanmean(arr[valid]))
    else:
        fill = 0.0
    arr[~np.isfinite(arr)] = fill
    return arr


def _gradient_mag(grid: np.ndarray, valid: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    filled = _fill_invalid(grid, valid)
    gy, gx = np.gradient(filled, max(float(dy), EPS), max(float(dx), EPS))
    gx = gx.astype(np.float32) * valid
    gy = gy.astype(np.float32) * valid
    mag = np.sqrt(gx**2 + gy**2 + EPS).astype(np.float32) * valid
    return mag, gx, gy


def _local_contrast_3x3(grid: np.ndarray, valid: np.ndarray) -> np.ndarray:
    filled = _fill_invalid(grid, valid)
    padded_values = np.pad(filled, 1, mode="edge")
    padded_valid = np.pad(valid, 1, mode="constant", constant_values=False)
    local_min = np.full_like(filled, np.inf, dtype=np.float32)
    local_max = np.full_like(filled, -np.inf, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            values = padded_values[dy : dy + filled.shape[0], dx : dx + filled.shape[1]]
            is_valid = padded_valid[dy : dy + filled.shape[0], dx : dx + filled.shape[1]]
            local_min = np.where(is_valid, np.minimum(local_min, values), local_min)
            local_max = np.where(is_valid, np.maximum(local_max, values), local_max)
    contrast = local_max - local_min
    contrast[~np.isfinite(contrast)] = 0.0
    return contrast.astype(np.float32) * valid


def compute_snapshot_features_2d(
    snapshot: dict[str, np.ndarray],
    projection_config: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Project one CFD condition to a 2D grid and compute fast 2D features.

    The saved output remains a flat valid-cell table so downstream symbolic
    regression code does not need to change.
    """
    cfg = projection_config or {}
    x_bins = int(cfg.get("x_bins", 384))
    y_bins = int(cfg.get("y_bins", 192))
    mask = _surface_mask(snapshot, cfg)
    if not np.any(mask):
        raise ValueError("Projection mask selected no points.")

    x = np.asarray(snapshot["x"], dtype=np.float32)[mask]
    y = np.asarray(snapshot["y"], dtype=np.float32)[mask]
    shape = (y_bins, x_bins)
    n_cells = x_bins * y_bins
    x_edges = np.linspace(float(x.min()), float(x.max()), x_bins + 1, dtype=np.float32)
    y_edges = np.linspace(float(y.min()), float(y.max()), y_bins + 1, dtype=np.float32)
    x_bin, y_bin, cell_id = _cell_ids(x, y, x_edges, y_edges)
    valid_grid = (np.bincount(cell_id, minlength=n_cells).reshape(shape) > 0)
    flat_valid = valid_grid.reshape(-1)
    rows, cols = np.nonzero(valid_grid)
    dx_grid = float(np.mean(np.diff(x_edges))) if x_edges.size > 1 else 1.0
    dy_grid = float(np.mean(np.diff(y_edges))) if y_edges.size > 1 else 1.0

    grids: dict[str, np.ndarray] = {}
    for name in ("x", "y", "z", "nx", "ny", "nz", "Mach", "AoA", "pi_scaled", "Cp", "Cfx", "Cfy", "Cfz"):
        grids[name] = _mean_grid(np.asarray(snapshot[name])[mask], cell_id, n_cells, shape)
    point_grid = _first_point_id(np.asarray(snapshot["point_id"])[mask], cell_id, n_cells, shape)

    cp_grad_mag, cp_grad_x, _ = _gradient_mag(grids["Cp"], valid_grid, dx_grid, dy_grid)
    cf_flat = compute_cf_directional_features(
        grids["Cfx"].reshape(-1)[flat_valid],
        grids["Cfy"].reshape(-1)[flat_valid],
        grids["Cfz"].reshape(-1)[flat_valid],
        np.column_stack(
            [
                grids["nx"].reshape(-1)[flat_valid],
                grids["ny"].reshape(-1)[flat_valid],
                grids["nz"].reshape(-1)[flat_valid],
            ]
        ),
        grids["AoA"].reshape(-1)[flat_valid],
    )
    cf_mag_grid = np.full(shape, np.nan, dtype=np.float32)
    cf_mag_grid.reshape(-1)[flat_valid] = cf_flat["Cf_mag"]
    cf_grad_mag, _, _ = _gradient_mag(cf_mag_grid, valid_grid, dx_grid, dy_grid)

    features: dict[str, np.ndarray] = {
        "point_id": point_grid[valid_grid].astype(np.int64),
        "grid_row": rows.astype(np.int64),
        "grid_col": cols.astype(np.int64),
        "grid_shape": np.asarray(shape, dtype=np.int64),
        "x_edges": x_edges.astype(np.float32),
        "y_edges": y_edges.astype(np.float32),
        "valid_mask_2d": valid_grid.astype(np.uint8),
        "grad_Cp_mag": cp_grad_mag[valid_grid].astype(np.float32),
        "grad_Cp_streamwise": cp_grad_x[valid_grid].astype(np.float32),
        "local_Cp_contrast": _local_contrast_3x3(grids["Cp"], valid_grid)[valid_grid].astype(np.float32),
        "grad_Cf_mag": cf_grad_mag[valid_grid].astype(np.float32),
        "local_Cf_contrast": _local_contrast_3x3(cf_mag_grid, valid_grid)[valid_grid].astype(np.float32),
        "feature_names": np.asarray(GRID_FEATURE_NAMES),
        "projection_mode": np.asarray("grid2d"),
    }
    for name in ("Cp", "Cfx", "Cfy", "Cfz", "x", "y", "z", "nx", "ny", "nz", "Mach", "AoA", "pi_scaled"):
        features[name] = grids[name][valid_grid].astype(np.float32)
    features.update({name: values.astype(np.float32) for name, values in cf_flat.items()})
    return features
