"""2D wing Cp comparison plots for true and predicted pressure fields."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
_CACHE = Path(os.environ.get("MPLCONFIGDIR", ".matplotlib_cache"))
if (_CACHE.exists() and not os.access(_CACHE, os.W_OK)) or (not _CACHE.exists() and not os.access(_CACHE.parent, os.W_OK)):
    _CACHE = Path(".matplotlib_cache")
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_CACHE)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


ABS_ERROR_CMAP = LinearSegmentedColormap.from_list("gray_red_abs_error", ["#eeeeee", "#fdae61", "#d73027"])


def finite_minmax(values: np.ndarray, default: tuple[float, float] = (-1.0, 1.0)) -> tuple[float, float]:
    """Return exact finite min/max with a small guard for constant fields."""
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return default
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        eps = max(abs(vmin) * 1.0e-6, 1.0e-6)
        return vmin - eps, vmax + eps
    return vmin, vmax


def sample_indices(n_points: int, max_points: int | None = None, seed: int = 42) -> np.ndarray:
    """Return deterministic sorted plotting indices."""
    if max_points is None or n_points <= max_points:
        return np.arange(n_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_points, size=int(max_points), replace=False))


def surface_mask(snapshot: dict[str, np.ndarray], config: dict[str, Any] | None = None) -> np.ndarray:
    """Build a plotting mask from surface and coordinate filters."""
    config = config or {}
    n_points = len(snapshot["x"])
    mask = np.ones(n_points, dtype=bool)
    surface = str(config.get("surface", "all")).lower()
    nz_threshold = float(config.get("normal_z_threshold", 0.0))
    if surface == "upper":
        mask &= np.asarray(snapshot["nz"]) >= nz_threshold
    elif surface == "lower":
        mask &= np.asarray(snapshot["nz"]) <= -nz_threshold
    elif surface != "all":
        raise ValueError("surface must be one of: all, upper, lower")

    for name in ("x", "y", "z"):
        values = np.asarray(snapshot[name])
        min_value = config.get(f"min_{name}")
        max_value = config.get(f"max_{name}")
        if min_value is not None:
            mask &= values >= float(min_value)
        if max_value is not None:
            mask &= values <= float(max_value)
    return mask


def load_cp_prediction(
    prediction_path: str | Path | None,
    case_start: int,
    case_stop: int,
    n_case_points: int,
    case_id: str | None = None,
    key: str | None = None,
    column: int = 0,
) -> np.ndarray | None:
    """Load a Cp prediction vector from `.npy` or `.npz`.

    Accepted shapes:
    - one value per current case,
    - one value per full split, sliced by case start/stop,
    - `.npz` with one array keyed by case_id, e.g. `test_0012`,
    - 2D arrays where `column` selects the Cp column.
    """
    if prediction_path is None:
        return None
    path = Path(prediction_path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r")
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if key is not None and key in payload.files:
                array = np.asarray(payload[key])
            else:
                candidates = []
                if case_id is not None:
                    candidates.extend([case_id, f"{case_id}_cp_pred", f"cp_pred_{case_id}"])
                candidates.extend(["Cp_pred", "cp_pred", "pred", "prediction", "Y_pred", "y_pred"])
                selected = next((name for name in candidates if name in payload.files), None)
                if selected is None:
                    requested = [key] if key is not None else []
                    raise KeyError(f"No Cp prediction key found in {path}. Tried: {requested + candidates}")
                array = np.asarray(payload[selected])
    else:
        raise ValueError(f"Unsupported prediction extension: {path.suffix}")

    if array.ndim == 2:
        array = array[:, int(column)]
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    if array.shape[0] == n_case_points:
        return array
    if array.shape[0] >= case_stop:
        return np.asarray(array[case_start:case_stop], dtype=np.float32)
    raise ValueError(
        f"Prediction length {array.shape[0]} is incompatible with case length {n_case_points} "
        f"and split stop index {case_stop}."
    )


def cp_error_metrics(cp_true: np.ndarray, cp_pred: np.ndarray) -> dict[str, float]:
    """Compute simple Cp comparison metrics."""
    err = np.asarray(cp_pred, dtype=np.float64) - np.asarray(cp_true, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs_error": float(np.max(np.abs(err))),
        "bias": float(np.mean(err)),
    }


def save_cp_comparison_2d(
    snapshot: dict[str, np.ndarray],
    output_path: str | Path,
    cp_pred: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    max_points: int | None = 160_000,
    seed: int = 42,
    title: str | None = None,
    cmap: str = "jet",
    robust_percentiles: tuple[float, float] = (1.0, 99.0),
    point_size: float = 0.55,
) -> dict[str, float]:
    """Save a planform 2D Cp comparison plot and return metrics."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(snapshot["x"], dtype=np.float32)
    y = np.asarray(snapshot["y"], dtype=np.float32)
    cp_true = np.asarray(snapshot["Cp"], dtype=np.float32)
    if mask is None:
        mask = np.ones_like(cp_true, dtype=bool)
    idx_pool = np.flatnonzero(mask)
    idx = idx_pool[sample_indices(idx_pool.size, max_points=max_points, seed=seed)]

    metrics: dict[str, float] = {}
    if cp_pred is None:
        values = cp_true[idx]
        vmin, vmax = finite_minmax(values)
        fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
        sc = ax.scatter(x[idx], y[idx], c=values, s=point_size, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(title or "Cp true")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=ax, shrink=0.86, label="Cp")
        fig.savefig(out, dpi=260)
        plt.close(fig)
        return metrics

    cp_pred = np.asarray(cp_pred, dtype=np.float32)
    metrics = cp_error_metrics(cp_true[mask], cp_pred[mask])
    cp_values = np.concatenate([cp_true[idx], cp_pred[idx]])
    vmin, vmax = finite_minmax(cp_values)
    err = cp_pred - cp_true
    abs_err = np.abs(err)
    err_values = abs_err[idx]
    finite_err = np.isfinite(err_values)
    err_lim = float(np.max(err_values[finite_err])) if np.any(finite_err) else 1.0
    err_lim = max(err_lim, 1.0e-8)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)
    fields = [
        ("Cp true", cp_true, cmap, vmin, vmax),
        ("Cp pred", cp_pred, cmap, vmin, vmax),
        ("|Cp pred - true|", abs_err, ABS_ERROR_CMAP, 0.0, err_lim),
    ]
    for ax, (label, field, field_cmap, field_vmin, field_vmax) in zip(axes, fields):
        sc = ax.scatter(
            x[idx],
            y[idx],
            c=field[idx],
            s=point_size,
            cmap=field_cmap,
            vmin=field_vmin,
            vmax=field_vmax,
            linewidths=0,
        )
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=ax, shrink=0.82)
    fig.suptitle(
        title
        or f"Cp comparison | MAE={metrics['mae']:.4g}, RMSE={metrics['rmse']:.4g}, bias={metrics['bias']:.4g}",
        fontsize=12,
    )
    fig.savefig(out, dpi=260)
    plt.close(fig)
    return metrics


def save_critical_cp_grid_2d(
    cases: list[dict[str, Any]],
    snapshots: list[dict[str, np.ndarray]],
    output_path: str | Path,
    cp_predictions: list[np.ndarray | None] | None = None,
    mask_config: dict[str, Any] | None = None,
    max_points_per_case: int | None = 80_000,
    seed: int = 42,
    robust_percentiles: tuple[float, float] = (1.0, 99.0),
    point_size: float = 0.55,
    title: str = "Critical ONERA CRM WBPN cases",
) -> dict[str, Any]:
    """Save a multi-case 2D grid of Cp truth/prediction/error maps.

    Rows correspond to CFD conditions. Columns are `Truth Cp`, `Predicted Cp`,
    and absolute prediction error when predictions are provided for every case;
    otherwise the grid contains only the true Cp maps.
    """
    if not cases:
        raise ValueError("No cases provided for Cp grid plotting.")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cp_predictions = cp_predictions or [None] * len(snapshots)
    has_predictions = all(pred is not None for pred in cp_predictions)
    if has_predictions:
        n_cols = 3
        n_rows = len(snapshots)
    else:
        n_cols = min(3, len(snapshots))
        n_rows = int(np.ceil(len(snapshots) / n_cols))

    selected_indices: list[np.ndarray] = []
    cp_for_scale: list[np.ndarray] = []
    err_for_scale: list[np.ndarray] = []
    row_metrics: list[dict[str, Any]] = []

    for row, (case, snapshot, pred) in enumerate(zip(cases, snapshots, cp_predictions)):
        mask = surface_mask(snapshot, mask_config)
        idx_pool = np.flatnonzero(mask)
        idx = idx_pool[sample_indices(idx_pool.size, max_points=max_points_per_case, seed=seed + row)]
        selected_indices.append(idx)
        cp_true = np.asarray(snapshot["Cp"], dtype=np.float32)
        cp_for_scale.append(cp_true[idx])
        row_payload: dict[str, Any] = {
            "case_id": case["case_id"],
            "Mach": float(case["Mach"]),
            "AoA": float(case["AoA"]),
            "pi_scaled": float(case["pi_scaled"]),
            "points_plotted": int(idx.size),
        }
        if pred is not None:
            pred_arr = np.asarray(pred, dtype=np.float32)
            cp_for_scale.append(pred_arr[idx])
            err = np.abs(pred_arr - cp_true)
            err_for_scale.append(err[idx])
            row_payload.update(cp_error_metrics(cp_true[mask], pred_arr[mask]))
        row_metrics.append(row_payload)

    cp_values = np.concatenate([values[np.isfinite(values)] for values in cp_for_scale if values.size])
    cp_vmin, cp_vmax = finite_minmax(cp_values)
    if err_for_scale:
        err_values = np.concatenate([values[np.isfinite(values)] for values in err_for_scale if values.size])
        err_lim = float(np.max(err_values)) if err_values.size else 1.0
    else:
        err_lim = 1.0
    err_lim = max(err_lim, 1.0e-8)

    fig_width = 16.2 if has_predictions else 4.8 * n_cols
    fig_height = max(3.2, (2.65 if has_predictions else 3.2) * n_rows + 0.8)
    fig = plt.figure(figsize=(fig_width, fig_height))
    if has_predictions:
        fig.subplots_adjust(left=0.035, right=0.992, bottom=0.045, top=0.93, hspace=0.42, wspace=0.18)
        outer = gridspec.GridSpec(
            n_rows,
            n_cols + 1,
            figure=fig,
            hspace=0.42,
            wspace=0.18,
            width_ratios=[0.82, 3.0, 3.0, 3.0],
        )
    else:
        fig.subplots_adjust(left=0.06, right=0.985, bottom=0.045, top=0.93, hspace=0.48, wspace=0.26)
        outer = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.48, wspace=0.26)

    for row, (case, snapshot, pred, idx) in enumerate(zip(cases, snapshots, cp_predictions, selected_indices)):
        x = np.asarray(snapshot["x"], dtype=np.float32)
        y = np.asarray(snapshot["y"], dtype=np.float32)
        cp_true = np.asarray(snapshot["Cp"], dtype=np.float32)
        row_title = f"{case['case_id']} | M={float(case['Mach']):.2f} | AoA={float(case['AoA']):.1f} | pi={float(case['pi_scaled']):.1f}"
        row_label = (
            f"{case['case_id']}\n"
            f"M={float(case['Mach']):.2f}\n"
            f"AoA={float(case['AoA']):.1f}\n"
            f"pi={float(case['pi_scaled']):.1f}"
        )
        fields: list[tuple[str, np.ndarray, str, float, float]] = [
            ("$C_p$ real", cp_true, "jet", float(cp_vmin), float(cp_vmax)),
        ]
        if has_predictions and pred is not None:
            pred_arr = np.asarray(pred, dtype=np.float32)
            fields.append(("$C_p$ predicho", pred_arr, "jet", float(cp_vmin), float(cp_vmax)))
            fields.append(("Error absoluto", np.abs(pred_arr - cp_true), ABS_ERROR_CMAP, 0.0, err_lim))

        if has_predictions:
            cell_row = row
            cell_col_offset = 1
            ax_meta = fig.add_subplot(outer[cell_row, 0])
            ax_meta.axis("off")
            ax_meta.text(0.98, 0.5, row_label, ha="right", va="center", fontsize=8, linespacing=1.25)
        else:
            cell_row = row // n_cols
            cell_col_offset = row % n_cols

        for col, (label, values, cmap, vmin, vmax) in enumerate(fields):
            ax = fig.add_subplot(outer[cell_row, cell_col_offset + col])
            sc = ax.scatter(x[idx], y[idx], c=values[idx], s=point_size, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            if has_predictions:
                ax.set_title(label if row == 0 else "", fontsize=8, pad=5)
            else:
                ax.set_title(row_title, fontsize=8, pad=5)
            cbar = fig.colorbar(sc, ax=ax, shrink=0.80, pad=0.015)
            cbar.ax.tick_params(labelsize=6)

    fig.suptitle(title, fontsize=13, y=0.985)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"path": str(out), "has_predictions": bool(has_predictions), "cases": row_metrics}
