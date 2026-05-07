"""Small diagnostics helpers for symbolic shock sensor outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shock_symbolic.visualization.scatter_plots import save_case_scatter_suite

CP_CMAP = "jet"
FP_COLOR = "#d73027"
FN_COLOR = "#2166ac"
TP_COLOR = "#4d4d4d"
TN_COLOR = "#d9d9d9"


def save_prediction_diagnostics(
    output_dir: str | Path,
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    symbolic_score: np.ndarray,
    threshold: float,
    max_points: int = 80_000,
) -> None:
    """Save standard diagnostic plots including symbolic score and mask."""
    save_case_scatter_suite(
        output_dir,
        features,
        labels=labels,
        scores={
            "symbolic_score": np.asarray(symbolic_score),
            "symbolic_mask": (np.asarray(symbolic_score) >= float(threshold)).astype(np.float32),
        },
        max_points=max_points,
    )


def _sample_indices(n_points: int, max_points: int, seed: int) -> np.ndarray:
    if n_points <= max_points:
        return np.arange(n_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_points, size=max_points, replace=False))


def save_symbolic_prediction_grid(
    output_path: str | Path,
    rows: list[dict[str, object]],
    max_points: int = 80_000,
    seed: int = 42,
    title: str = "PySR shock sensor: real vs prediction",
) -> None:
    """Save a compact multi-case grid comparing shock labels and predictions.

    Each row is one CFD case. Columns show the same Cp field for context,
    the thresholded symbolic prediction, and a categorical error map.
    """
    if not rows:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cp_values = []
    for item in rows:
        features = item["features"]
        cp = np.asarray(features["Cp"], dtype=np.float32)  # type: ignore[index]
        cp_values.append(cp[np.isfinite(cp)])
    cp_all = np.concatenate([values for values in cp_values if values.size])
    cp_vmin, cp_vmax = np.percentile(cp_all, [1.0, 99.0]) if cp_all.size else (-1.0, 1.0)

    n_rows = len(rows)
    fig = plt.figure(figsize=(13.3, max(3.0, 2.55 * n_rows + 0.8)))
    grid = fig.add_gridspec(n_rows, 4, width_ratios=[0.9, 3.0, 3.0, 3.0], hspace=0.42, wspace=0.18)
    fig.subplots_adjust(left=0.035, right=0.992, bottom=0.045, top=0.92)

    column_titles = ["$C_p$ real", "predicted", "error"]
    for row_idx, item in enumerate(rows):
        features = item["features"]
        labels = item["labels"]
        scores = np.asarray(item["scores"], dtype=np.float32)
        threshold = float(item["threshold"])
        metrics = item.get("metrics", {})
        x = np.asarray(features["x"], dtype=np.float32)  # type: ignore[index]
        y = np.asarray(features["y"], dtype=np.float32)  # type: ignore[index]
        cp = np.asarray(features["Cp"], dtype=np.float32)  # type: ignore[index]
        shock_label = np.asarray(labels["shock_label"], dtype=np.float32).astype(bool)  # type: ignore[index]
        pred_mask = (scores >= threshold)
        idx = _sample_indices(len(cp), max_points=max_points, seed=seed + row_idx)
        idx_set = np.zeros(len(cp), dtype=bool)
        idx_set[idx] = True

        meta_ax = fig.add_subplot(grid[row_idx, 0])
        meta_ax.axis("off")
        case_id = str(item["case_id"])
        f1 = float(metrics.get("f1", np.nan)) if isinstance(metrics, dict) else np.nan
        iou = float(metrics.get("iou", np.nan)) if isinstance(metrics, dict) else np.nan
        meta_ax.text(0.98, 0.5, f"{case_id}\nF1={f1:.3f}\nIoU={iou:.3f}", ha="right", va="center", fontsize=8)

        axes = [fig.add_subplot(grid[row_idx, col_idx + 1]) for col_idx in range(3)]
        for col_idx, ax in enumerate(axes):
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(column_titles[col_idx] if row_idx == 0 else "", fontsize=9, pad=5)

        sc_cp = axes[0].scatter(x[idx], y[idx], c=cp[idx], s=0.9, cmap=CP_CMAP, vmin=cp_vmin, vmax=cp_vmax, linewidths=0)
        true_idx = idx_set & shock_label
        if np.any(true_idx):
            axes[0].scatter(x[true_idx], y[true_idx], s=6.5, c=TP_COLOR, linewidths=0, label="shock real")
        cbar = fig.colorbar(sc_cp, ax=axes[0], shrink=0.80, pad=0.015)
        cbar.ax.tick_params(labelsize=6)

        sc_pred = axes[1].scatter(x[idx], y[idx], c=cp[idx], s=0.9, cmap=CP_CMAP, vmin=cp_vmin, vmax=cp_vmax, linewidths=0)
        pred_idx = idx_set & pred_mask
        if np.any(pred_idx):
            axes[1].scatter(x[pred_idx], y[pred_idx], s=6.5, c=TP_COLOR, linewidths=0, label="shock pred")
        cbar = fig.colorbar(sc_pred, ax=axes[1], shrink=0.80, pad=0.015)
        cbar.ax.tick_params(labelsize=6)

        axes[2].scatter(x[idx], y[idx], s=0.9, c=TN_COLOR, linewidths=0)
        tp_idx = idx_set & shock_label & pred_mask
        fp_idx = idx_set & ~shock_label & pred_mask
        fn_idx = idx_set & shock_label & ~pred_mask
        if np.any(tp_idx):
            axes[2].scatter(x[tp_idx], y[tp_idx], s=6.5, c=TP_COLOR, linewidths=0, label="acierto")
        if np.any(fp_idx):
            axes[2].scatter(x[fp_idx], y[fp_idx], s=6.5, c=FP_COLOR, linewidths=0, label="falso +")
        if np.any(fn_idx):
            axes[2].scatter(x[fn_idx], y[fn_idx], s=6.5, c=FN_COLOR, linewidths=0, label="falso -")
        if row_idx == 0:
            axes[2].legend(loc="upper right", fontsize=6, frameon=True)

    fig.suptitle(title, fontsize=13, y=0.985)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
