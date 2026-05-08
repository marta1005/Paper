from __future__ import annotations

from pathlib import Path

import numpy as np

from cp_shock_project.visualization.scatter import scatter_view


def plot_improvement_map(X: np.ndarray, y_true: np.ndarray, baseline_pred: np.ndarray, main_pred: np.ndarray, out_dir: str | Path, view: str = "xy") -> None:
    """Plot positive values where the main model improves over baseline."""
    improvement = np.abs(y_true - baseline_pred) - np.abs(y_true - main_pred)
    scatter_view(X, improvement, Path(out_dir) / f"improvement_{view}.png", f"Improvement ({view})", view=view, cmap="PiYG")
