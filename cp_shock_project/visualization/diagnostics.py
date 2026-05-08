from __future__ import annotations

from pathlib import Path

import numpy as np

from cp_shock_project.visualization.scatter import scatter_view


def plot_prediction_diagnostics(X: np.ndarray, predictions: dict[str, np.ndarray], out_dir: str | Path, views: list[str] | None = None) -> None:
    views = views or ["xy", "xz"]
    root = Path(out_dir)
    cp_true = predictions["Cp"]
    cp_pred = predictions["Cp_pred"]
    fields = {
        "Cp_true": cp_true,
        "Cp_pred": cp_pred,
        "Cp_error": cp_pred - cp_true,
        "abs_error": np.abs(cp_pred - cp_true),
    }
    for optional in ["oracle_shock_score", "symbolic_chi", "chi", "delta_Cp", "Cp_smooth"]:
        if optional in predictions:
            fields[optional] = predictions[optional]
    for name, values in fields.items():
        for view in views:
            scatter_view(X, values, root / f"{name}_{view}.png", f"{name} ({view})", view=view)
