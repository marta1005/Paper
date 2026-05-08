from __future__ import annotations

import numpy as np
import pandas as pd

from cp_shock_project.metrics.regression import regression_metrics


def per_case_metrics(y_true: np.ndarray, y_pred: np.ndarray, case_ids: np.ndarray, eps: float = 1e-12) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for cid in np.unique(case_ids):
        mask = case_ids == cid
        metrics = regression_metrics(y_true[mask], y_pred[mask], prefix="Cp")
        rmae = float(np.sum(np.abs(y_true[mask] - y_pred[mask])) / (np.sum(np.abs(y_true[mask])) + eps))
        rows.append({"case_id": int(cid), **metrics, "rMAE_Cp": rmae, "n_points": int(mask.sum())})
    df = pd.DataFrame(rows)
    summary = {
        "wrMAE_Cp": float(df["rMAE_Cp"].max()) if len(df) else 0.0,
        "mean_rMAE_Cp": float(df["rMAE_Cp"].mean()) if len(df) else 0.0,
        "median_rMAE_Cp": float(df["rMAE_Cp"].median()) if len(df) else 0.0,
    }
    return df, summary
