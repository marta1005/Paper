from __future__ import annotations

import numpy as np

from cp_shock_project.metrics.regression import regression_metrics


def shock_region_metrics(y_true: np.ndarray, y_pred: np.ndarray, oracle_shock_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    score = np.asarray(oracle_shock_score).reshape(-1)
    shock = score >= threshold
    non = ~shock
    out: dict[str, float] = {}
    if shock.any():
        m = regression_metrics(true[shock], pred[shock], prefix="Cp_shock")
        out["MAE_Cp_shock"] = m["MAE_Cp_shock"]
        out["RMSE_Cp_shock"] = m["RMSE_Cp_shock"]
    else:
        out["MAE_Cp_shock"] = 0.0
        out["RMSE_Cp_shock"] = 0.0
    if non.any():
        m = regression_metrics(true[non], pred[non], prefix="Cp_nonshock")
        out["MAE_Cp_nonshock"] = m["MAE_Cp_nonshock"]
        out["RMSE_Cp_nonshock"] = m["RMSE_Cp_nonshock"]
    else:
        out["MAE_Cp_nonshock"] = 0.0
        out["RMSE_Cp_nonshock"] = 0.0
    out["shock_error_ratio"] = float(out["MAE_Cp_shock"] / (out["MAE_Cp_nonshock"] + 1e-12))
    return out


def gradient_error(y_true: np.ndarray, y_pred: np.ndarray, neighbor_indices: np.ndarray, neighbor_distances: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-8) -> float:
    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    rows = np.arange(len(true)) if mask is None else np.flatnonzero(mask)
    total = 0.0
    count = 0
    for i in rows:
        nidx = neighbor_indices[i]
        valid = (nidx >= 0) & np.isfinite(neighbor_distances[i])
        valid &= np.isfinite(pred[nidx]) & np.isfinite(true[nidx]) & np.isfinite(pred[i]) & np.isfinite(true[i])
        if not np.any(valid):
            continue
        err = np.abs(((pred[nidx[valid]] - pred[i]) - (true[nidx[valid]] - true[i])) / (neighbor_distances[i, valid] + eps))
        total += float(np.sum(err))
        count += int(err.size)
    return total / max(count, 1)
