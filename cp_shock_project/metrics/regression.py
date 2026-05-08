from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "Cp") -> dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err**2)) / denom if denom > 0 else 0.0
    return {f"MAE_{prefix}": mae, f"RMSE_{prefix}": rmse, f"R2_{prefix}": float(r2)}
