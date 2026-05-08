from __future__ import annotations

import numpy as np


def sensor_metrics(symbolic_chi: np.ndarray, oracle_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    chi = np.asarray(symbolic_chi).reshape(-1)
    oracle = np.asarray(oracle_score).reshape(-1)
    if chi.size == 0:
        return {}
    corr = float(np.corrcoef(chi, oracle)[0, 1]) if np.std(chi) > 0 and np.std(oracle) > 0 else 0.0
    mae = float(np.mean(np.abs(chi - oracle)))
    pred = chi >= threshold
    true = oracle >= threshold
    tp = float(np.sum(pred & true))
    fp = float(np.sum(pred & ~true))
    fn = float(np.sum(~pred & true))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    iou = tp / (tp + fp + fn + 1e-12)
    return {
        "correlation": corr,
        "MAE": mae,
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),
        "IoU": float(iou),
    }
