"""Threshold calibration for symbolic sensor scores."""

from __future__ import annotations

import numpy as np

from shock_symbolic.metrics.classification import binary_classification_metrics


def calibrate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    metric: str = "f1",
    num_thresholds: int = 200,
) -> dict[str, float]:
    """Find a scalar threshold maximizing a binary classification metric."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    finite = np.isfinite(s)
    if not np.any(finite):
        return {"threshold": 0.0, metric: 0.0}
    lo, hi = float(np.min(s[finite])), float(np.max(s[finite]))
    if hi <= lo:
        thresholds = np.asarray([lo])
    else:
        thresholds = np.linspace(lo, hi, int(num_thresholds))
    best: dict[str, float] = {"threshold": float(thresholds[0]), metric: -1.0}
    for threshold in thresholds:
        row = binary_classification_metrics(y, s >= threshold)
        value = float(row.get(metric, row["f1"]))
        if value > best[metric]:
            best = {"threshold": float(threshold), metric: value, **row}
    return best
