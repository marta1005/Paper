"""Robust scaling and normalization utilities."""

from __future__ import annotations

import numpy as np


def robust_minmax(values: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> np.ndarray:
    """Map values to [0, 1] using percentile clipping."""
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(arr[finite], lower_percentile))
    hi = float(np.percentile(arr[finite], upper_percentile))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    scaled = (arr - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def safe_standardize(values: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    """Standardize finite values and replace non-finite outputs with zero."""
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float32)
    if not np.any(finite):
        return out
    mean = float(np.mean(arr[finite]))
    std = float(np.std(arr[finite]))
    out[finite] = (arr[finite] - mean) / max(std, eps)
    return out
