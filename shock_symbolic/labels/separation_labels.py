"""Separation pseudo-labels from skin-friction cues."""

from __future__ import annotations

from typing import Any

import numpy as np

from shock_symbolic.features.scaling import robust_minmax


def generate_separation_labels(features: dict[str, np.ndarray], config: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Generate a simple separation pseudo-label from Cf magnitude/topology."""
    config = config or {}
    cf_mag = np.asarray(features["Cf_mag"], dtype=np.float32)
    cf_parallel = np.asarray(features["Cf_parallel"], dtype=np.float32)
    grad_cf = np.asarray(features["grad_Cf_mag"], dtype=np.float32)
    low_cf_percentile = float(config.get("low_cf_percentile", 20.0))
    high_grad_percentile = float(config.get("high_grad_cf_percentile", 95.0))
    low_cf_threshold = float(np.percentile(cf_mag[np.isfinite(cf_mag)], low_cf_percentile)) if np.any(np.isfinite(cf_mag)) else 0.0
    high_grad_threshold = float(np.percentile(grad_cf[np.isfinite(grad_cf)], high_grad_percentile)) if np.any(np.isfinite(grad_cf)) else 0.0
    low_cf = cf_mag <= low_cf_threshold
    high_grad = grad_cf >= high_grad_threshold
    reversed_cf = cf_parallel < float(config.get("reverse_parallel_threshold", 0.0))
    if bool(config.get("require_low_cf_for_reversal", False)):
        sep = low_cf & (high_grad | reversed_cf)
    else:
        sep = low_cf | (high_grad & reversed_cf)
    sep_score = np.maximum(1.0 - robust_minmax(cf_mag, 1.0, 99.0), robust_minmax(grad_cf, 50.0, 99.5))
    return {
        "sep_label": sep.astype(np.float32),
        "sep_score": sep_score.astype(np.float32),
        "sep_low_cf_threshold": np.asarray(low_cf_threshold, dtype=np.float32),
        "sep_high_grad_cf_threshold": np.asarray(high_grad_threshold, dtype=np.float32),
    }
