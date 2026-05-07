"""Shock pseudo-labels from local pressure-gradient features."""

from __future__ import annotations

from typing import Any

import numpy as np

from shock_symbolic.features.scaling import robust_minmax


def generate_shock_labels(features: dict[str, np.ndarray], config: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Generate per-condition shock labels and continuous shock score.

    The label threshold is a percentile of `grad_Cp_mag` within the current CFD
    condition. Optionally, labels are suppressed below `min_mach`.
    """
    config = config or {}
    grad = np.asarray(features["grad_Cp_mag"], dtype=np.float32)
    mach = np.asarray(features["Mach"], dtype=np.float32)
    percentile = float(config.get("percentile", 98.5))
    min_mach = config.get("min_mach", 0.7)
    threshold = float(np.percentile(grad[np.isfinite(grad)], percentile)) if np.any(np.isfinite(grad)) else 0.0
    label = (grad >= threshold).astype(np.float32)
    if min_mach is not None:
        label *= (mach >= float(min_mach)).astype(np.float32)
    score = robust_minmax(
        grad,
        lower_percentile=float(config.get("score_lower_percentile", 50.0)),
        upper_percentile=float(config.get("score_upper_percentile", 99.5)),
    )
    if min_mach is not None:
        score *= (mach >= float(min_mach)).astype(np.float32)
    return {
        "shock_label": label.astype(np.float32),
        "shock_score": score.astype(np.float32),
        "shock_grad_threshold": np.asarray(threshold, dtype=np.float32),
    }
