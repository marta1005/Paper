"""Per-case aggregation for symbolic shock sensors."""

from __future__ import annotations

import numpy as np

from shock_symbolic.metrics.classification import binary_classification_metrics


def per_case_metrics(
    case_id: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    metadata: dict[str, float] | None = None,
) -> dict[str, float | str]:
    """Compute per-case symbolic sensor metrics."""
    pred = np.asarray(scores) >= float(threshold)
    metrics = binary_classification_metrics(y_true, pred)
    row: dict[str, float | str] = {
        "case_id": case_id,
        "threshold": float(threshold),
        "predicted_shock_points": float(pred.sum()),
        "true_shock_points": float(np.asarray(y_true).astype(bool).sum()),
    }
    row.update(metrics)
    if metadata:
        row.update(metadata)
    return row


def aggregate_metric_rows(rows: list[dict[str, float | str]]) -> dict[str, float]:
    """Average scalar metric rows."""
    numeric_keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (float, int))})
    return {key: float(np.mean([float(row[key]) for row in rows if key in row])) for key in numeric_keys}
