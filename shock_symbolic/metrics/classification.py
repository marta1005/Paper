"""Binary classification metrics for symbolic shock masks."""

from __future__ import annotations

import numpy as np


def binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Compute precision, recall, F1 and IoU for binary arrays."""
    true = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_pred).astype(bool)
    tp = float(np.logical_and(true, pred).sum())
    fp = float(np.logical_and(~true, pred).sum())
    fn = float(np.logical_and(true, ~pred).sum())
    tn = float(np.logical_and(~true, ~pred).sum())
    eps = 1.0e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}precision": precision,
        f"{p}recall": recall,
        f"{p}f1": f1,
        f"{p}iou": iou,
        f"{p}tp": tp,
        f"{p}fp": fp,
        f"{p}fn": fn,
        f"{p}tn": tn,
    }
