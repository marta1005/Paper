"""Small diagnostics helpers for symbolic shock sensor outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shock_symbolic.visualization.scatter_plots import save_case_scatter_suite


def save_prediction_diagnostics(
    output_dir: str | Path,
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    symbolic_score: np.ndarray,
    threshold: float,
    max_points: int = 80_000,
) -> None:
    """Save standard diagnostic plots including symbolic score and mask."""
    save_case_scatter_suite(
        output_dir,
        features,
        labels=labels,
        scores={
            "symbolic_score": np.asarray(symbolic_score),
            "symbolic_mask": (np.asarray(symbolic_score) >= float(threshold)).astype(np.float32),
        },
        max_points=max_points,
    )
