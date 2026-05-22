from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cp_shock_project.symbolic.expression import SYMBOLIC_VARIABLES
from cp_shock_project.utils.io import save_json

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cp_shock_matplotlib"))


DEFAULT_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)
DEFAULT_PERCENTILES = (50, 75, 90, 95, 97, 99)


def shock_score_diagnostics(
    oracle_shock_score: np.ndarray,
    thresholds: tuple[float, ...] | list[float] = DEFAULT_THRESHOLDS,
    percentiles: tuple[float, ...] | list[float] = DEFAULT_PERCENTILES,
) -> dict:
    """Summarize target sparsity before symbolic regression."""
    score = np.asarray(oracle_shock_score, dtype=np.float64).reshape(-1)
    finite = np.isfinite(score)
    valid = score[finite]
    if valid.size == 0:
        raise ValueError("oracle_shock_score has no finite values")
    stats = {
        "n_total": int(score.size),
        "n_finite": int(valid.size),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid)),
        "percentiles": {str(p): float(np.percentile(valid, p)) for p in percentiles},
        "thresholds": {},
    }
    for threshold in thresholds:
        mask = valid >= float(threshold)
        n_shock = int(mask.sum())
        stats["thresholds"][str(threshold)] = {
            "threshold": float(threshold),
            "n_shock": n_shock,
            "n_nonshock": int(valid.size - n_shock),
            "shock_fraction": float(n_shock / max(valid.size, 1)),
            "nonshock_fraction": float(1.0 - n_shock / max(valid.size, 1)),
            "enough_for_symbolic_regression": bool(n_shock >= max(100, 0.01 * valid.size)),
        }
    return stats


def write_score_diagnostics(
    oracle_shock_score: np.ndarray,
    out_dir: str | Path,
    thresholds: tuple[float, ...] | list[float] = DEFAULT_THRESHOLDS,
    bins: int = 80,
) -> dict:
    """Write JSON/CSV/PNG diagnostics for the shock-score target."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    score = np.asarray(oracle_shock_score, dtype=np.float64).reshape(-1)
    finite = score[np.isfinite(score)]
    stats = shock_score_diagnostics(finite, thresholds=thresholds)
    save_json(stats, root / "oracle_shock_score_stats.json")
    pd.DataFrame(stats["thresholds"].values()).to_csv(root / "threshold_diagnostics.csv", index=False)
    hist, edges = np.histogram(finite, bins=bins, range=(0.0, 1.0))
    pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": hist}).to_csv(root / "oracle_shock_score_histogram.csv", index=False)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
        ax.hist(finite, bins=bins, range=(0.0, 1.0), color="#2f6c9e")
        ax.set_xlabel("oracle_shock_score")
        ax.set_ylabel("count")
        ax.set_yscale("log")
        ax.set_title("Oracle shock score distribution")
        fig.tight_layout()
        fig.savefig(root / "oracle_shock_score_histogram.png")
        plt.close(fig)
    except Exception:
        pass
    return stats


def balanced_sensor_dataframe(
    X: np.ndarray,
    oracle_shock_score: np.ndarray,
    case_ids: np.ndarray | None = None,
    max_samples: int = 100_000,
    shock_threshold: float = 0.5,
    shock_fraction: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a balanced tabular dataset for symbolic sensor training."""
    rng = np.random.default_rng(seed)
    score = np.asarray(oracle_shock_score).reshape(-1)
    shock_idx = np.flatnonzero(score >= shock_threshold)
    nonshock_idx = np.flatnonzero(score < shock_threshold)
    n_shock = min(len(shock_idx), int(max_samples * shock_fraction))
    n_non = min(len(nonshock_idx), max_samples - n_shock)
    chosen_parts: list[np.ndarray] = []
    if n_shock:
        chosen_parts.append(rng.choice(shock_idx, size=n_shock, replace=False))
    if n_non:
        chosen_parts.append(rng.choice(nonshock_idx, size=n_non, replace=False))
    if not chosen_parts:
        chosen = rng.choice(np.arange(len(score)), size=min(max_samples, len(score)), replace=False)
    else:
        chosen = np.concatenate(chosen_parts)
        rng.shuffle(chosen)
    df = pd.DataFrame(np.asarray(X[chosen, :9], dtype=np.float32), columns=SYMBOLIC_VARIABLES)
    df["oracle_shock_score"] = score[chosen].astype(np.float32)
    df["shock_label"] = (df["oracle_shock_score"] >= shock_threshold).astype(np.int8)
    if case_ids is not None:
        df["case_id"] = np.asarray(case_ids)[chosen]
    df.attrs["sampling_info"] = {
        "max_samples": int(max_samples),
        "shock_threshold": float(shock_threshold),
        "requested_shock_fraction": float(shock_fraction),
        "available_shock": int(len(shock_idx)),
        "available_nonshock": int(len(nonshock_idx)),
        "sampled_shock": int(np.sum(df["shock_label"].to_numpy() == 1)),
        "sampled_nonshock": int(np.sum(df["shock_label"].to_numpy() == 0)),
        "sampled_total": int(len(df)),
    }
    return df


def write_sensor_splits(df: pd.DataFrame, out_dir: str | Path, val_fraction: float = 0.2, seed: int = 42) -> tuple[Path, Path]:
    """Write train/validation parquet files for PySR."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = int(round(len(df) * val_fraction))
    val = df.iloc[idx[:n_val]].reset_index(drop=True)
    train = df.iloc[idx[n_val:]].reset_index(drop=True)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / "train_sensor.parquet"
    val_path = root / "val_sensor.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    return train_path, val_path
