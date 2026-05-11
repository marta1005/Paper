from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cp_shock_project.symbolic.expression import SYMBOLIC_VARIABLES


def balanced_sensor_dataframe(
    X: np.ndarray,
    oracle_shock_score: np.ndarray,
    case_ids: np.ndarray | None = None,
    max_samples: int = 100_000,
    shock_threshold: float = 0.5,
    shock_fraction: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a balanced tabular dataset for PySR sensor training."""
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
    if case_ids is not None:
        df["case_id"] = np.asarray(case_ids)[chosen]
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
