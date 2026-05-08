from __future__ import annotations

import numpy as np

from cp_shock_project.data.case_indexing import CaseIndex


def train_val_case_split(
    index: CaseIndex, val_fraction: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Split complete CFD cases, never individual points."""
    rng = np.random.default_rng(seed)
    case_ids = np.arange(index.n_cases)
    rng.shuffle(case_ids)
    n_val = max(1, int(round(index.n_cases * val_fraction))) if index.n_cases > 1 else 0
    val_cases = np.sort(case_ids[:n_val])
    train_cases = np.sort(case_ids[n_val:])
    return train_cases, val_cases


def point_indices_for_cases(index: CaseIndex, case_ids: np.ndarray) -> np.ndarray:
    """Collect point indices for a list of complete cases."""
    if len(case_ids) == 0:
        return np.array([], dtype=np.int64)
    parts = [index.indices_for_case(int(cid)) for cid in case_ids]
    return np.concatenate(parts).astype(np.int64)
