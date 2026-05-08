from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CASE_COLUMNS = (6, 7, 8)
CASE_NAMES = ("Mach", "AoA", "pi")


@dataclass(frozen=True)
class CaseIndex:
    """Case grouping by complete aerodynamic condition."""

    case_ids: np.ndarray
    unique_conditions: np.ndarray
    sorted_indices: np.ndarray
    offsets: np.ndarray

    def indices_for_case(self, case_id: int) -> np.ndarray:
        start = int(self.offsets[case_id])
        end = int(self.offsets[case_id + 1])
        return self.sorted_indices[start:end]

    @property
    def n_cases(self) -> int:
        return int(self.unique_conditions.shape[0])


def case_keys(X: np.ndarray) -> np.ndarray:
    """Extract Mach/AoA/pi case keys."""
    if X.shape[1] <= max(CASE_COLUMNS):
        raise ValueError("X must include Mach, AoA, pi columns at indices 6, 7, 8")
    return np.asarray(X[:, CASE_COLUMNS])


def build_case_index(X: np.ndarray, decimals: int | None = 12) -> CaseIndex:
    """Group points by unique Mach/AoA/pi without mixing conditions."""
    keys = case_keys(X)
    if decimals is not None:
        keys = np.round(keys.astype(np.float64), decimals=decimals)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=unique.shape[0])
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return CaseIndex(
        case_ids=inverse.astype(np.int64),
        unique_conditions=unique.astype(np.float64),
        sorted_indices=order.astype(np.int64),
        offsets=offsets,
    )


def case_table(index: CaseIndex) -> pd.DataFrame:
    """Return one row per CFD condition."""
    rows = []
    for cid, cond in enumerate(index.unique_conditions):
        rows.append(
            {
                "case_id": cid,
                "Mach": cond[0],
                "AoA": cond[1],
                "pi": cond[2],
                "n_points": int(index.offsets[cid + 1] - index.offsets[cid]),
            }
        )
    return pd.DataFrame(rows)


def save_case_index(index: CaseIndex, out_dir: str | Path, prefix: str) -> None:
    """Save case metadata and point-to-case arrays."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    case_table(index).to_csv(root / f"{prefix}_cases.csv", index=False)
    np.savez_compressed(
        root / f"{prefix}_indices.npz",
        case_ids=index.case_ids,
        unique_conditions=index.unique_conditions,
        sorted_indices=index.sorted_indices,
        offsets=index.offsets,
    )


def load_case_index(path: str | Path) -> CaseIndex:
    """Load a saved CaseIndex npz file."""
    data = np.load(path)
    return CaseIndex(
        case_ids=data["case_ids"],
        unique_conditions=data["unique_conditions"],
        sorted_indices=data["sorted_indices"],
        offsets=data["offsets"],
    )
