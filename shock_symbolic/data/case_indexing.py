"""Identify CFD conditions and contiguous case snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from shock_symbolic.data.load_arrays import load_memmap_arrays, resolve_array_paths
from shock_symbolic.utils.io import load_csv_rows, save_csv

CONDITION_COLUMNS = (6, 7, 8)


def _same_condition(a: np.ndarray, b: np.ndarray, atol: float) -> bool:
    return bool(np.allclose(a, b, rtol=0.0, atol=atol))


def build_case_index_for_split(
    x_array: np.ndarray,
    split: str,
    atol: float = 1.0e-7,
    batch_size: int = 1_000_000,
    max_cases: int | None = None,
) -> list[dict[str, Any]]:
    """Build a case index by grouping contiguous rows with equal flow condition.

    kNN features must never mix conditions, so every row range returned here is
    a complete CFD snapshot under one `(Mach, AoA, pi)` tuple.
    """
    n_rows = int(x_array.shape[0])
    if n_rows == 0:
        return []

    change_rows: list[int] = []
    previous_last = np.asarray(x_array[0, CONDITION_COLUMNS], dtype=np.float64)

    for batch_start in range(0, n_rows, batch_size):
        batch_stop = min(n_rows, batch_start + batch_size)
        cond_batch = np.asarray(x_array[batch_start:batch_stop, CONDITION_COLUMNS], dtype=np.float64)
        if batch_start > 0 and not _same_condition(cond_batch[0], previous_last, atol):
            change_rows.append(batch_start)
        if cond_batch.shape[0] > 1:
            changes = np.any(np.abs(cond_batch[1:] - cond_batch[:-1]) > atol, axis=1)
            local_changes = np.flatnonzero(changes) + 1
            change_rows.extend((batch_start + local_changes).astype(int).tolist())
        previous_last = cond_batch[-1]
        if max_cases is not None and len(change_rows) >= max_cases:
            break

    boundaries = [0] + change_rows + [n_rows]
    cases: list[dict[str, Any]] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop <= start:
            continue
        current = np.asarray(x_array[start, CONDITION_COLUMNS], dtype=np.float64)
        cases.append(
            {
                "case_id": f"{split}_{len(cases):04d}",
                "split": split,
                "start": int(start),
                "stop": int(stop),
                "n_points": int(stop - start),
                "Mach": float(current[0]),
                "AoA": float(current[1]),
                "pi_scaled": float(current[2]),
            }
        )
        if max_cases is not None and len(cases) >= max_cases:
            break
    return cases[:max_cases] if max_cases is not None else cases


def build_case_index(
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs/symbolic/case_index",
    atol: float = 1.0e-7,
    batch_size: int = 1_000_000,
    max_cases_per_split: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build and save train/test case indices."""
    arrays = load_memmap_arrays(resolve_array_paths(data_dir))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "train": build_case_index_for_split(arrays["X_train"], "train", atol=atol, batch_size=batch_size, max_cases=max_cases_per_split),
        "test": build_case_index_for_split(arrays["X_test"], "test", atol=atol, batch_size=batch_size, max_cases=max_cases_per_split),
    }
    save_csv(output / "case_index_train.csv", result["train"])
    save_csv(output / "case_index_test.csv", result["test"])
    return result


def load_case_index(path: str | Path) -> list[dict[str, Any]]:
    """Load a case-index CSV with typed numeric fields."""
    rows = load_csv_rows(path)
    typed: list[dict[str, Any]] = []
    for row in rows:
        typed.append(
            {
                "case_id": row["case_id"],
                "split": row["split"],
                "start": int(row["start"]),
                "stop": int(row["stop"]),
                "n_points": int(row["n_points"]),
                "Mach": float(row["Mach"]),
                "AoA": float(row["AoA"]),
                "pi_scaled": float(row["pi_scaled"]),
            }
        )
    return typed
