from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

X_COLUMNS = ["x", "y", "z", "nx", "ny", "nz", "Mach", "AoA", "pi"]
Y_COLUMNS = ["Cp", "Cfx", "Cfy", "Cfz"]


@dataclass(frozen=True)
class ArrayBundle:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray


def load_array(path: str | Path, mmap: bool = True) -> np.ndarray:
    """Load a NumPy array, using mmap by default for large CFD arrays."""
    return np.load(Path(path), mmap_mode="r" if mmap else None)


def load_arrays(data_dir: str | Path = "data", mmap: bool = True) -> ArrayBundle:
    """Load train/test arrays using the filenames from the challenge data."""
    root = Path(data_dir)
    return ArrayBundle(
        X_train=load_array(root / "X_train.npy", mmap=mmap),
        Y_train=load_array(root / "Ytrain.npy", mmap=mmap),
        X_test=load_array(root / "X_test.npy", mmap=mmap),
        Y_test=load_array(root / "Ytest.npy", mmap=mmap),
    )


def validate_shapes(X: np.ndarray, Y: np.ndarray) -> None:
    """Validate expected ONERA CRM WBPN pointwise schema."""
    if X.ndim != 2 or X.shape[1] < 9:
        raise ValueError(f"X must have shape (N, >=9), got {X.shape}")
    if Y.ndim != 2 or Y.shape[1] < 4:
        raise ValueError(f"Y must have shape (N, >=4), got {Y.shape}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y row counts differ: {X.shape[0]} != {Y.shape[0]}")


def inspect_array(name: str, arr: np.ndarray, columns: list[str] | None = None) -> dict:
    """Return a compact JSON-serializable inspection summary."""
    summary: dict[str, object] = {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "nan_count": int(np.isnan(arr).sum()),
    }
    if arr.ndim == 2:
        cols = columns or [f"col_{i}" for i in range(arr.shape[1])]
        stats: dict[str, Mapping[str, float]] = {}
        for i, col in enumerate(cols[: arr.shape[1]]):
            values = arr[:, i]
            stats[col] = {"min": float(np.nanmin(values)), "max": float(np.nanmax(values))}
        summary["columns"] = stats
    return summary
