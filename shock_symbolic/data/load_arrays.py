"""Memory-efficient loading and inspection of ONERA CRM WBPN arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


X_COLUMNS = {
    0: "x",
    1: "y",
    2: "z",
    3: "nx",
    4: "ny",
    5: "nz",
    6: "Mach",
    7: "AoA",
    8: "pi_scaled",
}

Y_COLUMNS = {
    0: "Cp",
    1: "Cfx",
    2: "Cfy",
    3: "Cfz",
}


@dataclass(frozen=True)
class ONERAArrayPaths:
    """Resolved train/test array paths for the ONERA dataset."""

    root: Path
    x_train: Path
    x_test: Path
    y_train: Path
    y_test: Path


def resolve_array_paths(data_dir: str | Path = "data") -> ONERAArrayPaths:
    """Resolve expected array names, accepting legacy `Ytrain/Ytest` aliases."""
    root = Path(data_dir)
    y_train = root / "Y_train.npy"
    y_test = root / "Y_test.npy"
    if not y_train.exists() and (root / "Ytrain.npy").exists():
        y_train = root / "Ytrain.npy"
    if not y_test.exists() and (root / "Ytest.npy").exists():
        y_test = root / "Ytest.npy"
    paths = ONERAArrayPaths(
        root=root,
        x_train=root / "X_train.npy",
        x_test=root / "X_test.npy",
        y_train=y_train,
        y_test=y_test,
    )
    missing = [str(path) for path in [paths.x_train, paths.x_test, paths.y_train, paths.y_test] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing ONERA array files: {missing}")
    return paths


def load_memmap_arrays(paths: ONERAArrayPaths) -> dict[str, np.ndarray]:
    """Open train/test arrays in mmap read-only mode."""
    return {
        "X_train": np.load(paths.x_train, mmap_mode="r"),
        "X_test": np.load(paths.x_test, mmap_mode="r"),
        "Y_train": np.load(paths.y_train, mmap_mode="r"),
        "Y_test": np.load(paths.y_test, mmap_mode="r"),
    }


def _column_stats(array: np.ndarray, column: int, sample_size: int = 250_000) -> dict[str, Any]:
    n_rows = int(array.shape[0])
    if n_rows <= sample_size:
        values = np.asarray(array[:, column], dtype=np.float64)
    else:
        idx = np.linspace(0, n_rows - 1, sample_size, dtype=np.int64)
        values = np.asarray(array[idx, column], dtype=np.float64)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size == 0:
        return {"nan_count_sample": int((~finite).sum()), "min": None, "max": None, "mean": None, "std": None}
    return {
        "nan_count_sample": int((~finite).sum()),
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
    }


def inspect_arrays(data_dir: str | Path = "data", sample_size: int = 250_000) -> dict[str, Any]:
    """Inspect shapes, dtypes, NaNs and sampled ranges for the ONERA arrays."""
    paths = resolve_array_paths(data_dir)
    arrays = load_memmap_arrays(paths)
    payload: dict[str, Any] = {"data_dir": str(paths.root), "paths": {k: str(v) for k, v in paths.__dict__.items() if k != "root"}}
    for name, array in arrays.items():
        column_names = X_COLUMNS if name.startswith("X") else Y_COLUMNS
        payload[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "columns": {
                column_names[col]: _column_stats(array, col, sample_size=sample_size)
                for col in range(min(array.shape[1], len(column_names)))
            },
        }
    if arrays["X_train"].shape[0] != arrays["Y_train"].shape[0]:
        payload["train_row_mismatch"] = [int(arrays["X_train"].shape[0]), int(arrays["Y_train"].shape[0])]
    if arrays["X_test"].shape[0] != arrays["Y_test"].shape[0]:
        payload["test_row_mismatch"] = [int(arrays["X_test"].shape[0]), int(arrays["Y_test"].shape[0])]
    return payload
