from __future__ import annotations

from pathlib import Path

import numpy as np


def make_synthetic_arrays(n_cases: int = 3, points_per_case: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_parts = []
    Y_parts = []
    for c in range(n_cases):
        mach = 0.72 + 0.03 * c
        aoa = 1.0 + 0.5 * c
        pi_param = 0.2 + 0.1 * c
        x = np.linspace(0.0, 1.0, points_per_case)
        y = 0.1 * c + 0.05 * rng.normal(size=points_per_case)
        z = 0.02 * np.sin(2 * np.pi * x)
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (points_per_case, 1))
        shock_x = 0.45 + 0.04 * c
        cp = -0.25 * x + 0.05 * aoa - 0.65 * (x > shock_x).astype(float)
        cp += 0.02 * rng.normal(size=points_per_case)
        cf = 0.01 + 0.002 * rng.normal(size=(points_per_case, 3))
        X_case = np.column_stack(
            [
                x,
                y,
                z,
                normals,
                np.full(points_per_case, mach),
                np.full(points_per_case, aoa),
                np.full(points_per_case, pi_param),
            ]
        )
        Y_case = np.column_stack([cp, cf])
        X_parts.append(X_case.astype(np.float32))
        Y_parts.append(Y_case.astype(np.float32))
    return np.vstack(X_parts), np.vstack(Y_parts)


def write_synthetic_data(root: Path):
    data_dir = root / "data"
    data_dir.mkdir()
    X, Y = make_synthetic_arrays()
    X_test, Y_test = make_synthetic_arrays(n_cases=2, points_per_case=30, seed=10)
    np.save(data_dir / "X_train.npy", X)
    np.save(data_dir / "Ytrain.npy", Y)
    np.save(data_dir / "X_test.npy", X_test)
    np.save(data_dir / "Ytest.npy", Y_test)
    return data_dir, X, Y, X_test, Y_test
