from __future__ import annotations

import numpy as np

from shock_symbolic.symbolic.build_table import build_symbolic_table
from shock_symbolic.utils.io import load_table


def test_symbolic_table_balanced_sampling(tmp_path) -> None:
    features = {
        "case_id": np.asarray("train_0000"),
        "point_id": np.arange(10),
        "Cp": np.linspace(0, 1, 10, dtype=np.float32),
        "Cf_mag": np.linspace(1, 2, 10, dtype=np.float32),
        "Mach": np.full(10, 0.8, dtype=np.float32),
        "AoA": np.full(10, 2.0, dtype=np.float32),
        "pi_scaled": np.ones(10, dtype=np.float32),
    }
    labels = {
        "shock_label": np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "shock_score": np.linspace(1, 0, 10, dtype=np.float32),
    }
    f_path = tmp_path / "case.npz"
    l_path = tmp_path / "case_labels.npz"
    np.savez_compressed(f_path, **features)
    np.savez_compressed(l_path, **labels)
    table_path = build_symbolic_table(
        [f_path],
        [l_path],
        tmp_path / "table",
        feature_names=["Cp", "Cf_mag"],
        negative_ratio=2.0,
        seed=3,
    )
    table = load_table(table_path)
    assert len(table["Cp"]) == 6
    assert np.asarray(table["shock_label"]).sum() == 2
