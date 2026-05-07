from __future__ import annotations

import numpy as np

from shock_symbolic.features.grid2d_features import compute_snapshot_features_2d


def test_compute_snapshot_features_2d_flattened_valid_cells() -> None:
    x = np.linspace(0.0, 1.0, 20, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    n = xx.size
    snapshot = {
        "point_id": np.arange(n, dtype=np.int64),
        "x": xx.reshape(-1),
        "y": yy.reshape(-1),
        "z": np.zeros(n, dtype=np.float32),
        "nx": np.zeros(n, dtype=np.float32),
        "ny": np.zeros(n, dtype=np.float32),
        "nz": np.ones(n, dtype=np.float32),
        "Mach": np.full(n, 0.8, dtype=np.float32),
        "AoA": np.full(n, 2.0, dtype=np.float32),
        "pi_scaled": np.ones(n, dtype=np.float32),
        "Cp": xx.reshape(-1).astype(np.float32),
        "Cfx": np.ones(n, dtype=np.float32),
        "Cfy": np.zeros(n, dtype=np.float32),
        "Cfz": np.zeros(n, dtype=np.float32),
    }
    features = compute_snapshot_features_2d(snapshot, {"surface": "upper", "x_bins": 20, "y_bins": 10})
    assert features["Cp"].shape[0] == 200
    assert features["valid_mask_2d"].shape == (10, 20)
    assert float(features["grad_Cp_mag"].mean()) > 0.0
    assert "grid2d" == str(features["projection_mode"])
