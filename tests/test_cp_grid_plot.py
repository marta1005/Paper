from __future__ import annotations

import numpy as np

from shock_symbolic.visualization.cp_comparison import save_critical_cp_grid_2d
from shock_symbolic.visualization.cp_comparison import load_cp_prediction


def _snapshot(offset: float = 0.0) -> dict[str, np.ndarray]:
    x = np.linspace(0, 1, 16, dtype=np.float32)
    y = np.linspace(0, 1, 8, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    cp = np.sin(xx * np.pi).astype(np.float32) + offset
    n = cp.size
    return {
        "x": xx.reshape(-1),
        "y": yy.reshape(-1),
        "z": np.zeros(n, dtype=np.float32),
        "nz": np.ones(n, dtype=np.float32),
        "Cp": cp.reshape(-1),
    }


def test_save_critical_cp_grid_truth_only(tmp_path) -> None:
    cases = [
        {"case_id": "test_0000", "Mach": 0.8, "AoA": 2.0, "pi_scaled": 1.0},
        {"case_id": "test_0001", "Mach": 0.85, "AoA": 5.0, "pi_scaled": 1.0},
    ]
    out = tmp_path / "grid.png"
    payload = save_critical_cp_grid_2d(cases, [_snapshot(), _snapshot(0.1)], out, max_points_per_case=None)
    assert out.exists()
    assert payload["has_predictions"] is False


def test_save_critical_cp_grid_with_predictions(tmp_path) -> None:
    snap = _snapshot()
    cases = [{"case_id": "test_0000", "Mach": 0.8, "AoA": 2.0, "pi_scaled": 1.0}]
    pred = snap["Cp"] + 0.1
    out = tmp_path / "grid_pred.png"
    payload = save_critical_cp_grid_2d(cases, [snap], out, cp_predictions=[pred], max_points_per_case=None)
    assert out.exists()
    assert payload["has_predictions"] is True
    assert payload["cases"][0]["mae"] > 0.0


def test_load_cp_prediction_from_case_keyed_npz(tmp_path) -> None:
    path = tmp_path / "predictions.npz"
    expected = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    np.savez_compressed(path, test_0003=expected)
    loaded = load_cp_prediction(path, case_start=24, case_stop=32, n_case_points=8, case_id="test_0003")
    assert np.allclose(loaded, expected)
