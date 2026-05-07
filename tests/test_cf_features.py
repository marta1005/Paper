from __future__ import annotations

import numpy as np

from shock_symbolic.features.cf_features import compute_cf_directional_features


def test_cf_parallel_for_flat_surface() -> None:
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (3, 1))
    out = compute_cf_directional_features(
        np.array([1.0, -1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        normals,
        np.zeros(3, dtype=np.float32),
    )
    assert np.allclose(out["Cf_mag"][:2], 1.0, atol=1.0e-5)
    assert out["Cf_parallel"][0] > 0
    assert out["Cf_parallel"][1] < 0
    assert out["Cf_perp"][2] > 0.9
