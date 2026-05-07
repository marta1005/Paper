from __future__ import annotations

import numpy as np

from shock_symbolic.labels.shock_labels import generate_shock_labels


def test_shock_labels_percentile_and_mach_gate() -> None:
    features = {
        "grad_Cp_mag": np.array([0.0, 1.0, 2.0, 100.0], dtype=np.float32),
        "Mach": np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float32),
    }
    labels = generate_shock_labels(features, {"percentile": 90.0, "min_mach": 0.7})
    assert labels["shock_label"].sum() == 1.0
    assert labels["shock_score"][-1] == 1.0
    features["Mach"][:] = 0.5
    labels = generate_shock_labels(features, {"percentile": 90.0, "min_mach": 0.7})
    assert labels["shock_label"].sum() == 0.0
