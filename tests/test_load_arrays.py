from __future__ import annotations

import numpy as np

from shock_symbolic.data.load_arrays import inspect_arrays, resolve_array_paths


def test_resolve_legacy_y_names(tmp_path) -> None:
    np.save(tmp_path / "X_train.npy", np.zeros((4, 9), dtype=np.float32))
    np.save(tmp_path / "X_test.npy", np.zeros((2, 9), dtype=np.float32))
    np.save(tmp_path / "Ytrain.npy", np.zeros((4, 4), dtype=np.float32))
    np.save(tmp_path / "Ytest.npy", np.zeros((2, 4), dtype=np.float32))
    paths = resolve_array_paths(tmp_path)
    assert paths.y_train.name == "Ytrain.npy"
    payload = inspect_arrays(tmp_path, sample_size=10)
    assert payload["X_train"]["shape"] == [4, 9]
    assert payload["Y_test"]["shape"] == [2, 4]
