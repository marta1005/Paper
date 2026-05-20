import numpy as np

from cp_shock_project.data.case_indexing import build_case_index, case_table
from cp_shock_project.data.splits import train_val_case_split

from tests.conftest import make_synthetic_arrays


def test_case_indexing_groups_complete_conditions():
    X, _ = make_synthetic_arrays(n_cases=4, points_per_case=8)
    index = build_case_index(X)
    assert index.n_cases == 4
    table = case_table(index)
    assert table["n_points"].tolist() == [8, 8, 8, 8]
    for cid in range(index.n_cases):
        idx = index.indices_for_case(cid)
        assert np.unique(X[idx, 6]).size == 1
        assert np.unique(X[idx, 7]).size == 1
        assert np.unique(X[idx, 8]).size == 1


def test_zero_validation_fraction_uses_all_cases_for_training():
    X, _ = make_synthetic_arrays(n_cases=4, points_per_case=8)
    index = build_case_index(X)
    train_cases, val_cases = train_val_case_split(index, val_fraction=0.0)
    assert train_cases.tolist() == [0, 1, 2, 3]
    assert len(val_cases) == 0
