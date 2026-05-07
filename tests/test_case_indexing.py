from __future__ import annotations

import numpy as np

from shock_symbolic.data.case_indexing import build_case_index_for_split


def test_case_indexing_contiguous_conditions() -> None:
    x = np.zeros((10, 9), dtype=np.float32)
    x[:4, 6:9] = [0.8, 2.0, 1.0]
    x[4:, 6:9] = [0.85, 4.0, 1.0]
    cases = build_case_index_for_split(x, "train", batch_size=3)
    assert len(cases) == 2
    assert cases[0]["start"] == 0
    assert cases[0]["stop"] == 4
    assert cases[1]["n_points"] == 6
