import numpy as np

from cp_shock_project.metrics.per_case import per_case_metrics
from cp_shock_project.metrics.regression import regression_metrics
from cp_shock_project.metrics.sensor_metrics import sensor_metrics
from cp_shock_project.metrics.shock_metrics import shock_region_metrics


def test_metrics_keys():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.1, 1.9, 3.1, 3.8])
    case = np.array([0, 0, 1, 1])
    score = np.array([0.0, 1.0, 0.2, 0.9])
    assert "R2_Cp" in regression_metrics(y, p)
    _, summary = per_case_metrics(y, p, case)
    assert "wrMAE_Cp" in summary
    assert "shock_error_ratio" in shock_region_metrics(y, p, score)
    assert "F1" in sensor_metrics(score, score)
