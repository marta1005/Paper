import numpy as np

from cp_shock_project.symbolic.build_sensor_dataset import balanced_sensor_dataframe
from cp_shock_project.symbolic.build_sensor_dataset import shock_score_diagnostics
from cp_shock_project.symbolic.gplearn_sensor import _degeneracy_report, _regression_metrics
from cp_shock_project.symbolic.expression import SymbolicExpression

from tests.conftest import make_synthetic_arrays


def test_symbolic_expression_uses_pi_param_alias():
    X, _ = make_synthetic_arrays(n_cases=1, points_per_case=5)
    expr = SymbolicExpression("add(x, mul(0.0, pi_param))", clip_min=None, clip_max=None)
    values = expr.evaluate(X)
    assert np.allclose(values, X[:, 0])


def test_symbolic_expression_supports_gplearn_protected_div():
    X, _ = make_synthetic_arrays(n_cases=1, points_per_case=5)
    expr = SymbolicExpression("div(1.0, add(abs(x), 1.0))", clip_min=None, clip_max=None)
    values = expr.evaluate(X)
    assert np.all(np.isfinite(values))


def test_sensor_dataframe_avoids_reserved_pi_name():
    X, _ = make_synthetic_arrays(n_cases=1, points_per_case=5)
    df = balanced_sensor_dataframe(X, np.linspace(0, 1, X.shape[0]), max_samples=5)
    assert "pi_param" in df.columns
    assert "pi" not in df.columns


def test_shock_score_diagnostics_reports_threshold_counts():
    score = np.array([0.0, 0.1, 0.3, 0.8, 1.0])
    stats = shock_score_diagnostics(score, thresholds=[0.2, 0.7])
    assert stats["thresholds"]["0.2"]["n_shock"] == 3
    assert stats["thresholds"]["0.7"]["n_shock"] == 2


def test_degeneracy_report_flags_constant_zero_sensor():
    y_true = np.array([0.0, 0.0, 0.8, 1.0])
    y_pred = np.zeros_like(y_true)
    metrics = _regression_metrics(y_true, y_pred, threshold=0.3)
    report = _degeneracy_report("div(sub(ny, ny), x)", y_pred, metrics, 1e-4, 1e-3, 0.05)
    assert report["is_degenerate"]
    assert "expression_contains_self_subtraction" in report["reasons"]
