import numpy as np

from cp_shock_project.symbolic.build_sensor_dataset import balanced_sensor_dataframe
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
