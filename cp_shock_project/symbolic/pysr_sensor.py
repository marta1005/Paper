from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cp_shock_project.symbolic.expression import VARIABLES, SymbolicExpression
from cp_shock_project.utils.io import save_json


def train_pysr_sensor(
    train_parquet: str | Path,
    val_parquet: str | Path,
    out_dir: str | Path,
    pysr_kwargs: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Train a deployable symbolic shock sensor with PySR."""
    try:
        from pysr import PySRRegressor
    except Exception as exc:
        raise RuntimeError(
            "PySR is not available in this environment. Install/configure PySR and Julia, "
            "or run the rest of the pipeline with oracle_gated_residual or dummy_sensor mode."
        ) from exc

    train = pd.read_parquet(train_parquet)
    val = pd.read_parquet(val_parquet)
    X_train = train[VARIABLES].to_numpy(dtype=np.float64)
    y_train = train["oracle_shock_score"].to_numpy(dtype=np.float64)
    X_val = val[VARIABLES].to_numpy(dtype=np.float64)
    y_val = val["oracle_shock_score"].to_numpy(dtype=np.float64)
    kwargs = {
        "niterations": 40,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["sin", "cos", "exp", "log", "sqrt", "tanh"],
        "model_selection": "best",
        "verbosity": 1,
    }
    kwargs.update(pysr_kwargs or {})
    model = PySRRegressor(**kwargs)
    model.fit(X_train, y_train, variable_names=VARIABLES)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    equations = model.equations_
    equations.to_csv(root / "equations.csv", index=False)
    best_expr = str(model.sympy())
    pred = np.clip(model.predict(X_val), 0.0, 1.0)
    val_mae = float(np.mean(np.abs(pred - y_val)))
    (root / "best_equation.txt").write_text(best_expr + "\n", encoding="utf-8")
    try:
        latex = str(model.latex())
    except Exception:
        latex = best_expr
    (root / "best_equation.tex").write_text(latex + "\n", encoding="utf-8")
    save_json(
        {
            "expression": best_expr,
            "variables": VARIABLES,
            "clip_min": 0.0,
            "clip_max": 1.0,
            "val_mae": val_mae,
        },
        root / "best_sensor.json",
    )
    save_json({"threshold": 0.5, "target": "oracle_shock_score"}, root / "threshold.json")
    return {
        "equations": root / "equations.csv",
        "best_equation": root / "best_equation.txt",
        "best_sensor": root / "best_sensor.json",
        "threshold": root / "threshold.json",
    }


def evaluate_symbolic_sensor(sensor_json: str | Path, X: np.ndarray) -> np.ndarray:
    return SymbolicExpression.from_json(sensor_json).evaluate(X)
