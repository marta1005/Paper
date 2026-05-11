from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cp_shock_project.symbolic.expression import FEATURE_COLUMNS, SYMBOLIC_VARIABLES, SymbolicExpression
from cp_shock_project.utils.io import save_json


def _load_sensor_tables(train_parquet: str | Path, val_parquet: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = pd.read_parquet(train_parquet).rename(columns={"pi": "pi_param"})
    val = pd.read_parquet(val_parquet).rename(columns={"pi": "pi_param"})
    missing = [c for c in SYMBOLIC_VARIABLES if c not in train.columns or c not in val.columns]
    if missing:
        raise ValueError(f"Missing symbolic sensor columns: {missing}")
    X_train = train[SYMBOLIC_VARIABLES].to_numpy(dtype=np.float64)
    y_train = train["oracle_shock_score"].to_numpy(dtype=np.float64)
    X_val = val[SYMBOLIC_VARIABLES].to_numpy(dtype=np.float64)
    y_val = val["oracle_shock_score"].to_numpy(dtype=np.float64)
    return X_train, y_train, X_val, y_val


def train_gplearn_sensor(
    train_parquet: str | Path,
    val_parquet: str | Path,
    out_dir: str | Path,
    gplearn_kwargs: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Train a deployable symbolic shock sensor with gplearn."""
    try:
        from gplearn.genetic import SymbolicRegressor
    except Exception as exc:
        raise RuntimeError(
            "gplearn is not installed. Install it with `python -m pip install gplearn`, "
            "then rerun scripts/05_train_symbolic_sensor.py."
        ) from exc

    X_train, y_train, X_val, y_val = _load_sensor_tables(train_parquet, val_parquet)
    kwargs: dict[str, Any] = {
        "population_size": 1000,
        "generations": 20,
        "tournament_size": 20,
        "stopping_criteria": 0.0,
        "const_range": (-1.0, 1.0),
        "init_depth": (2, 5),
        "init_method": "half and half",
        "function_set": ("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "inv", "max", "min", "sin", "cos"),
        "metric": "mean absolute error",
        "parsimony_coefficient": 0.001,
        "p_crossover": 0.9,
        "p_subtree_mutation": 0.01,
        "p_hoist_mutation": 0.01,
        "p_point_mutation": 0.01,
        "max_samples": 0.9,
        "feature_names": SYMBOLIC_VARIABLES,
        "n_jobs": 1,
        "verbose": 1,
        "random_state": 42,
    }
    kwargs.update(gplearn_kwargs or {})
    if isinstance(kwargs.get("function_set"), list):
        kwargs["function_set"] = tuple(kwargs["function_set"])
    if isinstance(kwargs.get("init_depth"), list):
        kwargs["init_depth"] = tuple(kwargs["init_depth"])
    if isinstance(kwargs.get("const_range"), list):
        kwargs["const_range"] = tuple(kwargs["const_range"])

    model = SymbolicRegressor(**kwargs)
    model.fit(X_train, y_train)

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    expression = str(model._program)
    pred = np.clip(model.predict(X_val), 0.0, 1.0)
    val_mae = float(np.mean(np.abs(pred - y_val)))
    val_rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
    program = model._program
    hall = pd.DataFrame(
        [
            {
                "backend": "gplearn",
                "expression": expression,
                "fitness": float(getattr(program, "fitness_", np.nan)),
                "raw_fitness": float(getattr(program, "raw_fitness_", np.nan)),
                "depth": int(program.depth_),
                "length": int(program.length_),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }
        ]
    )
    hall.to_csv(root / "equations.csv", index=False)
    hall.to_csv(root / "hall_of_fame.csv", index=False)
    (root / "best_equation.txt").write_text(expression + "\n", encoding="utf-8")
    (root / "best_equation.tex").write_text(expression + "\n", encoding="utf-8")
    save_json(
        {
            "backend": "gplearn",
            "expression": expression,
            "variables": SYMBOLIC_VARIABLES,
            "feature_columns": FEATURE_COLUMNS,
            "clip_min": 0.0,
            "clip_max": 1.0,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
        },
        root / "best_sensor.json",
    )
    save_json({"threshold": 0.5, "target": "oracle_shock_score"}, root / "threshold.json")
    return {
        "equations": root / "equations.csv",
        "hall_of_fame": root / "hall_of_fame.csv",
        "best_equation": root / "best_equation.txt",
        "best_sensor": root / "best_sensor.json",
        "threshold": root / "threshold.json",
    }


def evaluate_gplearn_sensor(sensor_json: str | Path, X: np.ndarray) -> np.ndarray:
    return SymbolicExpression.from_json(sensor_json).evaluate(X)
