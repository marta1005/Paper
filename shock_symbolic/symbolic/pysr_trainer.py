"""PySR training wrapper for the symbolic shock sensor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from shock_symbolic.symbolic.expression_io import save_sensor_json
from shock_symbolic.symbolic.threshold import calibrate_threshold
from shock_symbolic.utils.io import load_table


def train_pysr_sensor(
    table_path: str | Path,
    output_dir: str | Path,
    feature_names: list[str],
    target_name: str = "shock_score",
    pysr_config: dict[str, Any] | None = None,
    threshold_metric: str = "f1",
) -> dict[str, Any]:
    """Train PySR and export the best symbolic shock sensor.

    PySR is imported only here. If it is missing, the function raises a clear
    RuntimeError so the rest of the pipeline remains usable.
    """
    try:
        from pysr import PySRRegressor  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PySR is not installed. Install it with `pip install pysr` and ensure Julia/PySR "
            "can initialize, then rerun scripts/05_train_symbolic_sensor.py."
        ) from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = load_table(table_path)
    x = np.column_stack([np.asarray(table[name], dtype=np.float64) for name in feature_names])
    y = np.asarray(table[target_name], dtype=np.float64)
    cfg = dict(pysr_config or {})
    model = PySRRegressor(**cfg)
    model.fit(x, y, variable_names=feature_names)

    equations = model.equations_
    equations_path = output / "equations.csv"
    equations.to_csv(equations_path, index=False)
    best = model.get_best()
    expression = str(best.get("equation", ""))
    latex = ""
    try:
        latex = str(model.latex())
    except Exception:
        latex = expression

    (output / "best_equation.txt").write_text(expression, encoding="utf-8")
    (output / "best_equation.tex").write_text(latex, encoding="utf-8")
    scores = model.predict(x)
    threshold = calibrate_threshold(scores, np.asarray(table["shock_label"], dtype=np.float64), metric=threshold_metric)
    save_sensor_json(output / "threshold.json", threshold)
    sensor = {
        "expression": expression,
        "latex": latex,
        "feature_names": feature_names,
        "target_name": target_name,
        "threshold": threshold["threshold"],
        "threshold_metric": threshold_metric,
        "equations_path": str(equations_path),
    }
    save_sensor_json(output / "best_sensor.json", sensor)
    return sensor
