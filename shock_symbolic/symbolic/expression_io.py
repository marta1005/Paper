"""Save, load and evaluate symbolic sensor expressions."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

SAFE_NAMESPACE = {
    "np": np,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": lambda x: np.log(np.maximum(x, 1.0e-12)),
    "sqrt": lambda x: np.sqrt(np.maximum(x, 0.0)),
    "abs": np.abs,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "pow": np.power,
    "pi": math.pi,
}


def sanitize_name(name: str) -> str:
    """Convert a feature name into a valid Python variable name."""
    clean = re.sub(r"\W+", "_", name)
    if clean and clean[0].isdigit():
        clean = f"f_{clean}"
    return clean


def evaluate_expression(expression: str, features: dict[str, np.ndarray], feature_names: list[str]) -> np.ndarray:
    """Evaluate a symbolic expression using feature names and x0/x1 aliases."""
    expression = expression.replace("^", "**")
    namespace: dict[str, Any] = dict(SAFE_NAMESPACE)
    for idx, name in enumerate(feature_names):
        arr = np.asarray(features[name], dtype=np.float64)
        namespace[f"x{idx}"] = arr
        namespace[sanitize_name(name)] = arr
    try:
        scores = eval(expression, {"__builtins__": {}}, namespace)
    except Exception as exc:
        raise RuntimeError(f"Could not evaluate symbolic expression: {expression}") from exc
    return np.asarray(scores, dtype=np.float64)


def save_sensor_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save symbolic sensor metadata as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_sensor_json(path: str | Path) -> dict[str, Any]:
    """Load symbolic sensor metadata."""
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload
