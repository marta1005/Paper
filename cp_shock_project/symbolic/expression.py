from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cp_shock_project.utils.io import load_json

VARIABLES = ["x", "y", "z", "nx", "ny", "nz", "Mach", "AoA", "pi"]


def _safe_namespace(X: np.ndarray) -> dict[str, Any]:
    ns: dict[str, Any] = {name: X[:, i] for i, name in enumerate(VARIABLES)}
    ns.update(
        {
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": lambda v: np.log(np.abs(v) + 1e-12),
            "sqrt": lambda v: np.sqrt(np.abs(v)),
            "abs": np.abs,
            "tanh": np.tanh,
            "minimum": np.minimum,
            "maximum": np.maximum,
            "pow": np.power,
            "np": np,
        }
    )
    return ns


@dataclass(frozen=True)
class SymbolicExpression:
    """Safe NumPy evaluator for a PySR-exported expression."""

    expression: str
    clip_min: float | None = 0.0
    clip_max: float | None = 1.0
    scale: float = 1.0
    offset: float = 0.0

    @classmethod
    def from_json(cls, path: str | Path) -> "SymbolicExpression":
        data = load_json(path)
        expr = data.get("expression") or data.get("equation")
        if not expr:
            raise ValueError(f"No expression/equation field found in {path}")
        return cls(
            expression=str(expr),
            clip_min=data.get("clip_min", 0.0),
            clip_max=data.get("clip_max", 1.0),
            scale=float(data.get("scale", 1.0)),
            offset=float(data.get("offset", 0.0)),
        )

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < len(VARIABLES):
            raise ValueError(f"Expected at least {len(VARIABLES)} X columns, got {arr.shape[1]}")
        expr = self.expression.replace("^", "**")
        values = eval(expr, {"__builtins__": {}}, _safe_namespace(arr))
        out = np.asarray(values, dtype=np.float64)
        if out.shape == ():
            out = np.full(arr.shape[0], float(out), dtype=np.float64)
        out = self.scale * out + self.offset
        if self.clip_min is not None or self.clip_max is not None:
            out = np.clip(
                out,
                -np.inf if self.clip_min is None else self.clip_min,
                np.inf if self.clip_max is None else self.clip_max,
            )
        return out.astype(np.float32).reshape(-1)
