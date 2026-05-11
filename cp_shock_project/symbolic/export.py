from __future__ import annotations

from pathlib import Path

from cp_shock_project.symbolic.expression import SymbolicExpression
from cp_shock_project.utils.io import save_json


def export_sensor(sensor_json: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Export the symbolic sensor as JSON, Python, and LaTeX-ish text."""
    sensor = SymbolicExpression.from_json(sensor_json)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "expression": sensor.expression,
            "clip_min": sensor.clip_min,
            "clip_max": sensor.clip_max,
            "scale": sensor.scale,
            "offset": sensor.offset,
        },
        root / "best_sensor.json",
    )
    py = (
        "import numpy as np\n\n"
        "def div(a, b):\n"
        "    aa, bb = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))\n"
        "    return np.divide(aa, bb, out=np.ones_like(aa, dtype=float), where=np.abs(bb) > 1e-12)\n\n"
        "def inv(v):\n"
        "    vv = np.asarray(v, dtype=float)\n"
        "    return np.divide(1.0, vv, out=np.zeros_like(vv, dtype=float), where=np.abs(vv) > 1e-12)\n\n"
        "add = np.add\n"
        "sub = np.subtract\n"
        "mul = np.multiply\n"
        "neg = np.negative\n"
        "sqrt = lambda v: np.sqrt(np.abs(v))\n"
        "log = lambda v: np.log(np.abs(v) + 1e-12)\n"
        "abs = np.abs\n"
        "sin = np.sin\n"
        "cos = np.cos\n"
        "tan = np.tan\n"
        "max = np.maximum\n"
        "min = np.minimum\n\n"
        "def symbolic_shock_sensor(x, y, z, nx, ny, nz, Mach, AoA, pi_param):\n"
        f"    chi = {sensor.expression.replace('^', '**')}\n"
        f"    return np.clip(chi, {sensor.clip_min}, {sensor.clip_max})\n"
    )
    (root / "best_sensor.py").write_text(py, encoding="utf-8")
    (root / "best_equation.tex").write_text(sensor.expression + "\n", encoding="utf-8")
    return {"json": root / "best_sensor.json", "python": root / "best_sensor.py", "latex": root / "best_equation.tex"}
