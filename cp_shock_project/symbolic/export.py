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
        "def symbolic_shock_sensor(x, y, z, nx, ny, nz, Mach, AoA, pi):\n"
        f"    chi = {sensor.expression.replace('^', '**')}\n"
        f"    return np.clip(chi, {sensor.clip_min}, {sensor.clip_max})\n"
    )
    (root / "best_sensor.py").write_text(py, encoding="utf-8")
    (root / "best_equation.tex").write_text(sensor.expression + "\n", encoding="utf-8")
    return {"json": root / "best_sensor.json", "python": root / "best_sensor.py", "latex": root / "best_equation.tex"}
