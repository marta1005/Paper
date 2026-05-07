#!/usr/bin/env python
"""Export symbolic shock sensor formula files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shock_symbolic.symbolic.expression_io import load_sensor_json
from shock_symbolic.utils.config import load_config
from shock_symbolic.utils.io import save_json
from shock_symbolic.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.get("logging", {}).get("level", "INFO"))
    sensor_path = Path(cfg.get("sensor_path", Path(cfg.get("output_dir", "outputs/symbolic/pysr")) / "best_sensor.json"))
    sensor = load_sensor_json(sensor_path)
    export_dir = Path(cfg.get("export_dir", "outputs/symbolic/export"))
    export_dir.mkdir(parents=True, exist_ok=True)
    expression = sensor["expression"]
    feature_names = sensor["feature_names"]
    function = [
        "import numpy as np",
        "",
        "def shock_sensor(features):",
        "    \"\"\"Evaluate exported symbolic shock sensor on a feature dictionary.\"\"\"",
    ]
    for idx, name in enumerate(feature_names):
        function.append(f"    x{idx} = np.asarray(features[{name!r}], dtype=float)")
    function.append(f"    score = {expression}")
    function.append(f"    return score, score >= {float(sensor['threshold'])!r}")
    (export_dir / "sensor.py").write_text("\n".join(function) + "\n", encoding="utf-8")
    (export_dir / "best_equation.txt").write_text(expression, encoding="utf-8")
    (export_dir / "best_equation.tex").write_text(str(sensor.get("latex", expression)), encoding="utf-8")
    save_json(export_dir / "best_sensor.json", sensor)
    print({"export_dir": str(export_dir)})


if __name__ == "__main__":
    main()
