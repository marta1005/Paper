#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from cp_shock_project.data.case_indexing import build_case_index, case_table
from cp_shock_project.data.load_arrays import X_COLUMNS, Y_COLUMNS, inspect_array, load_arrays, validate_shapes
from cp_shock_project.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    arrays = load_arrays(cfg.get("data_dir", "data"), mmap=True)
    for name, X, Y in [("train", arrays.X_train, arrays.Y_train), ("test", arrays.X_test, arrays.Y_test)]:
        validate_shapes(X, Y)
        print(json.dumps(inspect_array(f"X_{name}", X, X_COLUMNS), indent=2))
        print(json.dumps(inspect_array(f"Y_{name}", Y, Y_COLUMNS), indent=2))
        idx = build_case_index(X)
        table = case_table(idx)
        print(f"{name}: {idx.n_cases} unique conditions")
        print(table[["case_id", "Mach", "AoA", "pi", "n_points"]].describe(include="all").to_string())


if __name__ == "__main__":
    main()
