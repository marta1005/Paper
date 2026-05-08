#!/usr/bin/env python
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from cp_shock_project.data.case_indexing import build_case_index, save_case_index
from cp_shock_project.data.load_arrays import load_arrays
from cp_shock_project.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    arrays = load_arrays(cfg.get("data_dir", "data"), mmap=True)
    out_dir = cfg.get("case_index_dir", "processed/case_indices")
    save_case_index(build_case_index(arrays.X_train), out_dir, "train")
    save_case_index(build_case_index(arrays.X_test), out_dir, "test")
    print(f"Saved case indices to {out_dir}")


if __name__ == "__main__":
    main()
