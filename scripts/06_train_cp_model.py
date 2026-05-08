#!/usr/bin/env python
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from cp_shock_project.training.train_cp import train_from_config
from cp_shock_project.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    metrics = train_from_config(load_config(args.config))
    print(metrics)


if __name__ == "__main__":
    main()
