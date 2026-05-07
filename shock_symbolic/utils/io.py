"""Filesystem, JSON, CSV and optional tabular IO helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a JSON payload."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON payload."""
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Save flat dictionaries as CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Load CSV rows as dictionaries."""
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_table(path_without_suffix: str | Path, table: dict[str, np.ndarray]) -> Path:
    """Save a tabular dictionary as parquet when possible, otherwise CSV.

    The implementation prefers pandas/pyarrow when installed, but remains usable
    in lightweight environments by writing CSV with Python's standard library.
    """
    base = Path(path_without_suffix)
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(table)
        try:
            out = base.with_suffix(".parquet")
            df.to_parquet(out, index=False)
            return out
        except Exception:
            out = base.with_suffix(".csv")
            df.to_csv(out, index=False)
            return out
    except Exception:
        out = base.with_suffix(".csv")
        keys = list(table.keys())
        n_rows = len(next(iter(table.values()))) if table else 0
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for idx in range(n_rows):
                writer.writerow([table[key][idx] for key in keys])
        return out


def load_table(path: str | Path) -> dict[str, np.ndarray]:
    """Load a CSV/parquet table into NumPy arrays."""
    table_path = Path(path)
    if table_path.suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore

            df = pd.read_parquet(table_path)
            return {col: df[col].to_numpy() for col in df.columns}
        except Exception as exc:
            raise RuntimeError("Reading parquet requires pandas plus pyarrow or fastparquet.") from exc
    with table_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    cols = reader.fieldnames or []
    result: dict[str, np.ndarray] = {}
    for col in cols:
        values = [row[col] for row in rows]
        try:
            result[col] = np.asarray(values, dtype=np.float64)
        except ValueError:
            result[col] = np.asarray(values)
    return result
