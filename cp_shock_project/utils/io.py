from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON with stable indentation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> Any:
    """Load JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """Save compressed NumPy arrays."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)
