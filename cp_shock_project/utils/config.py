from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return data


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Write a YAML config file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, sort_keys=False)


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively update a config dictionary."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def get_path(config: Mapping[str, Any], key: str, default: str | None = None) -> Path:
    """Return a path from config, expanding user markers."""
    value = config.get(key, default)
    if value is None:
        raise KeyError(key)
    return Path(str(value)).expanduser()
