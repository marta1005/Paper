"""2D and 3D scatter plots for pointwise/scattered CFD snapshots."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
_CACHE = Path(os.environ.get("MPLCONFIGDIR", ".matplotlib_cache"))
if (_CACHE.exists() and not os.access(_CACHE, os.W_OK)) or (not _CACHE.exists() and not os.access(_CACHE.parent, os.W_OK)):
    _CACHE = Path(".matplotlib_cache")
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_CACHE)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))

import matplotlib.pyplot as plt
import numpy as np


def _sample(n_points: int, max_points: int, seed: int = 42) -> np.ndarray:
    if n_points <= max_points:
        return np.arange(n_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_points, size=max_points, replace=False))


def save_scatter_2d(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    path: str | Path,
    title: str,
    cmap: str = "viridis",
    max_points: int = 80_000,
) -> None:
    """Save an x-y scatter plot colored by a scalar field."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    idx = _sample(len(values), max_points)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    sc = ax.scatter(x[idx], y[idx], c=values[idx], s=1.2, cmap=cmap, linewidths=0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, shrink=0.88)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_scatter_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    path: str | Path,
    title: str,
    cmap: str = "viridis",
    max_points: int = 80_000,
) -> None:
    """Save a 3D scatter plot colored by a scalar field."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    idx = _sample(len(values), max_points)
    fig = plt.figure(figsize=(8.0, 5.4), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x[idx], y[idx], z[idx], c=values[idx], s=1.0, cmap=cmap, linewidths=0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, shrink=0.75)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_case_scatter_suite(
    output_dir: str | Path,
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray] | None = None,
    scores: dict[str, np.ndarray] | None = None,
    max_points: int = 80_000,
) -> None:
    """Save standard Cp/Cf/grad/label/sensor scatter diagnostics for one case."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields: list[tuple[str, np.ndarray, str]] = [
        ("Cp", features["Cp"], "jet"),
        ("Cf_mag", features["Cf_mag"], "viridis"),
        ("grad_Cp_mag", features["grad_Cp_mag"], "inferno"),
    ]
    if labels is not None:
        fields.append(("shock_label", labels["shock_label"], "magma"))
        if "sep_label" in labels:
            fields.append(("sep_label", labels["sep_label"], "magma"))
    if scores is not None:
        for name, value in scores.items():
            fields.append((name, value, "magma"))
    for name, value, cmap in fields:
        save_scatter_2d(features["x"], features["y"], value, out / f"{name}_xy.png", name, cmap=cmap, max_points=max_points)
        save_scatter_3d(features["x"], features["y"], features["z"], value, out / f"{name}_3d.png", name, cmap=cmap, max_points=max_points)
