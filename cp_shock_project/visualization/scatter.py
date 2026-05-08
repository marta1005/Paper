from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cp_shock_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def scatter_view(
    X: np.ndarray,
    values: np.ndarray,
    path: str | Path,
    title: str,
    view: str = "xy",
    s: float = 3.0,
    cmap: str = "coolwarm",
) -> None:
    axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    a, b = axes[view]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    sc = ax.scatter(X[:, a], X[:, b], c=np.asarray(values).reshape(-1), s=s, cmap=cmap, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(["x", "y", "z"][a])
    ax.set_ylabel(["x", "y", "z"][b])
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
