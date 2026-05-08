from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cp_shock_project.utils.io import load_json, save_json


@dataclass
class StandardScaler:
    mean_: list[float] | None = None
    scale_: list[float] | None = None
    eps: float = 1e-12

    def fit(self, x: np.ndarray) -> "StandardScaler":
        arr = np.asarray(x, dtype=np.float64)
        self.mean_ = arr.mean(axis=0).tolist()
        scale = arr.std(axis=0)
        scale = np.where(scale < self.eps, 1.0, scale)
        self.scale_ = scale.tolist()
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted")
        return (np.asarray(x, dtype=np.float32) - np.asarray(self.mean_, dtype=np.float32)) / np.asarray(self.scale_, dtype=np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted")
        return np.asarray(x, dtype=np.float32) * np.asarray(self.scale_, dtype=np.float32) + np.asarray(self.mean_, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        save_json(asdict(self), path)

    @classmethod
    def load(cls, path: str | Path) -> "StandardScaler":
        return cls(**load_json(path))
