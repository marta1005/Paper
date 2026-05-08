from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetScalers:
    x_scaler: Any | None = None
    cp_scaler: Any | None = None


class CpSurfaceDataset(Dataset):
    """Pointwise Cp dataset with optional neighbor features for graph models."""

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        indices: np.ndarray | None = None,
        case_ids: np.ndarray | None = None,
        neighbor_indices: np.ndarray | None = None,
        neighbor_distances: np.ndarray | None = None,
        oracle_shock_score: np.ndarray | None = None,
        symbolic_chi: np.ndarray | None = None,
        scalers: DatasetScalers | None = None,
    ):
        self.X = X
        self.Y = Y
        self.indices = np.asarray(indices if indices is not None else np.arange(X.shape[0]), dtype=np.int64)
        self.case_ids = case_ids
        self.neighbor_indices = neighbor_indices
        self.neighbor_distances = neighbor_distances
        self.oracle_shock_score = oracle_shock_score
        self.symbolic_chi = symbolic_chi
        self.scalers = scalers or DatasetScalers()

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _transform_x(self, x: np.ndarray) -> np.ndarray:
        if self.scalers.x_scaler is not None:
            return self.scalers.x_scaler.transform(x)
        return x

    def _transform_cp(self, cp: np.ndarray) -> np.ndarray:
        if self.scalers.cp_scaler is not None:
            return self.scalers.cp_scaler.transform(cp.reshape(-1, 1)).reshape(-1)
        return cp

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        x_raw = np.asarray(self.X[idx], dtype=np.float32).copy()
        x = np.asarray(self._transform_x(x_raw[None, :])[0], dtype=np.float32).copy()
        cp = np.asarray([self.Y[idx, 0]], dtype=np.float32)
        cp = np.asarray(self._transform_cp(cp), dtype=np.float32)
        out: dict[str, torch.Tensor] = {
            "X": torch.from_numpy(x),
            "X_raw": torch.from_numpy(x_raw),
            "Cp": torch.from_numpy(cp),
            "point_id": torch.tensor(idx, dtype=torch.long),
        }
        if self.case_ids is not None:
            out["case_id"] = torch.tensor(int(self.case_ids[idx]), dtype=torch.long)
        if self.oracle_shock_score is not None:
            out["oracle_shock_score"] = torch.tensor([float(self.oracle_shock_score[idx])], dtype=torch.float32)
        if self.symbolic_chi is not None:
            out["symbolic_chi"] = torch.tensor([float(self.symbolic_chi[idx])], dtype=torch.float32)
        if self.neighbor_indices is not None:
            nidx = np.asarray(self.neighbor_indices[idx], dtype=np.int64)
            valid = nidx >= 0
            safe_nidx = np.where(valid, nidx, idx)
            nx_raw = np.asarray(self.X[safe_nidx], dtype=np.float32).copy()
            nx = np.asarray(self._transform_x(nx_raw), dtype=np.float32).copy()
            out["neighbor_indices"] = torch.from_numpy(nidx)
            out["neighbor_X"] = torch.from_numpy(nx)
            out["neighbor_X_raw"] = torch.from_numpy(nx_raw)
            if self.neighbor_distances is not None:
                out["neighbor_distances"] = torch.from_numpy(np.asarray(self.neighbor_distances[idx], dtype=np.float32))
            else:
                out["neighbor_distances"] = torch.ones(len(nidx), dtype=torch.float32)
        return out
