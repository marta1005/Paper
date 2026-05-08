from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cp_shock_project.data.case_indexing import CaseIndex
from cp_shock_project.features.geometry import cf_magnitude


@dataclass(frozen=True)
class ShockFeatures:
    grad_Cp_mag_approx: np.ndarray
    local_Cp_contrast: np.ndarray
    grad_Cf_mag_approx: np.ndarray
    local_Cf_contrast: np.ndarray
    oracle_shock_score: np.ndarray
    shock_label: np.ndarray


def neighbor_mean_abs_gradient(values: np.ndarray, neighbor_indices: np.ndarray, neighbor_distances: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    out = np.zeros(vals.shape[0], dtype=np.float64)
    for i in range(vals.shape[0]):
        nidx = neighbor_indices[i]
        mask = (nidx >= 0) & np.isfinite(neighbor_distances[i])
        if not np.any(mask):
            continue
        diff = np.abs(vals[nidx[mask]] - vals[i])
        out[i] = float(np.mean(diff / (neighbor_distances[i, mask] + eps)))
    return out


def local_contrast(values: np.ndarray, neighbor_indices: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    out = np.zeros(vals.shape[0], dtype=np.float64)
    for i in range(vals.shape[0]):
        nidx = neighbor_indices[i]
        mask = nidx >= 0
        if not np.any(mask):
            continue
        local = vals[nidx[mask]]
        out[i] = float(np.max(local) - np.min(local))
    return out


def robust_normalize(values: np.ndarray, lower: float = 50.0, upper: float = 99.5, eps: float = 1e-12) -> np.ndarray:
    """Robust percentile normalization clipped to [0, 1]."""
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = np.nanpercentile(arr, [lower, upper])
    if hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)


def compute_oracle_shock_score(
    X: np.ndarray,
    Y: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    case_index: CaseIndex | None = None,
    lower_percentile: float = 50.0,
    upper_percentile: float = 99.5,
    label_percentile: float = 98.5,
    low_mach_downweight: float | None = None,
    low_mach_threshold: float = 0.7,
    eps: float = 1e-8,
) -> ShockFeatures:
    cp = np.asarray(Y[:, 0], dtype=np.float64)
    cf = cf_magnitude(Y)
    grad_cp = neighbor_mean_abs_gradient(cp, neighbor_indices, neighbor_distances, eps=eps)
    contrast_cp = local_contrast(cp, neighbor_indices)
    grad_cf = neighbor_mean_abs_gradient(cf, neighbor_indices, neighbor_distances, eps=eps)
    contrast_cf = local_contrast(cf, neighbor_indices)
    score = np.zeros_like(cp, dtype=np.float32)
    label = np.zeros_like(cp, dtype=bool)
    if case_index is None:
        score[:] = robust_normalize(grad_cp, lower_percentile, upper_percentile)
        threshold = np.percentile(score, label_percentile)
        label[:] = score > threshold
    else:
        for cid in range(case_index.n_cases):
            idx = case_index.indices_for_case(cid)
            s = robust_normalize(grad_cp[idx], lower_percentile, upper_percentile)
            if low_mach_downweight is not None and float(case_index.unique_conditions[cid, 0]) < low_mach_threshold:
                s = s * float(low_mach_downweight)
            score[idx] = s
            threshold = np.percentile(s, label_percentile)
            label[idx] = s > threshold
    return ShockFeatures(
        grad_Cp_mag_approx=grad_cp.astype(np.float32),
        local_Cp_contrast=contrast_cp.astype(np.float32),
        grad_Cf_mag_approx=grad_cf.astype(np.float32),
        local_Cf_contrast=contrast_cf.astype(np.float32),
        oracle_shock_score=score.astype(np.float32),
        shock_label=label,
    )
