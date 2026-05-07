"""Skin-friction derived pointwise features."""

from __future__ import annotations

import numpy as np

from shock_symbolic.features.geometry_features import tangent_freestream_direction

EPS = 1.0e-8


def cf_magnitude(cfx: np.ndarray, cfy: np.ndarray, cfz: np.ndarray) -> np.ndarray:
    """Compute full skin-friction vector magnitude."""
    return np.sqrt(cfx.astype(np.float32) ** 2 + cfy.astype(np.float32) ** 2 + cfz.astype(np.float32) ** 2 + EPS).astype(np.float32)


def compute_cf_directional_features(
    cfx: np.ndarray,
    cfy: np.ndarray,
    cfz: np.ndarray,
    normals: np.ndarray,
    aoa_degrees: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Compute Cf magnitude, tangent-streamwise and crossflow components."""
    cf_vec = np.column_stack([cfx, cfy, cfz]).astype(np.float32)
    cf_mag = cf_magnitude(cfx, cfy, cfz)
    t_stream = tangent_freestream_direction(normals, aoa_degrees)
    cf_parallel = np.sum(cf_vec * t_stream, axis=1).astype(np.float32)
    cf_perp_sq = np.maximum(cf_mag**2 - cf_parallel**2, 0.0)
    cf_perp = np.sqrt(cf_perp_sq + EPS).astype(np.float32)
    cf_angle_stream = np.arctan2(cf_perp, cf_parallel + EPS).astype(np.float32)
    return {
        "Cf_mag": cf_mag,
        "Cf_parallel": cf_parallel,
        "Cf_perp": cf_perp,
        "Cf_angle_stream": cf_angle_stream,
    }
