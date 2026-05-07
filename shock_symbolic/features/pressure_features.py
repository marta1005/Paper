"""Combined pointwise pressure, Cf, geometry and kNN features."""

from __future__ import annotations

from typing import Any

import numpy as np

from shock_symbolic.features.cf_features import compute_cf_directional_features
from shock_symbolic.features.geometry_features import tangent_freestream_direction
from shock_symbolic.features.knn_features import knn_indices_distances, local_contrast, local_streamwise_gradient, local_weighted_gradient


BASE_FEATURE_NAMES = [
    "Cp",
    "Cfx",
    "Cfy",
    "Cfz",
    "Cf_mag",
    "Cf_parallel",
    "Cf_perp",
    "Cf_angle_stream",
    "grad_Cp_mag",
    "grad_Cp_streamwise",
    "local_Cp_contrast",
    "grad_Cf_mag",
    "local_Cf_contrast",
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "Mach",
    "AoA",
    "pi_scaled",
]


def compute_snapshot_features(
    snapshot: dict[str, np.ndarray],
    k_neighbors: int = 16,
    batch_size: int = 4096,
    max_numpy_knn_points: int = 20_000,
) -> dict[str, np.ndarray]:
    """Compute all first-version physics features for one CFD condition."""
    coords = np.column_stack([snapshot["x"], snapshot["y"], snapshot["z"]]).astype(np.float32)
    normals = np.column_stack([snapshot["nx"], snapshot["ny"], snapshot["nz"]]).astype(np.float32)
    cf = compute_cf_directional_features(snapshot["Cfx"], snapshot["Cfy"], snapshot["Cfz"], normals, snapshot["AoA"])
    neighbor_idx, neighbor_dist = knn_indices_distances(
        coords,
        k_neighbors=k_neighbors,
        batch_size=batch_size,
        max_numpy_points=max_numpy_knn_points,
    )
    stream_t = tangent_freestream_direction(normals, snapshot["AoA"])
    features: dict[str, np.ndarray] = {
        "point_id": np.asarray(snapshot["point_id"], dtype=np.int64),
        "Cp": np.asarray(snapshot["Cp"], dtype=np.float32),
        "Cfx": np.asarray(snapshot["Cfx"], dtype=np.float32),
        "Cfy": np.asarray(snapshot["Cfy"], dtype=np.float32),
        "Cfz": np.asarray(snapshot["Cfz"], dtype=np.float32),
        "x": np.asarray(snapshot["x"], dtype=np.float32),
        "y": np.asarray(snapshot["y"], dtype=np.float32),
        "z": np.asarray(snapshot["z"], dtype=np.float32),
        "nx": np.asarray(snapshot["nx"], dtype=np.float32),
        "ny": np.asarray(snapshot["ny"], dtype=np.float32),
        "nz": np.asarray(snapshot["nz"], dtype=np.float32),
        "Mach": np.asarray(snapshot["Mach"], dtype=np.float32),
        "AoA": np.asarray(snapshot["AoA"], dtype=np.float32),
        "pi_scaled": np.asarray(snapshot["pi_scaled"], dtype=np.float32),
    }
    features.update(cf)
    features["grad_Cp_mag"] = local_weighted_gradient(features["Cp"], neighbor_idx, neighbor_dist)
    features["grad_Cp_streamwise"] = local_streamwise_gradient(features["Cp"], coords, stream_t, neighbor_idx, neighbor_dist)
    features["local_Cp_contrast"] = local_contrast(features["Cp"], neighbor_idx)
    features["grad_Cf_mag"] = local_weighted_gradient(features["Cf_mag"], neighbor_idx, neighbor_dist)
    features["local_Cf_contrast"] = local_contrast(features["Cf_mag"], neighbor_idx)
    features["feature_names"] = np.asarray(BASE_FEATURE_NAMES)
    return features


def save_case_features(path: str, case: dict[str, Any], features: dict[str, np.ndarray]) -> None:
    """Save case features plus scalar metadata to compressed `.npz`."""
    payload = dict(features)
    payload["case_id"] = np.asarray(case["case_id"])
    payload["split"] = np.asarray(case["split"])
    payload["case_start"] = np.asarray(int(case["start"]), dtype=np.int64)
    payload["case_stop"] = np.asarray(int(case["stop"]), dtype=np.int64)
    np.savez_compressed(path, **payload)
