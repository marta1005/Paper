"""Geometry and freestream projection features."""

from __future__ import annotations

import numpy as np

EPS = 1.0e-8


def freestream_direction(aoa_degrees: np.ndarray | float) -> np.ndarray:
    """Return freestream direction vectors from AoA in degrees.

    The first version assumes the freestream lies in the x-z plane. This is a
    local convention and can be changed here without touching the symbolic
    dataset or downstream scripts.
    """
    aoa = np.asarray(aoa_degrees, dtype=np.float32)
    alpha = np.deg2rad(aoa)
    return np.stack([np.cos(alpha), np.zeros_like(alpha), np.sin(alpha)], axis=-1).astype(np.float32)


def normalize_vectors(vectors: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Normalize row-wise vectors safely."""
    norm = np.linalg.norm(vectors.astype(np.float32), axis=1, keepdims=True)
    return vectors / np.maximum(norm, eps)


def tangent_freestream_direction(normals: np.ndarray, aoa_degrees: np.ndarray | float) -> np.ndarray:
    """Project freestream direction into the local tangent plane."""
    n_hat = normalize_vectors(normals)
    e_inf = freestream_direction(aoa_degrees)
    if e_inf.ndim == 1:
        e_inf = np.broadcast_to(e_inf[None, :], n_hat.shape)
    dot = np.sum(e_inf * n_hat, axis=1, keepdims=True)
    tangent = e_inf - dot * n_hat
    return normalize_vectors(tangent)
