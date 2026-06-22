import numpy as np
import logging
from config import PREPROCESSING_CONFIG

logger = logging.getLogger(__name__)


class CFDPreprocessor:
    def __init__(self, spatial_stats=None):
        self.gamma = PREPROCESSING_CONFIG['gamma']
        self.spatial_stats = spatial_stats  # fitted bounds for x_norm, span_norm

    def fit_spatial(self, X):
        """Compute chord/span normalization bounds from the full training set."""
        self.spatial_stats = {
            'x_min': float(X[:, 0].min()),
            'x_max': float(X[:, 0].max()),
            'y_min': float(X[:, 1].min()),
            'y_max': float(X[:, 1].max()),
        }
        logger.info(
            f"Spatial stats fitted:  "
            f"x=[{self.spatial_stats['x_min']:.3f}, {self.spatial_stats['x_max']:.3f}]  "
            f"y=[{self.spatial_stats['y_min']:.3f}, {self.spatial_stats['y_max']:.3f}]"
        )
        return self.spatial_stats

    def compute_derived_features(self, X):
        """
        Adds derived features to the 9-column raw array.

        Returns (n, 16) when spatial_stats are available, else (n, 14).

        Cols 9-13  — physics scalars from flight conditions (no geometry needed):
          9: q_dyn       = 0.5 * Mach^2
         10: Pi_norm     = Pi / (1 + 0.5*(γ-1)*Mach^2)
         11: AoA_sin     = sin(AoA_rad)
         12: L_factor    = sqrt(1-Mach^2) / (1 + 0.5*(γ-1)*Mach^2)  [Laitone]
         13: Cp_crit     = critical Cp at sonic condition (function of Mach only)

        Cols 14-15 — geometry context (requires fitted spatial_stats):
         14: x_norm      = (x - x_min) / (x_max - x_min)   [chord position, 0=LE, 1=TE]
         15: span_norm   = (y - y_min) / (y_max - y_min)   [span position, 0=root, 1=tip]
        """
        g    = self.gamma
        Mach = X[:, 6]
        AoA  = X[:, 7]
        Pi   = X[:, 8]

        q_dyn       = 0.5 * Mach ** 2
        Pi_norm     = Pi / (1.0 + 0.5 * (g - 1.0) * Mach ** 2)
        AoA_sin     = np.sin(np.deg2rad(AoA))
        L_factor    = np.sqrt(np.maximum(1.0 - Mach ** 2, 0.0)) / (1.0 + 0.5 * (g - 1.0) * Mach ** 2)
        sonic_ratio = (2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * Mach ** 2)
        Cp_crit     = (2.0 / (g * np.maximum(Mach ** 2, 1e-6))) * (sonic_ratio ** (g / (g - 1.0)) - 1.0)

        cols = [q_dyn, Pi_norm, AoA_sin, L_factor, Cp_crit]

        if self.spatial_stats is not None:
            ss  = self.spatial_stats
            eps = 1e-8
            x_range    = max(ss['x_max'] - ss['x_min'], eps)
            y_range    = max(ss['y_max'] - ss['y_min'], eps)
            x_norm    = np.clip((X[:, 0] - ss['x_min']) / x_range,    0.0, 1.0)
            span_norm = np.clip((X[:, 1] - ss['y_min']) / y_range, 0.0, 1.0)
            cols += [x_norm, span_norm]

        derived = np.column_stack(cols)
        X_out   = np.hstack([X, derived]).astype(np.float32)
        logger.info(f"Derived features: {X.shape} -> {X_out.shape}")
        return X_out
