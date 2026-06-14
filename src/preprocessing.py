import numpy as np
import logging
from config import PREPROCESSING_CONFIG

logger = logging.getLogger(__name__)


class CFDPreprocessor:
    def __init__(self):
        self.gamma = PREPROCESSING_CONFIG['gamma']

    def compute_derived_features(self, X):
        """
        Adds 5 physics-based features derived solely from X (no Y used).
        Returns array of shape (n, 14).
        """
        g = self.gamma
        Mach = X[:, 6]
        AoA  = X[:, 7]
        Pi   = X[:, 8]

        q_dyn    = 0.5 * Mach ** 2
        Pi_norm  = Pi / (1.0 + 0.5 * (g - 1.0) * Mach ** 2)
        AoA_sin  = np.sin(np.deg2rad(AoA))
        L_factor = np.sqrt(np.maximum(1.0 - Mach ** 2, 0.0)) / (1.0 + 0.5 * (g - 1.0) * Mach ** 2)

        sonic_ratio = (2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * Mach ** 2)
        Cp_crit = (2.0 / (g * np.maximum(Mach ** 2, 1e-6))) * (sonic_ratio ** (g / (g - 1.0)) - 1.0)

        derived = np.column_stack([q_dyn, Pi_norm, AoA_sin, L_factor, Cp_crit])
        X_out = np.hstack([X, derived]).astype(np.float32)
        logger.info(f"Derived features: {X.shape} -> {X_out.shape}")
        return X_out
