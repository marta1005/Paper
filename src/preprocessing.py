"""
Preprocessing: cálculo de variables derivadas Tier 1
"""
import numpy as np
import logging
from scipy.ndimage import uniform_filter1d
from config import PREPROCESSING_CONFIG

logger = logging.getLogger(__name__)


class CFDPreprocessor:
    """Calcula variables derivadas a partir de X (inputs) e Y (outputs)"""
    
    def __init__(self):
        self.gamma = PREPROCESSING_CONFIG['gamma']
    
    def compute_derived_features(self, X, Y, compute_all=True):
        """
        Calcula variables derivadas
        
        Args:
            X: array (n_samples, 9) - [Mach, AoA, Pi, x, y, z, nx, ny, nz]
            Y: array (n_samples, 4) - [Cp, Cfx, Cfy, Cfz]
            compute_all: bool, calcular todos o solo algunos
        
        Returns:
            X_derived: array (n_samples, 19) - X original + 10 features derivados
        """
        n = len(X)
        derived = []
        
        Mach = X[:, 0]
        AoA = X[:, 1]
        Pi = X[:, 2]
        
        Cp = Y[:, 0]
        Cfx = Y[:, 1]
        Cfy = Y[:, 2]
        Cfz = Y[:, 3]
        
        logger.info("Computing derived features...")
        
        # 1. Mach local (isentrópico)
        if PREPROCESSING_CONFIG.get('compute_mach_local', True):
            M_local = self._compute_mach_local(Mach, Cp)
            derived.append(M_local.reshape(-1, 1))
        
        # 2. Gradiente de presión (suavizado)
        if PREPROCESSING_CONFIG.get('compute_pressure_gradient', True):
            grad_p = self._compute_pressure_gradient(Pi)
            derived.append(grad_p.reshape(-1, 1))
        
        # 3. Pérdida de presión de remanso
        if PREPROCESSING_CONFIG.get('compute_cp_loss', True):
            cp_loss = self._compute_cp_loss(Mach, Cp)
            derived.append(cp_loss.reshape(-1, 1))
        
        # 4. Indicador de choque (basado en gradiente)
        if PREPROCESSING_CONFIG.get('compute_shock_indicator', True):
            shock_ind = self._compute_shock_indicator(Pi, Cp)
            derived.append(shock_ind.reshape(-1, 1))
        
        # 5. Magnitud de fricción
        Cf_mag = np.sqrt(Cfx**2 + Cfy**2 + Cfz**2)
        derived.append(Cf_mag.reshape(-1, 1))
        
        # 6. Número de presión dinámica
        q_dyn = 0.5 * (1 + (self.gamma - 1) * 0.5 * Mach**2) ** (self.gamma / (self.gamma - 1))
        derived.append(q_dyn.reshape(-1, 1))
        
        # 7-10. Features adicionales
        # Presión normalizada por Mach
        Pi_norm = Pi / (1 + (self.gamma - 1) * 0.5 * Mach**2)
        derived.append(Pi_norm.reshape(-1, 1))
        
        # Ángulo de ataque normalizado
        AoA_norm = AoA / (Mach + 1e-6)
        derived.append(AoA_norm.reshape(-1, 1))
        
        # Gradiente de fricción
        grad_cf = self._compute_pressure_gradient(Cf_mag)
        derived.append(grad_cf.reshape(-1, 1))
        
        # Factor de compresibilidad Laitone
        L_factor = (1 - Mach**2) ** 0.5 / (1 + 0.5 * (self.gamma - 1) * Mach**2)
        derived.append(L_factor.reshape(-1, 1))
        
        # Concatenar con X original
        X_derived = np.hstack([X, np.hstack(derived)])
        
        logger.info(f"Derived features shape: {X_derived.shape}")
        return X_derived
    
    def _compute_mach_local(self, M_inf, Cp):
        """
        Mach local isentrópico
        M_local^2 = 2/(gamma-1) * ((1 + (gamma-1)/2 * M_inf^2) / (1 - Cp) - 1)
        """
        gamma = self.gamma
        numerator = 2 / (gamma - 1) * (
            (1 + (gamma - 1) / 2 * M_inf**2) / (1 - Cp) - 1
        )
        M_local = np.sqrt(np.maximum(numerator, 0.001))  # Evitar raíz negativa
        return M_local.astype(np.float32)
    
    def _compute_pressure_gradient(self, P):
        """
        Gradiente de presión (aproximación simple: diferencias finitas suavizadas)
        """
        # Usar ventana de suavizado
        window = PREPROCESSING_CONFIG.get('pressure_gradient_window', 5)
        
        # Aproximación numérica: diferencias suavizadas
        grad = np.abs(np.gradient(P))
        grad_smooth = uniform_filter1d(grad, size=window, mode='nearest')
        
        return grad_smooth.astype(np.float32)
    
    def _compute_cp_loss(self, M_inf, Cp):
        """
        Pérdida de presión de remanso (indicador de irreversibilidad de choque)
        Cp_loss = Cp_isentropic - Cp_real
        """
        gamma = self.gamma
        
        # Cp isentrópico
        Cp_isen = 2 / (gamma * M_inf**2) * (
            (2 + (gamma - 1) * M_inf**2) / (gamma + 1)
        ) ** (gamma / (gamma - 1)) - 1
        
        # Pérdida (positiva = irreversibilidad)
        cp_loss = np.maximum(Cp_isen - Cp, 0)
        
        return cp_loss.astype(np.float32)
    
    def _compute_shock_indicator(self, Pi, Cp):
        """
        Indicador de choque combinado:
        - Presión elevada
        - Gradiente alto
        - Pérdida termodinámica
        """
        # Normalizar indicadores individuales
        grad_p = self._compute_pressure_gradient(Pi)
        grad_p_norm = (grad_p - np.mean(grad_p)) / (np.std(grad_p) + 1e-6)
        
        cp_loss = self._compute_cp_loss(np.ones_like(Cp), Cp)
        cp_loss_norm = (cp_loss - np.mean(cp_loss)) / (np.std(cp_loss) + 1e-6)
        
        # Presión normalizada
        pi_norm = (Pi - np.mean(Pi)) / (np.std(Pi) + 1e-6)
        
        # Combinación
        shock_indicator = 0.5 * grad_p_norm + 0.3 * cp_loss_norm + 0.2 * pi_norm
        
        return shock_indicator.astype(np.float32)
    
    def normalize_data(self, X, Y, fit=False, stats=None):
        """
        Normaliza inputs y outputs
        
        Args:
            X, Y: arrays
            fit: bool, calcular estadísticas si True
            stats: dict con estadísticas precalculadas
        
        Returns:
            X_norm, Y_norm, stats
        """
        if fit or stats is None:
            X_mean = np.mean(X, axis=0, dtype=np.float32)
            X_std = np.std(X, axis=0, dtype=np.float32)
            X_std[X_std == 0] = 1.0
            
            Y_mean = np.mean(Y, axis=0, dtype=np.float32)
            Y_std = np.std(Y, axis=0, dtype=np.float32)
            Y_std[Y_std == 0] = 1.0
            
            stats = {
                'X_mean': X_mean,
                'X_std': X_std,
                'Y_mean': Y_mean,
                'Y_std': Y_std,
            }
        
        X_norm = (X - stats['X_mean']) / stats['X_std']
        Y_norm = (Y - stats['Y_mean']) / stats['Y_std']
        
        return X_norm, Y_norm, stats


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test
    from src.data_loader import load_data_with_sampling
    
    X_train, Y_train, _, _, _, _ = load_data_with_sampling(sample_fraction=0.01)
    
    preprocessor = CFDPreprocessor()
    X_derived = preprocessor.compute_derived_features(X_train, Y_train)
    print(f"Original: {X_train.shape} -> Derived: {X_derived.shape}")
