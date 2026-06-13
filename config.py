"""
Configuración centralizada del proyecto
"""
import os
from pathlib import Path

# ============ PATHS ============
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
RESULT_DIR = OUTPUT_DIR / "results"
PLOT_DIR = OUTPUT_DIR / "plots"

# Crear directorios si no existen
for path in [MODEL_DIR, RESULT_DIR, PLOT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# ============ DATA CONFIG ============
DATA_CONFIG = {
    'X_train_path': DATA_DIR / "X_train.npy",
    'X_test_path': DATA_DIR / "X_test.npy",
    'Y_train_path': DATA_DIR / "Ytrain.npy",
    'Y_test_path': DATA_DIR / "Ytest.npy",
    'dataset_csv': DATA_DIR / "dataset.csv",
    
    # Descripción de features
    'input_features': ['Mach', 'AoA', 'Pi', 'x', 'y', 'z', 'nx', 'ny', 'nz'],
    'output_features': ['Cp', 'Cfx', 'Cfy', 'Cfz'],
    'input_dim': 9,
    'output_dim': 4,
    
    # Sampling estratégico (81M puntos es demasiado, subsampleamos)
    'train_sample_fraction': 0.05,  # Usar 5% (~4M puntos) para demo rápida. Aumentar a 0.2-0.5 para producción
    'val_split': 0.1,
    'test_sample_fraction': 0.1,  # Usar 10% de test para demo rápida. Usar 1.0 para evaluación completa
}

# ============ MODEL CONFIG ============
MODEL_CONFIG = {
    'autoencoder': {
        'latent_dim': 32,
        'encoder_dims': [19, 128, 64, 32],  # 19 = 9 inputs + 10 derived features
        'decoder_dims': [32, 64, 128, 19],
        'batch_norm': True,
        'dropout': 0.1,
    },
    'moe': {
        'num_experts': 4,
        'expert_dims': [32, 64, 128, 64],
        'gating_dims': [32, 64, 4],
        'expert_output_dim': 16,
    },
    'sensor': {
        'shock_head_dims': [32, 16, 8, 1],
        'intensity_head_dims': [32, 16, 8, 1],
        'separation_head_dims': [32, 16, 8, 1],
    }
}

# ============ TRAINING CONFIG ============
TRAINING_CONFIG = {
    'num_epochs': 20,  # Demo rápida. Aumentar a 50-100 para producción
    'batch_size': 256,  # Reducido para demo
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'early_stopping_patience': 5,  # Demo rápida
    'early_stopping_delta': 1e-4,
    
    # Scheduler
    'scheduler': 'cosine',  # 'cosine' o 'exponential'
    'warmup_epochs': 5,
    
    # Loss weights
    'loss_reconstruction_weight': 1.0,
    'loss_shock_weight': 1.0,
    'loss_moe_weight': 0.5,
    
    # Validation
    'validate_every': 5,
    'save_every': 5,
}

# ============ PREPROCESSING CONFIG ============
PREPROCESSING_CONFIG = {
    'normalize_inputs': True,
    'normalize_outputs': True,
    'normalization_method': 'standardization',  # 'standardization' o 'minmax'
    
    # Variables derivadas (Tier 1)
    'compute_pressure_gradient': True,
    'compute_mach_local': True,
    'compute_cp_loss': True,
    'compute_shock_indicator': True,
    
    # Physics parameters
    'gamma': 1.4,
    'pressure_gradient_window': 5,  # puntos vecinos para calcular gradiente
}

# ============ DEVICE ============
DEVICE = 'cuda'  # 'cuda' o 'cpu'

# ============ LOGGING ============
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'save_log': True,
    'log_file': OUTPUT_DIR / "training.log",
}

# ============ RANDOM SEED ============
SEED = 42
