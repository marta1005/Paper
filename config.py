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
    # Sobrescribible con env var PAPER_TRAIN_FRACTION (ej: 0.50 para producción)
    'train_sample_fraction': float(os.environ.get('PAPER_TRAIN_FRACTION', 0.05)),
    'val_split': 0.1,
    'test_sample_fraction': float(os.environ.get('PAPER_TEST_FRACTION', 0.1)),
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
        'output_dim': 4,  # Cp, Cfx, Cfy, Cfz
    },
    'sensor': {
        'shock_head_dims': [32, 16, 8, 1],
        'intensity_head_dims': [32, 16, 8, 1],
        'separation_head_dims': [32, 16, 8, 1],
    }
}

# ============ TRAINING CONFIG ============
TRAINING_CONFIG = {
    'num_epochs': int(os.environ.get('PAPER_EPOCHS', 20)),   # Demo rápida. 50-100 para producción
    'batch_size': int(os.environ.get('PAPER_BATCH_SIZE', 256)),
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'early_stopping_patience': 5,
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

    # DataLoader workers (0 = sin multiprocessing; aumentar a 4-8 en GPU linux)
    # Sobrescribible con env var PAPER_NUM_WORKERS
    'num_workers': int(os.environ.get('PAPER_NUM_WORKERS', 0)),

    # LR Scheduler
    'lr_min': 1e-6,  # Mínimo para CosineAnnealingLR
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

# ============ DERIVED FEATURE INDICES ============
# Orden exacto de columnas en X_derived (X_original [0-8] + derived [9-18])
# X_original: [Mach(0), AoA(1), Pi(2), x(3), y(4), z(5), nx(6), ny(7), nz(8)]
DERIVED_FEATURE_INDICES = {
    'M_local':          9,   # Mach local isentrópico
    'grad_p':          10,   # Gradiente de presión suavizado
    'cp_loss':         11,   # Pérdida de presión de remanso
    'shock_indicator': 12,   # Indicador combinado de choque
    'Cf_mag':          13,   # Magnitud de fricción (proxy separación)
    'q_dyn':           14,   # Número de presión dinámica
    'Pi_norm':         15,   # Presión normalizada por Mach
    'AoA_norm':        16,   # AoA normalizado por Mach
    'grad_cf':         17,   # Gradiente de fricción
    'L_factor':        18,   # Factor de compresibilidad de Laitone
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
