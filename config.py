import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
RESULT_DIR = OUTPUT_DIR / "results"
PLOT_DIR = OUTPUT_DIR / "plots"

for path in [MODEL_DIR, RESULT_DIR, PLOT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

DATA_CONFIG = {
    'X_train_path': DATA_DIR / "X_train.npy",
    'X_test_path':  DATA_DIR / "X_test.npy",
    'Y_train_path': DATA_DIR / "Ytrain.npy",
    'Y_test_path':  DATA_DIR / "Ytest.npy",
    'dataset_csv':  DATA_DIR / "dataset.csv",
    'input_features':  ['x', 'y', 'z', 'nx', 'ny', 'nz', 'Mach', 'AoA', 'Pi_1e-5'],
    'output_features': ['Cp', 'Cfx', 'Cfy', 'Cfz'],
    'input_dim':  9,
    'output_dim': 4,
    'train_sample_fraction': float(os.environ.get('PAPER_TRAIN_FRACTION', 0.05)),
    'val_split':             float(os.environ.get('PAPER_VAL_SPLIT', 0.1)),
    'test_sample_fraction':  float(os.environ.get('PAPER_TEST_FRACTION', 0.1)),
}

MODEL_CONFIG = {
    'autoencoder': {
        'input_dim':  14,
        'latent_dim': 32,
        'batch_norm': True,
        'dropout':    0.1,
        'l1_lambda':  1e-4,   # sparsity on latent activations
    },
    'moe': {
        'num_experts':       4,
        'expert_output_dim': 32,
        'output_dim':        4,
    },
    'sensor': {
        'head_hidden': [64, 32, 16],
    },
    'surrogate': {
        'num_experts':      4,
        'output_dim':       4,
        'indicator_hidden': [64, 32, 16],  # ShockIndicator MLP dims
        'expert_hidden':    [128, 256, 128],  # each expert MLP dims
        'shock_weight':     0.1,   # λ weighting shock BCE vs aero MSE
        'shock_pos_weight': 5.0,   # pos_weight for BCE (19% positive shock labels)
    },
}

TRAINING_CONFIG = {
    'num_epochs':              int(os.environ.get('PAPER_EPOCHS', 20)),
    'batch_size':              int(os.environ.get('PAPER_BATCH_SIZE', 256)),
    'learning_rate':           1e-3,
    'weight_decay':            1e-5,
    'early_stopping_patience': 5,
    'early_stopping_delta':    1e-4,
    'validate_every':          5,
    'num_workers':             int(os.environ.get('PAPER_NUM_WORKERS', 4)),
    'lr_min':                  1e-6,
}

PREPROCESSING_CONFIG = {
    'gamma': 1.4,
}

# Columns of the derived feature array (9 original + 5 derived from X only — no Y)
DERIVED_FEATURE_INDICES = {
    'q_dyn':    9,   # 0.5 * Mach^2
    'Pi_norm': 10,   # Pi / (1 + 0.5*(gamma-1)*Mach^2)
    'AoA_sin': 11,   # sin(AoA in radians)
    'L_factor':12,   # Laitone compressibility correction
    'Cp_crit': 13,   # Critical Cp at sonic condition (function of Mach only)
}

DEVICE = 'cuda'

LOGGING_CONFIG = {
    'log_level': 'INFO',
    'save_log':  True,
    'log_file':  OUTPUT_DIR / "training.log",
}

SEED = 42
