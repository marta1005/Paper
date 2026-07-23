import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data"
CACHE_DIR    = PROJECT_ROOT / "cache"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
MODEL_DIR    = OUTPUT_DIR / "models"
RESULT_DIR   = OUTPUT_DIR / "results"
PLOT_DIR     = OUTPUT_DIR / "plots"

for p in [MODEL_DIR, RESULT_DIR, PLOT_DIR, CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

N_TRAIN  = 312
N_TEST   = 156
N_P      = 260_774   # nodes per simulation
KNN_K    = int(os.environ.get('V2_KNN_K', 8))
SEED     = 42

DATA_CONFIG = {
    'X_train_path':  DATA_DIR / 'X_train.npy',
    'X_test_path':   DATA_DIR / 'X_test.npy',
    'Y_train_path':  DATA_DIR / 'Ytrain.npy',
    'Y_test_path':   DATA_DIR / 'Ytest.npy',
    'dataset_csv':   DATA_DIR / 'dataset.csv',
}

PREPROCESSING_CONFIG = {
    'gamma': 1.4,
    'raw_dim':     9,
    'derived_dim': 16,
    # spatial_stats fitted from training data and saved to MODEL_DIR/spatial_stats.npy
}

MODEL_CONFIG = {
    'node_in_dim':  16,   # 9 raw + 7 derived features
    'edge_in_dim':  5,    # dx, dy, dz, dist, n_dot
    'node_enc_dim': 128,  # intermediate node encoder
    'latent_dim':   256,  # d — node latent after backbone
    'edge_enc_dim': 128,  # edge encoder output
    'n_gnn_layers': 8,
    'moe': {
        'num_experts':        4,
        'output_dim':         1,    # Cp only (friction handled by dedicated head)
        'expert_hidden':      [256, 256, 128],
        'shock_expert_hidden':[128, 128],
        'indicator_hidden':   [128, 64, 32],
        'shock_weight':       0.1,
        'shock_pos_weight':   5.0,
        'load_balance_weight':0.01,
        'shock_mse_weight':   5.0,
        'gumbel_tau_start':   1.0,
        'gumbel_tau_end':     0.3,
    },
    'friction': {
        'hidden': [256, 256, 128],
        'loss_weight': 3.0,   # λ_fric
    },
}

TRAINING_CONFIG = {
    'num_epochs':               int(os.environ.get('PAPER_EPOCHS', 50)),
    'learning_rate':            1e-3,
    'weight_decay':             1e-5,
    'grad_clip':                1.0,
    'warmup_epochs':            3,
    'early_stopping_patience':  8,
    'validate_every':           2,   # epochs
    'val_sims':                 16,  # sims to use for quick validation each check
    'num_workers':              int(os.environ.get('PAPER_NUM_WORKERS', 4)),
}

LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file':  OUTPUT_DIR / 'training_v2.log',
}
