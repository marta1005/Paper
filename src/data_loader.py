import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from config import DATA_CONFIG, TRAINING_CONFIG

logger = logging.getLogger(__name__)


class CFDDataset(Dataset):
    def __init__(self, X, Y, scaler=None):
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        Y = np.nan_to_num(np.asarray(Y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        if scaler is None:
            X_std = X.std(axis=0).astype(np.float32)
            Y_std = Y.std(axis=0).astype(np.float32)
            X_std[X_std == 0] = 1.0
            Y_std[Y_std == 0] = 1.0
            scaler = {
                'X_mean': X.mean(axis=0).astype(np.float32),
                'X_std':  X_std,
                'Y_mean': Y.mean(axis=0).astype(np.float32),
                'Y_std':  Y_std,
            }

        self.scaler = scaler
        self.X = torch.from_numpy((X - scaler['X_mean']) / scaler['X_std'])
        self.Y = torch.from_numpy((Y - scaler['Y_mean']) / scaler['Y_std'])
        logger.info(f"Dataset: X={self.X.shape}, Y={self.Y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_data_with_sampling(sample_fraction=1.0):
    from pathlib import Path
    data_dir = Path(DATA_CONFIG['X_train_path']).parent

    X_tr_path  = data_dir / "X_train_derived.npy"
    X_te_path  = data_dir / "X_test_derived.npy"

    if X_tr_path.exists():
        X_train_full = np.load(X_tr_path, mmap_mode='r')
        n_feat = X_train_full.shape[1]
        if n_feat < 16:
            logger.warning(
                f"X_train_derived.npy has {n_feat} features (expected 16). "
                "Re-run preprocess_data.py to add x_norm and span_norm."
            )
        else:
            logger.info(f"Loading preprocessed data ({n_feat} features)...")
    else:
        logger.warning("X_train_derived.npy not found — run preprocess_data.py first")
        X_train_full = np.load(DATA_CONFIG['X_train_path'], mmap_mode='r')

    if X_te_path.exists():
        X_test_full = np.load(X_te_path, mmap_mode='r')
    else:
        X_test_full = np.load(DATA_CONFIG['X_test_path'], mmap_mode='r')

    Y_train_full = np.load(DATA_CONFIG['Y_train_path'], mmap_mode='r')
    Y_test_full  = np.load(DATA_CONFIG['Y_test_path'],  mmap_mode='r')

    n_train  = len(X_train_full)
    n_sample = int(n_train * sample_fraction)

    if sample_fraction < 1.0:
        idx     = np.sort(np.random.choice(n_train, size=n_sample, replace=False))
        X_train = np.asarray(X_train_full[idx], dtype=np.float32)
        Y_train = np.asarray(Y_train_full[idx], dtype=np.float32)
    else:
        X_train = np.asarray(X_train_full[:], dtype=np.float32)
        Y_train = np.asarray(Y_train_full[:], dtype=np.float32)

    n_val   = int(len(X_train) * DATA_CONFIG.get('val_split', 0.1))
    X_val   = X_train[-n_val:]
    Y_val   = Y_train[-n_val:]
    X_train = X_train[:-n_val]
    Y_train = Y_train[:-n_val]

    test_frac = DATA_CONFIG.get('test_sample_fraction', 1.0)
    n_test    = int(len(X_test_full) * test_frac)
    idx_te    = np.sort(np.random.choice(len(X_test_full), size=n_test, replace=False))
    X_test    = np.asarray(X_test_full[idx_te], dtype=np.float32)
    Y_test    = np.asarray(Y_test_full[idx_te],  dtype=np.float32)

    logger.info(f"Sizes — train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def get_dataloaders(sample_fraction=None, scaler=None):
    if sample_fraction is None:
        sample_fraction = DATA_CONFIG.get('train_sample_fraction', 0.05)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data_with_sampling(sample_fraction)

    train_ds = CFDDataset(X_train, Y_train, scaler=scaler)
    scaler   = train_ds.scaler
    val_ds   = CFDDataset(X_val,   Y_val,   scaler=scaler)
    test_ds  = CFDDataset(X_test,  Y_test,  scaler=scaler)

    nw = TRAINING_CONFIG.get('num_workers', 0)
    kw = dict(batch_size=TRAINING_CONFIG['batch_size'], num_workers=nw,
              pin_memory=True, persistent_workers=(nw > 0))

    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
        scaler,
    )
