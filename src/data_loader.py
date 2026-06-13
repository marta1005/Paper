"""
Data loader optimizado para archivos NPY enormes (81M puntos)
Usa memory mapping y sampling estratégico
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from config import DATA_CONFIG, TRAINING_CONFIG, PREPROCESSING_CONFIG

logger = logging.getLogger(__name__)


class CFDDataset(Dataset):
    """
    Dataset para datos CFD
    Usa memory mapping para evitar cargar todo en RAM
    """
    
    def __init__(self, X, Y, normalize=True, scaler=None):
        """
        Args:
            X: array (n_samples, 9) o ruta a .npy
            Y: array (n_samples, 4) o ruta a .npy
            normalize: bool, normalizar datos
            scaler: dict con estadísticas para normalización
        """
        # Cargar datos
        if isinstance(X, str):
            self.X = np.load(X, mmap_mode='r')  # Memory mapping
        else:
            self.X = X
            
        if isinstance(Y, str):
            self.Y = np.load(Y, mmap_mode='r')
        else:
            self.Y = Y
        
        logger.info(f"Dataset loaded: X.shape={self.X.shape}, Y.shape={self.Y.shape}")
        
        self.normalize = normalize
        self.scaler = scaler
        
        # Normalización
        if normalize and scaler is None:
            logger.info("Computing normalization statistics...")
            self.scaler = self._compute_scaler()
    
    def _compute_scaler(self):
        """Computa media y std (eficiente para arrays grandes)"""
        scaler = {}
        
        # Inputs
        X_sample = self.X[::1000]  # Sample cada 1000 puntos
        scaler['X_mean'] = np.mean(self.X, axis=0, dtype=np.float32)
        scaler['X_std'] = np.std(self.X, axis=0, dtype=np.float32)
        scaler['X_std'][scaler['X_std'] == 0] = 1.0  # Evitar división por cero
        
        # Outputs
        scaler['Y_mean'] = np.mean(self.Y, axis=0, dtype=np.float32)
        scaler['Y_std'] = np.std(self.Y, axis=0, dtype=np.float32)
        scaler['Y_std'][scaler['Y_std'] == 0] = 1.0
        
        return scaler
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx].astype(np.float32)
        Y = self.Y[idx].astype(np.float32)
        
        # Normalizar si aplica
        if self.normalize and self.scaler:
            X = (X - self.scaler['X_mean']) / self.scaler['X_std']
            Y = (Y - self.scaler['Y_mean']) / self.scaler['Y_std']
        
        return torch.from_numpy(X), torch.from_numpy(Y)


def load_data_with_sampling(sample_fraction=1.0):
    """
    Carga datos con sampling estratégico
    
    Args:
        sample_fraction: fracción de datos a usar (0-1)
    
    Returns:
        (X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler)
    """
    logger.info("Loading data files...")
    
    # Intentar cargar datos con features derivados (precompilados)
    # Si no existen, usar datos originales (9 features)
    from pathlib import Path
    data_dir = Path(DATA_CONFIG['X_train_path']).parent
    
    X_train_derived_path = data_dir / "X_train_derived.npy"
    X_test_derived_path = data_dir / "X_test_derived.npy"
    
    if X_train_derived_path.exists() and X_test_derived_path.exists():
        logger.info("Loading PREPROCESSED data (19 features)...")
        X_train_full = np.load(X_train_derived_path, mmap_mode='r')
        X_test_full = np.load(X_test_derived_path, mmap_mode='r')
    else:
        logger.info("Loading ORIGINAL data (9 features)...")
        X_train_full = np.load(DATA_CONFIG['X_train_path'], mmap_mode='r')
        X_test_full = np.load(DATA_CONFIG['X_test_path'], mmap_mode='r')
    
    # Cargar labels con memory mapping
    Y_train_full = np.load(DATA_CONFIG['Y_train_path'], mmap_mode='r')
    Y_test_full = np.load(DATA_CONFIG['Y_test_path'], mmap_mode='r')
    
    logger.info(f"Raw sizes: X_train={X_train_full.shape}, Y_train={Y_train_full.shape}")
    
    # Sampling estratégico
    n_train = len(X_train_full)
    n_sample = int(n_train * sample_fraction)
    
    if sample_fraction < 1.0:
        indices = np.random.choice(n_train, size=n_sample, replace=False)
        indices = np.sort(indices)
        logger.info(f"Sampling {n_sample} from {n_train} training samples ({100*sample_fraction:.1f}%)")
        
        X_train = X_train_full[indices]
        Y_train = Y_train_full[indices]
    else:
        X_train = np.array(X_train_full)
        Y_train = np.array(Y_train_full)
    
    # Convertir a arrays de memoria normal (no memory map) después del sampling
    X_train = np.asarray(X_train, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.float32)
    X_test_full = np.asarray(X_test_full, dtype=np.float32)
    Y_test_full = np.asarray(Y_test_full, dtype=np.float32)
    
    # Train/Val split
    n_val = int(len(X_train) * DATA_CONFIG.get('val_split', 0.1))
    X_val = X_train[-n_val:]
    Y_val = Y_train[-n_val:]
    X_train = X_train[:-n_val]
    Y_train = Y_train[:-n_val]
    
    # Test sampling
    test_sample = DATA_CONFIG.get('test_sample_fraction', 1.0)
    if test_sample < 1.0:
        n_test = int(len(X_test_full) * test_sample)
        indices_test = np.random.choice(len(X_test_full), size=n_test, replace=False)
        X_test = X_test_full[indices_test]
        Y_test = Y_test_full[indices_test]
    else:
        X_test = X_test_full
        Y_test = Y_test_full
    
    logger.info(f"Final sizes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def get_dataloaders(sample_fraction=None):
    """
    Crea DataLoaders para train/val/test
    
    Args:
        sample_fraction: fracción de datos (si None, usa config)
    
    Returns:
        (train_loader, val_loader, test_loader, scaler)
    """
    if sample_fraction is None:
        sample_fraction = DATA_CONFIG.get('train_sample_fraction', 0.1)
    
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data_with_sampling(
        sample_fraction=sample_fraction
    )
    
    # Crear scaler a partir de train
    train_dataset = CFDDataset(
        X_train, Y_train,
        normalize=PREPROCESSING_CONFIG['normalize_inputs'],
        scaler=None
    )
    scaler = train_dataset.scaler
    
    # Aplicar mismo scaler a val y test
    val_dataset = CFDDataset(
        X_val, Y_val,
        normalize=PREPROCESSING_CONFIG['normalize_inputs'],
        scaler=scaler
    )
    
    test_dataset = CFDDataset(
        X_test, Y_test,
        normalize=PREPROCESSING_CONFIG['normalize_inputs'],
        scaler=scaler
    )
    
    num_workers = TRAINING_CONFIG.get('num_workers', 0)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    
    return train_loader, val_loader, test_loader, scaler


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    
    train_loader, val_loader, test_loader, scaler = get_dataloaders(sample_fraction=0.05)
    
    # Test
    for X_batch, Y_batch in train_loader:
        print(f"Batch: X.shape={X_batch.shape}, Y.shape={Y_batch.shape}")
        break
