"""
dataset.py — SimulationDataset for AeroSurrogate v2.

Each sample is one full CFD simulation (260,774 nodes).
kNN edges are loaded from the pre-built cache (see build_knn_cache.py).
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class SimulationDataset(Dataset):
    """
    Returns one simulation at a time.

    Item dict:
      x          [N_P, 16]   derived node features (float32)
      y          [N_P, 4]    target coefficients (float32)
      edge_index [2, E]      kNN edge indices (int64)
      edge_attr  [E, 5]      edge features: dx,dy,dz,dist,n_dot (float32)
      y_shock    [N_P]       binary shock label (float32)
      weight     scalar      confidence_weight_simple from dataset.csv
      sim_idx    int
    """

    def __init__(self, split, cache_dir, data_dir, sim_weights,
                 preprocessor, scaler, k=8):
        assert split in ('train', 'test')
        self.split       = split
        self.cache_dir   = Path(cache_dir)
        self.k           = k
        self.scaler      = scaler
        self.sim_weights = sim_weights  # dict {sim_idx: weight}

        n_sims = 312 if split == 'train' else 156
        n_p    = 260_774

        # Memory-mapped raw arrays
        key = 'train' if split == 'train' else 'test'
        self.X_raw = np.load(
            str(Path(data_dir) / f'X_{key}.npy'), mmap_mode='r'
        ).reshape(n_sims, n_p, 9)
        self.Y_raw = np.load(
            str(Path(data_dir) / f'Y{key}.npy'), mmap_mode='r'
        ).reshape(n_sims, n_p, 4)

        self.preprocessor = preprocessor
        self.n_sims       = n_sims
        self.n_p          = n_p

        # Cp_crit col index in derived features (col 13)
        self._cp_crit_col = 13
        self._mach_col    = 6   # raw col

    def __len__(self):
        return self.n_sims

    def __getitem__(self, idx):
        X_sim = np.array(self.X_raw[idx], dtype=np.float32)  # [N_P, 9]  (copy: mmap→writable)
        Y_sim = np.array(self.Y_raw[idx], dtype=np.float32)  # [N_P, 4]

        # Derive 16 features (preprocessing identical to v1)
        X_derived = self.preprocessor.compute_derived_features(X_sim)  # [N_P, 16]

        # Normalise node features
        X_mean = np.array(self.scaler['X_mean'], dtype=np.float32)
        X_std  = np.array(self.scaler['X_std'],  dtype=np.float32)
        Y_mean = np.array(self.scaler['Y_mean'], dtype=np.float32)
        Y_std  = np.array(self.scaler['Y_std'],  dtype=np.float32)

        X_norm = (X_derived - X_mean) / X_std
        Y_norm = (Y_sim - Y_mean) / Y_std

        # Shock label: Cp_RANS < Cp_crit AND Mach > 0.75
        Cp_true  = Y_sim[:, 0]
        Cp_crit  = X_derived[:, self._cp_crit_col]
        Mach_raw = X_sim[:, self._mach_col]
        y_shock  = ((Cp_true < Cp_crit) & (Mach_raw > 0.75)).astype(np.float32)

        # Shock weight per node: 1 + 5 * shock_label (× sim confidence weight later)
        shock_node_w = 1.0 + 5.0 * y_shock   # [N_P]

        # Load cached edges
        edge_index, edge_attr = self._load_edges(idx)

        weight = float(self.sim_weights.get(idx, 1.0))

        return {
            'x':           torch.from_numpy(X_norm),
            'y':           torch.from_numpy(Y_norm),
            'y_phys':      torch.from_numpy(Y_sim),
            'edge_index':  torch.from_numpy(edge_index).long(),
            'edge_attr':   torch.from_numpy(edge_attr),
            'y_shock':     torch.from_numpy(y_shock),
            'shock_node_w':torch.from_numpy(shock_node_w),
            'weight':      weight,
            'sim_idx':     idx,
        }

    def _load_edges(self, idx):
        ei_path = self.cache_dir / f'edges_{self.split}_s{idx:03d}_k{self.k}.npy'
        ea_path = self.cache_dir / f'edge_attr_{self.split}_s{idx:03d}_k{self.k}.npy'
        if not ei_path.exists():
            raise FileNotFoundError(
                f"kNN cache missing: {ei_path}\n"
                f"Run: python build_knn_cache.py --split {self.split}"
            )
        edge_index = np.load(str(ei_path))   # [2, E]
        edge_attr  = np.load(str(ea_path))   # [E, 5]
        return edge_index, edge_attr


def load_sim_weights(dataset_csv, split):
    """
    Returns dict {local_sim_idx: confidence_weight_simple}.
    Sim indices are relative to the split (0-based).
    """
    import pandas as pd
    df     = pd.read_csv(dataset_csv, index_col=0)
    is_train = split == 'train'
    sub    = df[df['Train'] == is_train].reset_index(drop=True)
    return {i: float(sub.loc[i, 'confidence_weight_simple']) for i in range(len(sub))}


def collate_single(batch):
    """Collate fn for batch_size=1 (one simulation per step)."""
    assert len(batch) == 1
    return batch[0]
