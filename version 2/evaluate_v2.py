#!/usr/bin/env python3
"""
evaluate_v2.py — Full evaluation of AeroSurrogate v2.

Evaluates on the full test set (all 156 sims × 260,774 nodes).
Also evaluates on the v1 comparable sample (SEED=42, fraction=0.1, 4,068,074 pts)
so the two versions can be compared in the same table.

Usage:
  python evaluate_v2.py                              # best checkpoint
  python evaluate_v2.py --ckpt path/to/model.pt
  python evaluate_v2.py --compare-v1                 # print v1 numbers alongside
"""
import os; os.environ['PAPER_NUM_WORKERS'] = '0'
import sys
import argparse
import numpy as np
import torch
from pathlib import Path


class _Tee:
    """Duplicates writes to stdout and a log file simultaneously."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

sys.path.insert(0, str(Path(__file__).parent))

from config import (DATA_DIR, CACHE_DIR, MODEL_DIR, RESULT_DIR,
                    MODEL_CONFIG, KNN_K, N_P, N_TEST, SEED)
from src.preprocessing import CFDPreprocessor
from src.models_v2 import AeroSurrogatev2
from src.dataset import SimulationDataset, load_sim_weights

COEFF_NAMES   = ['Cp', 'Cfx', 'Cfy', 'Cfz']
MACH_THRESH   = 0.75
TEST_FRACTION = 0.1   # for v1-comparable sample


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def mae_score(y_true, y_pred):
    return np.abs(y_true - y_pred).mean(axis=0)


@torch.no_grad()
def evaluate_full(model, test_ds, scaler, device):
    """Evaluate on all 156 test simulations."""
    model.eval()
    Y_std  = np.array(scaler['Y_std'],  dtype=np.float32)
    Y_mean = np.array(scaler['Y_mean'], dtype=np.float32)

    all_y_true, all_y_pred = [], []
    all_x_phys             = []

    for idx in range(len(test_ds)):
        batch      = test_ds[idx]
        x          = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_attr  = batch['edge_attr'].to(device)

        out    = model(x, edge_index, edge_attr)
        y_pred = out['pred'].cpu().numpy()           # [N_P, 4] normalised

        y_pred_phys = y_pred * Y_std[:4] + Y_mean[:4]
        y_true_phys = batch['y_phys'].numpy()

        # Reconstruct physical X (unnorm) for mask computation
        X_mean = np.array(scaler['X_mean'], dtype=np.float32)
        X_std  = np.array(scaler['X_std'],  dtype=np.float32)
        x_phys = batch['x'].numpy() * X_std + X_mean

        all_y_true.append(y_true_phys)
        all_y_pred.append(y_pred_phys)
        all_x_phys.append(x_phys)

        if (idx + 1) % 20 == 0:
            print(f"  Evaluated {idx+1}/{len(test_ds)} sims")

    Y_true = np.vstack(all_y_true)     # [N_test_total, 4]
    Y_pred = np.vstack(all_y_pred)
    X_phys = np.vstack(all_x_phys)

    return Y_true, Y_pred, X_phys


def get_masks(X_phys, Y_true):
    Mach   = X_phys[:, 6]
    Cp_crit= X_phys[:, 13]
    Cp_true= Y_true[:, 0]
    shock  = (Cp_true < Cp_crit) & (Mach > MACH_THRESH)
    trans  = Mach > MACH_THRESH
    return {
        'global':    np.ones(len(X_phys), dtype=bool),
        'shock':     shock & trans,
        'transonic': trans,
        'subsonic':  ~trans,
    }


def print_table(Y_true, Y_pred, X_phys, label):
    masks = get_masks(X_phys, Y_true)
    col_w = 38
    print(f'\n{"─"*100}')
    print(f'{label}')
    print(f'{"─"*100}')
    hdr = (f'{"Subset":<14}  {"N pts":>10}  ' +
           '  '.join(f'{"R²("+c+")":>9}' for c in COEFF_NAMES) +
           '  ' +
           '  '.join(f'{"MAE("+c+")":>10}' for c in COEFF_NAMES))
    print(hdr)
    print('-' * len(hdr))
    for name, mask in masks.items():
        n   = mask.sum()
        r2  = r2_score(Y_true[mask], Y_pred[mask])
        mae = mae_score(Y_true[mask], Y_pred[mask])
        r2s  = '  '.join(f'{v:9.4f}' for v in r2)
        maes = '  '.join(f'{v:10.6f}' for v in mae)
        print(f'{name:<14}  {n:>10,}  {r2s}  {maes}')
    return masks


def weighted_r2(Y_true, Y_pred, X_phys, test_ds):
    """ONERA-comparable weighted R² using confidence_weight_simple per sim."""
    ss_res_w = np.zeros(4)
    ss_tot_w = np.zeros(4)
    for s in range(N_TEST):
        sl     = slice(s * N_P, (s + 1) * N_P)
        yt, yp = Y_true[sl], Y_pred[sl]
        w      = float(test_ds.sim_weights.get(s, 1.0))
        ss_res_w += w * np.sum((yt - yp) ** 2, axis=0)
        ss_tot_w += w * np.sum((yt - yt.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res_w / (ss_tot_w + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--compare-v1', action='store_true')
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path    = RESULT_DIR / 'v2_evaluation.txt'
    log_file    = open(out_path, 'w')
    real_stdout = sys.stdout
    sys.stdout  = _Tee(real_stdout, log_file)
    try:
        _run(args, out_path)
    finally:
        sys.stdout = real_stdout
        log_file.close()
    print(f'\nResults saved to {out_path}')


def _run(args, out_path):
    device    = torch.device('cpu')
    ckpt_path = Path(args.ckpt) if args.ckpt else MODEL_DIR / 'surrogate_v2_best.pt'
    print(f"Checkpoint: {ckpt_path}")

    scaler_path = MODEL_DIR / 'scaler_v2.npy'
    spatial_path= MODEL_DIR / 'spatial_stats_v2.npy'
    scaler      = np.load(str(scaler_path),  allow_pickle=True).item()
    spatial_stats = np.load(str(spatial_path), allow_pickle=True).item()

    preprocessor = CFDPreprocessor(spatial_stats=spatial_stats)
    test_weights = load_sim_weights(DATA_DIR / 'dataset.csv', 'test')

    test_ds  = SimulationDataset('test',  CACHE_DIR, DATA_DIR,
                                  test_weights, preprocessor, scaler, k=KNN_K)

    model = AeroSurrogatev2(MODEL_CONFIG).to(device).eval()
    sd    = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print("\nEvaluating on full test set (156 sims × 260,774 nodes)...")
    Y_true, Y_pred, X_phys = evaluate_full(model, test_ds, scaler, device)

    print_table(Y_true, Y_pred, X_phys, f'AeroSurrogate v2 — full test set ({len(Y_true):,} pts)')

    # Weighted R² (ONERA metric)
    r2_w = weighted_r2(Y_true, Y_pred, X_phys, test_ds)
    print(f'\nWeighted R² (confidence_weight):  ' +
          '  '.join(f'R²({c})={r2_w[i]:.4f}' for i, c in enumerate(COEFF_NAMES)))

    # v1-comparable sample: same SEED=42, fraction=0.1
    print(f"\nBuilding v1-comparable sample (SEED={SEED}, fraction={TEST_FRACTION})...")
    n_total = len(Y_true)
    n_samp  = int(n_total * TEST_FRACTION)
    np.random.seed(SEED)
    idx_samp = np.sort(np.random.choice(n_total, n_samp, replace=False))
    print_table(Y_true[idx_samp], Y_pred[idx_samp], X_phys[idx_samp],
                f'AeroSurrogate v2 — v1-comparable sample ({n_samp:,} pts, '
                f'SEED={SEED}, frac={TEST_FRACTION})')


if __name__ == '__main__':
    main()
