#!/usr/bin/env python3
"""
baseline_nearest_condition.py — how much of the surrogate's R² is the split?

Every test simulation in this database has a TRAINING simulation at the same
(Mach, Pi), differing only in angle of attack: the median normalised distance
from a test condition to its nearest training condition is 0.033 of the flight
envelope.  The test set therefore measures interpolation between near neighbours,
not extrapolation, and a reader is entitled to ask what a model that does no
learning at all would score under the same protocol.

This computes that floor.  For each test simulation it finds the nearest training
simulation in normalised (Mach, AoA, Pi) and copies its surface field verbatim —
node i of the test sim is predicted by node i of the neighbour, which is exact
because every simulation shares the same 260,774-node mesh.

The number to report is the gap between this and the model, not the model's R²
on its own.

Usage:
  python baseline_nearest_condition.py                # held-out 140 sims
  python baseline_nearest_condition.py --all-sims     # all 156
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, N_P, N_TEST, SEED, TRAINING_CONFIG

COEFF_NAMES = ['Cp', 'Cfx', 'Cfy', 'Cfz']
MACH_THRESH = 0.75
GAMMA       = 1.4


def validation_sim_indices():
    rng = np.random.default_rng(SEED)
    return sorted(rng.choice(N_TEST, TRAINING_CONFIG['val_sims'],
                             replace=False).tolist())


def load_conditions():
    """(train_conditions, test_conditions) in dataset.csv order, as v2 indexes them."""
    rows = list(csv.DictReader(open(DATA_DIR / 'dataset.csv')))
    tr, te = [], []
    for r in rows:
        c = (float(r['Mach']), float(r['AoA']), float(r['Pi *1e-5']))
        (tr if r['Train'].strip().lower() == 'true' else te).append(c)
    return np.array(tr), np.array(te)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def cp_crit(mach):
    s = (2.0 / (GAMMA + 1.0)) * (1.0 + 0.5 * (GAMMA - 1.0) * mach ** 2)
    return (2.0 / (GAMMA * max(mach ** 2, 1e-6))) * (s ** (GAMMA / (GAMMA - 1.0)) - 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--all-sims', action='store_true',
                   help='Use all 156 test sims instead of the held-out 140.')
    args = p.parse_args()

    tr_cond, te_cond = load_conditions()
    val = set(validation_sim_indices())
    sims = ([i for i in range(N_TEST)] if args.all_sims
            else [i for i in range(N_TEST) if i not in val])
    print(f'Test simulations scored: {len(sims)}'
          + ('' if args.all_sims else f' (excluding the {len(val)} validation sims)'))

    # Nearest training condition, each axis normalised by the training range so
    # Mach, AoA and Pi contribute comparably.
    lo, hi = tr_cond.min(0), tr_cond.max(0)
    rng_   = np.where(hi - lo > 0, hi - lo, 1.0)
    trn    = (tr_cond - lo) / rng_
    ten    = (te_cond - lo) / rng_
    d      = np.sqrt(((ten[:, None, :] - trn[None, :, :]) ** 2).sum(-1))
    nn_idx = d.argmin(1)
    nn_d   = d.min(1)

    print(f'Normalised distance to the nearest training condition: '
          f'median {np.median(nn_d[sims]):.4f}, max {nn_d[sims].max():.4f}')

    Y_tr = np.load(DATA_DIR / 'Ytrain.npy', mmap_mode='r')
    Y_te = np.load(DATA_DIR / 'Ytest.npy',  mmap_mode='r')

    true_blocks, pred_blocks, mach_blocks = [], [], []
    for n, s in enumerate(sims):
        j  = int(nn_idx[s])
        yt = np.asarray(Y_te[s * N_P:(s + 1) * N_P], dtype=np.float64)
        yp = np.asarray(Y_tr[j * N_P:(j + 1) * N_P], dtype=np.float64)
        true_blocks.append(yt)
        pred_blocks.append(yp)
        mach_blocks.append(np.full(N_P, te_cond[s, 0]))
        if (n + 1) % 20 == 0:
            print(f'  {n + 1}/{len(sims)} sims')

    Y_true = np.vstack(true_blocks)
    Y_pred = np.vstack(pred_blocks)
    Mach   = np.concatenate(mach_blocks)

    cp_c  = np.array([cp_crit(m) for m in np.unique(Mach)])
    crit  = dict(zip(np.unique(Mach), cp_c))
    Cp_cr = np.array([crit[m] for m in Mach])
    trans = Mach > MACH_THRESH
    masks = {
        'global':    np.ones(len(Y_true), dtype=bool),
        'shock':     (Y_true[:, 0] < Cp_cr) & trans,
        'transonic': trans,
        'subsonic':  ~trans,
    }

    print(f'\n{"=" * 78}')
    print(f'NEAREST-TRAINING-CONDITION BASELINE — no learning, copies the neighbour')
    print(f'{"=" * 78}')
    hdr = f'{"Subset":<12}{"N pts":>13}  ' + '  '.join(f'{"R²(" + c + ")":>9}'
                                                       for c in COEFF_NAMES)
    print(hdr)
    print('-' * len(hdr))
    for name, m in masks.items():
        r2 = r2_score(Y_true[m], Y_pred[m])
        print(f'{name:<12}{m.sum():>13,}  ' + '  '.join(f'{v:9.4f}' for v in r2))

    print('\nCompare against AeroSurrogate on the same held-out sims:')
    print('  global    0.9729   0.9172   0.9027   0.9209')
    print('  shock     0.9350   0.8923   0.8692   0.9076')


if __name__ == '__main__':
    main()
