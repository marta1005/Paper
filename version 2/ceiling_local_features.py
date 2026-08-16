#!/usr/bin/env python3
"""
ceiling_local_features.py — is p_s recoverable from local features AT ALL?

The symbolic search plateaus at R^2 ~= 0.37 against the model's p_s, and barely
improves from complexity 8 to 22.  Two explanations fit that shape:

  (a) symbolic regression is the wrong tool / the budget was too small, or
  (b) the information simply is not in the local physical features, because the
      ShockIndicator reads a latent built from eight rounds of message passing.

They are distinguishable.  Fit a deliberately over-powered black box — gradient
boosting, then a small MLP — to exactly the same features and the same targets.
Whatever they reach is the ceiling for ANY function of these features, symbolic
or not.  If the ceiling is also ~0.37, (b) is established and the negative result
is about the physics, not about PySR.  If the ceiling is far higher, (a) holds and
the search needs more budget before anything can be concluded.

Usage:
  python ceiling_local_features.py --sims 15
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('PAPER_NUM_WORKERS', '0')

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import DATA_DIR, MODEL_DIR, RESULT_DIR, MODEL_CONFIG, N_P, N_TRAIN, SEED
from src.preprocessing import CFDPreprocessor
from src.models_v2 import AeroSurrogatev2
from distill_symbolic_v2 import build_edges, SR_FEATURES, SR_IDX, MACH_THRESH

CACHE_NPZ = ROOT / 'outputs' / 'symbolic_targets.npz'


def r2(y, yp):
    return 1.0 - np.sum((y - yp) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)


@torch.no_grad()
def extract(n_sims, per_sim):
    scaler = np.load(str(MODEL_DIR / 'scaler_v2.npy'), allow_pickle=True).item()
    spat   = np.load(str(MODEL_DIR / 'spatial_stats_v2.npy'), allow_pickle=True).item()
    pre    = CFDPreprocessor(spatial_stats=spat)
    model  = AeroSurrogatev2(MODEL_CONFIG).eval()
    model.load_state_dict(torch.load(MODEL_DIR / 'surrogate_v2_moefix_long.pt',
                                     map_location='cpu', weights_only=False), strict=False)

    X_mmap = np.load(DATA_DIR / 'X_train.npy', mmap_mode='r')
    Y_mmap = np.load(DATA_DIR / 'Ytrain.npy',  mmap_mode='r')
    X_mean = np.array(scaler['X_mean'], dtype=np.float32)
    X_std  = np.array(scaler['X_std'],  dtype=np.float32)
    rng    = np.random.default_rng(SEED)

    F, P, S = [], [], []
    for n, s in enumerate(np.linspace(0, N_TRAIN - 1, n_sims).astype(int)):
        t0    = time.time()
        X_sim = np.asarray(X_mmap.reshape(N_TRAIN, N_P, 9)[s], dtype=np.float32)
        Y_sim = np.asarray(Y_mmap.reshape(N_TRAIN, N_P, 4)[s], dtype=np.float32)
        X_der = pre.compute_derived_features(X_sim)
        ei, ea = build_edges(X_sim)
        h = model.backbone(torch.from_numpy((X_der - X_mean) / X_std),
                           torch.from_numpy(ei).long(), torch.from_numpy(ea))
        _, p = model.shock_indicator(h)
        p = p.squeeze(-1).numpy()
        shock = ((Y_sim[:, 0] < X_der[:, 13]) & (X_sim[:, 6] > MACH_THRESH))
        take = rng.choice(N_P, min(per_sim, N_P), replace=False)
        F.append(X_der[take][:, SR_IDX]); P.append(p[take]); S.append(shock[take])
        print(f'  sim {s:>3} ({n+1}/{n_sims})  ({time.time()-t0:.0f}s)', flush=True)
    return np.vstack(F), np.concatenate(P), np.concatenate(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sims', type=int, default=15)
    ap.add_argument('--per-sim', type=int, default=20000)
    ap.add_argument('--refresh', action='store_true', help='ignore the cached sample')
    args = ap.parse_args()

    if CACHE_NPZ.exists() and not args.refresh:
        d = np.load(CACHE_NPZ)
        X, p, shock = d['X'], d['p'], d['shock']
        print(f'Reusing cached sample: {len(p):,} nodes')
    else:
        print(f'Extracting p_s over {args.sims} training simulations...')
        X, p, shock = extract(args.sims, args.per_sim)
        CACHE_NPZ.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE_NPZ, X=X, p=p, shock=shock)
        print(f'Cached to {CACHE_NPZ}')

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(p))
    cut = int(0.75 * len(p))
    tr, te = idx[:cut], idx[cut:]
    print(f'\n{len(tr):,} train / {len(te):,} test nodes, {X.shape[1]} local features')

    lines = [f'Features: {SR_FEATURES}',
             f'Nodes: {len(p):,}  ({len(tr):,} train / {len(te):,} test)',
             f'Target: p_s from the GNN-latent ShockIndicator', '']

    def report(name, pred):
        v = r2(p[te], pred)
        pos = pred > 0.5
        tp = (pos & shock[te]).sum(); fp = (pos & ~shock[te]).sum()
        fn = (~pos & shock[te]).sum()
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        f1 = 2 * pr * rc / max(pr + rc, 1e-9)
        line = f'{name:<34} R2 = {v:6.4f}   F1 vs true shock = {f1:.3f}'
        print(line); lines.append(line)
        return v

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    print('\nFitting the ceiling models...')
    t0 = time.time()
    gb = HistGradientBoostingRegressor(max_iter=400, max_depth=None,
                                       learning_rate=0.1, random_state=SEED)
    gb.fit(X[tr], p[tr])
    r_gb = report('Gradient boosting (400 trees)', gb.predict(X[te]))
    print(f'  ({time.time()-t0:.0f}s)')

    t0 = time.time()
    sc = StandardScaler().fit(X[tr])
    mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=60,
                       random_state=SEED, early_stopping=True)
    mlp.fit(sc.transform(X[tr]), p[tr])
    r_mlp = report('MLP 256-128-64 on the same features', mlp.predict(sc.transform(X[te])))
    print(f'  ({time.time()-t0:.0f}s)')

    r_sym = 0.3671
    lines.append('')
    lines.append(f'{"Best symbolic expression (PySR)":<34} R2 = {r_sym:6.4f}')
    ceiling = max(r_gb, r_mlp)
    lines.append('')
    verdict = (
        'CEILING REACHED: a flexible black box on the same local features does no\n'
        'better than the closed-form expression. The information the ShockIndicator\n'
        'uses is genuinely not present in local physical features — it comes from the\n'
        'eight-hop neighbourhood the GNN aggregates. The negative distillation result\n'
        'is a statement about the physics, not about symbolic regression.'
        if ceiling - r_sym < 0.10 else
        'HEADROOM REMAINS: a black box beats the expression by a wide margin, so the\n'
        'local features do carry the signal and the symbolic search was underpowered.\n'
        'Increase niterations/maxsize before concluding anything.')
    print(f'\nCeiling for ANY function of these features: R2 = {ceiling:.4f}')
    print(f'Symbolic expression reaches:                R2 = {r_sym:.4f}')
    print(f'Gap: {ceiling - r_sym:.4f}\n')
    print(verdict)
    lines.append(f'Ceiling for any function of these features: R2 = {ceiling:.4f}')
    lines.append(f'Gap to the symbolic expression: {ceiling - r_sym:.4f}')
    lines.append('')
    lines.append(verdict)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / 'symbolic_v2_ceiling.txt'
    out.write_text('\n'.join(lines) + '\n')
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
