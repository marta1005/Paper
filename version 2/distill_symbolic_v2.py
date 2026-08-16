#!/usr/bin/env python3
"""
distill_symbolic_v2.py — Distil the v2 ShockIndicator into a closed-form expression.

This is the interpretability claim of the paper, redone for the current model.

Why it is a harder and more interesting claim than the v1 version.  In v1 the
ShockIndicator read the 16 physical node features directly, so a symbolic
expression over those same features was fitting a function to itself in a
different basis.  In v2 the indicator reads h_i, the 256-dimensional latent
produced after eight rounds of message passing over the kNN graph — it has seen
an 8-hop neighbourhood before deciding.  Recovering that decision as an algebraic
expression over purely LOCAL physical features is therefore a real compression
claim: it says the neighbourhood context the GNN gathers is, for the purpose of
shock detection, summarisable in closed form.

Protocol
  * Targets are the model's own soft p_s, taken on TRAINING simulations.  The
    formula never sees the test set, so the downstream evaluation stays honest.
  * kNN graphs are built on the fly (~1.5 s/sim) rather than cached: the cache
    holds test graphs only, and 312 training graphs would cost ~18 GB on disk for
    a one-off fit.  Construction is copied verbatim from build_knn_cache.py so
    the edges are identical to the ones the model trained on.
  * Features offered to the search are physically interpretable ones only.  A
    formula in terms of a latent coordinate would be no more interpretable than
    the network it replaces.

Stage 2 (--evaluate) substitutes the discovered expression for the neural
indicator inside the full model and re-scores the held-out test set, which is the
number the paper must report — not the fidelity of the fit to p_s.

Usage
  python distill_symbolic_v2.py --sims 40                # extract + fit
  python distill_symbolic_v2.py --evaluate               # downstream substitution
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault('PAPER_NUM_WORKERS', '0')

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (DATA_DIR, CACHE_DIR, MODEL_DIR, RESULT_DIR, MODEL_CONFIG,
                    TRAINING_CONFIG, KNN_K, N_P, N_TRAIN, N_TEST, SEED)
from src.preprocessing import CFDPreprocessor
from src.models_v2 import AeroSurrogatev2
from src.dataset import SimulationDataset, load_sim_weights

DEFAULT_CKPT = MODEL_DIR / 'surrogate_v2_moefix_long.pt'
OUT_PKL      = MODEL_DIR / 'shock_sensor_symbolic_v2.pkl'
OUT_TXT      = RESULT_DIR / 'symbolic_v2_distillation.txt'

# Physically interpretable features offered to the symbolic search, as columns of
# the 16-feature derived array.  Same set v1 used, so the two formulas are
# directly comparable.
SR_FEATURES = ['Mach', 'AoA', 'x_norm', 'span_norm', 'nz',
               'Cp_crit', 'q_dyn', 'AoA_sin', 'L_factor']
SR_IDX      = [6, 7, 14, 15, 5, 13, 9, 11, 12]

COEFF_NAMES = ['Cp', 'Cfx', 'Cfy', 'Cfz']
MACH_THRESH = 0.75


def build_edges(X_sim, k=KNN_K):
    """kNN graph for one simulation — identical construction to build_knn_cache.py."""
    coords  = X_sim[:, :3].astype(np.float64)
    normals = X_sim[:, 3:6].astype(np.float32)

    tree        = cKDTree(coords)
    dists, inds = tree.query(coords, k=k + 1, workers=-1)
    dists, inds = dists[:, 1:], inds[:, 1:]

    src = np.repeat(np.arange(N_P, dtype=np.int32), k)
    dst = inds.ravel().astype(np.int32)
    d   = dists.ravel().astype(np.float32)

    dx = (coords[dst, 0] - coords[src, 0]).astype(np.float32)
    dy = (coords[dst, 1] - coords[src, 1]).astype(np.float32)
    dz = (coords[dst, 2] - coords[src, 2]).astype(np.float32)
    n_dot = (normals[src] * normals[dst]).sum(axis=1).astype(np.float32)

    return (np.stack([src, dst], axis=0),
            np.column_stack([dx, dy, dz, d, n_dot]))


def load_model(ckpt_path, device):
    model = AeroSurrogatev2(MODEL_CONFIG).to(device).eval()
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False), strict=False)
    return model


def r2(y, yp):
    return 1.0 - np.sum((y - yp) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)


# ── stage 1: extract soft targets and fit ────────────────────────────────────

@torch.no_grad()
def extract(model, scaler, preprocessor, n_sims, per_sim, rng, device):
    """Collect (physical features, p_s, true shock label) over training sims."""
    X_mmap = np.load(DATA_DIR / 'X_train.npy', mmap_mode='r')
    Y_mmap = np.load(DATA_DIR / 'Ytrain.npy',  mmap_mode='r')
    X_mean = np.array(scaler['X_mean'], dtype=np.float32)
    X_std  = np.array(scaler['X_std'],  dtype=np.float32)

    sims = np.linspace(0, N_TRAIN - 1, n_sims).astype(int)
    F, P, S = [], [], []

    for n, s in enumerate(sims):
        t0    = time.time()
        X_sim = np.asarray(X_mmap.reshape(N_TRAIN, N_P, 9)[s], dtype=np.float32)
        Y_sim = np.asarray(Y_mmap.reshape(N_TRAIN, N_P, 4)[s], dtype=np.float32)

        X_der  = preprocessor.compute_derived_features(X_sim)
        X_norm = (X_der - X_mean) / X_std
        ei, ea = build_edges(X_sim)

        h    = model.backbone(torch.from_numpy(X_norm),
                              torch.from_numpy(ei).long(),
                              torch.from_numpy(ea))
        _, p = model.shock_indicator(h)
        p    = p.squeeze(-1).numpy()

        shock = ((Y_sim[:, 0] < X_der[:, 13]) & (X_sim[:, 6] > MACH_THRESH))

        take = rng.choice(N_P, min(per_sim, N_P), replace=False)
        F.append(X_der[take][:, SR_IDX])
        P.append(p[take])
        S.append(shock[take])
        print(f'  sim {s:>3} ({n + 1}/{len(sims)})  '
              f'p_s mean {p.mean():.3f}  shock {100 * shock.mean():5.1f}%  '
              f'({time.time() - t0:.0f}s)', flush=True)

    return np.vstack(F), np.concatenate(P), np.concatenate(S)


def fit_pysr(X, y, niterations, maxsize, out_dir,
             populations=40, population_size=100, batch_size=10000):
    from pysr import PySRRegressor
    model = PySRRegressor(
        niterations=niterations,
        binary_operators=['+', '-', '*', '/'],
        unary_operators=['exp', 'tanh', 'sqrt', 'square'],
        populations=populations,
        population_size=population_size,
        maxsize=maxsize,
        elementwise_loss='loss(x, y) = (x - y)^2',
        model_selection='best',
        # Evaluating every candidate on the full sample would dominate the search;
        # PySR scores on random mini-batches instead, which is the standard way to
        # run it at this many points.
        batching=True,
        batch_size=batch_size,
        random_state=SEED,
        deterministic=True,
        parallelism='serial',
        progress=False,
        verbosity=1,
        output_directory=str(out_dir),
    )
    model.fit(X, y, variable_names=SR_FEATURES)
    return model


def stage_fit(args):
    device = torch.device('cpu')
    scaler = np.load(str(MODEL_DIR / 'scaler_v2.npy'), allow_pickle=True).item()
    spat   = np.load(str(MODEL_DIR / 'spatial_stats_v2.npy'), allow_pickle=True).item()
    pre    = CFDPreprocessor(spatial_stats=spat)
    model  = load_model(Path(args.ckpt), device)
    rng    = np.random.default_rng(SEED)

    cache = ROOT / 'outputs' / 'symbolic_targets.npz'
    print(f'Checkpoint: {args.ckpt}')
    if args.reuse_cache and cache.exists():
        d = np.load(cache)
        X, p, shock = d['X'], d['p'], d['shock']
        print(f'Reusing the cached sample from {cache.name}: {len(p):,} nodes')
    else:
        print(f'Extracting soft p_s over {args.sims} training simulations '
              f'({args.per_sim:,} nodes each)...')
        X, p, shock = extract(model, scaler, pre, args.sims, args.per_sim, rng, device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, p=p, shock=shock)
    print(f'\nCollected {len(p):,} nodes.  p_s mean {p.mean():.4f}  '
          f'true shock {100 * shock.mean():.1f}%')

    n_fit = min(args.fit_points, len(p))
    sel   = rng.choice(len(p), n_fit, replace=False)
    hold  = np.setdiff1d(np.arange(len(p)), sel)
    print(f'Fitting PySR on {n_fit:,} nodes, holding out {len(hold):,} for scoring.\n')

    t0  = time.time()
    sr  = fit_pysr(X[sel], p[sel], args.niterations, args.maxsize,
                   ROOT / 'outputs' / 'pysr',
                   populations=args.populations,
                   population_size=args.population_size,
                   batch_size=args.batch_size)
    dt  = time.time() - t0

    # PySR's own "best" trades loss against complexity by a fixed heuristic.  The
    # paper needs the fidelity/interpretability curve, so score every point on the
    # Pareto front on held-out nodes and let the trade-off be visible.
    print(f'\n{"=" * 92}')
    print(f'PARETO FRONT, each scored on {len(hold):,} held-out nodes '
          f'({dt / 60:.1f} min of search)')
    print(f'{"=" * 92}')
    print(f'{"cplx":>5}  {"R2 vs p_s":>10}  {"prec":>6}  {"rec":>6}  {"F1":>6}  equation')
    print('-' * 92)

    rows = []
    for i, row in sr.equations_.iterrows():
        try:
            pred = np.asarray(sr.predict(X[hold], index=i), dtype=np.float64)
        except Exception:
            continue
        fid_i = r2(p[hold], pred)
        pos   = pred > 0.5
        tp = (pos & shock[hold]).sum(); fp = (pos & ~shock[hold]).sum()
        fn = (~pos & shock[hold]).sum()
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        f1 = 2 * pr * rc / max(pr + rc, 1e-9)
        rows.append(dict(complexity=int(row['complexity']), fidelity=float(fid_i),
                         precision=float(pr), recall=float(rc), f1=float(f1),
                         equation=str(row['equation']),
                         sympy=str(row['sympy_format'])))
        print(f'{int(row["complexity"]):>5}  {fid_i:>10.4f}  {pr:>6.3f}  {rc:>6.3f}  '
              f'{f1:>6.3f}  {row["equation"]}')

    pick = max(rows, key=lambda r: r['fidelity']) if rows else None
    best = sr.get_best()
    expr = pick['sympy'] if pick else str(best['sympy_format'])
    fid  = pick['fidelity'] if pick else float('nan')
    prec = pick['precision'] if pick else float('nan')
    rec  = pick['recall'] if pick else float('nan')

    print(f'\nHighest fidelity: complexity {pick["complexity"]}, R2 = {fid:.4f}')
    print(f'  {expr}')

    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, 'wb') as fh:
        pickle.dump({'expr': expr, 'features': SR_FEATURES, 'sr_idx': SR_IDX,
                     'complexity': pick['complexity'] if pick else None,
                     'fidelity_r2': float(fid), 'pareto': rows}, fh)
    print(f'\nSaved {OUT_PKL}')

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_TXT, 'w') as fh:
        fh.write(f'Checkpoint: {args.ckpt}\n')
        fh.write(f'Training sims sampled: {args.sims}, nodes: {len(p):,}\n')
        fh.write(f'Features: {SR_FEATURES}\n\n')
        fh.write(f'HIGHEST-FIDELITY EXPRESSION (complexity '
                 f'{pick["complexity"] if pick else "?"}):\n{expr}\n\n')
        fh.write(f'Fidelity to neural p_s (held-out nodes): R2 = {fid:.4f}\n')
        fh.write(f'Against true shock label: precision {prec:.4f} recall {rec:.4f}\n\n')
        fh.write(f'PARETO FRONT, each scored on {len(hold):,} held-out nodes\n')
        fh.write(f'{"cplx":>5}  {"R2":>9}  {"prec":>6}  {"rec":>6}  {"F1":>6}  equation\n')
        for r in rows:
            fh.write(f'{r["complexity"]:>5}  {r["fidelity"]:>9.4f}  '
                     f'{r["precision"]:>6.3f}  {r["recall"]:>6.3f}  {r["f1"]:>6.3f}  '
                     f'{r["equation"]}\n')
    print(f'Saved {OUT_TXT}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=str(DEFAULT_CKPT))
    p.add_argument('--sims', type=int, default=40,
                   help='training simulations to sample (default 40)')
    p.add_argument('--per-sim', type=int, default=20000,
                   help='nodes sampled per simulation (default 20000)')
    p.add_argument('--fit-points', type=int, default=200000,
                   help='nodes actually given to PySR (default 200000)')
    p.add_argument('--niterations', type=int, default=60)
    p.add_argument('--maxsize', type=int, default=20)
    p.add_argument('--populations', type=int, default=40)
    p.add_argument('--population-size', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=10000,
                   help='nodes each candidate is scored on; too small makes fitness '
                        'noisy and the search stalls on shallow expressions')
    p.add_argument('--reuse-cache', action='store_true',
                   help='reuse outputs/symbolic_targets.npz instead of re-extracting')
    args = p.parse_args()
    stage_fit(args)


if __name__ == '__main__':
    main()
