#!/usr/bin/env python3
"""
evaluate_symbolic_gate_v2.py — Score the model with the neural ShockIndicator
replaced by the distilled closed-form expression.

This is the number the paper must report about symbolic distillation.  Fidelity
of the formula to p_s is an intermediate quantity; what a reader needs to know is
what the substitution costs in aerodynamic accuracy.

The v1 paper reported that the symbolic and neural gates scored identically to
four decimal places.  They do not: version 1/outputs/results/ablations.md records
why — SurrogateTrainer.train() reloaded a hardcoded "surrogate_best.pt" instead of
self.save_name, so the symbolic run was overwritten with the neural weights and
the "comparison" evaluated the same model twice.  This script exists so the v2
claim rests on an actual substitution.

What is substituted
  p_s enters the model in two places (models_v2.py:246-247): it is concatenated
  onto the gate input, and it scales the additive shock residual.  Both are driven
  from the symbolic value here, so nothing of the neural indicator survives in the
  Cp path.  The friction head never sees p_s and is therefore unaffected by
  construction — which the results below should confirm.

Usage
  python evaluate_symbolic_gate_v2.py                    # held-out 140 sims
  python evaluate_symbolic_gate_v2.py --all-sims
  python evaluate_symbolic_gate_v2.py --complexity 11    # a specific Pareto point
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault('PAPER_NUM_WORKERS', '0')

import numpy as np
import sympy
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (DATA_DIR, CACHE_DIR, MODEL_DIR, RESULT_DIR, MODEL_CONFIG,
                    TRAINING_CONFIG, KNN_K, N_P, N_TEST, SEED)
from src.preprocessing import CFDPreprocessor
from src.models_v2 import AeroSurrogatev2
from src.dataset import SimulationDataset, load_sim_weights

DEFAULT_CKPT = MODEL_DIR / 'surrogate_v2_moefix_long.pt'
SYMBOLIC_PKL = MODEL_DIR / 'shock_sensor_symbolic_v2.pkl'

COEFF_NAMES = ['Cp', 'Cfx', 'Cfy', 'Cfz']
MACH_THRESH = 0.75


def validation_sim_indices():
    rng = np.random.default_rng(SEED)
    return sorted(rng.choice(N_TEST, TRAINING_CONFIG['val_sims'],
                             replace=False).tolist())


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def mae_score(y_true, y_pred):
    return np.abs(y_true - y_pred).mean(axis=0)


def load_symbolic(pkl_path, complexity=None):
    """Return (callable over the SR feature block, expression string, complexity)."""
    with open(pkl_path, 'rb') as fh:
        d = pickle.load(fh)

    expr_str, cplx = d['expr'], d.get('complexity')
    if complexity is not None:
        match = [r for r in d.get('pareto', []) if r['complexity'] == complexity]
        if not match:
            avail = sorted(r['complexity'] for r in d.get('pareto', []))
            raise SystemExit(f'no Pareto entry of complexity {complexity}; have {avail}')
        expr_str, cplx = match[0]['sympy'], complexity

    names = d['features']
    syms  = sympy.symbols(names)
    expr  = sympy.sympify(expr_str)
    fn    = sympy.lambdify(syms, expr, modules='numpy')

    def evaluate(block):
        # block is [N, len(names)] of PHYSICAL features, columns in `names` order.
        out = fn(*[block[:, i] for i in range(block.shape[1])])
        out = np.asarray(out, dtype=np.float64)
        if out.ndim == 0:                      # a constant expression
            out = np.full(len(block), float(out))
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    return evaluate, expr_str, cplx, d


@torch.no_grad()
def run(model, test_ds, scaler, sim_indices, symbolic, sr_idx, device):
    """Evaluate twice per sim — neural p_s and symbolic p_s — sharing the backbone."""
    Y_std  = np.array(scaler['Y_std'],  dtype=np.float32)
    Y_mean = np.array(scaler['Y_mean'], dtype=np.float32)
    X_mean = np.array(scaler['X_mean'], dtype=np.float32)
    X_std  = np.array(scaler['X_std'],  dtype=np.float32)

    true, neu, sym, mach, ps_n, ps_s = [], [], [], [], [], []

    for n, idx in enumerate(sim_indices):
        b  = test_ds[idx]
        x  = b['x'].to(device)
        h  = model.backbone(x, b['edge_index'].to(device), b['edge_attr'].to(device))

        _, p_neural = model.shock_indicator(h)

        X_phys = b['x'].numpy() * X_std + X_mean
        p_sym  = symbolic(X_phys[:, sr_idx])
        p_sym_t = torch.from_numpy(p_sym.astype(np.float32)).unsqueeze(1)

        cp_n, _ = model.moe(h, p_neural)
        cp_s, _ = model.moe(h, p_sym_t)
        cf      = model.friction_head(h)

        pred_n = torch.cat([cp_n, cf], dim=-1).cpu().numpy() * Y_std[:4] + Y_mean[:4]
        pred_s = torch.cat([cp_s, cf], dim=-1).cpu().numpy() * Y_std[:4] + Y_mean[:4]

        true.append(b['y_phys'].numpy())
        neu.append(pred_n)
        sym.append(pred_s)
        mach.append(X_phys[:, 6])
        ps_n.append(p_neural.squeeze(-1).numpy())
        ps_s.append(p_sym)

        if (n + 1) % 20 == 0:
            print(f'  {n + 1}/{len(sim_indices)} sims', flush=True)

    return (np.vstack(true), np.vstack(neu), np.vstack(sym),
            np.concatenate(mach), np.concatenate(ps_n), np.concatenate(ps_s))


def subset_masks(Y_true, Mach, cp_crit_of):
    trans = Mach > MACH_THRESH
    crit  = np.array([cp_crit_of[m] for m in Mach])
    return {
        'global':    np.ones(len(Y_true), dtype=bool),
        'shock':     (Y_true[:, 0] < crit) & trans,
        'transonic': trans,
        'subsonic':  ~trans,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=str(DEFAULT_CKPT))
    p.add_argument('--symbolic', default=str(SYMBOLIC_PKL))
    p.add_argument('--complexity', type=int, default=None,
                   help='use a specific Pareto point instead of the highest-fidelity one')
    p.add_argument('--all-sims', action='store_true')
    p.add_argument('--sims', type=int, nargs='+', default=None, metavar='IDX',
                   help='score only these test sims (for quick checks)')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    device = torch.device('cpu')
    scaler = np.load(str(MODEL_DIR / 'scaler_v2.npy'), allow_pickle=True).item()
    spat   = np.load(str(MODEL_DIR / 'spatial_stats_v2.npy'), allow_pickle=True).item()
    pre    = CFDPreprocessor(spatial_stats=spat)
    test_ds = SimulationDataset('test', CACHE_DIR, DATA_DIR,
                                load_sim_weights(DATA_DIR / 'dataset.csv', 'test'),
                                pre, scaler, k=KNN_K)

    model = AeroSurrogatev2(MODEL_CONFIG).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device,
                                     weights_only=False), strict=False)

    symbolic, expr_str, cplx, meta = load_symbolic(Path(args.symbolic), args.complexity)
    sr_idx = meta['sr_idx']

    val = set(validation_sim_indices())
    if args.sims is not None:
        sims = args.sims
    elif args.all_sims:
        sims = list(range(len(test_ds)))
    else:
        sims = [i for i in range(len(test_ds)) if i not in val]

    lines = []
    def emit(s=''):
        print(s)
        lines.append(s)

    emit(f'Checkpoint: {args.ckpt}')
    emit(f'Symbolic expression (complexity {cplx}):')
    emit(f'  p_s = {expr_str}')
    emit(f'Features: {meta["features"]}')
    emit(f'\nSimulations: {len(sims)}'
         + ('' if args.all_sims else f' (held out; the {len(val)} selection sims excluded)'))

    Y_true, Y_neu, Y_sym, Mach, ps_n, ps_s = run(
        model, test_ds, scaler, sims, symbolic, sr_idx, device)

    g = 1.4
    cp_crit_of = {}
    for m in np.unique(Mach):
        s = (2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * m ** 2)
        cp_crit_of[m] = (2.0 / (g * max(m ** 2, 1e-6))) * (s ** (g / (g - 1.0)) - 1.0)
    masks = subset_masks(Y_true, Mach, cp_crit_of)

    emit(f'\nAgreement of the two gates on p_s itself: '
         f'R2 = {1.0 - np.sum((ps_n - ps_s) ** 2) / (np.sum((ps_n - ps_n.mean()) ** 2) + 1e-12):.4f}'
         f'   (neural mean {ps_n.mean():.4f}, symbolic mean {ps_s.mean():.4f})')

    emit(f'\n{"=" * 104}')
    emit('DOWNSTREAM COST OF THE SUBSTITUTION')
    emit(f'{"=" * 104}')
    hdr = (f'{"Subset":<11}{"Gate":<11}{"N pts":>13}  '
           + '  '.join(f'{"R2(" + c + ")":>9}' for c in COEFF_NAMES)
           + '  ' + '  '.join(f'{"MAE(" + c + ")":>10}' for c in COEFF_NAMES))
    emit(hdr)
    emit('-' * len(hdr))
    for name, m in masks.items():
        if not m.any():
            # An empty subset would otherwise print R2 = 1.0000 from 0/0, which
            # reads as a perfect score rather than as no data.
            emit(f'{name:<11}{"—":<11}{0:>13}  (no nodes in this subset)')
            emit()
            continue
        for tag, Y in (('neural', Y_neu), ('symbolic', Y_sym)):
            r = r2_score(Y_true[m], Y[m]); e = mae_score(Y_true[m], Y[m])
            emit(f'{name if tag == "neural" else "":<11}{tag:<11}{m.sum():>13,}  '
                 + '  '.join(f'{v:9.4f}' for v in r) + '  '
                 + '  '.join(f'{v:10.6f}' for v in e))
        rn = r2_score(Y_true[m], Y_neu[m]); rs = r2_score(Y_true[m], Y_sym[m])
        emit(f'{"":<11}{"delta":<11}{"":>13}  '
             + '  '.join(f'{v:+9.4f}' for v in (rs - rn)))
        emit()

    out = Path(args.out) if args.out else RESULT_DIR / 'symbolic_gate_v2_evaluation.txt'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
