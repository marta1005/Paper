#!/usr/bin/env python3
"""
evaluate_subsets.py — R², MAE on subsets of the held-out test set.

Reproduces EXACTLY the 4,068,074-point sample used in the paper:
  SEED=42, fraction=0.1 of 40,680,744 total test points.

Subsets evaluated:
  global     — all sampled points
  shock      — y_shock=1[Cp<Cp_crit] AND Mach>0.75
  transonic  — Mach>0.75
  subsonic   — Mach≤0.75

Usage:
  python evaluate_subsets.py                          # neural + symbolic (paper models)
  python evaluate_subsets.py --ckpt path/to/foo.pt --label "ablation-X"
  python evaluate_subsets.py --gate-mode constant:0.0 --ckpt outputs/models/surrogate_gate_const0.0_best.pt
"""
import os; os.environ['PAPER_NUM_WORKERS'] = '0'
import argparse
import pickle
import numpy as np
import torch
from pathlib import Path

from config import MODEL_DIR, MODEL_CONFIG, DATA_CONFIG, RESULT_DIR
from src.models import AeroSurrogate, PySRWrapper  # noqa: F401

SEED              = 42
TEST_FRACTION     = 0.1
COEFF_NAMES       = ['Cp', 'Cfx', 'Cfy', 'Cfz']
MACH_THRESH       = 0.75
DEFAULT_SYM_PKL   = MODEL_DIR / 'shock_sensor_symbolic_surrogate_base.pkl'


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_test_sample():
    data_dir = Path(DATA_CONFIG['X_train_path']).parent
    X_full   = np.load(str(data_dir / 'X_test_derived.npy'), mmap_mode='r')
    Y_full   = np.load(str(DATA_CONFIG['Y_test_path']),      mmap_mode='r')

    n_total = len(X_full)
    n_load  = int(n_total * TEST_FRACTION)

    # Reproduce paper's sampling: seed then np.random.choice (legacy API)
    np.random.seed(SEED)
    idx = np.sort(np.random.choice(n_total, n_load, replace=False))

    X = np.asarray(X_full[idx], dtype=np.float32)
    Y = np.asarray(Y_full[idx], dtype=np.float32)
    print(f"  Loaded {len(X):,} points  (SEED={SEED}, fraction={TEST_FRACTION})")
    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# Label and subset masks
# ─────────────────────────────────────────────────────────────────────────────

def get_masks(X_phys, Y_phys):
    """Return boolean masks for each subset."""
    Mach_real  = X_phys[:, 6]          # physical Mach (X_test_derived not normalised)
    Cp_real    = Y_phys[:, 0]          # physical Cp
    Cp_crit    = X_phys[:, 13]         # physical Cp_crit (derived col 13)
    shock      = Cp_real < Cp_crit     # y_shock label
    trans      = Mach_real > MACH_THRESH

    return {
        'global':    np.ones(len(X_phys), dtype=bool),
        'shock':     shock & trans,
        'transonic': trans,
        'subsonic':  ~trans,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model loading and inference
# ─────────────────────────────────────────────────────────────────────────────

def build_model(disable_shock_expert=False):
    cfg = MODEL_CONFIG['surrogate']
    return AeroSurrogate(
        in_dim=MODEL_CONFIG['autoencoder']['input_dim'],
        num_experts=cfg['num_experts'],
        output_dim=cfg['output_dim'],
        indicator_hidden=cfg.get('indicator_hidden'),
        expert_hidden=cfg.get('expert_hidden'),
        shock_expert_hidden=cfg.get('shock_expert_hidden'),
        disable_shock_expert=disable_shock_expert,
    )


def load_checkpoint(ckpt_path, device, disable_shock_expert=False):
    model = build_model(disable_shock_expert=disable_shock_expert)
    sd    = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def predict_batch(model, X_norm, device, batch_size=8192,
                  gate_mode='neural', symbolic_sensor=None, X_phys=None):
    """Run inference. gate_mode controls how shock_prob is sourced."""
    preds = []
    for i in range(0, len(X_norm), batch_size):
        xb = torch.from_numpy(X_norm[i:i+batch_size]).float().to(device)

        if gate_mode.startswith('constant:'):
            val = float(gate_mode.split(':')[1])
            sp  = torch.full((len(xb), 1), val, device=device)
            pred, _ = model.moe(xb, sp)

        elif gate_mode == 'random':
            # Fixed seed for deterministic eval
            g  = torch.Generator()
            g.manual_seed(SEED + i)
            sp = torch.rand(len(xb), 1, generator=g)
            pred, _ = model.moe(xb, sp)

        elif gate_mode == 'symbolic' and symbolic_sensor is not None:
            sr_idx  = symbolic_sensor['sr_idx']
            X_raw_b = X_phys[i:i+batch_size][:, sr_idx]
            proba   = symbolic_sensor['clf'].predict_proba(X_raw_b)[:, 1]
            sp_cal  = symbolic_sensor['calibrator'].predict(proba).astype(np.float32)
            sp_t    = torch.from_numpy(sp_cal[:, None]).to(device)
            pred, _ = model.moe(xb, sp_t)

        else:  # neural
            out  = model(xb)
            pred = out['pred']

        preds.append(pred.cpu().numpy())
    return np.vstack(preds)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def mae_score(y_true, y_pred):
    return np.abs(y_true - y_pred).mean(axis=0)


def evaluate_one(label, ckpt_path, X_phys, Y_phys, scaler, device,
                 gate_mode='neural', symbolic_sensor=None,
                 disable_shock_expert=False):
    print(f"\n  [{label}]  ckpt={Path(ckpt_path).name}  gate={gate_mode}")
    model = load_checkpoint(ckpt_path, device,
                            disable_shock_expert=disable_shock_expert)

    X_mean = np.array(scaler['X_mean'], dtype=np.float32)
    X_std  = np.array(scaler['X_std'],  dtype=np.float32)
    Y_mean = np.array(scaler['Y_mean'], dtype=np.float32)
    Y_std  = np.array(scaler['Y_std'],  dtype=np.float32)

    n_feat = len(X_mean)
    X_norm = (X_phys[:, :n_feat] - X_mean) / X_std

    pred_norm = predict_batch(model, X_norm, device,
                              gate_mode=gate_mode,
                              symbolic_sensor=symbolic_sensor,
                              X_phys=X_phys)
    Y_pred = pred_norm * Y_std[:4] + Y_mean[:4]
    Y_true = Y_phys[:, :4]

    masks   = get_masks(X_phys, Y_phys)
    results = {}
    for subset, mask in masks.items():
        n = mask.sum()
        if n == 0:
            results[subset] = {'r2': np.zeros(4), 'mae': np.zeros(4), 'n': 0}
            continue
        r2  = r2_score(Y_true[mask], Y_pred[mask])
        mae = mae_score(Y_true[mask], Y_pred[mask])
        results[subset] = {'r2': r2, 'mae': mae, 'n': n}
        r2_str = '  '.join(f'{v:.4f}' for v in r2)
        print(f"    {subset:<12} n={n:>9,}   R²: {r2_str}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_table(all_results):
    """Markdown-compatible table."""
    col_w = 30
    sub_w = 12
    n_w   = 10
    c_w   = 8

    hdr = (f"{'Model':<{col_w}}  {'Subset':<{sub_w}}  {'N pts':>{n_w}}  " +
           '  '.join(f"{'R²('+c+')':>{c_w}}" for c in COEFF_NAMES) +
           '    ' +
           '  '.join(f"{'MAE('+c+')':>{c_w}}" for c in COEFF_NAMES))
    sep = '-' * len(hdr)

    lines = [hdr, sep]
    for model_name, subsets in all_results.items():
        for subset_name in ['global', 'shock', 'transonic', 'subsonic']:
            m   = subsets[subset_name]
            r2s = '  '.join(f'{v:{c_w}.4f}' for v in m['r2'])
            ms  = '  '.join(f'{v:{c_w}.4f}' for v in m['mae'])
            lines.append(
                f"{model_name:<{col_w}}  {subset_name:<{sub_w}}  {m['n']:{n_w},}  "
                f"{r2s}    {ms}"
            )
        lines.append('')
    return '\n'.join(lines)


def format_ablation_md(all_results):
    """Compact markdown for paper / ablation table.
    Columns: R²(Cp) global/shock, R²(Cfy) global (largest delta), R²(Cfx) global.
    """
    lines = [
        '| Model | R²(Cp) global | R²(Cp) shock | R²(Cfy) global | R²(Cfx) global | MAE(Cp) shock |',
        '|---|---|---|---|---|---|',
    ]
    for name, subsets in all_results.items():
        g  = subsets['global']
        sh = subsets['shock']
        lines.append(
            f"| {name} "
            f"| {g['r2'][0]:.4f} "
            f"| {sh['r2'][0]:.4f} "
            f"| {g['r2'][2]:.4f} "
            f"| {g['r2'][1]:.4f} "
            f"| {sh['mae'][0]:.4f} |"
        )
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',      nargs='+', default=None,
                        help='Extra checkpoint(s) to evaluate')
    parser.add_argument('--label',     nargs='+', default=None,
                        help='Label(s) for extra checkpoints')
    parser.add_argument('--gate-mode', nargs='+', default=None,
                        help='Gate mode for extra checkpoints (neural/symbolic/constant:X/random)')
    parser.add_argument('--no-shock-expert', action='store_true',
                        help='Load checkpoint as no-shock-expert ablation')
    parser.add_argument('--skip-defaults', action='store_true',
                        help='Skip the default neural/symbolic models')
    args = parser.parse_args()

    device = torch.device('cpu')

    print('Loading test sample...')
    X_phys, Y_phys = load_test_sample()

    scaler = np.load(str(MODEL_DIR / 'scaler.npy'), allow_pickle=True).item()

    # Load symbolic sensor once
    symbolic_sensor = None
    if DEFAULT_SYM_PKL.exists():
        with open(DEFAULT_SYM_PKL, 'rb') as f:
            symbolic_sensor = pickle.load(f)

    # Default models (paper baselines)
    # NOTE: 'symbolic gate' uses the PySR formula — NOT model(X) which would use the
    # frozen random ShockIndicator (that bug produced the misleading 0.9506 in the
    # original surrogate_symbolic_evaluation.txt). The constant:0.41 entry below
    # reproduces that bug intentionally for documentation.
    runs = []
    if not args.skip_defaults:
        runs = [
            ('AeroSurrogate (neural gate)',
             MODEL_DIR / 'surrogate_best.pt',
             'neural', None, False),
            ('AeroSurrogate (symbolic gate — PySR)',
             MODEL_DIR / 'surrogate_symbolic_best.pt',
             'symbolic', symbolic_sensor, False),
            ('Symbolic ckpt / fixed p_s=0.41 [ablation]',
             MODEL_DIR / 'surrogate_symbolic_best.pt',
             'constant:0.41', None, False),
            ('Neural ckpt + symbolic routing',
             MODEL_DIR / 'surrogate_best.pt',
             'symbolic', symbolic_sensor, False),
        ]

    # Extra ablation checkpoints
    if args.ckpt:
        labels     = args.label     or [Path(c).stem for c in args.ckpt]
        gate_modes = args.gate_mode or ['neural'] * len(args.ckpt)
        for ckpt, lbl, gm in zip(args.ckpt, labels, gate_modes):
            sym = symbolic_sensor if gm == 'symbolic' else None
            runs.append((lbl, Path(ckpt), gm, sym, args.no_shock_expert))

    all_results = {}
    for name, ckpt, gm, sym, no_se in runs:
        if not Path(ckpt).exists():
            print(f"  SKIP {name}: {ckpt} not found")
            continue
        all_results[name] = evaluate_one(
            name, ckpt, X_phys, Y_phys, scaler, device,
            gate_mode=gm, symbolic_sensor=sym,
            disable_shock_expert=no_se,
        )

    full_table = format_table(all_results)
    md_table   = format_ablation_md(all_results)

    print('\n' + '=' * 80)
    print(full_table)

    # Save
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = RESULT_DIR / 'subset_metrics.txt'
    md_path  = RESULT_DIR / 'ablations.md'

    with open(txt_path, 'w') as f:
        f.write(f'Subset metrics — AeroSurrogate\n')
        f.write(f'Test sample: SEED={SEED}, fraction={TEST_FRACTION}, '
                f'n={int(len(X_phys)):,}\n')
        f.write(f'Shock = y_shock=1[Cp<Cp_crit] AND Mach>0.75\n\n')
        f.write(full_table + '\n')
    print(f'\nFull table  → {txt_path}')

    # Append / rewrite ablation markdown
    existing = ''
    if md_path.exists():
        existing = md_path.read_text()

    with open(md_path, 'w') as f:
        f.write('# Ablation results — AeroSurrogate\n\n')
        f.write('> **Evaluation bug (fixed 2026-07-17):** The original `surrogate_symbolic_evaluation.txt`\n')
        f.write('> reported R²(Cp)=0.9506 for the symbolic gate — INCORRECT.\n')
        f.write('> Root cause: `SurrogateTrainer.train()` called `self.load_model()` with the hardcoded\n')
        f.write('> default `"surrogate_best.pt"` instead of `self.save_name`, so the trainer\'s model\n')
        f.write('> was overwritten with the neural gate weights before evaluation. The symbolic checkpoint\n')
        f.write('> file on disk (`surrogate_symbolic_best.pt`) was correctly saved during training.\n')
        f.write('> Fix: `self.load_model(self.save_name)` in `train()` + `ModelEvaluator` accepts\n')
        f.write('> `symbolic_sensor` to use PySR formula at inference.\n')
        f.write('> **Correct symbolic gate R²(Cp) = 0.9421.**\n\n')
        f.write('## Summary table\n\n')
        f.write(md_table + '\n\n')
        f.write('## Full subset breakdown\n\n```\n')
        f.write(full_table + '\n```\n')
    print(f'Ablation MD → {md_path}')


if __name__ == '__main__':
    main()
