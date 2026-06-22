#!/usr/bin/env python3
"""
Inference comparison: CFD Cp vs Model Cp vs Error  (3D scatter, same style as visualization.py)

Usage:
    python infer_aero.py --list                      # list all test conditions with index
    python infer_aero.py --conditions 0 5 12 30      # pick conditions by index
    python infer_aero.py --model surrogate            # use surrogate instead of AE+MoE
    python infer_aero.py --fraction 1.0              # use full data (recommended on server)
"""
import os; os.environ['PAPER_NUM_WORKERS'] = '0'
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)
from collections import defaultdict

from config import MODEL_DIR, MODEL_CONFIG, PLOT_DIR
from src.models import ShockAutoencoder, MixtureOfExperts, AeroSurrogate


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_type, device):
    cfg = MODEL_CONFIG
    if model_type == 'surrogate':
        model = AeroSurrogate(
            in_dim=cfg['autoencoder']['input_dim'],
            num_experts=cfg['surrogate']['num_experts'],
            output_dim=cfg['surrogate']['output_dim'],
            indicator_hidden=cfg['surrogate']['indicator_hidden'],
            expert_hidden=cfg['surrogate']['expert_hidden'],
        )
        ckpt = MODEL_DIR / 'surrogate_best.pt'
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} not found — train the surrogate first")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        return model.to(device).eval()

    ae = ShockAutoencoder(
        input_dim=cfg['autoencoder']['input_dim'],
        latent_dim=cfg['autoencoder']['latent_dim'],
    )
    ae.load_state_dict(torch.load(MODEL_DIR / 'autoencoder_best.pt', map_location=device))
    moe = MixtureOfExperts(
        latent_dim=cfg['autoencoder']['latent_dim'],
        num_experts=cfg['moe']['num_experts'],
        expert_output_dim=cfg['moe']['expert_output_dim'],
        output_dim=cfg['moe']['output_dim'],
    )
    moe.load_state_dict(torch.load(MODEL_DIR / 'moe_best.pt', map_location=device))

    class AEMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ae = ae
            self.moe = moe
        def forward(self, x):
            z = self.ae.encoder(x)
            pred, _ = self.moe(z)
            return {'pred': pred}

    return AEMoE().to(device).eval()


# ──────────────────────────────────────────────────────────────────────────────
# Data collection
# ──────────────────────────────────────────────────────────────────────────────

def load_test_data(scaler, fraction=1.0):
    """Load test data directly from .npy files and group per-point by (Mach, AoA, Pi_norm)."""
    from pathlib import Path
    from config import DATA_CONFIG

    data_dir = Path(DATA_CONFIG['X_train_path']).parent
    X_te_path = data_dir / 'X_test_derived.npy'
    Y_te_path = Path(DATA_CONFIG['Y_test_path'])

    print(f"  Reading {X_te_path.name}...")
    X_full = np.load(str(X_te_path), mmap_mode='r')
    print(f"  Reading {Y_te_path.name}...")
    Y_full = np.load(str(Y_te_path), mmap_mode='r')

    n_total = len(X_full)
    n_load  = int(n_total * fraction)
    if fraction < 1.0:
        idx = np.sort(np.random.choice(n_total, n_load, replace=False))
        X_phys = np.asarray(X_full[idx],  dtype=np.float32)
        Y_phys = np.asarray(Y_full[idx],  dtype=np.float32)
    else:
        X_phys = np.asarray(X_full[:], dtype=np.float32)
        Y_phys = np.asarray(Y_full[:], dtype=np.float32)

    print(f"  Loaded {len(X_phys):,} points")

    X_mean = np.array(scaler['X_mean'], dtype=np.float32)
    X_std  = np.array(scaler['X_std'],  dtype=np.float32)
    Y_mean = np.array(scaler['Y_mean'], dtype=np.float32)
    Y_std  = np.array(scaler['Y_std'],  dtype=np.float32)

    X_norm = (X_phys - X_mean) / X_std   # normalised (model input)

    # Group per-point by (Mach, AoA, Pi_norm) — cols 6, 7, 10 of X_phys
    Machs = np.round(X_phys[:, 6], 2)
    AoAs  = np.round(X_phys[:, 7], 1)
    Pis   = np.round(X_phys[:, 10], 1)

    print("  Grouping by condition...")
    from collections import defaultdict
    groups = defaultdict(list)
    for i, (m, a, p) in enumerate(zip(Machs, AoAs, Pis)):
        groups[(float(m), float(a), float(p))].append(i)

    data = {}
    for key, idxs in groups.items():
        idxs = np.array(idxs)
        data[key] = {
            'X_norm': X_norm[idxs],
            'X_phys': X_phys[idxs],
            'Y_phys': Y_phys[idxs],
        }
    return data


@torch.no_grad()
def predict(model, X_norm, device, batch_size=4096):
    preds = []
    for i in range(0, len(X_norm), batch_size):
        xb  = torch.from_numpy(X_norm[i:i + batch_size]).float().to(device)
        out = model(xb)
        preds.append(out['pred'].cpu().numpy())
    return np.vstack(preds)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting (3D scatter, same style as visualization.py)
# ──────────────────────────────────────────────────────────────────────────────

def plot_condition(fig, n_rows, row, X_phys, Cp_cfd, Cp_pred, cond, cond_idx,
                   cp_lim=None, err_lim=None):
    """Fill one row: Truth Cp | Predicted Cp | Signed Error  (2D top-down XY view)."""
    x = X_phys[:, 0]   # streamwise
    y = X_phys[:, 1]   # spanwise

    mach, aoa, pi = cond
    err = Cp_cfd - Cp_pred
    mae = float(np.abs(err).mean())

    cp_min, cp_max = cp_lim if cp_lim is not None else (
        float(np.percentile(Cp_cfd, 2)), float(np.percentile(Cp_cfd, 98)))
    err_abs = round(
        err_lim if err_lim is not None else float(np.percentile(np.abs(err), 98)), 3)

    ax1 = fig.add_subplot(n_rows, 3, row * 3 + 1)
    ax2 = fig.add_subplot(n_rows, 3, row * 3 + 2)
    ax3 = fig.add_subplot(n_rows, 3, row * 3 + 3)

    kw = dict(s=1, alpha=0.9, rasterized=True, linewidths=0)

    sc1 = ax1.scatter(x, y, c=Cp_cfd,  cmap='RdBu_r',  vmin=cp_min,   vmax=cp_max,  **kw)
    sc2 = ax2.scatter(x, y, c=Cp_pred, cmap='RdBu_r',  vmin=cp_min,   vmax=cp_max,  **kw)
    sc3 = ax3.scatter(x, y, c=err,     cmap='coolwarm', vmin=-err_abs, vmax=err_abs, **kw)

    err_range = f'Error [{-err_abs:.3f}, {err_abs:.3f}]'

    ax1.set_title(r'Truth $C_p$', fontsize=9)
    ax2.set_title(
        f'cond {cond_idx} | M={mach:.2f} | AoA={aoa:.1f} | Pi={pi:.1f} | MAE={mae:.4f}',
        fontsize=8,
    )
    ax3.set_title(err_range, fontsize=9)

    for ax, sc, label in [
        (ax1, sc1, r'Truth $C_p$'),
        (ax2, sc2, r'Predicted $C_p$'),
        (ax3, sc3, err_range),
    ]:
        plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.03, fraction=0.046, label=label)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal', adjustable='datalim')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.annotate('upper', xy=(0.98, 0.02), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=7, color='gray', alpha=0.7)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      default='aemoe', choices=['aemoe', 'surrogate'])
    parser.add_argument('--conditions', type=int, nargs='+', default=None,
                        help='Indices of conditions to plot (see --list)')
    parser.add_argument('--list',       action='store_true',
                        help='List all available test conditions with their index and exit')
    parser.add_argument('--fraction',   type=float, default=1.0,
                        help='Fraction of test data to load (default 1.0 = all data)')
    parser.add_argument('--min-pts',    type=int,   default=5000,
                        help='Skip conditions with fewer points (default 5000)')
    args = parser.parse_args()

    device = torch.device('cpu')

    scaler_path = MODEL_DIR / 'scaler.npy'
    if not scaler_path.exists():
        raise FileNotFoundError("scaler.npy not found — copy it from the server first")
    scaler = np.load(str(scaler_path), allow_pickle=True).item()

    print(f"Loading test data (fraction={args.fraction})...")
    data = load_test_data(scaler, fraction=args.fraction)

    min_pts = args.min_pts
    all_conditions = sorted(
        [c for c in data if len(data[c]['Y_phys']) >= min_pts],
        key=lambda c: (c[0], c[1], c[2]),
    )
    skipped = len(data) - len(all_conditions)
    if skipped:
        print(f"Skipped {skipped} conditions with < {min_pts} points")

    if args.list:
        print(f"\n{'Idx':>4}  {'Mach':>6}  {'AoA':>6}  {'Pi':>6}  {'N points':>10}")
        print("-" * 44)
        for i, cond in enumerate(all_conditions):
            n_pts = len(data[cond]['Y_phys'])
            print(f"{i:>4}  {cond[0]:>6.2f}  {cond[1]:>6.1f}  {cond[2]:>6.1f}  {n_pts:>10,}")
        return

    if args.conditions is None:
        step    = max(1, len(all_conditions) // 6)
        indices = list(range(0, len(all_conditions), step))[:6]
    else:
        indices = args.conditions

    selected = []
    for idx in indices:
        if idx < 0 or idx >= len(all_conditions):
            print(f"Warning: index {idx} out of range (0–{len(all_conditions) - 1}), skipping")
            continue
        selected.append(all_conditions[idx])

    print(f"\nConditions selected:")
    for idx, c in zip(indices, selected):
        print(f"  [{idx}] Mach={c[0]:.2f}  AoA={c[1]:.1f}°  Pi={c[2]:.1f}  ({len(data[c]['Y_phys']):,} pts)")

    print("Loading model...")
    model = load_model(args.model, device)

    Y_mean = np.array(scaler['Y_mean'])
    Y_std  = np.array(scaler['Y_std'])

    # Pre-compute all predictions
    results = []
    for cond in selected:
        d = data[cond]
        Cp_pred_norm = predict(model, d['X_norm'], device)
        Cp_pred = Cp_pred_norm[:, 0] * Y_std[0] + Y_mean[0]
        results.append((d['X_phys'], d['Y_phys'][:, 0], Cp_pred))

    # Global shared color scales so colors are consistent across rows
    all_Cp  = np.concatenate([r[1] for r in results])
    cp_lim  = (float(np.percentile(all_Cp, 2)), float(np.percentile(all_Cp, 98)))
    all_err = np.concatenate([np.abs(r[1] - r[2]) for r in results])
    err_lim = round(float(np.percentile(all_err, 98)), 3)
    print(f"Global Cp range: [{cp_lim[0]:.3f}, {cp_lim[1]:.3f}]  |  Error range: ±{err_lim:.3f}")

    n   = len(selected)
    fig = plt.figure(figsize=(18, 5 * n))

    for row, (idx, cond, (X_phys, Cp_cfd, Cp_pred)) in enumerate(zip(indices, selected, results)):
        plot_condition(fig, n, row, X_phys, Cp_cfd, Cp_pred, cond, idx,
                       cp_lim=cp_lim, err_lim=err_lim)
        mae = float(np.abs(Cp_cfd - Cp_pred).mean())
        r2  = float(1 - np.var(Cp_cfd - Cp_pred) / np.var(Cp_cfd))
        print(f"  [{idx}] Mach={cond[0]:.2f}  AoA={cond[1]:.1f}°  Pi={cond[2]:.1f}  R²={r2:.4f}  MAE={mae:.4f}")

    fig.suptitle(
        'Full-aircraft upper shock-symbolic inference | multiple test conditions',
        fontsize=13, fontweight='bold', y=1.002,
    )
    plt.tight_layout()

    out = PLOT_DIR / f'cp_comparison_{args.model}_2d.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()
