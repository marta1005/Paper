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

def load_model(model_type, device, symbolic=False):
    cfg = MODEL_CONFIG
    if model_type == 'surrogate':
        model = AeroSurrogate(
            in_dim=cfg['autoencoder']['input_dim'],
            num_experts=cfg['surrogate']['num_experts'],
            output_dim=cfg['surrogate']['output_dim'],
            indicator_hidden=cfg['surrogate']['indicator_hidden'],
            expert_hidden=cfg['surrogate']['expert_hidden'],
        )
        ckpt_name = 'surrogate_symbolic_best.pt' if symbolic else 'surrogate_best.pt'
        ckpt = MODEL_DIR / ckpt_name
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} not found — run launch_gpu.sh first")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded checkpoint: {ckpt_name}")
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

    n_feat = len(X_mean)
    X_norm = (X_phys[:, :n_feat] - X_mean) / X_std   # normalised (model input)

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
def predict(model, X_norm, device, batch_size=4096, symbolic_sensor=None, X_phys=None):
    """Run AeroSurrogate forward. If symbolic_sensor is given, override shock_prob
    with the symbolic tree output before passing to the MoE."""
    preds, shock_probs, gate_weights = [], [], []
    for i in range(0, len(X_norm), batch_size):
        xb  = torch.from_numpy(X_norm[i:i + batch_size]).float().to(device)

        if symbolic_sensor is not None:
            # Get shock_prob from symbolic sensor instead of neural ShockIndicator
            sr_idx = symbolic_sensor['sr_idx']
            X_raw_b = X_phys[i:i + batch_size][:, sr_idx]
            proba   = symbolic_sensor['clf'].predict_proba(X_raw_b)[:, 1]
            sp_cal  = symbolic_sensor['calibrator'].predict(proba).astype(np.float32)
            sp_t    = torch.from_numpy(sp_cal[:, None]).to(device)

            # Run MoE directly with symbolic shock_prob (bypass ShockIndicator)
            pred, gates = model.moe(xb, sp_t)
            out = {'pred': pred, 'shock_prob': sp_t, 'gate_weights': gates}
        else:
            out = model(xb)

        preds.append(out['pred'].cpu().numpy())
        shock_probs.append(out['shock_prob'].cpu().numpy())
        gate_weights.append(out['gate_weights'].cpu().numpy())
    return np.vstack(preds), np.vstack(shock_probs), np.vstack(gate_weights)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting (3D scatter, same style as visualization.py)
# ──────────────────────────────────────────────────────────────────────────────

def plot_condition(fig, n_rows, row, X_phys, Cp_cfd, Cp_pred, cond, cond_idx,
                   cp_lim=None):
    """Fill one row: Truth Cp | Predicted Cp | Signed Error  (2D top-down XY view)."""
    x = X_phys[:, 0]   # streamwise
    y = X_phys[:, 1]   # spanwise

    mach, aoa, pi = cond
    err = Cp_cfd - Cp_pred
    mae = float(np.abs(err).mean())

    cp_min, cp_max = cp_lim if cp_lim is not None else (
        float(np.percentile(Cp_cfd, 2)), float(np.percentile(Cp_cfd, 98)))

    # Error as % of the global Cp range
    cp_range = cp_max - cp_min
    err_pct = 100.0 * err / cp_range
    err_pct_abs = round(float(np.percentile(np.abs(err_pct), 98)), 1)
    err_range = f'Error [{-err_pct_abs:.1f}%, {err_pct_abs:.1f}%]'
    mae_str = f'MAE={mae:.4f}'

    ax1 = fig.add_subplot(n_rows, 3, row * 3 + 1)
    ax2 = fig.add_subplot(n_rows, 3, row * 3 + 2)
    ax3 = fig.add_subplot(n_rows, 3, row * 3 + 3)

    kw = dict(s=1, alpha=0.9, rasterized=True, linewidths=0)

    # Pastel diverging colormap for error (muted blue–white–red)
    pastel_err = matplotlib.colors.LinearSegmentedColormap.from_list(
        'pastel_err', ['#3a78b5', 'white', '#c94040'])

    sc1 = ax1.scatter(x, y, c=Cp_cfd,    cmap='jet',      vmin=cp_min,      vmax=cp_max,      **kw)
    sc2 = ax2.scatter(x, y, c=Cp_pred,   cmap='jet',      vmin=cp_min,      vmax=cp_max,      **kw)
    sc3 = ax3.scatter(x, y, c=err_pct,   cmap=pastel_err, vmin=-err_pct_abs, vmax=err_pct_abs, **kw)

    ax1.set_title(r'Truth $C_p$', fontsize=9)
    ax2.set_title(
        f'cond {cond_idx} | M={mach:.2f} | AoA={aoa:.1f} | Pi={pi:.1f} | {mae_str}',
        fontsize=8,
    )
    ax3.set_title(err_range, fontsize=9)

    for ax, sc, label in [
        (ax1, sc1, r'Truth $C_p$'),
        (ax2, sc2, r'Predicted $C_p$'),
        (ax3, sc3, 'Error (% of $C_p$ range)'),
    ]:
        plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.03, fraction=0.046, label=label)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal', adjustable='datalim')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.annotate('upper', xy=(0.98, 0.02), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=7, color='gray', alpha=0.7)


# ──────────────────────────────────────────────────────────────────────────────
# Expert specialisation diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def _plot_expert_diagnostics(results, indices, selected, model_name):
    """3-col plot per condition: dominant expert | shock probability | gate entropy."""
    import matplotlib.colors as mcolors

    n = len(results)
    n_experts = results[0][4].shape[1]  # gate_w columns

    # Discrete colormap for experts
    expert_colors = plt.colormaps['tab10']
    expert_cmap   = mcolors.ListedColormap([expert_colors(i) for i in range(n_experts)])
    expert_bounds  = np.arange(-0.5, n_experts + 0.5, 1)
    expert_norm    = mcolors.BoundaryNorm(expert_bounds, n_experts)

    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    # Print per-condition expert usage stats
    print(f"\n{'Cond':>5}  {'Mach':>5}  {'AoA':>6}  " +
          "  ".join(f"Expert{i}" for i in range(n_experts)))
    print("-" * (30 + 10 * n_experts))

    for row, (idx, cond, (X_phys, Cp_cfd, Cp_pred, shock_prob, gate_w)) in \
            enumerate(zip(indices, selected, results)):

        x = X_phys[:, 0]
        y = X_phys[:, 1]
        dominant = gate_w.argmax(axis=1)

        # Gate entropy (nats): H = -sum(w * log(w + eps))
        eps     = 1e-9
        entropy = -(gate_w * np.log(gate_w + eps)).sum(axis=1)
        max_H   = np.log(n_experts)

        kw = dict(s=1, alpha=0.9, rasterized=True, linewidths=0)
        mach, aoa, pi = cond

        # Col 1: dominant expert
        sc1 = axes[row, 0].scatter(x, y, c=dominant, cmap=expert_cmap,
                                   norm=expert_norm, **kw)
        cb1 = plt.colorbar(sc1, ax=axes[row, 0], orientation='horizontal',
                           pad=0.03, fraction=0.046, ticks=range(n_experts))
        cb1.set_label('Dominant expert')

        # Col 2: shock probability
        sc2 = axes[row, 1].scatter(x, y, c=shock_prob[:, 0], cmap='plasma',
                                   vmin=0, vmax=1, **kw)
        plt.colorbar(sc2, ax=axes[row, 1], orientation='horizontal',
                     pad=0.03, fraction=0.046, label='Shock probability')

        # Col 3: gate entropy (normalised to [0,1])
        sc3 = axes[row, 2].scatter(x, y, c=entropy / max_H, cmap='viridis',
                                   vmin=0, vmax=1, **kw)
        plt.colorbar(sc3, ax=axes[row, 2], orientation='horizontal',
                     pad=0.03, fraction=0.046, label='Gate entropy (0=certain, 1=uniform)')

        titles = [
            f'Dominant expert | M={mach:.2f} AoA={aoa:.1f}° Pi={pi:.1f}',
            'Shock probability',
            'Gate entropy',
        ]
        for ax, title in zip(axes[row], titles):
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect('equal', adjustable='datalim')
            for sp in ax.spines.values():
                sp.set_visible(False)

        # Print usage fractions
        fracs = [(dominant == e).mean() for e in range(n_experts)]
        frac_str = "  ".join(f"{f:.3f}    " for f in fracs)
        print(f"  [{idx}]  {mach:.2f}  {aoa:+.1f}°   {frac_str}")

    fig.suptitle(
        f'Expert specialisation diagnostics — {model_name}',
        fontsize=13, fontweight='bold', y=1.002,
    )
    plt.tight_layout()
    out = PLOT_DIR / f'expert_diagnostics_{model_name}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved → {out}")


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
    parser.add_argument('--expert',     action='store_true',
                        help='Plot expert specialisation diagnostics instead of Cp comparison')
    parser.add_argument('--symbolic',   action='store_true',
                        help='Replace neural ShockIndicator with symbolic sensor (DT) for MoE gating')
    parser.add_argument('--sensor-pkl', default=None,
                        help='Path to symbolic sensor pkl (default: outputs/models/shock_sensor_symbolic_physics_knn.pkl)')
    args = parser.parse_args()

    device = torch.device('cpu')

    scaler_path = MODEL_DIR / 'scaler.npy'
    if not scaler_path.exists():
        raise FileNotFoundError("scaler.npy not found — copy it from the server first")
    scaler = np.load(str(scaler_path), allow_pickle=True).item()

    # Load symbolic sensor if requested
    symbolic_sensor = None
    if args.symbolic:
        import pickle
        pkl_path = args.sensor_pkl or str(MODEL_DIR / 'shock_sensor_symbolic_physics_knn.pkl')
        with open(pkl_path, 'rb') as f:
            symbolic_sensor = pickle.load(f)
        if symbolic_sensor.get('clf') is None:
            raise ValueError("Symbolic sensor pkl missing 'clf' — re-run symbolic_regression.py")
        print(f"Symbolic sensor loaded: {symbolic_sensor['sr_features']}")

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
    model = load_model(args.model, device, symbolic=args.symbolic)

    Y_mean = np.array(scaler['Y_mean'])
    Y_std  = np.array(scaler['Y_std'])

    # Pre-compute all predictions
    results = []
    for cond in selected:
        d = data[cond]
        pred_norm, shock_prob, gate_w = predict(
            model, d['X_norm'], device,
            symbolic_sensor=symbolic_sensor,
            X_phys=d['X_phys'],
        )
        Cp_pred = pred_norm[:, 0] * Y_std[0] + Y_mean[0]
        results.append((d['X_phys'], d['Y_phys'][:, 0], Cp_pred, shock_prob, gate_w))

    if args.expert:
        _plot_expert_diagnostics(results, indices, selected, args.model)
        return

    # Global shared color scales so colors are consistent across rows
    all_Cp  = np.concatenate([r[1] for r in results])
    cp_lim  = (float(np.percentile(all_Cp, 2)), float(np.percentile(all_Cp, 98)))
    print(f"Global Cp range: [{cp_lim[0]:.3f}, {cp_lim[1]:.3f}]")

    n   = len(selected)
    fig = plt.figure(figsize=(18, 5 * n))

    for row, (idx, cond, (X_phys, Cp_cfd, Cp_pred, shock_prob, gate_w)) in enumerate(zip(indices, selected, results)):
        plot_condition(fig, n, row, X_phys, Cp_cfd, Cp_pred, cond, idx,
                       cp_lim=cp_lim)
        mae = float(np.abs(Cp_cfd - Cp_pred).mean())
        r2  = float(1 - np.var(Cp_cfd - Cp_pred) / np.var(Cp_cfd))
        print(f"  [{idx}] Mach={cond[0]:.2f}  AoA={cond[1]:.1f}°  Pi={cond[2]:.1f}  R²={r2:.4f}  MAE={mae:.4f}")

    sensor_tag = '_symbolic' if args.symbolic else '_neural'
    fig.suptitle(
        f'Full-aircraft Cp inference | {args.model} | sensor={sensor_tag.strip("_")}',
        fontsize=13, fontweight='bold', y=1.002,
    )
    plt.tight_layout()

    idx_str = '_'.join(str(i) for i in indices)
    out = PLOT_DIR / f'cp_comparison_{args.model}{sensor_tag}_cond{idx_str}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()
