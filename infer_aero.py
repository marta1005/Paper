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
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)
from collections import defaultdict
from src.models import PySRWrapper  # noqa: F401 — ensures pkl unpickling works (class lives in src.models)

from config import MODEL_DIR, MODEL_CONFIG, PLOT_DIR
from src.models import ShockAutoencoder, MixtureOfExperts, AeroSurrogate


# ──────────────────────────────────────────────────────────────────────────────
# 3-D surface scatter — replicates ONERA createFigure_TB exactly
# ──────────────────────────────────────────────────────────────────────────────

def plot_combined_condition(X, Y, Z, Y_true, Y_pred, aeroCond, cond_idx, model_tag):
    """
    One figure per condition.
    Layout: 3 rows (CFD truth | Predicted | Rel. Error) × 4 cols (Cp, Cfx, Cfy, Cfz).
    Each cell: bottom+top 3D sub-views (ONERA style).
    Two colorbar rows at bottom (truth scale + error scale).
    """
    sMinf, sAoA, pressure = aeroCond

    cam = dict(elevation=90., azimuth=0., zoom=1.9,
               xoffsets=[0., 0.], yoffsets=[0., 4.], zoffsets=[0., 0.])

    xlim = (float(X.min()) + cam['xoffsets'][0], float(X.max()) + cam['xoffsets'][1])
    ylim = (float(Y.min()) + cam['yoffsets'][0], float(Y.min()) + cam['yoffsets'][1])
    zlim = (float(Z.min()) + cam['zoffsets'][0], float(Z.max()) + cam['zoffsets'][1])

    truth_scales = [(-1, 1), (-0.002, 0.007), (-0.002, 0.007), (-0.002, 0.007)]
    error_scales = [(0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0)]
    coeff_labels = [r'$C_p$', r'$Cf_x$', r'$Cf_y$', r'$Cf_z$']
    row_labels   = ['CFD truth', 'Predicted', 'Rel. Error']

    rel_err   = np.abs(Y_true - Y_pred) / (np.abs(Y_true) + 1e-6)
    data_rows = [Y_true, Y_pred, rel_err]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f'$M_\\infty$={sMinf:.2f}  AoA={sAoA:.1f}°  $p_i$={pressure:.1f}×10⁵'
        f'  |  {model_tag}  |  cond {cond_idx}',
        fontsize=11, fontweight='bold',
    )

    gs = gridspec.GridSpec(nrows=5, ncols=4,
                           height_ratios=[10, 10, 10, 0.7, 0.7],
                           hspace=0.12, wspace=0.05)

    # Column headers
    for ci, label in enumerate(coeff_labels):
        fig.text((ci + 0.5) / 4, 0.965, label,
                 ha='center', fontsize=11, fontweight='bold')

    for row_i, (pres_arr, row_label) in enumerate(zip(data_rows, row_labels)):
        scales = error_scales if row_i == 2 else truth_scales

        # Row label (left margin)
        fig.text(0.005, 1 - (row_i + 0.5) / 3 * 0.88 - 0.04, row_label,
                 va='center', ha='left', fontsize=9, fontweight='bold', rotation=90)

        for ci, (vmin, vmax) in enumerate(scales):
            gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[row_i, ci], wspace=0.)
            for j in range(2):
                ax = fig.add_subplot(gs00[0, j], projection='3d')
                sca = ax.scatter3D(X, Y, Z, vmin=vmin, vmax=vmax, c=pres_arr[:, ci],
                                   cmap='jet', clip_on=False, s=0.3, rasterized=True)
                ax.view_init(elev=cam['elevation'] * (-1) ** (j + 1),
                             azim=cam['azimuth'] + 180 * abs(j - 1))
                ax.set(xlim=xlim, ylim=ylim, zlim=zlim)
                ax.set_axis_off()
                limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
                ax.set_box_aspect(np.ptp(limits, axis=1), zoom=cam['zoom'])

    # Colorbar row 3: truth scale (shared for CFD + Predicted rows)
    for ci, ((tv_min, tv_max), label) in enumerate(zip(truth_scales, coeff_labels)):
        cax = fig.add_subplot(gs[3, ci])
        sm  = plt.cm.ScalarMappable(cmap='jet',
                                     norm=matplotlib.colors.Normalize(vmin=tv_min, vmax=tv_max))
        cb  = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cb.set_label(label + ' (CFD / Pred)', size=7)
        cb.ax.tick_params(labelsize=5)
        cb.ax.xaxis.set_label_position('top')

    # Colorbar row 4: error scale
    for ci, ((ev_min, ev_max), label) in enumerate(zip(error_scales, coeff_labels)):
        cax = fig.add_subplot(gs[4, ci])
        sm  = plt.cm.ScalarMappable(cmap='jet',
                                     norm=matplotlib.colors.Normalize(vmin=ev_min, vmax=ev_max))
        cb  = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cb.set_label('rel. error ' + label, size=7)
        cb.ax.tick_params(labelsize=5)
        cb.ax.xaxis.set_label_position('top')

    return fig


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
        missing, _ = model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        if missing:
            print(f"  (buffers not in checkpoint, using defaults: {missing})")
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
                   cp_lim=None, err_lim=None):
    """Fill one row: Truth Cp | Predicted Cp | Signed Error  (2D top-down XY view)."""
    x = X_phys[:, 0]   # streamwise
    y = X_phys[:, 1]   # spanwise

    mach, aoa, pi = cond
    err = Cp_cfd - Cp_pred
    mae = float(np.abs(err).mean())

    cp_min, cp_max = cp_lim if cp_lim is not None else (
        float(np.percentile(Cp_cfd, 2)), float(np.percentile(Cp_cfd, 98)))

    err_abs = err_lim if err_lim is not None else float(np.percentile(np.abs(err), 98))
    mae_str = f'MAE={mae:.4f}'

    ax1 = fig.add_subplot(n_rows, 3, row * 3 + 1)
    ax2 = fig.add_subplot(n_rows, 3, row * 3 + 2)
    ax3 = fig.add_subplot(n_rows, 3, row * 3 + 3)

    kw = dict(s=1, alpha=0.9, rasterized=True, linewidths=0)

    # Pastel diverging colormap for error (muted blue–white–red)
    pastel_err = matplotlib.colors.LinearSegmentedColormap.from_list(
        'pastel_err', ['#3a78b5', 'white', '#c94040'])

    sc1 = ax1.scatter(x, y, c=Cp_cfd,  cmap='jet',      vmin=cp_min,   vmax=cp_max,  **kw)
    sc2 = ax2.scatter(x, y, c=Cp_pred, cmap='jet',      vmin=cp_min,   vmax=cp_max,  **kw)
    sc3 = ax3.scatter(x, y, c=err,     cmap=pastel_err, vmin=-err_abs, vmax=err_abs, **kw)

    ax1.set_title(r'Truth $C_p$', fontsize=9)
    ax2.set_title(
        f'cond {cond_idx} | M={mach:.2f} | AoA={aoa:.1f} | Pi={pi:.1f} | {mae_str}',
        fontsize=8,
    )
    ax3.set_title(r'Error $C_p$', fontsize=9)

    for ax, sc, label in [
        (ax1, sc1, r'$C_p$'),
        (ax2, sc2, r'$C_p$'),
        (ax3, sc3, 'Error'),
    ]:
        plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.03, fraction=0.046, label=label)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal', adjustable='datalim')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.annotate('upper', xy=(0.98, 0.02), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=7, color='gray', alpha=0.7)


# ──────────────────────────────────────────────────────────────────────────────
# Span-cut Cp plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_spancuts(X_phys, Y_true, Y_pred, cond, cond_idx,
                 span_stations=None, band_frac=0.03, coeff_idx=0, coeff_name=r'$C_p$'):
    """
    For each span station η, extract a band of surface points, separate upper/lower
    surface by nz sign, and plot truth vs predicted coefficient along the chord.

    X_phys columns: 0=x, 1=y, 2=z, 3=nx, 4=ny, 5=nz, 6=Mach, 7=AoA, 14=x_norm, 15=span_norm
    """
    if span_stations is None:
        span_stations = [0.20, 0.35, 0.50, 0.65, 0.80, 0.95]

    # span_norm is col 15 (fitted [0=root, 1=tip])
    span_norm = X_phys[:, 15]
    x_norm    = X_phys[:, 14]   # chord position [0=LE, 1=TE]
    nz        = X_phys[:, 5]    # outward normal z — positive on upper surface

    true_c = Y_true[:, coeff_idx]
    pred_c = Y_pred[:, coeff_idx]

    mach, aoa, pi = cond
    n_stations = len(span_stations)
    ncols = min(3, n_stations)
    nrows = (n_stations + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(
        f'Spanwise cuts — {coeff_name} | cond {cond_idx} | '
        f'M={mach:.2f}  AoA={aoa:.1f}°  Pi={pi:.1f}',
        fontsize=12, fontweight='bold',
    )

    for s_idx, eta in enumerate(span_stations):
        ax = axes[s_idx // ncols][s_idx % ncols]

        # Band: span_norm within ±band_frac of eta
        band = np.abs(span_norm - eta) < band_frac
        if band.sum() < 20:
            ax.set_title(f'η={eta:.2f}  (no points)', fontsize=9)
            ax.axis('off')
            continue

        xn  = x_norm[band]
        nz_ = nz[band]
        tc  = true_c[band]
        pc  = pred_c[band]

        upper = nz_ >= 0
        lower = nz_ < 0

        # Bin x into N_BINS bins and take mean per bin — gives a clean line
        # instead of a noisy scatter from the unstructured mesh
        N_BINS  = 120
        x_lo, x_hi = xn.min(), xn.max()
        edges   = np.linspace(x_lo, x_hi, N_BINS + 1)
        x_mid   = 0.5 * (edges[:-1] + edges[1:])

        all_vals = []   # collect for y-axis autoscale

        for mask, label, color, ls in [
            (upper, 'upper', 'steelblue', '-'),
            (lower, 'lower', 'steelblue', '--'),
        ]:
            if mask.sum() < 5:
                continue
            bin_idx = np.digitize(xn[mask], edges) - 1
            bin_idx = np.clip(bin_idx, 0, N_BINS - 1)

            tc_bin, pc_bin, x_bin = [], [], []
            for b in range(N_BINS):
                pts = bin_idx == b
                if pts.sum() == 0:
                    continue
                tc_bin.append(tc[mask][pts].mean())
                pc_bin.append(pc[mask][pts].mean())
                x_bin.append(x_mid[b])

            if not x_bin:
                continue
            x_bin  = np.array(x_bin)
            tc_bin = np.array(tc_bin)
            pc_bin = np.array(pc_bin)
            all_vals.extend(tc_bin.tolist())
            all_vals.extend(pc_bin.tolist())

            ax.plot(x_bin, tc_bin, ls, color=color,   lw=1.8,
                    label=f'CFD {label}')
            ax.plot(x_bin, pc_bin, ls, color='tomato', lw=1.8, alpha=0.85,
                    label=f'Model {label}')

        # Cp_crit reference line (Mach only)
        g = 1.4
        sonic_ratio = (2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * mach ** 2)
        cp_crit = (2.0 / (g * max(mach ** 2, 1e-6))) * (sonic_ratio ** (g / (g - 1.0)) - 1.0)
        if coeff_idx == 0:
            ax.axhline(cp_crit, color='gray', lw=1.0, ls=':', alpha=0.7,
                       label=r'$C_p^{\rm crit}$')
            if all_vals:
                all_vals.append(cp_crit)

        # Y-axis: data range with 10% margin, capped at 5th/95th percentile to avoid outliers
        if all_vals:
            y_lo = float(np.percentile(all_vals,  2))
            y_hi = float(np.percentile(all_vals, 98))
            margin = 0.1 * max(abs(y_hi - y_lo), 0.01)
            ax.set_ylim(y_lo - margin, y_hi + margin)

        ax.invert_yaxis()   # aerodynamic convention: suction up
        ax.set_xlabel(r'$x_{\rm norm}$ (chord)', fontsize=8)
        ax.set_ylabel(coeff_name, fontsize=8)
        ax.set_xlim(x_lo, x_hi)   # auto from actual data range, not forced 0–1
        ax.set_title(f'η = {eta:.2f}  ({band.sum()} pts)', fontsize=9)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for s_idx in range(len(span_stations), nrows * ncols):
        axes[s_idx // ncols][s_idx % ncols].axis('off')

    plt.tight_layout()
    return fig


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
    parser.add_argument('--3d',         action='store_true', dest='plot3d',
                        help='3-D surface scatter (top+bottom views, all 4 coefficients)')
    parser.add_argument('--spancut',    action='store_true',
                        help='Plot Cp (and optionally Cfx) vs chord at several span stations')
    parser.add_argument('--span-stations', type=float, nargs='+', default=None,
                        metavar='ETA',
                        help='Span stations to cut (span_norm values, default: 0.20 0.35 0.50 0.65 0.80 0.95)')
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

    COEFF_NAMES = [r'$C_p$', r'$C_{fx}$', r'$C_{fy}$', r'$C_{fz}$']
    COEFF_KEYS  = ['Cp', 'Cfx', 'Cfy', 'Cfz']

    # Pre-compute all predictions (store all 4 coefficients)
    results = []
    for cond in selected:
        d = data[cond]
        pred_norm, shock_prob, gate_w = predict(
            model, d['X_norm'], device,
            symbolic_sensor=symbolic_sensor,
            X_phys=d['X_phys'],
        )
        # Denormalise all 4 outputs
        Y_pred_phys = pred_norm * Y_std[:4] + Y_mean[:4]   # [N, 4]
        results.append((d['X_phys'], d['Y_phys'][:, :4], Y_pred_phys, shock_prob, gate_w))

    if args.expert:
        # expert diagnostics still uses Cp (col 0) internally — rebuild compat tuple
        compat = [(r[0], r[1][:, 0], r[2][:, 0], r[3], r[4]) for r in results]
        _plot_expert_diagnostics(compat, indices, selected, args.model)
        return

    if args.plot3d:
        _sensor_tag = '_symbolic' if args.symbolic else '_neural'
        model_tag   = f'{args.model}{_sensor_tag}'
        for idx, cond, (X_phys, Y_true, Y_pred, shock_prob, gate_w) in \
                zip(indices, selected, results):
            mach, aoa, pi = cond
            X, Y, Z = X_phys[:, 0], X_phys[:, 1], X_phys[:, 2]
            fig = plot_combined_condition(
                X, Y, Z, Y_true, Y_pred, (mach, aoa, pi), idx, model_tag)
            out = PLOT_DIR / f'3d_{model_tag}_cond{idx}.png'
            fig.savefig(out, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved → {out}")
        return

    if args.spancut:
        _sensor_tag = '_symbolic' if args.symbolic else '_neural'
        stations = args.span_stations
        for idx, cond, (X_phys, Y_true, Y_pred, shock_prob, gate_w) in \
                zip(indices, selected, results):
            for coeff_idx, (ckey, cname) in enumerate(
                zip(['Cp', 'Cfx'], [r'$C_p$', r'$C_{fx}$'])
            ):
                fig = plot_spancuts(X_phys, Y_true, Y_pred, cond, idx,
                                    span_stations=stations,
                                    coeff_idx=coeff_idx, coeff_name=cname)
                out = PLOT_DIR / (
                    f'spancut_{ckey.lower()}_{args.model}{_sensor_tag}'
                    f'_cond{idx}.png'
                )
                fig.savefig(out, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved → {out}")
        return

    sensor_tag = '_symbolic' if args.symbolic else '_neural'
    idx_str    = '_'.join(str(i) for i in indices)
    n          = len(selected)

    pastel_err = matplotlib.colors.LinearSegmentedColormap.from_list(
        'pastel_err', ['#3a78b5', 'white', '#c94040'])

    # One figure per coefficient — shared color scales within each coefficient
    for coeff_idx, (cname, ckey) in enumerate(zip(COEFF_NAMES, COEFF_KEYS)):
        # Global scale for this coefficient (same across all conditions)
        all_true = np.concatenate([r[1][:, coeff_idx] for r in results])
        all_pred = np.concatenate([r[2][:, coeff_idx] for r in results])
        all_err  = all_true - all_pred
        val_lim  = (float(np.percentile(all_true, 2)), float(np.percentile(all_true, 98)))
        err_lim  = float(np.percentile(np.abs(all_err), 98))
        print(f"\n{ckey} | val range [{val_lim[0]:.4f}, {val_lim[1]:.4f}] | err ±{err_lim:.4f}")

        fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n),
                                 squeeze=False)

        for row, (idx, cond, (X_phys, Y_true, Y_pred, shock_prob, gate_w)) in \
                enumerate(zip(indices, selected, results)):

            x = X_phys[:, 0]
            y = X_phys[:, 1]
            true_c = Y_true[:, coeff_idx]
            pred_c = Y_pred[:, coeff_idx]
            err_c  = true_c - pred_c
            mae    = float(np.abs(err_c).mean())
            r2     = float(1 - np.var(err_c) / (np.var(true_c) + 1e-12))

            mach, aoa, pi = cond
            kw = dict(s=1, alpha=0.9, rasterized=True, linewidths=0)

            sc1 = axes[row, 0].scatter(x, y, c=true_c, cmap='jet',
                                       vmin=val_lim[0], vmax=val_lim[1], **kw)
            sc2 = axes[row, 1].scatter(x, y, c=pred_c, cmap='jet',
                                       vmin=val_lim[0], vmax=val_lim[1], **kw)
            sc3 = axes[row, 2].scatter(x, y, c=err_c,  cmap=pastel_err,
                                       vmin=-err_lim,   vmax=err_lim,    **kw)

            axes[row, 0].set_title(f'Truth {cname}', fontsize=9)
            axes[row, 1].set_title(
                f'cond {idx} | M={mach:.2f} AoA={aoa:.1f}° Pi={pi:.1f} | '
                f'R²={r2:.4f} MAE={mae:.4f}',
                fontsize=8,
            )
            axes[row, 2].set_title(f'Error {cname}', fontsize=9)

            for ax, sc, label in [
                (axes[row, 0], sc1, cname),
                (axes[row, 1], sc2, cname),
                (axes[row, 2], sc3, 'Error'),
            ]:
                plt.colorbar(sc, ax=ax, orientation='horizontal',
                             pad=0.03, fraction=0.046, label=label)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_aspect('equal', adjustable='datalim')
                for sp in ax.spines.values():
                    sp.set_visible(False)

            print(f"  [{idx}] M={mach:.2f} AoA={aoa:.1f}° {ckey}: R²={r2:.4f} MAE={mae:.4f}")

        fig.suptitle(
            f'{ckey} inference | {args.model} | sensor={sensor_tag.strip("_")}',
            fontsize=13, fontweight='bold', y=1.002,
        )
        plt.tight_layout()
        out = PLOT_DIR / f'{ckey.lower()}_comparison_{args.model}{sensor_tag}_cond{idx_str}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved → {out}")


if __name__ == '__main__':
    main()
