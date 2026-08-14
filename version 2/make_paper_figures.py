#!/usr/bin/env python3
"""
make_paper_figures.py — regenerate the paper's physics figures for AeroSurrogate v2.

Produces two families of figure, both written to outputs/plots/ under names that
do not collide with anything already there (every name is prefixed ``v2_``):

  1. v2_surface_fields_sim{NNN}.png   — one per selected test simulation.
     3 rows (CFD truth | AeroSurrogate v2 | signed error) x 2 columns
     (Cp, |Cf|).  Each cell is a pair of 3-D scatter views of the half-model,
     mirrored and abutted so they read as a whole aircraft: the left view is the
     lower surface, the right view the upper surface.  The rendering is the ONERA
     ``createFigure_TB`` one used for the v1 paper figures
     (``surface_fields_cond*.png``) — same camera, same ``jet`` colormap, same
     marker size (s=0.3), same ``set_box_aspect(ptp)`` — so a v2 panel and a v1
     panel of the same flow are directly comparable.

     Deliberate differences from v1, all visible on the page:
       * columns are (Cp, |Cf|), not (Cp, Cfx, Cfy, Cfz).  v2 predicts friction
         as a vector through a dedicated magnitude/direction head, so |Cf| is a
         quantity the model actually produces rather than three loosely-coupled
         scalars.  v1 could not show this.
       * the error row is SIGNED error on a diverging scale, not v1's relative
         error |y-ŷ|/(|y|+1e-6) on ``jet`` over [0, 1].  Relative error is
         unreadable wherever the true coefficient passes through zero, which for
         Cp is most of the wing.
       * the two half-model views actually meet on the seam.  v1's do not — see
         the CAM comment below for the measurement and the reason.

  2. v2_parity_sims_{...}.png         — one figure for the whole selection.
     2x2 parity plots (Cp, Cfx, Cfy, Cfz), pooled over every selected sim.
     Unlike the v1 parity figure (``parity_plots.png``, which came out of
     main_train.py) these axes are in PHYSICAL units, not z-scored ones, so the
     RMSE printed in each panel title is a physical RMSE and the R^2 is directly
     comparable with the tables produced by evaluate_v2.py.

Colour-scale policy
  * The CFD row and the model row share vmin/vmax for each field, so the two
    panels are visually comparable and a difference on the page is a real
    difference in the physics.  Defaults reproduce the v1 paper scales
    (Cp in [-1, 1]); pass --auto-scale to derive them from the data instead.
  * Error panels use a diverging colormap on a scale symmetric about zero, so
    white is exactly zero error and over/under-prediction are distinguishable.
    The half-range is the --err-pct percentile of |error| pooled over every
    selected sim, so all the per-sim figures in one run share an error scale.

Usage
    # the six flight conditions the paper reports
    python make_paper_figures.py

    # a single sim, quick check
    python make_paper_figures.py --sims 0

    # a different checkpoint (output names get the checkpoint stem appended)
    python make_paper_figures.py --ckpt outputs/models/surrogate_v2_best.pt

    # list the test conditions with their sim index
    python make_paper_figures.py --list-sims

Runs on CPU by default; one sim costs roughly a minute of forward pass plus the
3-D rendering.  No seaborn.
"""
import os
os.environ.setdefault('PAPER_NUM_WORKERS', '0')

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D, proj3d  # noqa: F401  (registers '3d')

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (DATA_DIR, CACHE_DIR, MODEL_DIR, PLOT_DIR,   # noqa: E402
                    MODEL_CONFIG, TRAINING_CONFIG, KNN_K, N_TEST, SEED)
from src.preprocessing import CFDPreprocessor                    # noqa: E402
from src.models_v2 import AeroSurrogatev2                        # noqa: E402
from src.dataset import SimulationDataset, load_sim_weights      # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CKPT = MODEL_DIR / 'surrogate_v2_moefix_long.pt'   # run 4, the reported model

# The six flight conditions the paper's surface-field figures show, expressed as
# v2 test-sim indices.  v1 indexed conditions by their rank in a list sorted by
# (Mach, AoA, Pi_norm); v2 indexes simulations by their row in
# dataset.csv[Train == False], which is sorted (Pi, Mach, AoA).  The indices are
# therefore different integers for the same physics — these are the v2 ones, and
# _check_default_conditions() re-derives them at runtime so a change to
# dataset.csv cannot silently repoint a figure at the wrong flow.
# Sim 112 (M=0.70, AoA=-10.0, Pi=4) is one of the 16 simulations early stopping
# selected the checkpoint on, so a figure of it shows the model on data it was
# tuned against. Replaced by 113, the nearest condition of the same family that
# is genuinely held out — and which additionally carries confidence weight 1.0
# rather than 112's 0.5.
DEFAULT_SIMS = [0, 113, 69, 78, 86, 147]
DEFAULT_CONDITIONS = [           # (Mach, AoA_deg, Pi*1e-5) each default sim must match
    (0.30,  -6.0, 1.0),          # subsonic, negative incidence
    (0.70,  -7.5, 4.0),          # high-subsonic, strong negative incidence
    (0.80,   0.0, 2.0),          # transonic cruise
    (0.84,   5.0, 2.0),          # transonic, moderate incidence
    (0.86,   0.0, 2.0),          # high-transonic
    (0.90,   6.0, 4.0),          # high Mach, high incidence
]

COEFF_NAMES  = ['Cp', 'Cfx', 'Cfy', 'Cfz']
COEFF_LABELS = [r'$C_p$', r'$C_{fx}$', r'$C_{fy}$', r'$C_{fz}$']

# Columns of the denormalised 16-feature vector.
MACH_COL    = 6
AOA_COL     = 7
PI_COL      = 8
CP_CRIT_COL = 13
MACH_THRESH = 0.75
# float32 round-trip through the scaler costs a few ULP on the flight-condition
# columns, so the CSV cross-check needs a tolerance rather than equality.
COND_TOL    = dict(Mach=1e-3, AoA=1e-2, Pi=1e-2)

# ONERA camera, from v1 infer_aero.py:plot_combined_condition — same top-down
# elevation, same zoom, same set_box_aspect(ptp) trick, so the rendering is
# isotropic and reads exactly like the v1 paper figures.
#
# The one thing NOT copied verbatim is the y half-width of the view box.  v1
# hard-coded 4.0 (``yoffsets=[0., 4.]``), which is what decides how far apart
# the two mirrored half-model views land: the mesh is a half aircraft, the
# scatter is drawn with clip_on=False so nothing is ever cut, and the pair only
# reads as one whole aircraft when the fuselage lines of the two views coincide
# on the seam.
#
# 4.0 is not usable any more, and the reason is a matplotlib change, not a
# layout one.  Measured on this mesh:
#   * the published v1 figure (figures/surface_fields_cond81.png, saved at
#     dpi=300) already does NOT abut — its two fuselage lines are 29 px ≈ 0.10 in
#     apart, a hairline split you can see running down the fuselage;
#   * re-running v1's own geometry (figsize 16x12, the 5x4 GridSpec) under
#     matplotlib 3.10.9 turns that 0.10 in into 1.31 in.  Same code, same 4.0,
#     13x the gap — the mpl 3-D projection/zoom behaviour moved under it, so 4.0
#     no longer reproduces even v1's own output;
#   * under this file's layout (2 columns, 4.0x4.6 in cells) 4.0 gives 1.65 in
#     and crops the model as well.
# Hence _solve_y_halfwidth() below solves for the half-width instead of trusting
# a constant.  The root is 37.655 m for this mesh and — measured over figsizes
# from 30x5 to 6x20 and grids from 1x6 to 4x1 — it does not depend on the figure
# geometry at all, only on the data extents and CAM['zoom'].  Solving it per run
# is therefore cheap insurance against the next mpl projection change rather
# than a per-layout tuning knob.
CAM = dict(elevation=90., azimuth=0., zoom=1.9,
           xoffsets=[0., 0.], zoffsets=[0., 0.])

VIEW_NAMES = ['bottom', 'top']

# Figure geometry for the surface-field figures.  GRID is in figure fractions;
# the cell size is what actually sets the scale of the rendering, because
# mpl3d fits the view box to the smaller axes dimension (here the width).
# CELL_H only has to be large enough that the aircraft, which ends up about
# 1.7x the sub-axes width tall, does not run into the row above.
GRID = dict(left=0.045, right=0.995, top=0.925, bottom=0.145,
            hspace=0.02, wspace=0.02)
CELL_W, CELL_H = 4.0, 4.6        # inches per (column, row)
N_FIELD_ROWS, N_FIELD_COLS = 3, 2   # CFD/model/error  x  Cp/|Cf|

# v1's error colormap (infer_aero.py, cp_comparison figures) — diverging,
# white at the centre.
PASTEL_ERR = matplotlib.colors.LinearSegmentedColormap.from_list(
    'pastel_err', ['#3a78b5', 'white', '#c94040'])


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def r2_score(y_true, y_pred):
    """Column-wise R^2; matches evaluate_v2.r2_score so numbers agree."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def rmse_score(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def cf_magnitude(Y):
    """|Cf| from the three friction components of a [N, 4] coefficient array."""
    return np.linalg.norm(Y[:, 1:4], axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# Data / model
# ──────────────────────────────────────────────────────────────────────────────

def validation_sim_indices():
    """The 16 test sims early stopping selected on (evaluate_v2 reproduces these)."""
    rng = np.random.default_rng(SEED)
    return sorted(rng.choice(N_TEST, TRAINING_CONFIG['val_sims'], replace=False).tolist())


def load_condition_table():
    """dataset.csv rows for the test split; row i == test sim i."""
    df = pd.read_csv(DATA_DIR / 'dataset.csv', index_col=0)
    return df[df['Train'] == False].reset_index(drop=True)   # noqa: E712


def sim_conditions(meta, sim):
    r = meta.loc[sim]
    return dict(Mach=float(r['Mach']),
                AoA=float(r['AoA']),
                Pi=float(r['Pi *1e-5']),          # column name really has that space
                weight=float(r['confidence_weight_simple']))


def _check_default_conditions(meta):
    """Fail loudly if DEFAULT_SIMS no longer point at the paper's conditions."""
    bad = []
    for sim, want in zip(DEFAULT_SIMS, DEFAULT_CONDITIONS):
        c = sim_conditions(meta, sim)
        got = (round(c['Mach'], 2), round(c['AoA'], 1), round(c['Pi'], 1))
        if got != want:
            bad.append(f'  sim {sim}: expected M/AoA/Pi {want}, dataset.csv says {got}')
    if bad:
        raise SystemExit(
            'DEFAULT_SIMS no longer match the flight conditions the paper reports.\n'
            'dataset.csv or the split order changed; re-derive the indices before\n'
            'regenerating figures, or the captions will describe the wrong flow.\n'
            + '\n'.join(bad))


def build_stack(ckpt_path, device, allow_partial_load=False):
    """Scaler, dataset and model, loaded once and reused for every sim."""
    scaler        = np.load(str(MODEL_DIR / 'scaler_v2.npy'),        allow_pickle=True).item()
    spatial_stats = np.load(str(MODEL_DIR / 'spatial_stats_v2.npy'), allow_pickle=True).item()

    test_ds = SimulationDataset(
        'test', CACHE_DIR, DATA_DIR,
        load_sim_weights(DATA_DIR / 'dataset.csv', 'test'),
        CFDPreprocessor(spatial_stats=spatial_stats),
        scaler, k=KNN_K,
    )

    model = AeroSurrogatev2(MODEL_CONFIG).to(device)
    model.eval()                       # MANDATORY: train() swaps the MoE gate to
                                       # gumbel_softmax and leaves dropout live.
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if (missing or unexpected) and not allow_partial_load:
        raise SystemExit(
            f'Checkpoint does not match the architecture:\n'
            f'  missing keys:    {list(missing)}\n'
            f'  unexpected keys: {list(unexpected)}\n'
            'Those tensors would keep their random initialisation and the figures '
            'would be meaningless. Pass --allow-partial-load only if you know why.')
    if missing or unexpected:
        print(f'  WARNING: loaded with {len(missing)} missing / {len(unexpected)} '
              f'unexpected keys (--allow-partial-load)')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  model: {n_params:,} parameters, tau={float(model.moe.tau):.4f}')
    return scaler, test_ds, model


@torch.no_grad()
def predict_sim(model, test_ds, scaler, sim, device):
    """Physical truth/prediction plus the extras the figures annotate with."""
    batch = test_ds[sim]

    X_mean = np.asarray(scaler['X_mean'], np.float32)
    X_std  = np.asarray(scaler['X_std'],  np.float32)
    Y_mean = np.asarray(scaler['Y_mean'], np.float32)
    Y_std  = np.asarray(scaler['Y_std'],  np.float32)

    out = model(batch['x'].to(device),
                batch['edge_index'].to(device),
                batch['edge_attr'].to(device))

    Y_pred = out['pred'].cpu().numpy() * Y_std + Y_mean     # [N, 4] physical
    Y_true = batch['y_phys'].numpy()                         # [N, 4] physical
    x_phys = batch['x'].numpy() * X_std + X_mean             # [N, 16] physical

    # Same shock definition evaluate_v2.get_masks uses, so the fraction quoted in
    # a caption matches the tables.
    shock = (Y_true[:, 0] < x_phys[:, CP_CRIT_COL]) & (x_phys[:, MACH_COL] > MACH_THRESH)

    return dict(
        Y_true=Y_true,
        Y_pred=Y_pred,
        shock_prob=out['shock_prob'].cpu().numpy().ravel(),
        shock_true=shock,
        gate_mean=out['gate_weights'].cpu().numpy().mean(axis=0),
        # Flight condition as the model actually saw it, for cross-checking the
        # caption against dataset.csv.
        feat_cond=dict(Mach=float(x_phys[0, MACH_COL]),
                       AoA=float(x_phys[0, AOA_COL]),
                       Pi=float(x_phys[0, PI_COL])),
    )


def surface_coords(test_ds, sim):
    """Exact node coordinates, straight off the memmap (no normalisation round-trip).

    Geometry is identical for every sim — only the flight-condition columns
    change — so this is read once and reused.
    """
    xyz = np.array(test_ds.X_raw[sim][:, :3], dtype=np.float32)
    return xyz[:, 0], xyz[:, 1], xyz[:, 2]


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — surface fields
# ──────────────────────────────────────────────────────────────────────────────

def _setup_view(ax, j, xlim, ylim, zlim):
    """Apply the ONERA camera to one 3-D sub-view. j=0 lower surface, j=1 upper."""
    ax.view_init(elev=CAM['elevation'] * (-1) ** (j + 1),
                 azim=CAM['azimuth'] + 180 * abs(j - 1))
    ax.set(xlim=xlim, ylim=ylim, zlim=zlim)
    ax.set_axis_off()
    limits = np.array([getattr(ax, f'get_{a}lim')() for a in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1), zoom=CAM['zoom'])


def _root_display_x(ax, x_mid, y_root, z_mid):
    """Display x (pixels) where the fuselage line lands in this sub-view."""
    a, b, _ = proj3d.proj_transform(x_mid, y_root, z_mid, ax.get_proj())
    return float(ax.transData.transform((a, b))[0])


def _solve_y_halfwidth(fig_size, gs_kw, n_row, n_col, xlim, zlim, y_root,
                       lo=1.0, hi=400.0, iters=40):
    """Length of the y view box that makes the two mirrored views abut.

    The gap between the two fuselage lines shrinks monotonically as the box
    grows, so a bisection on a throwaway figure of identical geometry pins it
    down exactly.  Only empty axes are drawn, so this costs milliseconds.

    Returns ``(halfwidth, solved)``.  ``solved`` is False when the bracket was
    lost and the caller is being handed v1's hard-coded 4.0 back — a value that
    for this layout leaves the two halves ~1.6 in apart and crops the aircraft,
    so the caller must say so rather than claim the seam was closed.
    """
    probe = plt.figure(figsize=fig_size)
    gs    = gridspec.GridSpec(n_row, n_col, figure=probe, **gs_kw)
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.)
    axes  = [probe.add_subplot(inner[0, j], projection='3d') for j in range(2)]
    x_mid = 0.5 * (xlim[0] + xlim[1])
    z_mid = 0.5 * (zlim[0] + zlim[1])

    def gap(ylen):
        for j, ax in enumerate(axes):
            _setup_view(ax, j, xlim, (y_root, y_root + ylen), zlim)
        probe.canvas.draw()
        return (_root_display_x(axes[1], x_mid, y_root, z_mid)
                - _root_display_x(axes[0], x_mid, y_root, z_mid))

    g_lo, g_hi = gap(lo), gap(hi)
    if g_lo < 0 or g_hi > 0:      # bracket lost (new mpl projection?) — fall back
        plt.close(probe)
        return 4.0, False         # v1's hard-coded value
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if gap(mid) > 0:
            lo = mid
        else:
            hi = mid
    plt.close(probe)
    return 0.5 * (lo + hi), True


def _draw_pair(fig, subspec, X, Y, Z, values, cmap, vmin, vmax, lims, annotate):
    """One cell: lower-surface and upper-surface 3-D views, abutted."""
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=subspec, wspace=0.)
    xlim, ylim, zlim = lims
    for j in range(2):
        ax = fig.add_subplot(inner[0, j], projection='3d')
        _setup_view(ax, j, xlim, ylim, zlim)
        ax.scatter3D(X, Y, Z, c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                     s=0.3, clip_on=False, rasterized=True)
        if annotate:
            ax.text2D(0.40 + 0.20 * j, 0.72, VIEW_NAMES[j], transform=ax.transAxes,
                      ha='center', va='center', fontsize=8, color='0.35')


def plot_surface_fields(coords, fields, cond, sim, field_lims, err_lims,
                        y_halfwidth, model_tag='AeroSurrogate v2'):
    """3 rows (CFD | model | error) x 2 columns (Cp, |Cf|)."""
    X, Y, Z = coords
    col_labels = [r'$C_p$', r'$|C_f|$']
    row_labels = ['CFD truth', model_tag, 'Error (CFD $-$ model)']
    n_col, n_row = N_FIELD_COLS, N_FIELD_ROWS

    xlim = (float(X.min()) + CAM['xoffsets'][0], float(X.max()) + CAM['xoffsets'][1])
    ylim = (float(Y.min()), float(Y.min()) + y_halfwidth)
    zlim = (float(Z.min()) + CAM['zoffsets'][0], float(Z.max()) + CAM['zoffsets'][1])

    fig = plt.figure(figsize=(CELL_W * n_col, CELL_H * n_row))
    fig.suptitle(
        f'$M_\\infty$={cond["Mach"]:.2f}  AoA={cond["AoA"]:.1f}°  '
        f'$p_i$={cond["Pi"]:.1f}×10⁵  |  {model_tag}  |  test sim {sim}',
        fontsize=11, fontweight='bold', y=0.985,
    )
    gs = gridspec.GridSpec(n_row, n_col, figure=fig, **GRID)

    for ci, label in enumerate(col_labels):
        fig.text(GRID['left'] + (ci + 0.5) * (GRID['right'] - GRID['left']) / n_col,
                 0.948, label, ha='center', va='center',
                 fontsize=12, fontweight='bold')

    rows = [
        ([fields['cp_true'], fields['cf_true']], 'jet',      field_lims, False),
        ([fields['cp_pred'], fields['cf_pred']], 'jet',      field_lims, False),
        ([fields['cp_err'],  fields['cf_err']],  PASTEL_ERR, err_lims,   True),
    ]
    row_h = (GRID['top'] - GRID['bottom']) / n_row

    for row_i, (arrays, cmap, cell_lims, symmetric) in enumerate(rows):
        fig.text(0.012, GRID['top'] - (row_i + 0.5) * row_h, row_labels[row_i],
                 va='center', ha='center', fontsize=10, fontweight='bold',
                 rotation=90)
        for ci, values in enumerate(arrays):
            vmin, vmax = ((-cell_lims[ci], cell_lims[ci]) if symmetric
                          else cell_lims[ci])
            _draw_pair(fig, gs[row_i, ci], X, Y, Z, values, cmap, vmin, vmax,
                       (xlim, ylim, zlim), annotate=(row_i == 0))

    # Two colorbar strips below the grid: the field scale (shared by the CFD row
    # and the model row, which is what makes those two panels comparable), then
    # the error scale, symmetric about zero.
    col_w = (GRID['right'] - GRID['left']) / n_col
    strips = [
        (0.086, 'jet', [(v[0], v[1]) for v in field_lims],
         [f'{lab} (CFD / model)' for lab in col_labels]),
        (0.030, PASTEL_ERR, [(-e, e) for e in err_lims],
         [f'error {lab} (CFD $-$ model)' for lab in col_labels]),
    ]
    for y0, cmap, ranges, labels in strips:
        for ci, ((vmin, vmax), label) in enumerate(zip(ranges, labels)):
            cax = fig.add_axes([GRID['left'] + (ci + 0.16) * col_w, y0,
                                0.68 * col_w, 0.011])
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
            cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
            cb.set_label(label, size=8)
            cb.ax.tick_params(labelsize=6)
            cb.ax.xaxis.set_label_position('top')

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — parity
# ──────────────────────────────────────────────────────────────────────────────

def plot_combined_conditions(results, sims, coeff, field_lim, err_lim,
                             y_halfwidth, model_tag='AeroSurrogate v2'):
    """One figure, one row per flight condition, columns CFD | model | error.

    This is the v1 ``cp_comparison_conditions`` layout, with one change: v1 gave
    every row its own pair of colorbars, which silently rescaled each condition
    and made the rows impossible to read against each other — a row could look
    identical to its neighbour while spanning half the Cp range.  Here all rows
    share one field scale and one error scale, so a redder error panel really
    does mean a worse prediction.
    """
    key      = 'cp' if coeff == 'Cp' else 'cf'
    label    = r'$C_p$' if coeff == 'Cp' else r'$|C_f|$'
    n_row    = len(sims)
    n_col    = 3
    col_labels = ['CFD truth', model_tag, 'Error (CFD $-$ model)']

    X, Y, Z = results['_coords']
    xlim = (float(X.min()) + CAM['xoffsets'][0], float(X.max()) + CAM['xoffsets'][1])
    ylim = (float(Y.min()), float(Y.min()) + y_halfwidth)
    zlim = (float(Z.min()) + CAM['zoffsets'][0], float(Z.max()) + CAM['zoffsets'][1])

    grid = dict(GRID)
    grid.update(top=0.945, bottom=0.085 if n_row > 3 else 0.145)

    fig = plt.figure(figsize=(CELL_W * n_col, CELL_H * n_row))
    fig.suptitle(f'{model_tag} — {label} across the Mach envelope  '
                 f'(shared colour scale)',
                 fontsize=13, fontweight='bold', y=0.985)
    gs = gridspec.GridSpec(n_row, n_col, figure=fig, **grid)

    for ci, txt in enumerate(col_labels):
        fig.text(grid['left'] + (ci + 0.5) * (grid['right'] - grid['left']) / n_col,
                 grid['top'] + 0.012, txt, ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    row_h = (grid['top'] - grid['bottom']) / n_row
    for row_i, s in enumerate(sims):
        res  = results[s]
        cond = res['cond']
        cells = [(res[f'{key}_true'], 'jet', field_lim),
                 (res[f'{key}_pred'], 'jet', field_lim),
                 (res[f'{key}_true'] - res[f'{key}_pred'], PASTEL_ERR,
                  (-err_lim, err_lim))]
        fig.text(0.012, grid['top'] - (row_i + 0.5) * row_h,
                 f'$M_\\infty$={cond["Mach"]:.2f}  AoA={cond["AoA"]:.1f}°  '
                 f'$p_i$={cond["Pi"]:.1f}×10⁵\nsim {s}   shock '
                 f'{100 * res["shock_true"].mean():.1f}%',
                 va='center', ha='center', fontsize=9, rotation=90)
        for ci, (values, cmap, (vmin, vmax)) in enumerate(cells):
            _draw_pair(fig, gs[row_i, ci], X, Y, Z, values, cmap, vmin, vmax,
                       (xlim, ylim, zlim), annotate=(row_i == 0))

    col_w = (grid['right'] - grid['left']) / n_col
    bars = [(0, 2, 'jet', field_lim, f'{label} (CFD / model)'),
            (2, 1, PASTEL_ERR, (-err_lim, err_lim),
             f'error {label} (CFD $-$ model)')]
    for ci, span, cmap, (vmin, vmax), txt in bars:
        cax = fig.add_axes([grid['left'] + (ci + 0.18) * col_w,
                            0.030 if n_row > 3 else 0.055,
                            (span - 0.36) * col_w, 0.009])
        sm  = plt.cm.ScalarMappable(
            cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
        cb  = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cb.set_label(txt, size=9)
        cb.ax.tick_params(labelsize=7)
        cb.ax.xaxis.set_label_position('top')

    return fig


def plot_parity(Y_true, Y_pred, sims, model_tag='AeroSurrogate v2',
                max_points=200_000, clip_pct=99.9, seed=SEED):
    """2x2 parity, one panel per coefficient, in PHYSICAL units.

    Metrics use every point; only the scatter is subsampled, so what the title
    reports is not a property of the sample drawn.
    """
    r2   = r2_score(Y_true, Y_pred)
    rmse = rmse_score(Y_true, Y_pred)

    n = len(Y_true)
    if max_points and n > max_points:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n, max_points, replace=False)
        n_drawn = max_points
    else:
        sel = slice(None)
        n_drawn = n

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        t, p = Y_true[:, i], Y_pred[:, i]

        # Robust limits: the friction head reconstructs its magnitude through an
        # exp(), so a handful of pathological nodes can otherwise squash every
        # real point into a dot at the origin.
        if clip_pct and clip_pct < 100:
            lo = float(min(np.percentile(t, 100 - clip_pct), np.percentile(p, 100 - clip_pct)))
            hi = float(max(np.percentile(t, clip_pct),       np.percentile(p, clip_pct)))
        else:
            lo = float(min(t.min(), p.min()))
            hi = float(max(t.max(), p.max()))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        lo, hi = lo - pad, hi + pad
        outside = int(np.count_nonzero((t < lo) | (t > hi) | (p < lo) | (p > hi)))

        ax.scatter(t[sel], p[sel], alpha=0.3, s=1, rasterized=True)
        ax.plot([lo, hi], [lo, hi], 'r--', lw=2, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_box_aspect(1)      # square panel, so the identity line reads at 45°
        ax.locator_params(axis='both', nbins=6)   # friction ticks are long; keep
                                                  # them from running together
        ax.set_xlabel(f'CFD {COEFF_LABELS[i]}')
        ax.set_ylabel(f'{model_tag} {COEFF_LABELS[i]}')
        title = f'{COEFF_NAMES[i]}   $R^2$={r2[i]:.4f}   RMSE={rmse[i]:.6f}'
        if outside:
            title += f'   ({outside} of {n:,} pts outside axes)'
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

    sim_str = ', '.join(str(s) for s in sims)
    # Say how many dots are actually on the page: the metrics are over every
    # node, the scatter is a subsample, and quoting only n would let a reader
    # take the cloud's density for the full data set.
    drawn = ('' if n_drawn == n
             else f' ({n_drawn:,} drawn)')
    fig.suptitle(
        f'{model_tag} — parity on physical units | test sims {sim_str} '
        f'| $R^2$/RMSE over {n:,} nodes{drawn}',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sims', type=int, nargs='+', default=None, metavar='IDX',
                   help='Test sim indices to render (0-155). Default: the six '
                        f'flight conditions the paper reports, {DEFAULT_SIMS}.')
    p.add_argument('--ckpt', default=str(DEFAULT_CKPT),
                   help=f'Checkpoint to render. Default {DEFAULT_CKPT.name} (run 4, '
                        'the model the paper reports).')
    p.add_argument('--device', default='cpu', help='torch device (default cpu)')
    p.add_argument('--out-dir', default=str(PLOT_DIR),
                   help='Where the PNGs go (default outputs/plots/)')
    p.add_argument('--tag', default=None,
                   help='Suffix appended to every filename. Defaults to the '
                        'checkpoint stem when a non-default checkpoint is used, so '
                        'two checkpoints never overwrite each other.')
    p.add_argument('--cp-lim', type=float, nargs=2, default=(-1.0, 1.0),
                   metavar=('VMIN', 'VMAX'),
                   help='Shared colour range for the Cp CFD and model panels '
                        '(default -1 1, the v1 paper scale).')
    p.add_argument('--cf-lim', type=float, nargs=2, default=(0.0, 0.007),
                   metavar=('VMIN', 'VMAX'),
                   help='Shared colour range for the |Cf| CFD and model panels '
                        '(default 0 0.007).')
    p.add_argument('--auto-scale', action='store_true',
                   help='Derive the field colour ranges from the data (2nd/98th '
                        'percentile of truth and prediction pooled over the '
                        'selected sims) instead of using the fixed defaults.')
    p.add_argument('--err-pct', type=float, default=98.0,
                   help='Percentile of |error| that sets the half-range of the '
                        'symmetric error scale (default 98).')
    p.add_argument('--err-lim', type=float, nargs=2, default=None,
                   metavar=('CP', 'CF'),
                   help='Override the error half-ranges for Cp and |Cf|.')
    p.add_argument('--stride', type=int, default=1,
                   help='Plot every Nth node in the 3-D views. Draft-mode speedup; '
                        'metrics always use every node. Default 1.')
    p.add_argument('--parity-points', type=int, default=200_000,
                   help='Nodes drawn in the parity scatter (metrics use all). '
                        'Default 200000; 0 draws everything.')
    p.add_argument('--parity-clip', type=float, default=99.9,
                   help='Percentile bounding the parity axes (100 = full range). '
                        'Default 99.9.')
    p.add_argument('--dpi', type=int, default=300, help='Surface-field DPI (default 300)')
    p.add_argument('--parity-dpi', type=int, default=150, help='Parity DPI (default 150)')
    p.add_argument('--no-parity', action='store_true', help='Skip the parity figure')
    p.add_argument('--no-surface', action='store_true', help='Skip the surface figures')
    p.add_argument('--no-combined', action='store_true',
                   help='Skip the combined figure that puts every condition in one '
                        'image, one row per condition (columns CFD | model | error).')
    p.add_argument('--allow-partial-load', action='store_true',
                   help='Tolerate a checkpoint that does not fully match the model.')
    p.add_argument('--list-sims', action='store_true',
                   help='Print the test conditions with their sim index and exit.')
    return p.parse_args()


def main():
    args = parse_args()
    meta = load_condition_table()

    if args.list_sims:
        val = set(validation_sim_indices())
        print(f'{"Idx":>4}  {"Mach":>5}  {"AoA":>6}  {"Pi":>4}  {"w":>4}  note')
        print('-' * 46)
        for i in range(len(meta)):
            c = sim_conditions(meta, i)
            note = 'val sim' if i in val else ''
            print(f'{i:>4}  {c["Mach"]:>5.2f}  {c["AoA"]:>6.1f}  {c["Pi"]:>4.1f}  '
                  f'{c["weight"]:>4.1f}  {note}')
        return

    sims = args.sims
    if sims is None:
        _check_default_conditions(meta)
        sims = list(DEFAULT_SIMS)
    for s in sims:
        if not 0 <= s < len(meta):
            raise SystemExit(f'sim {s} out of range 0-{len(meta) - 1}')

    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    if not ckpt.exists():
        raise SystemExit(f'checkpoint not found: {ckpt}')

    tag = args.tag
    if tag is None:
        tag = '' if ckpt.resolve() == DEFAULT_CKPT.resolve() else f'_{ckpt.stem}'
    elif tag and not tag.startswith('_'):
        tag = f'_{tag}'

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f'Checkpoint: {ckpt}')
    print(f'Device:     {device}')
    scaler, test_ds, model = build_stack(ckpt, device, args.allow_partial_load)

    val_sims = set(validation_sim_indices())
    flagged  = [s for s in sims if s in val_sims]
    if flagged:
        print(f'\n  CAVEAT: sims {flagged} are among the 16 early stopping selected '
              f'on.\n          Their agreement is optimistic — say so in the caption, '
              f'or pick\n          replacements from the held-out sims.')

    # ── predict every selected sim first: the colour scales are shared across
    #    the whole selection, so nothing can be drawn until all of them exist.
    print(f'\nPredicting {len(sims)} sim(s)...')
    coords  = surface_coords(test_ds, sims[0])   # geometry is sim-independent
    results = {}
    for s in sims:
        t0 = time.time()
        res = predict_sim(model, test_ds, scaler, s, device)
        res['cond'] = sim_conditions(meta, s)
        # The figure title comes from dataset.csv but the physics comes from
        # X_test.npy; if the two ever disagree the caption would describe a flow
        # the model was never shown.
        for k, tol in COND_TOL.items():
            if abs(res['cond'][k] - res['feat_cond'][k]) > tol:
                raise SystemExit(
                    f'sim {s}: dataset.csv says {k}={res["cond"][k]} but the node '
                    f'features say {k}={res["feat_cond"][k]}. The CSV row order no '
                    f'longer matches X_test.npy — every caption would be wrong.')
        results[s] = res
        r2 = r2_score(res['Y_true'], res['Y_pred'])
        cf_t, cf_p = cf_magnitude(res['Y_true']), cf_magnitude(res['Y_pred'])
        res['cf_true'], res['cf_pred'] = cf_t, cf_p
        r2_cf = float(r2_score(cf_t[:, None], cf_p[:, None])[0])
        c = res['cond']
        print(f"  sim {s:>3}  M={c['Mach']:.2f} AoA={c['AoA']:>6.1f}° Pi={c['Pi']:.1f}  "
              f"shock {100 * res['shock_true'].mean():5.1f}%  "
              f"R²: Cp={r2[0]:.4f} Cfx={r2[1]:.4f} Cfy={r2[2]:.4f} Cfz={r2[3]:.4f} "
              f"|Cf|={r2_cf:.4f}  ({time.time() - t0:.0f}s)")

    # ── colour scales, pooled over the selection ─────────────────────────────
    cp_true_all = np.concatenate([results[s]['Y_true'][:, 0] for s in sims])
    cp_pred_all = np.concatenate([results[s]['Y_pred'][:, 0] for s in sims])
    cf_true_all = np.concatenate([results[s]['cf_true'] for s in sims])
    cf_pred_all = np.concatenate([results[s]['cf_pred'] for s in sims])

    if args.auto_scale:
        def _rng(a, b):
            both = np.concatenate([a, b])
            return (float(np.percentile(both, 2)), float(np.percentile(both, 98)))
        field_lims = [_rng(cp_true_all, cp_pred_all), _rng(cf_true_all, cf_pred_all)]
    else:
        field_lims = [tuple(args.cp_lim), tuple(args.cf_lim)]

    cp_err_all = cp_true_all - cp_pred_all
    cf_err_all = cf_true_all - cf_pred_all
    if args.err_lim is not None:
        err_lims = [float(args.err_lim[0]), float(args.err_lim[1])]
    else:
        err_lims = [float(np.percentile(np.abs(cp_err_all), args.err_pct)),
                    float(np.percentile(np.abs(cf_err_all), args.err_pct))]

    def _clipped(arr, lo, hi):
        return 100.0 * np.count_nonzero((arr < lo) | (arr > hi)) / arr.size

    print('\nColour scales (shared by every figure in this run):')
    print(f'  Cp    field [{field_lims[0][0]:+.4f}, {field_lims[0][1]:+.4f}]  '
          f'error ±{err_lims[0]:.4f}  '
          f'({_clipped(cp_true_all, *field_lims[0]):.2f}% of CFD nodes saturate, '
          f'{_clipped(cp_err_all, -err_lims[0], err_lims[0]):.2f}% of errors)')
    print(f'  |Cf|  field [{field_lims[1][0]:+.4f}, {field_lims[1][1]:+.4f}]  '
          f'error ±{err_lims[1]:.4f}  '
          f'({_clipped(cf_true_all, *field_lims[1]):.2f}% of CFD nodes saturate, '
          f'{_clipped(cf_err_all, -err_lims[1], err_lims[1]):.2f}% of errors)')

    written = []

    # ── figure 1: surface fields, one per sim ────────────────────────────────
    if not args.no_surface:
        st = max(1, args.stride)
        xyz = tuple(c[::st] for c in coords)
        if st > 1:
            print(f'\n  stride={st}: drawing {len(xyz[0]):,} of {len(coords[0]):,} nodes')

        # Calibrate the view box once: this is what makes the two mirrored
        # half-model views meet on the seam and read as one aircraft.  Solve on
        # the coordinates that are actually drawn (xyz, i.e. after --stride), not
        # on the full set: a strided subsample has slightly different x/z extrema
        # and the seam is only exact for the limits the figure really uses.
        y_halfwidth, solved = _solve_y_halfwidth(
            fig_size=(CELL_W * N_FIELD_COLS, CELL_H * N_FIELD_ROWS), gs_kw=GRID,
            n_row=N_FIELD_ROWS, n_col=N_FIELD_COLS,
            xlim=(float(xyz[0].min()), float(xyz[0].max())),
            zlim=(float(xyz[2].min()), float(xyz[2].max())),
            y_root=float(xyz[1].min()))
        y_span = float(xyz[1].max() - xyz[1].min())
        if not solved:
            print('\n  WARNING: could not bracket the y half-width that makes the '
                  'two mirrored views\n           abut — the matplotlib 3-D '
                  f'projection has probably changed. Falling back\n           to v1\'s '
                  f'{y_halfwidth:.1f} m, which for this layout leaves the two halves '
                  '~1.6 in\n           apart and crops the model: the figures will NOT '
                  'read as a whole aircraft.')
        elif y_halfwidth < y_span:
            print(f'\n  WARNING: solved y half-width {y_halfwidth:.2f} m is smaller '
                  f'than the {y_span:.2f} m half-span,\n           so the outboard wing '
                  'falls outside the view box and (clip_on=False)\n           spills '
                  'across the seam into the other half.')
        print(f'\nRendering surface fields (view box y half-width '
              f'{y_halfwidth:.2f} m, '
              f'{"solved so the two views abut" if solved else "NOT solved — see warning"})...')
        for s in sims:
            t0  = time.time()
            res = results[s]
            fields = dict(
                cp_true=res['Y_true'][::st, 0],
                cp_pred=res['Y_pred'][::st, 0],
                cp_err=(res['Y_true'][:, 0] - res['Y_pred'][:, 0])[::st],
                cf_true=res['cf_true'][::st],
                cf_pred=res['cf_pred'][::st],
                cf_err=(res['cf_true'] - res['cf_pred'])[::st],
            )
            fig = plot_surface_fields(xyz, fields, res['cond'], s,
                                      field_lims, err_lims, y_halfwidth)
            out = out_dir / f'v2_surface_fields_sim{s:03d}{tag}.png'
            fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
            plt.close(fig)
            written.append(out)
            print(f'  saved {out.name}  ({out.stat().st_size / 1e6:.2f} MB, '
                  f'{time.time() - t0:.0f}s)')

    # ── figure 1b: every condition in one figure, one row each ───────────────
    if not args.no_combined:
        st  = max(1, args.stride)
        xyz = tuple(c[::st] for c in coords)
        # Solve the seam for THIS layout: the combined figure has a different
        # row/column count, and the probe geometry has to match what is drawn.
        y_hw_c, solved_c = _solve_y_halfwidth(
            fig_size=(CELL_W * 3, CELL_H * len(sims)),
            gs_kw=dict(GRID, top=0.945, bottom=0.085 if len(sims) > 3 else 0.145),
            n_row=len(sims), n_col=3,
            xlim=(float(xyz[0].min()), float(xyz[0].max())),
            zlim=(float(xyz[2].min()), float(xyz[2].max())),
            y_root=float(xyz[1].min()))
        if not solved_c:
            print('\n  WARNING: seam not solved for the combined layout; the two '
                  'half-model views will not abut.')

        strided = {'_coords': xyz}
        for s in sims:
            res = results[s]
            strided[s] = dict(
                cond=res['cond'], shock_true=res['shock_true'],
                cp_true=res['Y_true'][::st, 0], cp_pred=res['Y_pred'][::st, 0],
                cf_true=res['cf_true'][::st],   cf_pred=res['cf_pred'][::st],
            )

        print(f'\nRendering combined ({len(sims)} conditions in one figure)...')
        for ci, coeff in enumerate(['Cp', 'Cf']):
            t0  = time.time()
            fig = plot_combined_conditions(
                strided, sims, coeff, field_lims[ci], err_lims[ci], y_hw_c)
            out = out_dir / f'v2_conditions_{coeff.lower()}{tag}.png'
            fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
            plt.close(fig)
            written.append(out)
            print(f'  saved {out.name}  ({out.stat().st_size / 1e6:.2f} MB, '
                  f'{time.time() - t0:.0f}s)')

    # ── figure 2: parity over the whole selection ────────────────────────────
    if not args.no_parity:
        print('\nRendering parity...')
        Y_true = np.concatenate([results[s]['Y_true'] for s in sims])
        Y_pred = np.concatenate([results[s]['Y_pred'] for s in sims])
        fig = plot_parity(Y_true, Y_pred, sims,
                          max_points=args.parity_points,
                          clip_pct=args.parity_clip)
        sim_str = ('_'.join(str(s) for s in sims) if len(sims) <= 8
                   else f'{len(sims)}sims_{min(sims)}to{max(sims)}')
        out = out_dir / f'v2_parity_sims_{sim_str}{tag}.png'
        fig.savefig(out, dpi=args.parity_dpi, bbox_inches='tight')
        plt.close(fig)
        written.append(out)
        print(f'  saved {out.name}  ({out.stat().st_size / 1e6:.2f} MB)')

    print(f'\n{len(written)} file(s) written to {out_dir}')
    for w in written:
        print(f'  {w}  ({w.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
