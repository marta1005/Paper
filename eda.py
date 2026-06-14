#!/usr/bin/env python3
"""
Exploratory Data Analysis — ONERA CFD dataset
Genera figuras listas para paper en outputs/plots/eda/

Usage:
    python eda.py
    python eda.py --samples 500000   # más puntos (más lento)
    python eda.py --skip-derived     # solo features originales
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from pathlib import Path
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
X_FEATURES = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'Mach', 'AoA', 'Pi (×10⁻⁵)']
Y_FEATURES = ['Cp', 'Cfx', 'Cfy', 'Cfz']
DERIVED_FEATURES = ['q_dyn', 'Pi_norm', 'AoA_sin', 'L_factor', 'Cp_crit']
ALL_X_FEATURES = X_FEATURES + DERIVED_FEATURES

OUT_DIR = Path('outputs/plots/eda')
OUT_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f8f8',
    'axes.grid': True,
    'grid.color': 'white',
    'grid.linewidth': 0.8,
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
}
plt.rcParams.update(STYLE)

COLORS = plt.cm.tab10.colors


def load_sample(n=300_000):
    X_raw = np.load('data/X_train.npy', mmap_mode='r')
    Y_raw = np.load('data/Ytrain.npy',  mmap_mode='r')
    X_der = np.load('data/X_train_derived.npy', mmap_mode='r')

    total = len(X_raw)
    idx   = np.random.choice(total, min(n, total), replace=False)
    return X_raw[idx], Y_raw[idx], X_der[idx]


def savefig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {path}')


# ── 1. Summary statistics table ───────────────────────────────────────────────
def plot_summary_table(X, Y, Xd):
    data = np.hstack([X, Y])
    names = X_FEATURES + Y_FEATURES

    rows = []
    for i, n in enumerate(names):
        col = data[:, i]
        rows.append([n,
                     f'{col.min():.4g}', f'{col.max():.4g}',
                     f'{col.mean():.4g}', f'{col.std():.4g}',
                     f'{np.percentile(col, 5):.4g}', f'{np.percentile(col, 95):.4g}'])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis('off')
    cols = ['Feature', 'Min', 'Max', 'Mean', 'Std', 'P5', 'P95']
    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.4)

    # colour header row
    for j in range(len(cols)):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # colour X rows light blue, Y rows light orange
    for i in range(1, len(names) + 1):
        colour = '#dce8f5' if i <= len(X_FEATURES) else '#fde8cc'
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(colour)

    ax.set_title('Dataset summary statistics — inputs (blue) & outputs (orange)',
                 fontsize=11, pad=12, fontweight='bold')
    savefig(fig, '01_summary_statistics.png')


# ── 2. Input feature distributions ───────────────────────────────────────────
def plot_input_distributions(X):
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    axes = axes.flat
    for i, (ax, name) in enumerate(zip(axes, X_FEATURES)):
        col = X[:, i]
        ax.hist(col, bins=60, color=COLORS[i % 10], edgecolor='none', alpha=0.85)
        ax.set_title(name)
        ax.set_ylabel('Count')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    fig.suptitle('Distribution of input features (X)', fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    savefig(fig, '02_input_distributions.png')


# ── 3. Output feature distributions ──────────────────────────────────────────
def plot_output_distributions(Y):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for i, (ax, name) in enumerate(zip(axes, Y_FEATURES)):
        col = Y[:, i]
        ax.hist(col, bins=80, color=COLORS[i + 4], edgecolor='none', alpha=0.85)
        ax.set_title(name)
        ax.set_ylabel('Count')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.text(0.97, 0.97, f'μ={col.mean():.3f}\nσ={col.std():.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
                bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    fig.suptitle('Distribution of output targets (Y)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    savefig(fig, '03_output_distributions.png')


# ── 4. Cp vs Mach (scatter, coloured by AoA) ──────────────────────────────────
def plot_cp_vs_mach(X, Y):
    Mach = X[:, 6]
    AoA  = X[:, 7]
    Cp   = Y[:, 0]

    # downsample for scatter
    n   = min(50_000, len(Mach))
    idx = np.random.choice(len(Mach), n, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sc = axes[0].scatter(Mach[idx], Cp[idx], c=AoA[idx], cmap='coolwarm',
                         s=1.5, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=axes[0], label='AoA (°)')
    axes[0].set_xlabel('Mach')
    axes[0].set_ylabel('Cp')
    axes[0].set_title('Cp vs Mach, coloured by AoA')

    # 2D density
    xb = np.linspace(Mach.min(), Mach.max(), 80)
    yb = np.linspace(np.percentile(Cp, 1), np.percentile(Cp, 99), 80)
    h, xe, ye = np.histogram2d(Mach, Cp, bins=[xb, yb])
    im = axes[1].pcolormesh(xe, ye, h.T, cmap='Blues', norm=LogNorm())
    plt.colorbar(im, ax=axes[1], label='Count (log)')
    axes[1].set_xlabel('Mach')
    axes[1].set_ylabel('Cp')
    axes[1].set_title('Cp vs Mach — density (log scale)')

    fig.suptitle('Pressure coefficient vs Mach number', fontsize=12, fontweight='bold')
    fig.tight_layout()
    savefig(fig, '04_cp_vs_mach.png')


# ── 5. Correlation matrix ─────────────────────────────────────────────────────
def plot_correlation_matrix(X, Y):
    data   = np.hstack([X, Y])
    labels = X_FEATURES + Y_FEATURES
    corr   = np.corrcoef(data.T)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=6.5, color='black' if abs(v) < 0.7 else 'white')

    # separator between X and Y
    sep = len(X_FEATURES) - 0.5
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(False)
    ax.axhline(sep, color='black', linewidth=1.5)
    ax.axvline(sep, color='black', linewidth=1.5)

    ax.set_title('Pearson correlation matrix — X | Y', fontsize=12, fontweight='bold')
    fig.tight_layout()
    savefig(fig, '05_correlation_matrix.png')


# ── 6. Derived features distributions ────────────────────────────────────────
def plot_derived_distributions(Xd):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    for i, (ax, name) in enumerate(zip(axes, DERIVED_FEATURES)):
        col = Xd[:, 9 + i]
        col = col[np.isfinite(col)]
        ax.hist(col, bins=60, color=COLORS[(i + 2) % 10], edgecolor='none', alpha=0.85)
        ax.set_title(name)
        ax.set_ylabel('Count')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.text(0.97, 0.97, f'μ={col.mean():.3f}\nσ={col.std():.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
                bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    fig.suptitle('Distribution of physics-derived features (computed from X only)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    savefig(fig, '06_derived_feature_distributions.png')


# ── 7. Flight condition coverage ──────────────────────────────────────────────
def plot_flight_coverage(X):
    Mach = X[:, 6]
    AoA  = X[:, 7]
    Pi   = X[:, 8]

    unique_mach = np.unique(np.round(Mach, 2))
    unique_aoa  = np.unique(np.round(AoA, 1))

    fig = plt.figure(figsize=(14, 4))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Mach-AoA grid coverage
    ax0 = fig.add_subplot(gs[0])
    h, xe, ye = np.histogram2d(Mach, AoA, bins=40)
    ax0.pcolormesh(xe, ye, h.T, cmap='YlOrRd')
    ax0.set_xlabel('Mach')
    ax0.set_ylabel('AoA (°)')
    ax0.set_title(f'Mach–AoA coverage\n({len(unique_mach)} Mach × {len(unique_aoa)} AoA values)')

    # Points per Mach condition
    ax1 = fig.add_subplot(gs[1])
    mach_vals, mach_counts = np.unique(np.round(Mach, 2), return_counts=True)
    ax1.bar(range(len(mach_vals)), mach_counts / 1000, color='steelblue', edgecolor='none')
    ax1.set_xticks(range(0, len(mach_vals), max(1, len(mach_vals) // 10)))
    ax1.set_xticklabels([f'{mach_vals[i]:.2f}' for i in range(0, len(mach_vals), max(1, len(mach_vals) // 10))],
                        rotation=45, ha='right', fontsize=7)
    ax1.set_xlabel('Mach')
    ax1.set_ylabel('Points (×10³)')
    ax1.set_title('Points per Mach condition')

    # Pi distribution
    ax2 = fig.add_subplot(gs[2])
    ax2.hist(Pi, bins=40, color='darkorange', edgecolor='none', alpha=0.85)
    ax2.set_xlabel('Pi (×10⁻⁵)')
    ax2.set_ylabel('Count')
    ax2.set_title('Stagnation pressure distribution')

    fig.suptitle('Flight condition coverage', fontsize=12, fontweight='bold', y=1.03)
    savefig(fig, '07_flight_coverage.png')


# ── 8. Surface geometry ───────────────────────────────────────────────────────
def plot_geometry(X):
    x, y, z   = X[:, 0], X[:, 1], X[:, 2]
    nx, ny, nz = X[:, 3], X[:, 4], X[:, 5]

    n = min(30_000, len(x))
    idx = np.random.choice(len(x), n, replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # XY projection (planform)
    axes[0].scatter(x[idx], y[idx], s=0.3, c=z[idx], cmap='viridis',
                    alpha=0.5, rasterized=True)
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    axes[0].set_title('Surface points — XY (planform view)')
    axes[0].set_aspect('equal')

    # XZ projection (side)
    axes[1].scatter(x[idx], z[idx], s=0.3, c=y[idx], cmap='plasma',
                    alpha=0.5, rasterized=True)
    axes[1].set_xlabel('x'); axes[1].set_ylabel('z')
    axes[1].set_title('Surface points — XZ (side view)')
    axes[1].set_aspect('equal')

    # Normal vector components distribution
    for k, (comp, label) in enumerate(zip([nx, ny, nz], ['nx', 'ny', 'nz'])):
        axes[2].hist(comp[idx], bins=60, alpha=0.6, label=label, edgecolor='none')
    axes[2].set_xlabel('Component value')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Surface normal components')
    axes[2].legend(fontsize=8)

    fig.suptitle('Geometry: surface point distribution', fontsize=12, fontweight='bold')
    fig.tight_layout()
    savefig(fig, '08_geometry.png')


# ── 9. Cp spatial distribution (coloured by Cp value) ─────────────────────────
def plot_cp_surface(X, Y):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    Cp       = Y[:, 0]

    n   = min(40_000, len(x))
    idx = np.random.choice(len(x), n, replace=False)
    vmin, vmax = np.percentile(Cp, 2), np.percentile(Cp, 98)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sc0 = axes[0].scatter(x[idx], y[idx], c=Cp[idx], cmap='RdBu_r',
                          s=0.8, vmin=vmin, vmax=vmax, rasterized=True)
    plt.colorbar(sc0, ax=axes[0], label='Cp')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    axes[0].set_title('Cp on XY plane')
    axes[0].set_aspect('equal')

    sc1 = axes[1].scatter(x[idx], z[idx], c=Cp[idx], cmap='RdBu_r',
                          s=0.8, vmin=vmin, vmax=vmax, rasterized=True)
    plt.colorbar(sc1, ax=axes[1], label='Cp')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('z')
    axes[1].set_title('Cp on XZ plane')
    axes[1].set_aspect('equal')

    fig.suptitle('Pressure coefficient spatial distribution', fontsize=12, fontweight='bold')
    fig.tight_layout()
    savefig(fig, '09_cp_surface.png')


# ── 10. Cp percentiles across Mach ───────────────────────────────────────────
def plot_cp_mach_percentiles(X, Y):
    Mach = X[:, 7]
    Cp   = Y[:, 0]

    mach_vals = np.unique(np.round(Mach, 2))
    if len(mach_vals) > 30:
        mach_vals = np.percentile(mach_vals, np.linspace(0, 100, 30))

    p5, p50, p95, means = [], [], [], []
    for m in mach_vals:
        mask = np.abs(Mach - m) < 0.02
        if mask.sum() < 50:
            continue
        sub = Cp[mask]
        p5.append(np.percentile(sub, 5))
        p50.append(np.percentile(sub, 50))
        p95.append(np.percentile(sub, 95))
        means.append(sub.mean())

    mach_vals = mach_vals[:len(p50)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(mach_vals, p5, p95, alpha=0.2, color='steelblue', label='P5–P95')
    ax.plot(mach_vals, p50, color='steelblue', linewidth=2, label='Median Cp')
    ax.plot(mach_vals, means, color='tomato', linewidth=1.5, linestyle='--', label='Mean Cp')
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Mach')
    ax.set_ylabel('Cp')
    ax.legend()
    ax.set_title('Cp distribution across Mach numbers (P5 / median / P95)', fontsize=11, fontweight='bold')
    fig.tight_layout()
    savefig(fig, '10_cp_vs_mach_percentiles.png')


# ── 11. Dataset size & split ──────────────────────────────────────────────────
def plot_dataset_info():
    X_tr = np.load('data/X_train.npy', mmap_mode='r')
    X_te = np.load('data/X_test.npy',  mmap_mode='r')
    Y_tr = np.load('data/Ytrain.npy',  mmap_mode='r')

    train_n, test_n = len(X_tr), len(X_te)
    total = train_n + test_n

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Pie chart
    axes[0].pie([train_n, test_n],
                labels=[f'Train\n{train_n/1e6:.1f}M', f'Test\n{test_n/1e6:.1f}M'],
                autopct='%1.1f%%', colors=['steelblue', 'tomato'],
                startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    axes[0].set_title(f'Train/Test split\nTotal: {total/1e6:.1f}M points')

    # Feature count bar
    feature_groups = {
        'Geometry\n(x,y,z)': 3,
        'Normals\n(nx,ny,nz)': 3,
        'Flight cond.\n(AoA,Mach,Pi)': 3,
        'Derived\nphysics': 5,
        'Targets\n(Cp,Cf×3)': 4,
    }
    names = list(feature_groups.keys())
    counts = list(feature_groups.values())
    colors_bar = ['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974']
    axes[1].bar(names, counts, color=colors_bar, edgecolor='white', linewidth=0.8)
    axes[1].set_ylabel('Number of features')
    axes[1].set_title('Feature breakdown')
    for i, (n, c) in enumerate(zip(names, counts)):
        axes[1].text(i, c + 0.05, str(c), ha='center', fontweight='bold')

    fig.suptitle('Dataset overview', fontsize=12, fontweight='bold')
    fig.tight_layout()
    savefig(fig, '00_dataset_overview.png')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',       type=int, default=300_000)
    parser.add_argument('--skip-derived',  action='store_true')
    args = parser.parse_args()

    print(f'Loading {args.samples:,} samples...')
    X, Y, Xd = load_sample(args.samples)
    print(f'  X={X.shape}  Y={Y.shape}  Xd={Xd.shape}')
    print(f'Generating plots in {OUT_DIR}/\n')

    plot_dataset_info()
    plot_summary_table(X, Y, Xd)
    plot_input_distributions(X)
    plot_output_distributions(Y)
    plot_cp_vs_mach(X, Y)
    plot_correlation_matrix(X, Y)
    plot_flight_coverage(X)
    plot_geometry(X)
    plot_cp_surface(X, Y)
    plot_cp_mach_percentiles(X, Y)

    if not args.skip_derived:
        plot_derived_distributions(Xd)

    print(f'\nDone. {len(list(OUT_DIR.glob("*.png")))} plots saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
