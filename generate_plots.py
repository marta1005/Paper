#!/usr/bin/env python3
"""
Genera plots exploratorios del dataset para el paper.
Uso: python generate_plots.py
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

DATA_DIR   = Path('data')
PLOTS_DIR  = Path('outputs/plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE = 300_000
np.random.seed(42)

print(f"Cargando datos (muestra={SAMPLE:,})...")
X_raw = np.load(DATA_DIR / 'X_train.npy',    mmap_mode='r')
Y_raw = np.load(DATA_DIR / 'Ytrain.npy',     mmap_mode='r')
has_derived = (DATA_DIR / 'X_train_derived.npy').exists()
if has_derived:
    X_der_full = np.load(DATA_DIR / 'X_train_derived.npy', mmap_mode='r')

idx = np.random.choice(len(X_raw), min(SAMPLE, len(X_raw)), replace=False)
X = np.asarray(X_raw[idx], dtype=np.float32)
Y = np.asarray(Y_raw[idx], dtype=np.float32)
if has_derived:
    Xd = np.nan_to_num(np.asarray(X_der_full[idx], dtype=np.float32), nan=0.0)

df_meta = pd.read_csv(DATA_DIR / 'dataset.csv')
feat_names    = ['Mach', 'AoA', 'Pi', 'x', 'y', 'z', 'nx', 'ny', 'nz']
target_names  = ['Cp', 'Cfx', 'Cfy', 'Cfz']
derived_names = ['M_local','grad_p','cp_loss','shock_ind','Cf_mag',
                 'q_dyn','Pi_norm','AoA_norm','grad_cf','L_factor']

# ── Plot 1: Espacio de diseño Mach–AoA de las simulaciones ──────────────────
print("Plot 1: espacio de diseño...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

train_m = df_meta[df_meta.get('Train', pd.Series(1, index=df_meta.index)) == 1] \
          if 'Train' in df_meta.columns else df_meta
test_m  = df_meta[df_meta['Train'] == 0] if 'Train' in df_meta.columns else pd.DataFrame()

ax = axes[0]
ax.scatter(train_m['Mach'], train_m['AoA'], s=25, alpha=0.7, label='Train', zorder=3)
if not test_m.empty:
    ax.scatter(test_m['Mach'], test_m['AoA'], s=40, marker='^', alpha=0.9, label='Test', zorder=4)
ax.axvline(1.0, color='k', lw=1.2, linestyle=':', label='Mach=1 (sónico)')
ax.set_xlabel('Mach'); ax.set_ylabel('AoA (°)')
ax.set_title(f'Espacio de diseño ({len(df_meta)} simulaciones)')
ax.legend()

ax = axes[1]
ax.hist(df_meta['Mach'], bins=25, alpha=0.6, label='Mach', color='steelblue')
ax2 = ax.twinx()
ax2.hist(df_meta['AoA'], bins=25, alpha=0.4, label='AoA (°)', color='coral')
ax.set_xlabel('Valor'); ax.set_ylabel('N sim (Mach)', color='steelblue')
ax2.set_ylabel('N sim (AoA)', color='coral')
ax.set_title('Distribución de condiciones de vuelo')
l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax.legend(l1+l2, lb1+lb2)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '01_design_space.png', bbox_inches='tight')
plt.close()
print("  → 01_design_space.png")

# ── Plot 2: Distribución de features de entrada ──────────────────────────────
print("Plot 2: distribución features entrada...")
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()
for i, name in enumerate(feat_names):
    ax = axes[i]
    data = X[:, i]
    lo, hi = np.percentile(data, 0.5), np.percentile(data, 99.5)
    ax.hist(data[(data >= lo) & (data <= hi)], bins=80, color='steelblue', alpha=0.75)
    ax.axvline(data.mean(),   color='red',    lw=1.5, linestyle='--', label=f'μ={data.mean():.3f}')
    ax.axvline(np.median(data), color='orange', lw=1.5, linestyle=':',  label=f'med={np.median(data):.3f}')
    ax.set_title(name); ax.legend(fontsize=8)
plt.suptitle(f'Distribución de features de entrada X — {SAMPLE//1000}K muestra', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(PLOTS_DIR / '02_input_features.png', bbox_inches='tight')
plt.close()
print("  → 02_input_features.png")

# ── Plot 3: Distribución de targets aerodinámicos ────────────────────────────
print("Plot 3: distribución targets...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, name in enumerate(target_names):
    ax = axes[i]
    data = Y[:, i]
    lo, hi = np.percentile(data, 1), np.percentile(data, 99)
    ax.hist(data[(data >= lo) & (data <= hi)], bins=100, color='coral', alpha=0.75)
    ax.axvline(data.mean(), color='navy', lw=2, linestyle='--', label=f'μ={data.mean():.4f}')
    ax.set_title(name); ax.set_xlabel(name); ax.legend()
plt.suptitle(f'Distribución de outputs aerodinámicos Y — {SAMPLE//1000}K muestra', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(PLOTS_DIR / '03_output_targets.png', bbox_inches='tight')
plt.close()
print("  → 03_output_targets.png")

# ── Plot 4: Distribución de features derivados ───────────────────────────────
if has_derived:
    print("Plot 4: features derivados...")
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()
    for i, name in enumerate(derived_names):
        ax = axes[i]
        data = Xd[:, 9 + i]
        lo, hi = np.percentile(data, 1), np.percentile(data, 99)
        color = 'darkorange' if name == 'shock_ind' else 'mediumpurple'
        ax.hist(data[(data >= lo) & (data <= hi)], bins=80, color=color, alpha=0.75)
        ax.set_title(name, fontsize=9)
    plt.suptitle(f'Features derivados (física) — {SAMPLE//1000}K muestra', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '04_derived_features.png', bbox_inches='tight')
    plt.close()
    print("  → 04_derived_features.png")

# ── Plot 5: Análisis de choque ───────────────────────────────────────────────
if has_derived:
    print("Plot 5: análisis de choque...")
    shock_raw  = Xd[:, 12]
    s_norm     = (shock_raw - shock_raw.mean()) / (shock_raw.std() + 1e-8)
    shock_lbl  = (s_norm > 0.5).astype(int)
    m_local    = Xd[:, 9]

    print(f"  Puntos en choque (σ>0.5): {shock_lbl.mean()*100:.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.hist(s_norm[(s_norm > -5) & (s_norm < 5)], bins=100, color='steelblue', alpha=0.7)
    ax.axvline(0.5, color='red', lw=2, linestyle='--', label='umbral=0.5σ')
    ax.set_xlabel('shock_indicator (normalizado)'); ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución shock_indicator'); ax.legend()

    ax = axes[1]
    cp = Y[:, 0]
    lo, hi = np.percentile(cp, 1), np.percentile(cp, 99)
    m0 = (shock_lbl == 0) & (cp >= lo) & (cp <= hi)
    m1 = (shock_lbl == 1) & (cp >= lo) & (cp <= hi)
    ax.hist(cp[m0], bins=80, alpha=0.6, color='steelblue', label='No choque', density=True)
    ax.hist(cp[m1], bins=80, alpha=0.6, color='red', label='Choque', density=True)
    ax.set_xlabel('Cp'); ax.set_ylabel('Densidad')
    ax.set_title('Cp: choque vs no choque'); ax.legend()

    ax = axes[2]
    lo, hi = np.percentile(m_local, 1), np.percentile(m_local, 99)
    mask_m = (m_local >= lo) & (m_local <= hi)
    ax.hist(m_local[mask_m & (shock_lbl == 0)], bins=80, alpha=0.6,
            color='steelblue', label='No choque', density=True)
    ax.hist(m_local[mask_m & (shock_lbl == 1)], bins=80, alpha=0.6,
            color='red', label='Choque', density=True)
    ax.axvline(1.0, color='k', lw=1.5, linestyle=':', label='M=1')
    ax.set_xlabel('M_local'); ax.set_ylabel('Densidad')
    ax.set_title('M_local: choque vs no choque'); ax.legend()

    plt.suptitle('Análisis de indicador de choque', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '05_shock_analysis.png', bbox_inches='tight')
    plt.close()
    print("  → 05_shock_analysis.png")

# ── Plot 6: Distribución espacial coloreada por Cp ──────────────────────────
print("Plot 6: distribución espacial...")
N_SPATIAL = 80_000
idx2 = np.random.choice(len(X), N_SPATIAL, replace=False)
xs, ys, zs = X[idx2, 3], X[idx2, 4], X[idx2, 5]
cp_s = Y[idx2, 0]
cp_c = np.clip((cp_s - np.percentile(cp_s, 5)) /
               (np.percentile(cp_s, 95) - np.percentile(cp_s, 5) + 1e-8), 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sc = axes[0].scatter(xs, ys, c=cp_c, cmap='RdBu_r', s=0.8, alpha=0.5)
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
axes[0].set_title('Vista xy — coloreado por Cp')
plt.colorbar(sc, ax=axes[0], label='Cp (norm)')

sc = axes[1].scatter(xs, zs, c=cp_c, cmap='RdBu_r', s=0.8, alpha=0.5)
axes[1].set_xlabel('x'); axes[1].set_ylabel('z')
axes[1].set_title('Vista xz — coloreado por Cp')
plt.colorbar(sc, ax=axes[1], label='Cp (norm)')

# AoA vs Mach coloreado por Cp medio por simulación
if 'Pi' in feat_names:
    mach_col = X[idx2, 0]
    aoa_col  = X[idx2, 1]
    sc = axes[2].scatter(mach_col, aoa_col, c=cp_c, cmap='RdBu_r', s=0.8, alpha=0.3)
    axes[2].set_xlabel('Mach'); axes[2].set_ylabel('AoA')
    axes[2].set_title('Mach vs AoA — coloreado por Cp')
    plt.colorbar(sc, ax=axes[2], label='Cp (norm)')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '06_spatial_distribution.png', bbox_inches='tight')
plt.close()
print("  → 06_spatial_distribution.png")

# ── Plot 7: Matriz de correlaciones ─────────────────────────────────────────
print("Plot 7: correlaciones...")
all_data  = np.hstack([X, Y])
all_names = feat_names + target_names
corr      = np.corrcoef(all_data.T)

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(all_names))); ax.set_xticklabels(all_names, rotation=45, ha='right')
ax.set_yticks(range(len(all_names))); ax.set_yticklabels(all_names)
plt.colorbar(im, ax=ax, label='Correlación de Pearson')
ax.set_title('Matriz de correlaciones X ∪ Y')
for i in range(len(all_names)):
    for j in range(len(all_names)):
        ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                fontsize=7, color='black' if abs(corr[i,j]) < 0.7 else 'white')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '07_correlation_matrix.png', bbox_inches='tight')
plt.close()
print("  → 07_correlation_matrix.png")

# ── Plot 8: Presión Pi vs Cp por régimen Mach ────────────────────────────────
print("Plot 8: Pi vs Cp por régimen Mach...")
mach_vals = X[:, 0]
pi_vals   = X[:, 2]
cp_vals   = Y[:, 0]

regimes = {
    'Subsónico (M<0.8)':   mach_vals < 0.8,
    'Transónico (0.8-1.2)': (mach_vals >= 0.8) & (mach_vals < 1.2),
    'Supersónico (M>1.2)':  mach_vals >= 1.2,
}
colors = ['steelblue', 'darkorange', 'firebrick']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for (label, mask), c in zip(regimes.items(), colors):
    if mask.sum() > 100:
        lo_p, hi_p = np.percentile(pi_vals[mask], 1), np.percentile(pi_vals[mask], 99)
        lo_c, hi_c = np.percentile(cp_vals[mask], 1), np.percentile(cp_vals[mask], 99)
        m2 = mask & (pi_vals >= lo_p) & (pi_vals <= hi_p) & (cp_vals >= lo_c) & (cp_vals <= hi_c)
        n = min(m2.sum(), 30000)
        idxr = np.random.choice(np.where(m2)[0], n, replace=False)
        ax.scatter(pi_vals[idxr], cp_vals[idxr], s=0.5, alpha=0.3, c=c, label=label)
ax.set_xlabel('Pi (presión de remanso)'); ax.set_ylabel('Cp')
ax.set_title('Pi vs Cp por régimen Mach'); ax.legend(markerscale=10)

ax = axes[1]
for (label, mask), c in zip(regimes.items(), colors):
    if mask.sum() > 100:
        lo_m, hi_m = np.percentile(mach_vals[mask], 0.5), np.percentile(mach_vals[mask], 99.5)
        lo_c, hi_c = np.percentile(cp_vals[mask], 1), np.percentile(cp_vals[mask], 99)
        m2 = mask & (mach_vals >= lo_m) & (mach_vals <= hi_m) & (cp_vals >= lo_c) & (cp_vals <= hi_c)
        n = min(m2.sum(), 30000)
        idxr = np.random.choice(np.where(m2)[0], n, replace=False)
        ax.scatter(mach_vals[idxr], cp_vals[idxr], s=0.5, alpha=0.3, c=c, label=label)
ax.set_xlabel('Mach'); ax.set_ylabel('Cp')
ax.set_title('Mach vs Cp (regímenes aerodinámicos)'); ax.legend(markerscale=10)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '08_regimes.png', bbox_inches='tight')
plt.close()
print("  → 08_regimes.png")

# ── Resumen ───────────────────────────────────────────────────────────────────
print("\n=== Dataset Summary ===")
print(f"  Train points total:   {len(X_raw):,}")
print(f"  Test points total:    ~40M (separado)")
print(f"  Simulations:          {len(df_meta)}")
print(f"  Mach range:           {df_meta['Mach'].min():.2f} – {df_meta['Mach'].max():.2f}")
print(f"  AoA range:            {df_meta['AoA'].min():.2f}° – {df_meta['AoA'].max():.2f}°")
if has_derived:
    print(f"  Shock ratio (~0.5σ):  {shock_lbl.mean()*100:.1f}%")
print(f"\n✓ Plots guardados en: {PLOTS_DIR}/")
