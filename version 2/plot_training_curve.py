#!/usr/bin/env python3
"""
plot_training_curve.py — Recover the training curve from a training log.

main_train_v2.py logs one line per epoch and one per validation to stderr.
This parses that text (a SLURM .out file, a nohup capture, a redirected run, or
outputs/training_v2.log) into a CSV and a figure.

Usage:
  python plot_training_curve.py slurm-123456.out
  python plot_training_curve.py train.log --out-prefix outputs/plots/full_run
  python plot_training_curve.py train.log --csv-only
"""
import argparse
import csv
import re
import sys
from pathlib import Path

# Categorical slots 1-4 of the validated default palette (light surface).
# Validated: worst adjacent CVD dE 9.1, normal-vision dE 22.9 — both pass.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK_MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#d8d7d2', '#fcfcfb'

# "13:22:36 INFO Epoch  42/200  loss=0.1234  L_cp=0.0567  L_fric=0.0123  lr=1.2e-04  t=45s"
NUM     = r'([-+]?[\d.]+(?:[eE][-+]?\d+)?)'
RE_EP   = re.compile(r'Epoch\s+(\d+)\s*/\s*(\d+)')
RE_KV   = re.compile(rf'\b(loss|L_cp|L_shock|L_lb|L_fric|lr)\s*=\s*{NUM}')
# "  Val  R²(Cp)=0.9123  R²(Cfx)=0.8456 ..."   (tolerates R2 as well as R²)
RE_VAL  = re.compile(rf'R.?\((Cp|Cfx|Cfy|Cfz)\)\s*=\s*{NUM}')
RE_STOP = re.compile(r'Early stopping at epoch\s+(\d+)')
RE_BEST = re.compile(r'Saved best model')


def parse_log(path):
    """Return (rows, total_epochs, stop_epoch). One row per epoch seen."""
    rows, order = {}, []
    total_epochs = stop_epoch = None
    cur = None

    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            m = RE_EP.search(line)
            if m:
                cur = int(m.group(1))
                total_epochs = int(m.group(2))
                if cur not in rows:
                    rows[cur] = {'epoch': cur}
                    order.append(cur)
                for k, v in RE_KV.findall(line):
                    rows[cur][k] = float(v)
                continue

            # Validation lines carry no epoch number — attach to the last epoch.
            vals = RE_VAL.findall(line)
            if vals and cur is not None:
                for coef, v in vals:
                    rows[cur][f'val_R2_{coef}'] = float(v)
                continue

            if RE_BEST.search(line) and cur is not None:
                rows[cur]['is_best'] = 1

            m = RE_STOP.search(line)
            if m:
                stop_epoch = int(m.group(1))

    return [rows[e] for e in order], total_epochs, stop_epoch


def write_csv(rows, path):
    cols = ['epoch', 'loss', 'L_cp', 'L_shock', 'L_lb', 'L_fric', 'lr',
            'val_R2_Cp', 'val_R2_Cfx', 'val_R2_Cfy', 'val_R2_Cfz', 'is_best']
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def plot(rows, out_png, stop_epoch=None, title=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ep = [r['epoch'] for r in rows]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 6.6), dpi=200, sharex=True,
        gridspec_kw={'height_ratios': [1, 1.15], 'hspace': 0.12})
    fig.patch.set_facecolor(SURFACE)

    # ── Panel A: training losses. Separate panel rather than a second y-axis:
    # losses and R² have unrelated scales and must never share an axis.
    loss_keys = [k for k in ('loss', 'L_cp', 'L_fric', 'L_lb')
                 if any(k in r for r in rows)]
    for i, k in enumerate(loss_keys):
        xs = [r['epoch'] for r in rows if k in r]
        ys = [r[k] for r in rows if k in r]
        ax1.plot(xs, ys, color=SERIES[i % len(SERIES)], linewidth=2, label=k)
    if loss_keys and all(r.get(k, 1) > 0 for r in rows for k in loss_keys if k in r):
        ax1.set_yscale('log')
    _style(ax1)
    ax1.set_ylabel('training loss', color=INK_MUTED, fontsize=9)
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=len(loss_keys))
    if title:
        ax1.set_title(title, color=INK, fontsize=11, loc='left', pad=10)

    # ── Panel B: validation R², the early-stopping criterion.
    coefs = ['Cp', 'Cfx', 'Cfy', 'Cfz']
    best_ep = best_r2 = None
    labels = []
    for i, c in enumerate(coefs):
        key = f'val_R2_{c}'
        pts = [(r['epoch'], r[key]) for r in rows if key in r]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax2.plot(xs, ys, color=SERIES[i], linewidth=2, marker='o',
                 markersize=3, label=f'R²({c})')
        labels.append([xs[-1], ys[-1], f'R²({c})'])
        if c == 'Cp':
            best_r2 = max(ys)
            best_ep = xs[ys.index(best_r2)]

    # Direct labels at the line ends. The friction coefficients converge, so
    # place them top-down with a minimum gap or they overprint each other.
    if labels:
        lo, hi   = ax2.get_ylim()
        min_gap  = 0.052 * (hi - lo)
        labels.sort(key=lambda t: -t[1])
        for j in range(1, len(labels)):
            if labels[j][1] > labels[j - 1][1] - min_gap:
                labels[j][1] = labels[j - 1][1] - min_gap
        for x, y, text in labels:
            ax2.annotate(text, xy=(x, y), xytext=(5, 0),
                         textcoords='offset points', va='center',
                         fontsize=8, color=INK_MUTED)

    if best_ep is not None:
        ax2.axvline(best_ep, color=INK_MUTED, linewidth=1,
                    linestyle='--', alpha=0.7)
        ax2.annotate(f'best R²(Cp)={best_r2:.4f} @ epoch {best_ep}',
                     xy=(best_ep, best_r2), xytext=(-6, 14),
                     textcoords='offset points', ha='right',
                     fontsize=8, color=INK)
    if stop_epoch:
        ax2.annotate(f'early stop @ {stop_epoch}', xy=(stop_epoch, 0),
                     xytext=(0, 6), textcoords='offset points',
                     ha='center', fontsize=8, color=INK_MUTED)

    _style(ax2)
    ax2.set_xlabel('epoch', color=INK_MUTED, fontsize=9)
    ax2.set_ylabel('validation R²', color=INK_MUTED, fontsize=9)
    ax2.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=4,
               loc='lower right')
    ax2.margins(x=0.08)

    fig.savefig(out_png, bbox_inches='tight', facecolor=SURFACE)
    print(f'  figure -> {out_png}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('log', help='training log (SLURM .out, nohup capture, ...)')
    p.add_argument('--out-prefix', default=None,
                   help='output prefix (default: outputs/plots/<logname>)')
    p.add_argument('--csv-only', action='store_true')
    p.add_argument('--title', default=None)
    args = p.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        sys.exit(f'No such log: {log_path}')

    rows, total_epochs, stop_epoch = parse_log(log_path)
    if not rows:
        sys.exit('No "Epoch N/M" lines found — is this a training log?')

    n_val = sum(1 for r in rows if 'val_R2_Cp' in r)
    print(f'Parsed {len(rows)} epochs'
          + (f' of {total_epochs}' if total_epochs else '')
          + f', {n_val} validations'
          + (f', early stop at {stop_epoch}' if stop_epoch else ''))

    prefix = Path(args.out_prefix) if args.out_prefix else \
        Path('outputs/plots') / log_path.stem
    prefix.parent.mkdir(parents=True, exist_ok=True)

    write_csv(rows, f'{prefix}.csv')
    print(f'  csv    -> {prefix}.csv')
    if not args.csv_only:
        plot(rows, f'{prefix}.png', stop_epoch=stop_epoch, title=args.title)


if __name__ == '__main__':
    main()
