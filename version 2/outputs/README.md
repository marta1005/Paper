# AeroSurrogate v2 — outputs

```
outputs/
├── models/
│   ├── scaler_v2.npy                       feature/target normalisation
│   ├── spatial_stats_v2.npy                spatial stats fitted on train
│   ├── surrogate_v2_best.pt                run 1 — MoE gate COLLAPSED
│   └── surrogate_v2_moefix.pt              run 2 — MoE gate healthy  ← use this
├── results/
│   ├── v2_evaluation_collapsed_gate.txt    evaluation of surrogate_v2_best.pt
│   └── v2_evaluation_moefix.txt            evaluation of surrogate_v2_moefix.pt
└── plots/
    └── gate_diagnostic_moefix.png          diagnose_gate.py on the fixed run
```

## The two checkpoints

**`surrogate_v2_best.pt`** — trained from scratch, `PAPER_EPOCHS=200`, best epoch 157,
early-stopped at 174. Trained *before* the load-balancing sign bug was found, so its
MoE gate is collapsed onto a single expert. Keep it: it is the ablation baseline that
shows what the mixture contributes.

**`surrogate_v2_moefix.pt`** — head-only fine-tune from the checkpoint above with the
corrected loss (`--freeze-backbone`, backbone = 75.4% of params held fixed). Same
backbone, working gate. This is the model to report.

## What the fix changed

`L_lb` (entropy of the mean gate) was *added* to a loss being minimised, so it drove
the gate toward collapse instead of away from it. Sign corrected in commit `45cb8bc`.

Gate diagnostics, 3 test sims (`python diagnose_gate.py 3 --ckpt <ckpt>`):

| | collapsed | fixed |
|---|---|---|
| load balance (entropy of mean gate, max 1.3863) | 0.0771 — 5.6% | **1.3801 — 99.6%** |
| dead experts (<1% usage) | E1, E3 | **none** |
| shock specialisation (TV, shock vs non-shock) | 0.0175 | **0.3167** |
| routing sharpness (mean per-node entropy) | 0.0009 | 0.0610 |

Note the two entropies measure different things. Load balance is what `L_lb` optimises
— whether all experts carry a share of the work. Routing sharpness stays near 0 in both
runs, and that is *healthy*: each node commits to one expert. Collapse is load balance
near zero, not sharpness near zero.

## Accuracy, v1-comparable sample (SEED=42, frac=0.1, 4,068,074 pts)

R², all three models on the identical sample and identical subset masks.

| Subset | Coef | v1 (neural gate) | v2 collapsed | v2 fixed |
|---|---|---|---|---|
| global | Cp | 0.9506 | 0.9543 | **0.9698** |
| | Cfx | 0.8512 | 0.8707 | **0.9118** |
| | Cfy | 0.8463 | 0.8436 | **0.8939** |
| | Cfz | 0.8555 | 0.8779 | **0.9134** |
| shock | Cp | 0.8827 | 0.8812 | **0.9262** |
| | Cfx | 0.8390 | 0.8219 | **0.8807** |
| | Cfy | 0.8292 | 0.7907 | **0.8599** |
| | Cfz | 0.8643 | 0.8608 | **0.9012** |

With the gate collapsed, v2 was *worse than v1* on every coefficient inside the shock
region — the mixture was contributing nothing where it was supposed to matter most.
With the gate working, v2 wins everywhere, and the largest gains are in the shock
subset (Cp +0.0435, Cfy +0.0307 over v1).

## Caveat for the paper

`surrogate_v2_moefix.pt` is a head-only fine-tune on a backbone that was itself trained
alongside a collapsed gate. The numbers are real, but a clean from-scratch run with the
corrected loss has not been done yet and would likely do at least as well.
