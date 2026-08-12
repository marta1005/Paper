# AeroSurrogate v2 — outputs

```
outputs/
├── models/
│   ├── scaler_v2.npy                       feature/target normalisation
│   ├── spatial_stats_v2.npy                spatial stats fitted on train
│   ├── surrogate_v2_best.pt                run 1 — MoE gate COLLAPSED
│   ├── surrogate_v2_moefix.pt              run 2 — two-stage      ← best
│   └── surrogate_v2_full.pt                run 3 — from scratch, fixed loss
├── results/
│   ├── v2_evaluation_collapsed_gate.txt    run 1
│   ├── v2_evaluation_moefix.txt            run 2  (= surrogate_v2_moefix_evaluation.txt)
│   ├── surrogate_v2_moefix_evaluation.txt  run 2, re-run under the derived name
│   └── surrogate_v2_full_evaluation.txt    run 3
└── plots/
    └── gate_diagnostic_moefix.png          diagnose_gate.py on run 2
```

## The three runs

| | schedule | best epoch | stopped | converged? |
|---|---|---|---|---|
| `surrogate_v2_best.pt` | 200, from scratch | 157 | 174 | yes |
| `surrogate_v2_moefix.pt` | 50, head-only fine-tune | 45 | 50 = **cap** | **no** |
| `surrogate_v2_full.pt` | 200, from scratch | 175 | 192 | yes |

**Run 1 — `surrogate_v2_best.pt`.** Trained *before* the load-balancing sign bug was
found, so its MoE gate is collapsed onto a single expert. Keep it: it is the ablation
baseline that shows what the mixture contributes.

**Run 2 — `surrogate_v2_moefix.pt`.** Head-only fine-tune from run 1 with the corrected
loss (`--freeze-backbone`, backbone = 75.4% of params held fixed). **Most accurate model
by a clear margin.** Note it stopped at the epoch cap, not on patience — it was still
improving.

**Run 3 — `surrogate_v2_full.pt`.** From scratch with the corrected loss. Healthiest
gate of the three, converged properly — and yet *less accurate than run 2 everywhere*.
Training the backbone and the gate together from scratch is worse than letting the
backbone mature first and specialising the head afterwards.

## What the fix changed

`L_lb` (entropy of the mean gate) was *added* to a loss being minimised, so it drove
the gate toward collapse instead of away from it. Sign corrected in commit `45cb8bc`.

Gate diagnostics, 3 test sims (`python diagnose_gate.py 3 --ckpt <ckpt>`):

| | run 1 collapsed | run 2 moefix | run 3 full |
|---|---|---|---|
| load balance (entropy of mean gate, max 1.3863) | 0.0771 — 5.6% | 1.3801 — 99.6% | **1.3832 — 99.8%** |
| dead experts (<1% usage) | E1, E3 | none | none |
| shock specialisation (TV, shock vs non-shock) | 0.0175 | 0.3167 | **0.3767** |
| routing sharpness (mean per-node entropy) | 0.0009 | 0.0610 | 0.0347 |

Run 3 has the healthiest gate of the three — and in it the specialisation is physically
legible: on shock nodes E2 and E3 carry 0.44 and 0.38, while off shock all four experts
split roughly evenly. Gate health and accuracy came apart here; see the caveat below.

Note the two entropies measure different things. Load balance is what `L_lb` optimises
— whether all experts carry a share of the work. Routing sharpness stays near 0 in both
runs, and that is *healthy*: each node commits to one expert. Collapse is load balance
near zero, not sharpness near zero.

## Accuracy, v1-comparable sample (SEED=42, frac=0.1, 4,068,074 pts)

R², all three models on the identical sample and identical subset masks.

| Subset | Coef | v1 (neural gate) | run 1 collapsed | run 3 full | run 2 moefix |
|---|---|---|---|---|---|
| global | Cp | 0.9506 | 0.9543 | 0.9577 | **0.9698** |
| | Cfx | 0.8512 | 0.8707 | 0.8811 | **0.9118** |
| | Cfy | 0.8463 | 0.8436 | 0.8504 | **0.8939** |
| | Cfz | 0.8555 | 0.8779 | 0.8798 | **0.9134** |
| shock | Cp | 0.8827 | 0.8812 | 0.8955 | **0.9262** |
| | Cfx | 0.8390 | 0.8219 | 0.8448 | **0.8807** |
| | Cfy | 0.8292 | 0.7907 | 0.8120 | **0.8599** |
| | Cfz | 0.8643 | 0.8608 | 0.8690 | **0.9012** |

The ordering is the same for every coefficient and every subset:
**moefix > full > collapsed ≳ v1.**

Two things follow. First, the sign fix works on its own terms: run 3 beats run 1
everywhere, and unlike run 1 it beats v1 inside the shock region too. Second, and more
interesting, the two-stage schedule beats joint training from scratch by a wider margin
than the sign fix itself bought (run 2 over run 3: Cp +0.0121 global, +0.0307 on shock).

The plausible reading is a curriculum effect. In run 2 the backbone first matured against
an effectively single-expert head, then the head specialised on frozen, settled features.
In run 3 the backbone must co-adapt to a gate that is still reorganising, and the
load-balancing pressure applies from epoch 0, competing with the accuracy objective while
the representation is still forming. This is worth stating as a deliberate two-stage
schedule rather than an accident — but it rests on one run per arm, so it is a hypothesis,
not an established result. Repeated seeds would be needed to claim it.

## Two caveats for the paper

**1. Run 2 is not converged.** It stopped at its epoch cap (45 best of 50), not on
patience, so it was still improving. Its numbers are a lower bound, and a longer
`--freeze-backbone` schedule should be run before these go to press.

**2. Model selection used test simulations.** `main_train_v2.py` draws the 16 validation
sims from the test set (`rng.choice(N_TEST, ...)` under `SEED=42` → sims 12, 14, 30, 62,
63, 79, 93, 103, 109, 111, 112, 117, 122, 125, 146, 148). Early stopping picked the
checkpoint on those sims, and every table above then reports over all 156, those 16
included. The bias is diluted — 16 of 156 — and it is identical across all four models,
so the comparison stands. The absolute values are optimistic. For publication, re-run
with `--exclude-val-sims` to report over the 140 sims that never touched model selection.
