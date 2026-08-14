# AeroSurrogate v2 — outputs

```
outputs/
├── models/
│   ├── scaler_v2.npy                       feature/target normalisation
│   ├── spatial_stats_v2.npy                spatial stats fitted on train
│   ├── surrogate_v2_best.pt                run 1 — MoE gate COLLAPSED
│   ├── surrogate_v2_moefix.pt              run 2 — two-stage
│   ├── surrogate_v2_full.pt                run 3 — from scratch, fixed loss
│   └── surrogate_v2_moefix_long.pt         run 4 — two-stage, 200 epochs   ← best
├── results/
│   ├── v2_evaluation_collapsed_gate.txt    run 1
│   ├── v2_evaluation_moefix.txt            run 2  (= surrogate_v2_moefix_evaluation.txt)
│   ├── surrogate_v2_moefix_evaluation.txt  run 2, re-run under the derived name
│   ├── surrogate_v2_full_evaluation.txt    run 3
│   ├── surrogate_v2_moefix_long_evaluation.txt  run 4, held-out 140 sims
│   ├── collapsed_heldout.txt               run 1, held-out 140 sims
│   ├── full_heldout.txt                    run 3, held-out 140 sims
│   └── gate_moefix_long.txt                diagnose_gate.py on run 4
└── plots/
    ├── gate_diagnostic_moefix.png          diagnose_gate.py on run 2
    ├── moefix_long.png                     run 4 training curve
    └── moefix_long.csv                     run 4 per-epoch metrics
```

## The four runs

| | schedule | best epoch | stopped | converged? |
|---|---|---|---|---|
| `surrogate_v2_best.pt` | 200, from scratch | 157 | 174 | yes |
| `surrogate_v2_moefix.pt` | 50, head-only fine-tune | 45 | 50 = **cap** | **no** |
| `surrogate_v2_full.pt` | 200, from scratch | 175 | 192 | yes |
| `surrogate_v2_moefix_long.pt` | 200, head-only fine-tune | 118 | 134 | yes |

All four were run as a single process, not under `torchrun`. That matters because of the
DDP defect fixed in commit `9fe5f2e` (the wrapper was built and then bypassed, so
gradients were never all-reduced): with `world_size == 1` the wrapper is never
constructed and the sampler is a plain `RandomSampler`, so the defect was latent and
never affected these runs. Every one of them saw the full training set each epoch, and
they are directly comparable to each other.

**Run 1 — `surrogate_v2_best.pt`.** Trained *before* the load-balancing sign bug was
found, so its MoE gate is collapsed onto a single expert. Keep it: it is the ablation
baseline that shows what the mixture contributes.

**Run 2 — `surrogate_v2_moefix.pt`.** Head-only fine-tune from run 1 with the corrected
loss (`--freeze-backbone`, backbone = 75.4% of params held fixed). Superseded by run 4,
which is the same recipe given enough epochs to converge — run 2 stopped at its cap, not
on patience, so it was still improving when the budget ran out.

**Run 3 — `surrogate_v2_full.pt`.** From scratch with the corrected loss. Healthy gate,
converged properly — and yet *less accurate than run 2 everywhere*. Training the backbone
and the gate together from scratch came out worse than letting the backbone mature first
and specialising the head afterwards.

**Run 4 — `surrogate_v2_moefix_long.pt`.** Same two-stage recipe as run 2 (resume from
run 1, `--freeze-backbone`) with a 200-epoch budget instead of 50. Converged on patience
at epoch 134, best at 118 — so unlike run 2 it had room to finish. **This is the model to
report.**

## What the fix changed

`L_lb` (entropy of the mean gate) was *added* to a loss being minimised, so it drove
the gate toward collapse instead of away from it. Sign corrected in commit `45cb8bc`.

Gate diagnostics, 3 test sims (`python diagnose_gate.py 3 --ckpt <ckpt>`):

| | run 1 collapsed | run 2 moefix | run 3 full | run 4 moefix_long |
|---|---|---|---|---|
| load balance (entropy of mean gate, max 1.3863) | 0.0771 — 5.6% | 1.3801 — 99.6% | **1.3832 — 99.8%** | 1.3777 — 99.4% |
| dead experts (<1% usage) | E1, E3 | none | none | none |
| shock specialisation (TV, shock vs non-shock) | 0.0175 | 0.3167 | **0.3767** | 0.2789 |
| routing sharpness (mean per-node entropy) | 0.0009 | 0.0610 | 0.0347 | 0.0817 |
| shock indicator F1 | — | 0.9478 | 0.9254 | **0.9564** |
| shock residual contribution to Cp MAE | — | +59.3% | +62.3% | **+78.6%** |

Every run after the sign fix balances the gate to within 0.6% of uniform with no dead
experts, so load balance no longer discriminates between them. Run 3 has the sharpest
shock specialisation, and there it is physically legible: on shock nodes E2 and E3 carry
0.44 and 0.38, while off shock all four experts split roughly evenly. But gate health and
accuracy came apart — run 3 is the least accurate of the three fixed runs.

Note the two entropies measure different things. Load balance is what `L_lb` optimises
— whether all experts carry a share of the work. Routing sharpness stays near 0 in every
run, and that is *healthy*: each node commits to one expert. Collapse is load balance
near zero, not sharpness near zero.

## Accuracy, v1-comparable sample (SEED=42, frac=0.1, 4,068,074 pts)

R² over all 156 test sims, identical sample and identical subset masks. Run 4 is absent
here because it was evaluated on the held-out 140 — see the section after this one, and
caveat 1.

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

The plausible reading is a curriculum effect. In runs 2/4 the backbone first matured
against an effectively single-expert head, then the head specialised on frozen, settled
features. In run 3 the backbone must co-adapt to a gate that is still reorganising, and
the load-balancing pressure applies from epoch 0, competing with the accuracy objective
while the representation is still forming.

The arms are cleanly matched — same data, same loss, same single-process setup, both
converged on patience — so this is worth stating as a deliberate two-stage schedule
rather than an accident. The gap widens with run 4, which is the two-stage arm given
enough epochs to converge. What one run per arm cannot rule out is seed variance; see
caveat 2.

## The table to report — held-out test set

All three converged runs over the same 140 sims that never touched model selection
(`--exclude-val-sims`), 36,508,360 points, identical subset masks. **This is the
comparison for the paper**; everything above it is measured on all 156 and is
optimistic by the amount early stopping fitted to its 16 selection sims.

| Subset | Coef | run 1 collapsed | run 3 from scratch | run 4 two-stage |
|---|---|---|---|---|
| global | Cp | 0.9549 | 0.9580 | **0.9729** |
| | Cfx | 0.8665 | 0.8775 | **0.9172** |
| | Cfy | 0.8443 | 0.8509 | **0.9027** |
| | Cfz | 0.8796 | 0.8806 | **0.9209** |
| shock | Cp | 0.8814 | 0.8956 | **0.9350** |
| | Cfx | 0.8231 | 0.8445 | **0.8923** |
| | Cfy | 0.7867 | 0.8063 | **0.8692** |
| | Cfz | 0.8592 | 0.8677 | **0.9076** |
| transonic | Cp | 0.9669 | 0.9698 | **0.9842** |
| subsonic | Cp | 0.9259 | 0.9303 | **0.9471** |

Run 4 weighted R² (confidence_weight): Cp 0.9768, Cfx 0.9218, Cfy 0.9097, Cfz 0.9243.

The two effects separate cleanly on this table, and they are not the same size. Fixing
the load-balancing sign, holding the schedule fixed (run 1 → run 3), buys +0.0031 on
global Cp and +0.0142 on shock. Adding the two-stage schedule on top (run 3 → run 4)
buys a further +0.0149 and +0.0394. **The curriculum is worth about 2.8× the sign fix
inside the shock region** — the place the architecture exists to serve.

Note also that run 1, with its gate collapsed, sits at 0.8814 on shock Cp — still below
v1's 0.8827. The mixture only pays for itself once the gate is actually balanced.

Gate at run 4, over 8 test sims (`gate_moefix_long.txt`, a larger sample than the 3-sim
table above): load balance 1.3827 / 1.3863 — 99.7% of uniform, no dead experts, shock
specialisation 0.3581, shock indicator F1 0.9564. Ablating the additive residual
`p_s * shock_expert(h)` costs 78.6% of the Cp MAE on shock nodes — its largest
contribution across all four runs.

`outputs/plots/moefix_long.png` shows the gate load balance climbing from 0.50 to 1.379
within the first few epochs and holding there — the corrected loss doing its job, visible
per-epoch for the first time now that `L_lb` is logged.

## Two caveats for the paper

**1. The v1 baseline is not on the held-out set.** v1's published figures (global R²(Cp)
0.9506, shock 0.8827) come from its own protocol: a 10% sample of all 156 test sims,
4,068,074 points. The held-out table above uses 140 sims. Putting v1 on the same footing
would mean re-evaluating the v1 checkpoint, and v1's own validation split was different,
so "exclude these 16" is not even the right exclusion for it.

This does not threaten the conclusion. Run 4 is measured on the *stricter* set — it
excludes the sims its own early stopping fitted to, which v1's numbers do not — and still
leads by +0.022 on global Cp and +0.052 on shock. A margin that size does not turn on a
140-vs-156 population difference. State the protocol difference and the direction of the
bias, and the comparison holds.

**2. The curriculum result rests on one run per arm.** Runs 2/4 (two-stage) and run 3
(from scratch) are otherwise matched — same data, same loss, same single-process setup,
both converged — so the comparison is sound as far as it goes. What it cannot rule out is
seed variance: a single run per arm cannot separate a real effect from a lucky
initialisation. Repeated seeds are what would turn this from a well-supported observation
into a claim.

**3. Run 4 is the two-stage arm to quote, not run 2.** Run 2 stopped at its epoch cap
while still improving, so it understates the recipe. Any comparison against run 3 should
use run 4.
