# AeroSurrogate v2 — outputs

```
outputs/
├── models/
│   ├── scaler_v2.npy                       feature/target normalisation
│   ├── spatial_stats_v2.npy                spatial stats fitted on train
│   ├── surrogate_v2_best.pt                run 1 — MoE gate COLLAPSED
│   ├── surrogate_v2_moefix.pt              run 2 — two-stage
│   ├── surrogate_v2_full.pt                run 3 — from scratch, fixed loss
│   └── surrogate_v2_moefix_long.pt         run 4 — two-stage, 200ep, DDP fixed ← best
├── results/
│   ├── v2_evaluation_collapsed_gate.txt    run 1
│   ├── v2_evaluation_moefix.txt            run 2  (= surrogate_v2_moefix_evaluation.txt)
│   ├── surrogate_v2_moefix_evaluation.txt  run 2, re-run under the derived name
│   ├── surrogate_v2_full_evaluation.txt    run 3
│   ├── surrogate_v2_moefix_long_evaluation.txt  run 4, held-out 140 sims
│   └── gate_moefix_long.txt                diagnose_gate.py on run 4
└── plots/
    ├── gate_diagnostic_moefix.png          diagnose_gate.py on run 2
    ├── moefix_long.png                     run 4 training curve
    └── moefix_long.csv                     run 4 per-epoch metrics
```

## The four runs

| | schedule | best epoch | stopped | converged? | DDP |
|---|---|---|---|---|---|
| `surrogate_v2_best.pt` | 200, from scratch | 157 | 174 | yes | unsynced |
| `surrogate_v2_moefix.pt` | 50, head-only fine-tune | 45 | 50 = **cap** | **no** | unsynced |
| `surrogate_v2_full.pt` | 200, from scratch | 175 | 192 | yes | unsynced |
| `surrogate_v2_moefix_long.pt` | 200, head-only fine-tune | 118 | 134 | yes | **synced** |

"unsynced" means the run predates commit `9fe5f2e`: DDP was constructed and then
bypassed, so gradients were never all-reduced and each rank trained alone on
1/world_size of the sims. Those three checkpoints are penalised by however many GPUs
they used; treat their numbers as a floor.

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
run 1, `--freeze-backbone`) but with a 200-epoch budget and with DDP gradient
synchronisation fixed. Converged on patience at epoch 134, best at 118. **This is the
model to report.**

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

The plausible reading is a curriculum effect. In run 2 the backbone first matured against
an effectively single-expert head, then the head specialised on frozen, settled features.
In run 3 the backbone must co-adapt to a gate that is still reorganising, and the
load-balancing pressure applies from epoch 0, competing with the accuracy objective while
the representation is still forming. This is worth stating as a deliberate two-stage
schedule rather than an accident — but it rests on one run per arm, so it is a hypothesis,
not an established result. Repeated seeds would be needed to claim it.

## Run 4 — the numbers to report

Held-out test set: the 140 sims that never touched model selection
(`--exclude-val-sims`), 36,508,360 points.

| Subset | R²(Cp) | R²(Cfx) | R²(Cfy) | R²(Cfz) | MAE(Cp) |
|---|---|---|---|---|---|
| global | **0.9729** | 0.9172 | 0.9027 | 0.9209 | 0.027647 |
| shock | **0.9350** | 0.8923 | 0.8692 | 0.9076 | 0.040479 |
| transonic | 0.9842 | 0.9279 | 0.9167 | 0.9321 | 0.025027 |
| subsonic | 0.9471 | 0.9005 | 0.8839 | 0.9086 | 0.041453 |

Weighted R² (confidence_weight): Cp 0.9768, Cfx 0.9218, Cfy 0.9097, Cfz 0.9243.

These beat run 2 (0.9689 global / 0.9264 shock) *despite* being measured on the harder,
unbiased subset — run 2's figures include the 16 sims its own early stopping selected on.
The true gap is therefore wider than the difference in the numbers suggests.

Gate at run 4: load balance 1.3777 / 1.3863 (99.4% of uniform), no dead experts, shock
specialisation 0.2789, shock indicator F1 0.9564. Ablating the additive residual
`p_s * shock_expert(h)` costs 78.6% of the Cp MAE on shock nodes — its largest
contribution across all four runs.

`outputs/plots/moefix_long.png` shows the gate load balance climbing from 0.50 to 1.379
within the first few epochs and holding there — the corrected loss doing its job, visible
per-epoch for the first time now that `L_lb` is logged.

## Two caveats for the paper

**1. The comparison table is not measured on one common set.** Runs 1–3 were evaluated
over all 156 test sims; run 4 over the held-out 140. Because the 16 excluded sims are the
ones early stopping fit to, the all-156 figures flatter runs 1–3 and the held-out figures
are stricter on run 4 — so run 4's lead is real and understated, but the rows are not
strictly comparable. Before the table goes in the paper, re-run runs 1–3 with
`--exclude-val-sims` so every row is measured over the same 140 sims:

```bash
python evaluate_v2.py --ckpt outputs/models/surrogate_v2_full.pt --exclude-val-sims \
       --out outputs/results/full_heldout.txt
python evaluate_v2.py --ckpt outputs/models/surrogate_v2_best.pt --exclude-val-sims \
       --out outputs/results/collapsed_heldout.txt
```

Note also that under `--exclude-val-sims` the "v1-comparable sample" row is 3,650,836
points drawn from 140 sims, not the 4,068,074 from 156 that v1 reported. It is the
better number, but it is no longer a like-for-like sample against v1 — say so, or quote
the all-156 run for that one comparison.

**2. Runs 1–3 trained without DDP gradient synchronisation** (see the table above). If
they used more than one GPU their numbers are depressed, which also puts the run 2 vs
run 3 curriculum comparison in question: both carried the same handicap, so the ordering
probably survives, but the conclusion should be re-checked with synchronised runs before
it is claimed in the paper.
