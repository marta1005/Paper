# PAPER_V2_CHANGES.md — converting `main.tex` from v1 to v2

Target file: `/Users/martaarnabatmartin/Desktop/Paper/version 1/Overleaf_Paper/main.tex` (1356 lines).
Reported model: `version 2/outputs/models/surrogate_v2_moefix_long.pt` (run 4).
Every number below is traced to a file. Nothing here is inferred.

**Provenance key** — the only five sources any v2 number may come from:

| Tag | File |
|---|---|
| `HELDOUT4` | `version 2/outputs/results/surrogate_v2_moefix_long_evaluation.txt` (run 4, 140 sims) |
| `HELDOUT3` | `version 2/outputs/results/full_heldout.txt` (run 3, 140 sims) |
| `HELDOUT1` | `version 2/outputs/results/collapsed_heldout.txt` (run 1, 140 sims) |
| `GATE` | `version 2/outputs/results/gate_moefix_long.txt` (run 4 diagnostics, 8 sims) |
| `V1SUB` | `version 1/outputs/results/subset_metrics.txt` (the authoritative v1 table) |

`version 2/outputs/README.md` is **narrative, not ground truth** — it contradicts `GATE` in two
places (see §1.9). Quote the `.txt` files, not the README.

---

## Critical path — what blocks what

```
DAY 0   §0 decision: does the paper keep a symbolic section at all?
        ├─ NO  → title (L16), keywords (L37), contribution 4 (L198), §sec:sr (L773–928),
        │        conclusion 4 (L1172), discussion 3 (L1138), abstract (L52–56) all get cut
        │        together in one pass. Cheapest path. ~2 days.
        └─ YES → everything in §1 still has to be fixed first, because the retained
                 material is currently false. Adds ~4 days. See §5.

WEEK 1  §1 blocking falsehoods (independent of the §0 decision)
        §3 results tables — paste the LaTeX below, it is ready
        §2 methodology text edits (the non-figure ones)

WEEK 2  §2 the two TikZ figures (architecture + pipeline) — slowest single item
        §4 regenerate parity plots and the six surface-field figures from the v2 checkpoint
        run the one missing measurement: v2 inference timing (§6.M1)

WEEK 3  §6 limitations paragraph, §5 symbolic disposition, full read-through,
        cross-reference sweep (`\ref{eq:forward}`, `\ref{tab:results}`, `\ref{fig:pipeline}`)
```

**The single hardest dependency:** §4 (figure regeneration) needs a working
`evaluate_v2.py`-driven plotting path that does not exist yet — v1's plots came from
`infer_aero.py` against a point-wise checkpoint. Start §4 in week 1 even though it is listed
in week 2, or the figures will not land.

---

## §0. The decision that gates everything

The paper's **title** (L16) is *"Symbolic Shock Sensor for Physics-Informed
Mixture-of-Experts Surrogates in Transonic Aerodynamics"* and its **keywords** (L37–38)
include `symbolic regression`.

`grep -riE 'symbolic|pysr'` over the whole of `version 2/` (`.py`, `.md`, `.txt`) returns
**zero hits**. There is no symbolic gate in v2, and none can be lifted across: the v1
expression is a function of two named physical features (`Cp_crit`, `span_norm`), whereas
`ShockIndicatorSpatial.forward` (`src/models_v2.py:127`) consumes the 256-d GNN latent `h_i`.
The substitution has no v2 counterpart and none has been fitted.

So the title as it stands describes a component the reported model does not contain. Decide
this on day 0 — every other symbolic-related edit hangs off it.

Suggested replacement title if the symbolic material is cut:

> **Spatial Shock-Gated Mixture of Experts for Transonic Surface-Load Prediction**

Keywords become: `shock capture, mixture of experts, graph neural network, Gumbel-Softmax
routing, physics-informed machine learning, RANS surrogate`.

---

## §1. Blocking corrections — required regardless of v1/v2

These are false statements about work that was done. They must be fixed even if the paper
were to remain a v1 paper.

### 1.1 The symbolic-gate table row is fabricated — **L989**

This is the most serious item in the paper.

Currently at L988–989:

```latex
AeroSurrogate (neural gate)   & 0.9506 & 0.8508 & 0.8475 & 0.8557 \\
AeroSurrogate (symbolic gate) & 0.9506 & 0.8508 & 0.8475 & 0.8557 \\
```

The symbolic row is a **copy of the neural row**. It is not a measurement.

| | claimed (L989) | measured (`V1SUB` L12) | delta |
|---|---|---|---|
| R²(Cp) | 0.9506 | **0.9421** | −0.0085 |
| R²(Cfx) | 0.8508 | **0.8202** | −0.0310 |
| R²(Cfy) | 0.8475 | **0.7458** | **−0.1005** (−12% relative) |
| R²(Cfz) | 0.8557 | **0.8400** | −0.0155 |

On the shock subset (`V1SUB` L13) the gap widens: Cp 0.8659 vs 0.8827 (−0.0168),
Cfy 0.7111 vs 0.8292 (−0.1181).

**Root cause is already diagnosed in the repo.** `version 1/outputs/results/ablations.md`
L3–11 carries the correction banner: `SurrogateTrainer.train()` called `self.load_model()`
with the hardcoded default `"surrogate_best.pt"` instead of `self.save_name`, so the
trainer's weights were overwritten with the **neural** checkpoint before evaluation.
`version 1/outputs/results/surrogate_symbolic_evaluation.txt` (r2_Cp `0.950557`) is the
output of that bug. Compare it to `surrogate_evaluation.txt` (r2_Cp `0.950554`) — they agree
to the sixth decimal, which is the signature of evaluating the same weights twice.

**Action:** delete the row. For v2 there is no symbolic gate to report at all. If the paper
retains a v1 symbolic result anywhere, it must carry `0.9421 / 0.8202 / 0.7458 / 0.8400`
and cite `V1SUB`, never `surrogate_symbolic_evaluation.txt`.

### 1.2 "Identical to four decimal places" — **L994–1002**

```latex
The central observation of Table~\ref{tab:results} is that the neural
and symbolic gates achieve \emph{identical} scores to four decimal places.
```

False at every decimal place (§1.1). L998's *"without any measurable degradation"* is false —
the Cfy degradation is 0.1005 globally. L999–1002's *"the routing information the MoE
actually exploits is fully captured by the two physical variables"* is **inverted** by the
data: a 0.1005 Cfy loss is direct evidence the routing carries information those two
variables do not.

There is a second, independent reason L1000 cannot stand. The paper's own Setup at **L789–791**
says the SR stage regressed *"the hard physics label $y_{\mathrm{shock}}$ … not the soft
output of the neural ShockIndicator."* PySR never saw the routing signal, so the experiment
as run cannot support any claim about what the MoE exploits. That contradiction also
invalidates **L918** (*"The symbolic expression is an approximation of the neural
ShockIndicator"*), which directly contradicts L789.

**Action:** delete L994–1002 entirely. If a v1 symbolic result is retained, the defensible
sentence is:

> Substituting the algebraic gate costs 0.0085 in $R^2(C_p)$ and 0.1005 in $R^2(C_{fy})$:
> the pressure field survives the substitution nearly intact, the lateral friction does not.

Two further rows in `V1SUB` are worth keeping because they are genuinely informative and
correctly measured — they show how brittle the routing is:

| Ablation (`V1SUB`) | global Cfy | shock Cp |
|---|---|---|
| Symbolic ckpt, $p_s$ frozen at 0.41 (L17–18) | 0.5568 | 0.8459 |
| Neural ckpt driven by symbolic routing (L22–23) | 0.8414 | **0.7877** |

### 1.3 Contribution bullet 4 — **L198–204**

> *"Symbolic distillation with lossless redeployment … achieving $R^2$ scores identical to
> the neural gate on more than four million held-out test points."*

Same falsehood as §1.1/§1.2, promoted to a headline contribution, and it propagates into the
abstract. **Delete the bullet.** Replace with the two contributions v2 actually measures:

```latex
  \item A \emph{spatial backbone}: an eight-layer message-passing GNN over a
        per-simulation $k$-nearest-neighbour graph of the surface, which
        replaces v1's point-wise input and lifts $R^{2}(C_p)$ from 0.9506
        to 0.9729 and $R^{2}(C_{fy})$ from 0.8463 to 0.9027.
  \item A \emph{two-stage training schedule}---end-to-end pre-training followed
        by a head-only fine-tune with the backbone frozen---which outperforms
        joint training from scratch by $+0.0149$ in global $R^{2}(C_p)$ and
        $+0.0394$ inside the shock region on the held-out set.
```

### 1.4 Abstract — **L52–56, L58–61**

L52–56 carries *"retraining with this frozen symbolic gate yields aerodynamic accuracy
identical to the neural gate"* — delete. L58–61 carries three stale numbers: `R^{2}=0.951`,
`R^{2}\approx 0.85`, `more than four million`, `3\times 10^{5}` parameters. All four wrong
for v2. Replacement paragraph:

```latex
Trained on RANS surface data of the ONERA CRM WBPN database spanning Mach
$0.30$--$0.96$ and angle of attack from $-15^{\circ}$ to $+12.5^{\circ}$ on the
test partition, the model achieves $R^{2}=0.9729$ for $C_p$ and $R^{2}$ between
$0.9027$ and $0.9209$ for the skin-friction components over more than 36 million
held-out surface points, with $4.6\times 10^{6}$ parameters.
```

L49–50 (*"Mach-gated Gumbel-Softmax that applies stochastic exploration only in the
transonic regime"*) describes a mechanism that does not exist in v2 — see §2.7. Delete.

### 1.5 Conclusion bullet 4 — **L1172–1178**

> *"retraining with this frozen algebraic gate reproduces the neural-gate accuracy
> exactly---a data-driven confirmation that the network exploits genuine shock physics"*

Both halves fail: the first is §1.1's falsehood; the second is a non-sequitur, since PySR
fitted the RANS label (L789), not the network. **Delete the bullet.** Replace with the
shock-residual ablation, which is measured (`GATE` §[5]):

```latex
  \item Ablating the additive shock residual $p_s\,f^{\mathrm{shock}}(\mathbf{h}_i)$
        at inference raises the $C_p$ mean absolute error on shock nodes from
        $0.0458$ to $0.1370$, an increase of $66.6\%$---a direct measurement of
        what the explicit discontinuity mechanism contributes.
```

### 1.6 Forward reference to the false result — **L927**

> *"As shown in Section~\ref{sec:results}, this substitution is lossless in terms of
> aerodynamic accuracy."*

Delete. It points at the fabricated row.

### 1.7 The neural row does not match its own stated ground truth — **L988**

Even setting the symbolic row aside, L988 prints `0.9506 / 0.8508 / 0.8475 / 0.8557`,
which are the `surrogate_evaluation.txt` values. The authoritative `V1SUB` L7 gives
`0.9506 / 0.8512 / 0.8463 / 0.8555` on the same nominal sample. The Cfy discrepancy
(0.8475 vs 0.8463) is what props up the *"all three remain above $R^2=0.847$"* claim at
**L942** — which is false against `V1SUB`, since 0.8463 < 0.847.

**Action:** pick one protocol and use it everywhere. If v1 is retained as a baseline row,
use `V1SUB`: `0.9506 / 0.8512 / 0.8463 / 0.8555`. Delete the 0.847 floor sentence at L942 —
for v2 the friction floor is 0.9027 (Cfy, `HELDOUT4` L19).

### 1.8 The boxed equation is not what was deployed — **L836–841**

`shock_sensor_symbolic_surrogate_base.pkl` — the file `evaluate_subsets.py` actually loads
(`DEFAULT_SYM_PKL`, L33), and therefore the file behind the 0.9421 measurement — unpickles to:

```
expr_str   = tanh(exp(Cp_crit/span_norm))
calibrator = sklearn IsotonicRegression, 92 knots, output range [0.01552, 1.0]
```

The calibrator is applied at inference (`evaluate_subsets.py:126`:
`symbolic_sensor['calibrator'].predict(proba)`). The raw expression saturates at
$\tanh(1)=0.7318$ and never reaches 1; the calibrator maps it to $[0.0155, 1.0]$.

`shock_sensor_symbolic_protected.pkl` additionally clamps the denominator:
`tanh(exp(Cp_crit / maximum(span_norm, 0.1)))` — **seven** nodes, not five — because the
bare $1/\eta$ diverges at the wing root.

**Action:** the deployed gate is $p_s = \mathrm{Iso}_{92}\bigl(\tanh(\exp(C_p^{\mathrm{crit}}/\max(\eta,0.1)))\bigr)$.
State the guarded form and the calibration, or drop the boxed presentation. Every downstream
"five-node formula" claim (L901, L997, L1139) inherits this and must go with it.

### 1.9 The repo's own README disagrees with its measurement file — fix before quoting

`version 2/outputs/README.md` L157–161 attributes **F1 0.9564** and **+78.6%** to the
8-simulation run in `gate_moefix_long.txt`. That file (`GATE` L19, L42) actually reports
**F1 0.9421** (precision 0.8964, recall 0.9927) and **+66.58%**. The 0.9564 / 78.6% pair
comes from the 3-simulation diagnostic table higher up in the same README (L75–76).
The other two figures quoted in the same README sentence — load balance 1.3827,
specialisation 0.3581 — *are* the 8-sim values and are correct.

**Action:** use the 8-simulation values (F1 **0.9421**, ablation **+66.58%**) in the paper —
it is the larger sample — and fix the README paragraph so the two stop disagreeing.

### 1.10 Six figure captions label $\Pi_{\mathrm{norm}}$ as $\Pi$ — **L1075, 1086, 1095, 1104, 1113, 1122**

The paper states at **L237** that $\Pi \in \{1,2,4\}\times 10^{5}$ Pa, and
`version 2/data/dataset.csv` confirms exactly three distinct values. The captioned values
are $\Pi_{\mathrm{norm}} = \Pi/(1+0.2M^2)$ — the Mach-adjusted quantity defined at L272 —
mislabelled as $\Pi$. Verified against `dataset.csv`:

| Figure | line | caption says | true $\Pi$ | test sim | check |
|---|---|---|---|---|---|
| `cond0` | 1075 | $1.0{\times}10^{5}$ | $1{\times}10^{5}$ | 0 | ok as printed |
| `cond26` | 1086 | $3.6{\times}10^{5}$ | $4{\times}10^{5}$ | **112** | $4/1.098 = 3.64$ |
| `cond52` | 1095 | $1.8{\times}10^{5}$ | $2{\times}10^{5}$ | 69 | $2/1.128 = 1.77$ |
| `cond81` | 1104 | $1.8{\times}10^{5}$ | $2{\times}10^{5}$ | 78 | $2/1.141 = 1.75$ |
| `cond104` | 1113 | $1.7{\times}10^{5}$ | $2{\times}10^{5}$ | 86 | $2/1.148 = 1.74$ |
| `cond130` | 1122 | $3.4{\times}10^{5}$ | $4{\times}10^{5}$ | 147 | $4/1.162 = 3.44$ |

**Action:** relabel all six with the true $\Pi$ from `dataset.csv`.

### 1.11 One surface-field panel is a model-selection simulation — **L1027–1029, fig:cond26**

The condition $M{=}0.70$, $\alpha{=}{-}10^{\circ}$ resolves to **test simulation 112**
(verified against `dataset.csv`; it is the unique test sim with that $(M,\alpha)$ pair).
Simulation 112 is in the list of 16 that early stopping selected on:
`[12, 14, 30, 62, 63, 79, 93, 103, 109, 111, 112, 117, 122, 125, 146, 148]`
(stated at the top of all three held-out reports, e.g. `HELDOUT4` L4).

Showing it as a *"held-out flight condition"* for v2 would be showing a selection
simulation. The other five map to sims 0, 69, 78, 86, 147 — all outside the excluded 16,
all clean.

Note also that sim 112 carries `confidence_weight_simple = 0.5` (extreme AoA), so it is
one of the down-weighted conditions.

**Action:** swap it for a genuinely held-out condition, or label the panel explicitly as
coming from the selection set.

---

## §2. Methodology rewrite, section by section

Ordered by line number. All widths and counts below were verified by instantiating
`AeroSurrogatev2(MODEL_CONFIG)` and by reading `src/models_v2.py`, `src/training.py`,
`src/dataset.py`, `config.py`, `build_knn_cache.py`, `main_train_v2.py`.

### Measured parameter budget (use these numbers everywhere)

| Component | params | share |
|---|---:|---:|
| GNN backbone | 3,472,640 | 75.4% |
| — node encoder $16{\to}128{\to}256$ | 35,456 | |
| — edge encoder $5{\to}64{\to}128$ | 8,832 | |
| — 8 message-passing layers | 3,428,352 | |
| ShockIndicator | 43,713 | 0.9% |
| ShockGatedMoE | 755,529 | 16.4% |
| — gate | 41,924 | |
| — 4 smooth experts | 663,556 | |
| — shock expert | 50,049 | |
| FrictionHead (165,889 mag + 166,147 dir) | 332,036 | 7.2% |
| **Total** | **4,603,918** | |

The two-stage schedule freezes exactly the 3,472,640-parameter backbone share, so the
breakdown is worth printing.

### 2.1 Opening paragraph — **L373–379**

Currently: *"composed of two coupled subnetworks"*, MoE *"predicts the four aerodynamic
coefficients"*, *"305\,753"* parameters. All three wrong. v2 has four components, the MoE
has `output_dim = 1` (`config.py:45`, comment: *"Cp only (friction handled by dedicated
head)"*), and the model consumes a graph.

Replacement:

```latex
AeroSurrogate v2 is a single end-to-end trainable model with a spatial
backbone and three heads.  An eight-layer message-passing GNN over a
per-simulation $k$-nearest-neighbour graph ($k=8$) maps each surface node to
a 256-dimensional latent $\mathbf{h}_i$; a \emph{ShockIndicator} estimates
from $\mathbf{h}_i$ the probability $p_{s,i}$ that the node lies inside the
supersonic pocket; a \emph{shock-gated Mixture of Experts} predicts the
pressure coefficient $C_p$ from $[\mathbf{h}_i\,\|\,p_{s,i}]$; and a separate
\emph{FrictionHead} predicts the friction vector
$(C_{fx},C_{fy},C_{fz})$ from $\mathbf{h}_i$ as a magnitude and a unit
direction.  Figure~\ref{fig:architecture} shows the complete architecture and
data flow.  The total parameter count is 4\,603\,918, of which 3\,472\,640
($75.4\%$) lie in the GNN backbone.
```

### 2.2 Architecture figure — **L381–470** (full TikZ replacement)

Every block in the current figure is wrong. Specifically:

| line | current | actual |
|---|---|---|
| 392 | input $\mathbf{x}\in\mathbb{R}^{16}$ fanning to all heads | graph: $\mathbf{x}_i\in\mathbb{R}^{16}$, `edge_index` $[2,E]$, `edge_attr` $\in\mathbb{R}^5$; only the node encoder sees $\mathbf{x}$ |
| 398 | ShockIndicator $16{\to}64{\to}32{\to}16{\to}1$ | $256{\to}128{\to}64{\to}32{\to}1$ (`indicator_hidden=[128,64,32]`, `config.py:48`) |
| 401 | Gate $[\mathbf{x}\|p_s]\in\mathbb{R}^{17}{\to}64{\to}32{\to}4$ | $[\mathbf{h}_i\|p_s]\in\mathbb{R}^{257}{\to}128{\to}64{\to}4$ (`models_v2.py:151–155`) |
| 404 | experts $16{\to}128{\to}256{\to}128{\to}4$ | $256{\to}256{\to}256{\to}128{\to}1$ (`models_v2.py:159`) |
| 407 | shock expert $16{\to}128{\to}128{\to}4$ | $256{\to}128{\to}128{\to}1$ (`models_v2.py:162`) |
| 413 | $y_{\mathrm{shock}}=\mathbf{1}[C_p^{\mathrm{RANS}}<C_p^{\mathrm{crit}}]$ | $\mathbf{1}[C_p^{\mathrm{RANS}}<C_p^{\mathrm{crit}}(M)]\wedge\mathbf{1}[M>0.75]$ (`dataset.py:79`) |
| 419 | symbolic-gate variant node | does not exist — **delete** |
| 431 | all four coefficients from the MoE sum | only $\hat C_p$; friction is a parallel branch |
| 436 | input fans out to all four heads | $\mathbf{h}$ fans out to all five heads |

Ready-to-paste replacement (wrap in `\resizebox{\textwidth}{!}{…}`; it is wider than v1's):

```latex
\begin{figure}[ht]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
  font=\small,
  blk/.style={draw, rounded corners=2pt, align=center,
              inner sep=4pt, text width=44mm},
  op/.style={draw, circle, inner sep=1.2pt, fill=gray!10},
  arrow/.style={-{Stealth[length=2.2mm]}, thick}
]
% ---- Input: a graph, not a point ----
\node[blk, fill=gray!12, text width=34mm] (input) at (0,0)
  {\textbf{Graph} (one simulation)\\
  {\scriptsize nodes $\mathbf{x}_i\in\mathbb{R}^{16}$, $N=260\,774$}\\
  {\scriptsize kNN edges $k{=}8$, $E=2\,086\,192$}\\
  {\scriptsize $\mathbf{e}_{ij}=(\delta x,\delta y,\delta z,\|\boldsymbol{\delta}\|,
   \mathbf{n}_i\!\cdot\!\mathbf{n}_j)\in\mathbb{R}^{5}$}};

% ---- Backbone ----
\node[blk, fill=violet!12, text width=36mm] (bb) at (5.2,0)
  {\textbf{GNN backbone}\\
  {\scriptsize node enc.\ $16{\to}128{\to}256$}\\
  {\scriptsize edge enc.\ $5{\to}64{\to}128$}\\
  {\scriptsize $8\times$ message passing}\\
  {\scriptsize residual $+$ LayerNorm}};
\node (hlab) at (8.1,0.45) {$\mathbf{h}_i\!\in\!\mathbb{R}^{256}$};

% ---- Heads ----
\node[blk, fill=red!12]    (si)   at (11.6, 4.0)
  {\textbf{ShockIndicator}\\{\scriptsize $256{\to}128{\to}64{\to}32{\to}1$, sigmoid}};
\node[blk, fill=blue!8]    (gate) at (11.6, 2.1)
  {\textbf{Gate} (Gumbel-Softmax)\\
   {\scriptsize $[\mathbf{h}_i\,\|\,p_s]\in\mathbb{R}^{257}{\to}128{\to}64{\to}4$}};
\node[blk, fill=orange!15] (e1)   at (11.6, 0.2)
  {\textbf{Smooth experts} $f_k$, $k{=}1,\dots,4$\\
   {\scriptsize $256{\to}256{\to}256{\to}128{\to}1$ each}};
\node[blk, fill=red!12]    (se)   at (11.6,-1.7)
  {\textbf{Shock expert} $f^{\mathrm{shock}}$ (residual)\\
   {\scriptsize $256{\to}128{\to}128{\to}1$}};
\node[blk, fill=teal!14]   (fh)   at (11.6,-3.8)
  {\textbf{FrictionHead}\\
   {\scriptsize mag.\ $256{\to}\dots{\to}1$, dir.\ $256{\to}\dots{\to}3$}\\
   {\scriptsize $\exp(\mathrm{clip}(\cdot))\times$ unit direction}};

% ---- Training-only supervision ----
\node[align=center] (ysh) at (11.6,5.5)
  {\scriptsize $y_{\mathrm{shock}}=\mathbf{1}[C_p^{\mathrm{RANS}}<C_p^{\mathrm{crit}}(M)]
   \wedge \mathbf{1}[M>0.75]$ \;(training only)};
\draw[-{Stealth[length=2.2mm]}, thick, dashed] (ysh)
  -- node[right, pos=0.45]{\scriptsize BCE} (11.6,4.55);

% ---- Operators / outputs ----
\node (ps)    at (15.4, 4.0) {$p_s$};
\node[op] (mult)  at (15.4, 0.2)  {$\times$};
\node[op] (mult2) at (15.4,-1.7) {$\times$};
\node[op] (plus)  at (16.9,-0.75) {$+$};
\node[blk, fill=green!12, text width=22mm] (outcp) at (19.0,-0.75)
  {\textbf{$\hat{C}_p$}};
\node[blk, fill=green!12, text width=22mm] (outcf) at (19.0,-3.8)
  {\textbf{$[\hat{C}_{fx},\hat{C}_{fy},\hat{C}_{fz}]$}};

% ---- Wiring ----
\draw[arrow] (input) -- (bb);
\draw[arrow] (bb.east) -- ++(0.6,0) |- (si.west);
\draw[arrow] (bb.east) -- ++(0.6,0) |- (gate.west);
\draw[arrow] (bb.east) -- ++(0.6,0) |- (e1.west);
\draw[arrow] (bb.east) -- ++(0.6,0) |- (se.west);
\draw[arrow] (bb.east) -- ++(0.6,0) |- (fh.west);
\draw[arrow] (si.east) -- (ps);
\draw[arrow] (ps.south) -- ++(0,-1.3) -| ($(gate.east)+(0.4,0)$) -- (gate.east);
\draw[arrow] (ps.east) -- ++(0.7,0) -- (16.4,-2.4) -- (15.4,-2.4) -- (mult2.south);
\draw[arrow] (gate.south) -- (11.6,1.1) -- (15.4,1.1)
  -- node[right, pos=0.5]{\scriptsize $g_k$} (mult.north);
\draw[arrow] (e1.east) -- (mult.west);
\draw[arrow] (se.east) -- (mult2.west);
\draw[arrow] (mult)  -- node[sloped, above, pos=0.2]{\scriptsize $\textstyle\sum_k g_k f_k$} (plus);
\draw[arrow] (mult2) -- node[sloped, below, pos=0.75]{\scriptsize $p_s f^{\mathrm{shock}}$} (plus);
\draw[arrow] (plus) -- (outcp);
\draw[arrow] (fh.east) -- (outcf);
\end{tikzpicture}}
\caption{AeroSurrogate v2 architecture (4\,603\,918 parameters).  An
  eight-layer GNN maps the kNN graph of one surface to a 256-dimensional
  latent $\mathbf{h}_i$ per node.  The ShockIndicator maps $\mathbf{h}_i$ to a
  shock probability $p_s$.  The gate routes the four smooth $C_p$ experts
  using both $\mathbf{h}_i$ and $p_s$; the shock expert contributes an
  additive residual scaled by $p_s$, providing an explicit discontinuity
  mechanism.  The friction vector is produced by an independent head off
  $\mathbf{h}_i$ that is \emph{not} gated by $p_s$.  Dashed elements exist
  only at training time (BCE supervision).  At inference the model requires
  the surface point cloud and the flight condition---from which the kNN graph
  and all node and edge features are derived analytically---but no CFD solver
  output.}
\label{fig:architecture}
\end{figure}
```

The caption above replaces L461–468, which contained three errors in one sentence (the
count, the indicator's input, and the gate's conditioning) plus the non-existent symbolic
variant, plus an *"only $\mathbf{x}$ is required"* claim that understates the v2 requirement:
the model needs the whole surface at once and cannot score an isolated point.

### 2.3 New subsection: Spatial backbone — insert **before L496** (`\subsection{Shock-gated Mixture of Experts}`)

This is the largest structural gap in the paper. The backbone is 75.4% of the parameters and
the entire reason v2 outperforms v1, and it is described nowhere.

Content to cover (all from `src/models_v2.py:56–110`, `build_knn_cache.py:29–62`):

- **Graph.** One graph per simulation, $N = 260\,774$ nodes, built from the node
  coordinates alone with a `scipy.spatial.cKDTree` (`build_knn_cache.py:40–42`); self-edges
  excluded; $k=8$ so $E = 8N = 2\,086\,192$ directed edges. No mesh connectivity is used, so
  a bare point cloud suffices. Edge features
  $(\delta x, \delta y, \delta z, \|\boldsymbol\delta\|, \mathbf{n}_i\!\cdot\!\mathbf{n}_j)$.
- **Encoders.** node $16{\to}128{\to}256$, edge $5{\to}64{\to}128$.
- **Message passing**, one layer:
  $\mathbf{m}_{i\to j} = \mathrm{MLP}([\mathbf{h}_i\|\mathbf{e}_{ij}])$,
  $\mathbf{h}_j \leftarrow \mathrm{LN}\bigl(\mathbf{h}_j + \mathrm{MLP}([\mathbf{h}_j\|
  \mathrm{mean}\,\mathbf{m}\|\mathrm{max}\,\mathbf{m}])\bigr)$.
  **State the asymmetry explicitly:** edges are emitted as $(i \to j)$ for
  $j \in \mathrm{kNN}(i)$ (`build_knn_cache.py:45–46`), and messages are computed at the
  source and scattered to the destination (`models_v2.py:73–75`). Node $j$ therefore pools
  over its *reverse*-kNN neighbourhood, whose size varies across the surface because kNN is
  not a symmetric relation.
- **Depth.** 8 layers, giving an 8-hop receptive field. Residual + LayerNorm on every layer
  (`models_v2.py:77`) is the over-smoothing countermeasure — without it eight rounds of
  mean/max aggregation collapse the latents.
- **Memory.** Activation checkpointing (`models_v2.py:103–107`) trades one extra forward
  pass per layer for $O(1)$ activation memory; this is what makes a 260k-node,
  2.09M-edge graph fit in device memory as a single optimiser step. It is disabled when the
  backbone is frozen (`main_train_v2.py:206`), where the recompute would be pure overhead.

Add the defining equation just above Eq. (`eq:indicator`):

```latex
\begin{equation}
  \mathbf{h}_i = \mathrm{GNN}\bigl(\{\mathbf{x}_j\}_{j=1}^{N},\,\mathcal{E}\bigr)_i
  \in \mathbb{R}^{256}
  \label{eq:backbone}
\end{equation}
```

### 2.4 ShockIndicator — **L475–494**

- **L475–478.** Wrong widths and wrong input. Actual: `_mlp([256, 128, 64, 32, 1])`
  (LayerNorm + SiLU + Dropout(0.1) on hidden layers, linear final), on the backbone latent.
  Replace `$16\!\to\!64\!\to\!32\!\to\!16\!\to\!1$` with
  `$256\!\to\!128\!\to\!64\!\to\!32\!\to\!1$` and add:

  > Crucially, the indicator no longer reads the raw point features but the backbone latent
  > $\mathbf{h}_i$, which after eight rounds of message passing already summarises an 8-hop
  > neighbourhood of the surface. The sensor is therefore spatial: it can condition on the
  > local pressure-gradient structure that defines a shock, not just on the point's own state.

  *(Note: the class docstring at `models_v2.py:8` still says `d → 64 → 32 → 1`. It is stale —
  the constructed widths come from `config.py:48`. Worth fixing in code too.)*

- **L479–485.** Argument is now $\mathbf{h}_i$ (`models_v2.py:246`):
  $\mathrm{shock\_logit}_i = f_{\mathrm{SI}}(\mathbf{h}_i;\theta_{\mathrm{SI}})$,
  $p_{s,i}=\sigma(\mathrm{shock\_logit}_i)$.

- **L489–494.** *"no separate pretraining stage is needed"* directly contradicts the model
  being reported. Run 4 **is** a two-stage schedule: from-scratch joint training, then a
  head-only fine-tune with `--freeze-backbone` (`main_train_v2.py:130–133, 199–209`).
  Replace with an honest statement of the schedule and its measured effect (`HELDOUT3` vs
  `HELDOUT4`: +0.0149 global $C_p$, +0.0394 shock $C_p$), note that $p_s$ is used by the
  $C_p$ branch only — the FrictionHead is not gated by it — and add the measured sensor
  quality from `GATE` L19: **precision 0.8964, recall 0.9927, F1 0.9421** over 8 test
  simulations. Also from `GATE` L17–18: mean $p_s$ = 0.9765 on shock nodes, 0.0317 off.

### 2.5 Shock-gated MoE — **L499–526**

- **L499.** Append: *"…and predicts the pressure coefficient alone; the friction vector is
  produced by a separate head (Section~\ref{sec:friction})."*
- **L500–504.** Wrong on all three counts. Replace:

  > The gate receives $[\mathbf{h}_i\,\|\,p_{s,i}]\in\mathbb{R}^{257}$ and maps it through
  > $257\!\to\!128\!\to\!64\!\to\!4$ (LayerNorm + SiLU on hidden layers, no dropout) to $K$
  > routing logits. Routing is therefore conditioned on the node's aggregated spatial
  > context — which already encodes the flight condition, the chordwise and spanwise
  > position and the neighbourhood pressure structure — together with the estimated shock
  > state.

- **L505–506.** Experts: $256\!\to\!256\!\to\!256\!\to\!128\!\to\!1$ (LayerNorm + SiLU +
  Dropout(0.1)).
- **L507–511.** The regime-specialisation story was written for a gate that saw only the
  flight condition, so specialisation was necessarily per-simulation. The v2 gate is
  per-node, so specialisation is spatial — and it has been measured (`GATE` §[2]).
  Replace the speculative regime list with:

  | mean gate weight | $E_0$ | $E_1$ | $E_2$ | $E_3$ |
  |---|---|---|---|---|
  | shock nodes | 0.4722 | 0.1794 | 0.0631 | 0.2853 |
  | non-shock nodes | 0.1451 | 0.2861 | 0.3144 | 0.2544 |

  Total variation between the two rows: **0.3581**. Load balance (entropy of the mean gate):
  **1.3827 / 1.3863 = 99.7% of uniform**, no expert below 1% usage. Do **not** claim a
  flight-regime partition — none has been measured.

- **L511–515.** The $K=4$ justification rests on the discarded per-condition regime story,
  and no $K$ ablation exists anywhere in `version 2/`. Either run one ($K \in \{2,4,8\}$) or
  state plainly: *"$K=4$ was inherited unchanged from v1 and not re-tuned for v2; a sweep
  over $K$ is left to future work."* Given three weeks, take the second option.
- **L516–518.** Shock expert: $256\!\to\!128\!\to\!128\!\to\!1$; note the residual now
  corrects $C_p$ only.
- **L519–526.** The central equation. Replace `eq:forward` with a three-equation block:

```latex
\begin{equation}
  \hat{C}_{p,i} =
    \underbrace{\sum_{k=1}^{K} g_{k}(\mathbf{h}_i, p_{s,i})\,
                f_{k}(\mathbf{h}_i)}_{\text{smooth MoE output}}
    \;+\;
    \underbrace{p_{s,i}\, f^{\mathrm{shock}}(\mathbf{h}_i)}_{\text{shock residual}}
  \label{eq:forward_cp}
\end{equation}
\begin{equation}
  \hat{\mathbf{C}}_{f,i} =
    \exp\!\bigl(\mathrm{clip}(f_{\mathrm{mag}}(\mathbf{h}_i),\,-10,\,5)\bigr)\;
    \frac{f_{\mathrm{dir}}(\mathbf{h}_i)}{\|f_{\mathrm{dir}}(\mathbf{h}_i)\|},
  \qquad
  \hat{\mathbf{y}}_i = \bigl[\hat{C}_{p,i},\;\hat{\mathbf{C}}_{f,i}\bigr]
  \label{eq:forward_cf}
\end{equation}
```

  with Eq. (`eq:backbone`) from §2.3 above them.
  **Then sweep every `\ref{eq:forward}` in the file and re-point it** — L525 is referenced at
  L753 and implicitly at L1062.

- **L528–547.** *"The shock residual is the key architectural innovation"* — keep the
  argument, but anchor it, because v2 has a direct measurement far stronger than the
  qualitative case (`GATE` §[3], §[5]):

  > Ablating the residual term at inference raises the $C_p$ MAE on shock nodes from
  > $0.0458$ to $0.1370$, an increase of $66.6\%$; on those nodes the residual carries
  > $43\%$ of the magnitude of the smooth term ($|{\rm resid}|/|{\rm smooth}| = 0.4338$).

  Also change *"A smooth MLP can only approximate a discontinuous function by smearing it"*
  to refer to the smooth **decoders over the spatial latent** — the smoothing pathology now
  sits downstream of an 8-layer GNN, not on raw features.

### 2.6 New subsection: Friction head — insert **after** the MoE subsection

Label it `sec:friction` (referenced from §2.5's L499 edit). Cover
(`models_v2.py:180–199`, `training.py:24–44`):

- magnitude/direction factorisation: `mag_head` $256{\to}256{\to}256{\to}128{\to}1$
  produces a log-magnitude, clipped to $[-10, 5]$ then exponentiated; `dir_head`
  $256{\to}256{\to}256{\to}128{\to}3$ produces an unnormalised direction, L2-normalised.
- **why it exists:** friction magnitude spans several orders of magnitude, and the friction
  vector is tangent to the surface, so direction and magnitude are separately learnable.
  Its measured contribution: $C_{fy}$ 0.8463 (v1, `V1SUB` L7) $\to$ 0.9027
  (`HELDOUT4` L19).
- **it is not gated by $p_s$** — `cf_pred = self.friction_head(h)` (`models_v2.py:248`)
  never sees the shock probability or the gate. Say so; the current figure implies otherwise.
- 332,036 parameters (7.2% of the model).

### 2.7 Gumbel-Softmax routing — **L549–596**

There is **no Mach gating of the routing in v2**. `ShockGatedMoEv2.forward`
(`models_v2.py:164–170`) calls `F.gumbel_softmax(gate_logits, tau=tau, hard=False)` for
every node whenever `self.training` is true. There is no Mach branch anywhere in
`models_v2.py`.

- **L549.** Retitle `\subsection{Gumbel-Softmax routing}`.
- **L558–569.** Replace the piecewise Mach condition. The real branch is on train/eval mode:

```latex
\begin{equation}
  \mathbf{g}_i =
  \begin{cases}
    \mathrm{softmax}\!\bigl((\mathbf{l}_i+\mathbf{G}_i)/\tau\bigr)
      & \text{training (all nodes)}\\[2pt]
    \mathrm{softmax}(\mathbf{l}_i/\tau)
      & \text{inference}
  \end{cases}
  \qquad
  G_k = -\log(-\log u_k),\;\; u_k \sim \mathcal{U}(0,1)
  \label{eq:routing}
\end{equation}
```

  State that the straight-through variant is not used (`hard=False`).
- **L579–583.** $\tau$ anneals **linearly**, not exponentially
  (`training.py:161–164`: `frac = min(epoch / max(num_epochs-1, 1), 1.0)`;
  `tau = tau_start + frac * (tau_end - tau_start)`):

```latex
\begin{equation}
  \tau(t) = \tau_{0} + \frac{t}{T-1}\,(\tau_{T}-\tau_{0}),
  \qquad \tau_{0}=1.0,\quad \tau_{T}=0.3,\quad T=200,
  \label{eq:tau}
\end{equation}
```

  and note the schedule restarts at $\tau_0$ for the second (head-only) stage.
- **L584–586.** *"The freestream Mach number is recovered inside the forward pass by
  denormalising…"* — nothing in `models_v2.py` denormalises anything or reads a Mach
  feature; `forward` takes only `(x, edge_index, edge_attr)`. **Delete the sentence.**
- **L586–594.** The whole selective-noise justification is false for v2 — Gumbel noise is
  applied to every node during training. **Delete both sentences.** If the idea is worth
  keeping, move it to Future work and state that v2 applies the relaxation uniformly.
- **L594–596.** The deterministic-softmax half is right; $\tau_T = 0.3$ is not. `tau` is a
  registered buffer (`models_v2.py:156`) saved in the `state_dict`, and checkpoints are
  written at the best-validation epoch (`training.py:297–302`), not at the end of the
  schedule. Verified by loading each checkpoint:

  | checkpoint | stored `moe.tau` | implied best epoch (from $\tau$) | README best epoch |
  |---|---|---|---|
  | `surrogate_v2_moefix_long.pt` (**reported**) | **0.5884** | 118 | 118 ✓ |
  | `surrogate_v2_full.pt` | 0.3844 | 175 | 175 ✓ |
  | `surrogate_v2_best.pt` | 0.4477 | 157 | 157 ✓ |
  | `surrogate_v2_moefix.pt` | 0.3571 | — | — |

  None is 0.3. Replacement:

  > At inference the gate uses a deterministic softmax at the temperature stored in the
  > selected checkpoint. Because checkpoints are written at the best validation epoch rather
  > than at the end of the schedule, the reported model operates at $\tau = 0.59$; the
  > measured routing is nevertheless near-one-hot (mean per-node gate entropy 0.080 of a
  > maximum $\log K = 1.386$).

  (routing sharpness 0.0800 from `GATE` L26.)

### 2.8 Normalisation and activations — **L601–612**

Still broadly true, but incomplete and with one exception. Add:

> Each message-passing layer additionally applies LayerNorm to its residual sum, which
> together with the residual connection is what keeps eight rounds of aggregation from
> over-smoothing the latents. Dropout (0.1) is applied in every MLP except the gating
> network, which is built without it.

(`models_v2.py:77` for the residual LN; `models_v2.py:151–155` — the gate is a hand-built
`nn.Sequential` with LayerNorm + SiLU and no `nn.Dropout`.)

### 2.9 Loss function — **L617–650**

v2 minimises a **four**-term objective (`training.py:96–99`). Every term in the current
equation is wrong.

- **L617.** *"three-term"* → *"four-term"*.
- **L618–628.** Replace `eq:loss`:

```latex
\begin{equation}
  \mathcal{L} =
    w_f\underbrace{\frac{1}{N}\sum_{i=1}^{N} w_i\,
      \bigl(\hat{C}_{p,i} - C_{p,i}\bigr)^{2}}_{\text{shock-weighted }C_p\text{ MSE}}
    \;+\; \lambda_{s}\,
    \underbrace{\mathrm{BCE}\bigl(\mathrm{shock\_logit},\,
      y_{\mathrm{shock}}\bigr)}_{\text{indicator supervision}}
    \;-\; \lambda_{\mathrm{lb}}\,
    \underbrace{H(\bar{\mathbf{g}})}_{\text{load balance}}
    \;+\; \lambda_{\mathrm{fric}}\,w_f\,
    \underbrace{\mathcal{L}_{\mathrm{fric}}}_{\text{friction}}
  \label{eq:loss}
\end{equation}
```

  with

```latex
\begin{equation}
  H(\bar{\mathbf{g}}) = -\sum_{k=1}^{K}\bar{g}_k\log\bar{g}_k,
  \qquad
  \bar{g}_k = \frac{1}{N}\sum_{i=1}^{N} g_{i,k}
  \label{eq:lb}
\end{equation}
\begin{equation}
  \mathcal{L}_{\mathrm{fric}} =
    \mathrm{MSE}\bigl(\log\|\hat{\mathbf{c}}_i\|,\,\log\|\mathbf{c}_i\|\bigr)
    \;+\;
    \frac{1}{N}\sum_{i=1}^{N}
      \min\!\left(\frac{\|\mathbf{c}_i\|}{\overline{\|\mathbf{c}\|}},\,5\right)
      \bigl(1-\cos(\hat{\mathbf{c}}_i,\mathbf{c}_i)\bigr)
  \label{eq:fric}
\end{equation}
```

  evaluated after affine denormalisation to physical units (`training.py:31–32`).
  State in the text that the shock weight $w_i$ and the friction term do not interact:
  friction carries only the per-simulation weight $w_f$, never the node weight
  (`training.py:94`). Note also that the BCE term is **not** multiplied by $w_f$
  (`training.py:78–80`).

- **L630–637.** The explicit Mach indicator is gone from the weight because it now lives
  inside the label (`dataset.py:79, 82`: `shock_node_w = 1.0 + 5.0 * y_shock`, where
  `y_shock` already ANDs $M > 0.75$). And $\lambda_{\mathrm{fric}}$ is missing:

```latex
\begin{equation}
  w_i = 1 + \lambda_{\mathrm{sw}}\,y_{\mathrm{shock},i},
  \qquad
  \lambda_{s}=0.1,\quad \lambda_{\mathrm{lb}}=0.01,\quad
  \lambda_{\mathrm{sw}}=5.0,\quad \lambda_{\mathrm{fric}}=3.0
  \label{eq:weights}
\end{equation}
```

  Add a definition of the per-simulation confidence weight $w_f$:
  `confidence_weight_simple` from `dataset.csv`, which takes exactly two values,
  $1.0$ and $0.5$ (verified), the latter at extreme AoA. **Say that this is the ONERA
  challenge weighting** — that is what makes the weighted-$R^2$ metric in §3 comparable to
  the challenge's reference results.

- **L638–643.** Rewrite the mechanism:

  > The shock-weighted MSE assigns $6\times$ loss weight to points carrying a positive shock
  > label. Because the label itself conjoins $C_p < C_p^{\mathrm{crit}}$ with $M > 0.75$,
  > subsonic points whose $C_p$ dips below the critical value — physically ambiguous at low
  > Mach — are labelled negative and are therefore both weighted at unity and supervised as
  > negatives by the BCE term. The reweighting applies to the $C_p$ term only; the friction
  > objective carries the per-simulation weight alone.

- **L643–645.** `pos_weight = 5.0` is still correct (`config.py:50`), but the
  *"$\approx$14--19\% positives"* rate was measured under v1's label. Under the v2 label the
  measured rate is **21.0%** (438,984 of 2,086,192 nodes over 8 test simulations,
  `GATE` L14). Update the parenthetical, or re-measure over the training split and quote that.
  *(Caution: 2,086,192 appears twice in this document with different meanings — it is the
  node count over 8 simulations here, and coincidentally also the edge count $8N$ of a
  single simulation. Do not conflate them in the paper.)*

- **L645–650.** v2 uses a completely different regulariser (`training.py:82–88, 98`).
  Replace:

  > The load-balance term is the entropy $H(\bar{\mathbf{g}})$ of the batch-averaged gate
  > distribution, maximised (i.e. subtracted from the minimised objective). It attains its
  > maximum $\log K = 1.3863$ under perfectly uniform routing and vanishes when the gate
  > collapses onto a single expert.

  The Switch-Transformer citation (`\cite{Shazeer2017,Fedus2022}`) no longer describes the
  implemented term — either drop it or reframe it as related work. Report the measured
  value here: **1.3827 (99.7% of uniform), no expert below 1% usage** (`GATE` L25, L27).

  A sign error in exactly this term collapsed an entire run. That is worth a sentence in
  Results (see §3, `tab:ablation`): fixing it bought +0.0031 global $C_p$ and +0.0142 shock
  $C_p$ with the schedule held fixed.

### 2.10 Training procedure — **L655–664**

Steps 3 and 4 (PySR, symbolic-gate retraining) do not exist in v2. The actual workflow also
gains a step v1 did not need. Replace:

> The complete workflow consists of four steps (Figure~\ref{fig:pipeline}):
> (1) **preprocessing** — the seven physics and geometry features are derived from the nine
> raw columns, and the z-score scalers and chord/span bounds are fitted on the training
> split; (2) **graph construction** — a $k=8$ kNN graph per simulation is built with a
> KD-tree over the node coordinates ($\approx 1.5$ s per simulation), annotated with the
> five-dimensional edge features and cached to disk for all 468 simulations;
> (3) **stage A** — end-to-end training from scratch (200 epochs); and (4) **stage B** — a
> head-only fine-tune resuming from that checkpoint with the backbone frozen (200 epochs),
> which is the reported model. All four loss terms are active in both stages.

### 2.11 Pipeline figure — **L666–689**

Replace the four blocks:

```latex
\node[stepblk, fill=gray!12]  (s1) {\textbf{1. Preprocess}\\
  {\scriptsize 16 node features,\\ scalers, spatial bounds}};
\node[stepblk, fill=violet!10, right=5mm of s1]  (s2) {\textbf{2. Build kNN cache}\\
  {\scriptsize $k{=}8$ KD-tree, 468 graphs,\\ $2.09$M edges each}};
\node[stepblk, fill=blue!10, right=5mm of s2]  (s3) {\textbf{3. Stage A}\\
  {\scriptsize train end-to-end\\ from scratch}};
\node[stepblk, fill=green!12, right=5mm of s3]  (s4) {\textbf{4. Stage B}\\
  {\scriptsize freeze backbone,\\ fine-tune heads}};
```

Caption (replaces L686–687):

```latex
\caption{Training pipeline.  Steps 1--2 run once on CPU---the kNN cache is the
  dominant preprocessing cost---and steps 3--4 run on GPU, optionally under
  DDP across four devices.  Step 4 produces the reported model.}
```

### 2.12 Optimisation paragraph — **L691–698**

Six separate factual errors. Only the gradient clipping survives.

| claim (L691–696) | actual | source |
|---|---|---|
| Adam | **AdamW** | `training.py:136` |
| fixed lr $10^{-3}$ | 3 epochs linear warmup to $10^{-3}$, then `CosineAnnealingLR` to $\eta_{\min}=10^{-6}$ | `training.py:145–149, 166–170` |
| batch size 256 | `batch_size=1` with `collate_single` — **one full simulation graph** per optimiser step (260,774 nodes, 2,086,192 edges) | `training.py:265–266`, `dataset.py:126–129` |
| patience 5 | patience **8**, counted in validation *checks*, which occur every 2 epochs $\Rightarrow$ ~16 epochs | `config.py:68–69`, `training.py:288, 304–305` |
| tolerance $10^{-4}$ | no tolerance; strict `r2_cp > self.best_r2` | `training.py:297` |
| selection on validation MSE | selection on unweighted $R^{2}(C_p)$ | `training.py:254, 297` |
| clipping at 1.0 | correct | `training.py:193` |

Replacement:

> Optimisation uses AdamW ($\beta_1=0.9$, $\beta_2=0.999$) with peak learning rate $10^{-3}$
> after three epochs of linear warmup, cosine-annealed to $10^{-6}$; weight decay $10^{-5}$;
> gradient-norm clipping at 1.0; one full simulation graph (260\,774 nodes, 2\,086\,192
> edges) per optimiser step; and up to 200 epochs with validation every two epochs and early
> stopping after eight consecutive non-improving checks. Checkpoints are selected on
> unweighted $R^{2}(C_p)$ over a fixed subset of 16 validation simulations, so selection is
> not biased by the training-time shock reweighting or by the per-simulation confidence
> weights. Activation checkpointing in the backbone trades one extra forward pass per layer
> for $O(1)$ activation memory, which is what makes the full 260k-node graph fit in device
> memory.

### 2.13 Validation-split disclosure — insert at **L696**

This is a protocol disclosure the v2 methodology requires, and it currently exists only in
`outputs/README.md`. The 16 validation simulations are drawn from the **test** split
(`main_train_v2.py:180–181`: `rng.choice(N_TEST, n_val)`), so early stopping fits to test
simulations. Add:

> The 16 validation simulations used for checkpoint selection are drawn from the test
> split. All reported metrics are therefore computed on the remaining 140 simulations
> (36\,508\,360 points), which never influenced model selection. The v1 baseline figures are
> quoted from their own protocol---a 10\% sample of all 156 test simulations---so the v1
> numbers are, if anything, optimistic relative to v2's.

Put it in the methodology, not only in Results.

### 2.14 Architecture table — **L700–731**

Every row is stale and three components are missing. Full replacement in §3 below
(`tab:arch`).

### 2.15 Inference pipeline — **L736–770**

- **L736–742.** The three input sources are still correct, but incomplete in a way that
  matters. Add the graph as a derived requirement and state the change of granularity:

  > Unlike the point-wise v1, v2 consumes the surface as a whole: the $k=8$ neighbour graph
  > is rebuilt from the node coordinates alone (no mesh connectivity is needed, so a bare
  > point cloud suffices) and the five-dimensional edge features follow from the coordinates
  > and normals. The model cannot score an isolated point or an arbitrary subset.

- **L744–750.** Keep the analytical-feature item (`preprocessing.py` is unchanged from v1
  and still produces 9 raw + 7 derived = 16), and **insert a new item after it**:

  > \item \emph{Graph construction}: a $k=8$ nearest-neighbour graph is built over the node
  >       coordinates with a KD-tree (self-edges excluded), giving $E = 8N$ directed edges,
  >       and each edge is annotated with
  >       $(\delta x,\delta y,\delta z,\|\boldsymbol\delta\|,\mathbf{n}_i\!\cdot\!\mathbf{n}_j)$.
  >       This takes $\approx 1.5$ s per 260k-node surface and is cached.

- **L751–752.** Incomplete and reproducibility-critical: only the **node** features are
  z-scored (`dataset.py:72`). Edge features go to the edge encoder **raw, in physical
  units** (`dataset.py:110`, `models_v2.py:101`). Amend:

  > \item \emph{Normalisation} of the node features $\mathbf{x}$ with the stored training
  >       statistics. Edge features are fed to the edge encoder unnormalised, in physical units.

- **L753–756.** Replace:

  > \item \emph{A single forward pass over the whole graph}, producing $\hat{C}_p$ from the
  >       shock-gated mixture (Eq.~\ref{eq:forward_cp}) and
  >       $[\hat{C}_{fx},\hat{C}_{fy},\hat{C}_{fz}]$ from the friction head
  >       (Eq.~\ref{eq:forward_cf}), together with the per-node shock probability $p_s$ and
  >       the gate weights as interpretable by-products.

- **L761–765.** Delete the symbolic-gate variant passage entirely. It cannot be lifted
  across: the v1 formula is a function of two raw features, the v2 indicator a function of
  a 256-d learned latent, and substituting it would not save the dominant cost anyway since
  the 8-layer backbone still has to run.
- **L765–770.** *"operates on millisecond time scales per surface point batch"* is a v1
  timing claim carried onto a model 15× larger that runs 8 rounds of message passing over
  2.09M edges and needs a KD-tree build. **No v2 timing exists anywhere in `version 2/`.**
  See §6.M1 — this is the one measurement that must be run.

---

## §3. Results tables — ready-to-paste LaTeX

### 3.1 Sample-size and protocol corrections that touch Results prose

| line | current | correct for v2 | source |
|---|---|---|---|
| 244 | 156 held-out conditions | 140 held-out; 16 excluded (model selection) | `HELDOUT4` L4 |
| 247 | 4\,068\,074-point random sample | 36\,508\,360-point held-out set (or 3\,650\,836 for the v1-comparable 10% sample) | `HELDOUT4` L15, L29 |
| 938 | held-out test sample of 4\,068\,074 … neural-gate (step 2) and symbolic-gate (step 4) | held-out test set of 36\,508\,360 points (140 simulations disjoint from training and model selection); drop the step-2/step-4 pairing | `HELDOUT4` |
| 940–942 | $R^2=0.9506$ / $\approx 0.85$ / floor 0.847 | $R^2(C_p)=0.9729$, friction between 0.9027 and 0.9209 | `HELDOUT4` L19 |
| 943–945 | MSE 0.1228 / RMSE 0.3504 / MAE 0.1612 in normalised space | delete (these are the buggy file's, and normalised space is not comparable to anything else); report physical-unit MAE and weighted $R^2$ instead — `tab:mae` below | §1.1 |
| 1151 | more than four million held-out points | more than 36 million | `HELDOUT4` L15 |
| 1161, 58, 1131 | AoA up to $\pm 15^{\circ}$ | AoA $-15^{\circ}$ to $+12.5^{\circ}$ on the test partition (the full $\pm 15$ exists only across all 468 simulations) | `dataset.csv` |
| 1129–1132 | *"pointwise"*, *"compact"*, $3\times 10^{5}$ | not point-wise — a spatial GNN; 4\,603\,918 parameters | `models_v2.py`, measured |
| 61, 177, 461, 703 | $3\times 10^{5}$ / 305\,753 | 4\,603\,918 | measured |

### 3.2 Main results table — replaces `tab:results` (L976–992)

```latex
\begin{table}[ht]
\centering
\caption{Prediction accuracy on the held-out test set: 140 simulations
  (36\,508\,360 surface points) disjoint from both training and model
  selection.  \emph{Collapsed gate} is the run trained before the
  load-balancing sign correction; \emph{from scratch} is joint end-to-end
  training with the corrected loss; \emph{two-stage} is the reported
  model---end-to-end pre-training followed by a head-only fine-tune with the
  backbone frozen.  The v1 baseline is quoted from its own protocol (a 10\%
  sample of all 156 test simulations, 4\,068\,074 points) and is therefore
  \emph{not} on the same footing: it includes the simulations its own model
  selection saw, and is if anything optimistic relative to the v2 columns.}
\label{tab:results}
\begin{tabular}{llcccc}
\toprule
\textbf{Subset} & \textbf{Coef.} & \textbf{v1}$^{\dagger}$
  & \textbf{collapsed gate} & \textbf{from scratch} & \textbf{two-stage} \\
\midrule
global    & $R^{2}(C_p)$    & 0.9506 & 0.9549 & 0.9580 & \textbf{0.9729} \\
          & $R^{2}(C_{fx})$ & 0.8512 & 0.8665 & 0.8775 & \textbf{0.9172} \\
          & $R^{2}(C_{fy})$ & 0.8463 & 0.8443 & 0.8509 & \textbf{0.9027} \\
          & $R^{2}(C_{fz})$ & 0.8555 & 0.8796 & 0.8806 & \textbf{0.9209} \\
\midrule
shock     & $R^{2}(C_p)$    & 0.8827 & 0.8814 & 0.8956 & \textbf{0.9350} \\
          & $R^{2}(C_{fx})$ & 0.8390 & 0.8231 & 0.8445 & \textbf{0.8923} \\
          & $R^{2}(C_{fy})$ & 0.8292 & 0.7867 & 0.8063 & \textbf{0.8692} \\
          & $R^{2}(C_{fz})$ & 0.8643 & 0.8592 & 0.8677 & \textbf{0.9076} \\
\midrule
transonic & $R^{2}(C_p)$    & 0.9629 & 0.9669 & 0.9698 & \textbf{0.9842} \\
subsonic  & $R^{2}(C_p)$    & 0.9236 & 0.9259 & 0.9303 & \textbf{0.9471} \\
\bottomrule
\multicolumn{6}{l}{\footnotesize $^{\dagger}$ different evaluation protocol; see caption.}
\end{tabular}
\end{table}
```

*Sources, column by column:* v1 = `V1SUB` L7–10; collapsed = `HELDOUT1` L19–22;
from scratch = `HELDOUT3` L19–22; two-stage = `HELDOUT4` L19–22.

### 3.3 Error magnitudes and the ONERA-weighted score — new table

```latex
\begin{table}[ht]
\centering
\caption{Reported model (two-stage) on the held-out test set: mean absolute
  error in physical units, and the ONERA confidence-weighted $R^{2}$ that
  makes the score comparable to the regression-challenge reference results.}
\label{tab:mae}
\begin{tabular}{lrcccc}
\toprule
\textbf{Subset} & \textbf{$N$ pts} & MAE($C_p$) & MAE($C_{fx}$)
  & MAE($C_{fy}$) & MAE($C_{fz}$) \\
\midrule
global    & 36\,508\,360 & 0.0276 & $2.26{\times}10^{-4}$ & $1.64{\times}10^{-4}$ & $1.48{\times}10^{-4}$ \\
shock     &  6\,830\,670 & 0.0405 & $3.99{\times}10^{-4}$ & $2.76{\times}10^{-4}$ & $2.49{\times}10^{-4}$ \\
transonic & 26\,077\,400 & 0.0250 & $2.15{\times}10^{-4}$ & $1.57{\times}10^{-4}$ & $1.36{\times}10^{-4}$ \\
subsonic  & 10\,430\,960 & 0.0415 & $3.02{\times}10^{-4}$ & $2.35{\times}10^{-4}$ & $2.26{\times}10^{-4}$ \\
\midrule
\multicolumn{6}{l}{Weighted $R^{2}$ (\texttt{confidence\_weight}):
  $C_p$ 0.9768,\; $C_{fx}$ 0.9218,\; $C_{fy}$ 0.9097,\; $C_{fz}$ 0.9243} \\
\bottomrule
\end{tabular}
\end{table}
```

*Source:* `HELDOUT4` L19–24.

### 3.4 Ablation / mechanism table — new, replaces the anecdote at L1133–1137

```latex
\begin{table}[ht]
\centering
\caption{What each correction contributes, isolated on the held-out test set,
  and the inference-time ablation of the shock residual.}
\label{tab:ablation}
\begin{tabular}{lcc}
\toprule
\textbf{Change} & $\Delta R^{2}(C_p)$ global & $\Delta R^{2}(C_p)$ shock \\
\midrule
Load-balance sign fix (collapsed $\to$ from scratch) & $+0.0031$ & $+0.0142$ \\
Two-stage schedule (from scratch $\to$ two-stage)    & $+0.0149$ & $+0.0394$ \\
\midrule
\multicolumn{3}{l}{\emph{Inference-time ablation of $p_s\,f^{\mathrm{shock}}(\mathbf{h}_i)$
  on shock nodes, 8 simulations}}\\
MAE($C_p$) with residual    & \multicolumn{2}{c}{0.0458} \\
MAE($C_p$) without residual & \multicolumn{2}{c}{0.1370 \quad ($+66.6\%$)} \\
\bottomrule
\end{tabular}
\end{table}
```

Accompanying sentence: *"The curriculum is worth about 2.8$\times$ the sign fix inside the
shock region"* ($0.0394 / 0.0142 = 2.77$) — the place the architecture exists to serve.
*Sources:* deltas from `HELDOUT1`/`HELDOUT3`/`HELDOUT4` L19–20; ablation from `GATE` L40–42.

### 3.5 Gate behaviour table — new, supports §2.5

```latex
\begin{table}[ht]
\centering
\caption{Measured routing of the reported model over 8 test simulations
  (2\,086\,192 nodes, of which 21.0\% carry a positive shock label).  Load
  balance, the entropy of the mean gate, is $1.3827$ of a maximum
  $\log K = 1.3863$ ($99.7\%$ of uniform) with no expert below $1\%$ usage;
  routing is nevertheless near-one-hot per node (mean per-node entropy
  $0.080$).}
\label{tab:gate}
\begin{tabular}{lcccc}
\toprule
\textbf{Mean gate weight} & $E_0$ & $E_1$ & $E_2$ & $E_3$ \\
\midrule
shock nodes     & 0.4722 & 0.1794 & 0.0631 & 0.2853 \\
non-shock nodes & 0.1451 & 0.2861 & 0.3144 & 0.2544 \\
\bottomrule
\end{tabular}
\end{table}
```

Total variation between the rows: 0.3581. *Source:* `GATE` L14, L22–28.

### 3.6 Architecture table — replaces `tab:arch` (L700–731)

```latex
\begin{table}[ht]
\centering
\caption{Architecture and training hyperparameters of AeroSurrogate v2
  (4\,603\,918 parameters in total).}
\label{tab:arch}
\begin{tabular}{lll}
\toprule
\textbf{Component} & \textbf{Parameter} & \textbf{Value} \\
\midrule
Graph          & Neighbourhood & kNN, $k=8$, directed ($j\in\mathrm{kNN}(i)$) \\
               & Size per simulation & $N = 260\,774$ nodes, $E = 2\,086\,192$ edges \\
               & Edge features & $(\delta x,\delta y,\delta z,\|\boldsymbol\delta\|,
                                  \mathbf{n}_i\!\cdot\!\mathbf{n}_j)\in\mathbb{R}^{5}$ \\
\midrule
Backbone       & Node / edge encoder & $16{\to}128{\to}256$ \;/\; $5{\to}64{\to}128$ \\
               & Message passing & 8 layers; msg $384{\to}256{\to}256$,
                                   upd.\ $768{\to}256{\to}256$ \\
               & Aggregation & mean $\|$ max, residual $+$ LayerNorm \\
               & Latent / params & $d = 256$ \;/\; 3\,472\,640 \\
\midrule
ShockIndicator & Layers / params & $256{\to}128{\to}64{\to}32{\to}1$, sigmoid \;/\; 43\,713 \\
\midrule
Gate           & Layers & $257{\to}128{\to}64{\to}4$ \\
               & Routing & Gumbel-Softmax (\texttt{hard=False}) in training,
                           tempered softmax at inference \\
               & Temperature & $\tau: 1.0 \to 0.3$ (linear over $T$ epochs) \\
\midrule
Smooth experts & Number $K$ / params & 4 \;/\; 663\,556 \\
               & Layers (each) & $256{\to}256{\to}256{\to}128{\to}1$ \;($C_p$ only) \\
\midrule
Shock expert   & Layers / params & $256{\to}128{\to}128{\to}1$ \;/\; 50\,049 \\
               & Coupling & additive residual, scaled by $p_s$ \\
\midrule
Friction head  & Magnitude / direction & $256{\to}256{\to}256{\to}128{\to}1$ \;/\;
                                         $256{\to}256{\to}256{\to}128{\to}3$ \\
               & Output / params & $\exp(\mathrm{clip}(\cdot,-10,5))\times$ unit direction
                                   \;/\; 332\,036 \\
\midrule
Loss           & $\lambda_{s}$ / $\lambda_{\mathrm{lb}}$ / $\lambda_{\mathrm{sw}}$
                 / $\lambda_{\mathrm{fric}}$ & 0.1 / 0.01 / 5.0 / 3.0 \\
               & BCE positive-class weight & 5.0 \\
               & Load balance & entropy of the mean gate (maximised) \\
               & Shock label & $\mathbf{1}[C_p^{\mathrm{RANS}}<C_p^{\mathrm{crit}}(M)]
                               \wedge \mathbf{1}[M>0.75]$ \\
\midrule
Optimisation   & Optimiser & AdamW ($\beta_1{=}0.9$, $\beta_2{=}0.999$) \\
               & Learning rate & $10^{-3}$, 3-epoch linear warmup $+$ cosine to $10^{-6}$ \\
               & Weight decay / clipping & $10^{-5}$ \;/\; $\|\nabla\|_2 \le 1.0$ \\
               & Step / epochs & 1 simulation graph \;/\; 200,
                                 validate every 2, patience 8 checks \\
               & Memory & activation checkpointing in the backbone \\
\midrule
All MLP blocks & Hidden layers & LayerNorm $+$ SiLU $+$ Dropout(0.1)
                                 (gate: no dropout) \\
\bottomrule
\end{tabular}
\end{table}
```

Note the deletion: the **"Transonic threshold $M>0.75$"** row must leave the Gate block —
that threshold survives only inside the shock label, nowhere in the routing. It now appears
under Loss.

### 3.7 Discussion rewrite — **L1127–1142**

All three points need replacing.

- **First (L1129–1132).** Both claims invert in v2. Rewrite so that the point-wise-to-spatial
  move becomes the headline contribution rather than a caveat: the accuracy is obtained by a
  spatial GNN over the surface mesh with 4,603,918 parameters, and that is *why* the gain
  exists (v1 $\to$ v2: $C_p$ 0.9506 $\to$ 0.9729, $C_{fy}$ 0.8463 $\to$ 0.9027).
- **Second (L1133–1137).** *"during development, ablating either mechanism visibly
  degraded…"* is anecdotal — no numbers, no results file. Replace with `tab:ablation` (§3.4).
- **Third (L1138–1142).** Refuted (§1.1) and, for v2, describes nothing that exists.
  Replace with the curriculum finding from `tab:ablation`, and **carry its caveat**: one run
  per arm, seed variance not excluded (§6.L3).

### 3.8 Conclusions — **L1152–1179**

*"Four conclusions are drawn"* becomes bookkeeping once bullet 4 goes (§1.5) and bullet 3
(L1168–1171, LayerNorm *"first-order impact"*) is downgraded (§6.M2). The natural v2 set:

1. The spatial backbone beats the point-wise formulation (numbers from `tab:results`).
2. The shock-gated residual is worth +66.6% of the shock-node $C_p$ MAE.
3. The two-stage curriculum beats joint training by ~2.8× what the load-balance sign fix
   bought inside the shock region.
4. Physics-derived labels keep the inference path solver-free (this bullet, L1163–1167,
   survives v2 unchanged and is the only one that does).

L1158 also needs its mechanism fixed: routing is **not** Mach-gated; it is conditioned on
$[\mathbf{h}_i \| p_s]$ (`models_v2.py:165`).

---

## §4. Figures to regenerate

Every figure showing model output was rendered from v1's symbolic-gate checkpoint — the very
model the paper will no longer report. None can be shown alongside v2 numbers.

| figure | line | current source | action |
|---|---|---|---|
| `fig:parity` | 1011–1018 | `parity_plots.png`, v1 symbolic checkpoint (caption L1015 says so) | regenerate from `surrogate_v2_moefix_long.pt` on the 140-sim held-out set; drop *"symbolic-gate model"* from the caption |
| `fig:cond0` | 1071–1080 | v1 symbolic | regenerate; $\Pi = 1{\times}10^{5}$; test sim 0 — clean |
| `fig:cond26` | 1082–1089 | v1 symbolic | regenerate **and swap the condition** — test sim 112 is a model-selection sim (§1.11); $\Pi = 4{\times}10^{5}$ |
| `fig:cond52` | 1091–1098 | v1 symbolic | regenerate; $\Pi = 2{\times}10^{5}$; sim 69 — clean |
| `fig:cond81` | 1100–1107 | v1 symbolic | regenerate; $\Pi = 2{\times}10^{5}$; sim 78 — clean |
| `fig:cond104` | 1109–1116 | v1 symbolic | regenerate; $\Pi = 2{\times}10^{5}$; sim 86 — clean |
| `fig:cond130` | 1118–1125 | v1 symbolic | regenerate; $\Pi = 4{\times}10^{5}$; sim 147 — clean |
| `fig:architecture` | 381–470 | TikZ | full replacement, §2.2 |
| `fig:pipeline` | 666–689 | TikZ | full replacement, §2.11 |
| `fig:importance` | 823–831 | `feature_importance.png` | v1-only; keep only if §5 keeps a v1 symbolic section. Caption L827–829 (*"Red bars mark the two variables that appear in the selected symbolic expression"*) is fine as stated but must not imply they are the top two — see §5.3 |
| `fig:pareto` | 861–869 | `sr_pareto_front.png` | v1-only; same condition |
| `fig:symmap` | 904–913 | `symbolic_sensor_map.png` | v1-only; same condition, **and** its physical reading is wrong — see §5.4 |
| `fig:pathology` | 102–146 | TikZ sketch | keep — it is an illustrative sketch, still accurate as motivation |

**New figure worth adding (cheap, high value):** `outputs/plots/moefix_long.png` already
exists and shows the gate load balance climbing from 0.50 to 1.379 in the first few epochs
and holding. It is the visual evidence for the sign-fix story in §2.9 and costs nothing.

**Practical note:** there is no v2 surface-field plotting script. `version 2/` contains
`evaluate_v2.py`, `diagnose_gate.py` and `plot_training_curve.py` only. The v1 renderer lives
in `version 1/infer_aero.py` and is written against a point-wise checkpoint. Porting it is
the longest lead-time item in this whole list — start it in week 1.

---

## §5. What to do with the symbolic-distillation section (L773–928)

### 5.1 The structural fact

In v2 the ShockIndicator is `ShockIndicatorSpatial`, an MLP $256{\to}128{\to}64{\to}32{\to}1$
over the GNN latent produced by 8 message-passing layers. It is a function of the node's
whole neighbourhood, not of the nine raw features PySR searched over
(`symbolic_regression_surrogate_base.txt` L3). The v1 expression is therefore **not a
distillation of anything in v2** — it cannot be, because the v2 sensor's input space is not
spanned by those features. And nothing in v2 is retrained with a frozen algebraic gate, so
`sec:symgate` (step 4 of `fig:pipeline`) describes a step that does not exist.

### 5.2 Options, in order of preference

**(A) Cut it — recommended given a three-week budget.** Delete L773–928, contribution
bullet 4 (L198), conclusion bullet 4 (L1172), discussion point 3 (L1138), the abstract
sentence (L52–56), the symbolic node in `fig:architecture`, step 4 in `fig:pipeline`,
`fig:importance`, `fig:pareto`, `fig:symmap`, and change the title (§0). Add one honest
sentence to Future work:

> The closed-form sensor distilled in v1 does not transfer to v2, whose indicator reads a
> learned latent rather than named physical features; re-distilling an interpretable sensor
> from the spatial latent is left as future work.

**(B) Demote to an appendix as a clearly-labelled v1 result.** Requires fixing §1.1, §1.2,
§1.8, §5.3, §5.4 first — roughly four extra days.

**(C) Reframe as *"An interpretable shock criterion"***, which is what it always actually
was: a symbolic fit to the RANS physics label $y_{\mathrm{shock}}$, **not** a distillation of
any network (the paper says so itself at L789–791). Report it as a standalone sensor with
AUC 0.6727, and the gate-substitution experiment honestly at $-0.0085$ $C_p$ / $-0.1005$
$C_{fy}$.

**Under all three options, note what replaces it.** v2 measures a stronger
interpretability-adjacent result: the spatial sensor reaches precision 0.8964 / recall
0.9927 / **F1 0.9421** on the same physics label (`GATE` L19), against the symbolic sensor's
AUC 0.6727. That is the sentence to put in its place.

### 5.3 If any of it is retained, these four claims must still be fixed

- **L784, purpose (iii) *"lightweight deployment — elementary arithmetic, no neural
  inference"***. False on both counts. The deployed sensor appends a 92-knot isotonic step
  function (§1.8), which is not elementary arithmetic; and the gate feeds a MoE whose four
  experts and shock expert are all neural, so only the sensor MLP is avoided. For v2 it is
  void — the sensor consumes $\mathbf{h}_i$ from an 8-layer GNN, so an algebraic sensor saves
  nothing. **Delete purpose (iii).** Purposes (i) interpretability and (ii) physical
  validation survive if reframed.
- **L814–816 and L891–894, the feature-importance story.** Contradicted by the measured
  RandomForest importances in `symbolic_regression_surrogate_base.txt`:

  | rank | feature | importance |
  |---|---|---|
  | 1 | `x_norm` | 0.2123 |
  | 2 | `span_norm` | 0.1730 |
  | 3 | `Mach` | 0.1275 |
  | 4 | `L_factor` | 0.1176 |
  | 5 | `nz` | 0.0977 |
  | 6 | `Cp_crit` | **0.0965** |
  | 7 | `q_dyn` | 0.0889 |
  | 8–9 | `AoA`, `AoA_sin` | 0.0434, 0.0431 |

  `Cp_crit` ranks **sixth of nine**, below `L_factor` and `nz`. The top-ranked feature,
  `x_norm`, does not appear in the expression at all. So the formula couples the
  second-ranked factor with the sixth-ranked one and omits the first. L814's *"Chord-wise and
  span-wise position dominate, followed by the Mach number and its derived quantities"* is
  loose: the order interleaves non-Mach features. Give the numbers instead of the adjective.

- **L820, *"This analysis motivated the inclusion of $x_{\mathrm{norm}}$ and $\eta$"***.
  Circular twice over: the RandomForest was trained on a feature set that already contains
  both, and the SR stage runs *after* the surrogate is trained. Reword to *"is consistent
  with the inclusion of…"* or delete.
- **L853, the Pareto selection.** Scores match the source file, but complexity 5 differs
  from complexity 4 only by an outer `tanh`, a strictly monotone squash. As a classifier the
  two are **identical** — same ROC, same AUC, same ranking of every point. The
  0.0433 vs 0.0422 gap is an MSE artefact of squashing the output toward $[0,1]$, not an
  accuracy gain. State that the two candidates are ranking-equivalent and that the `tanh` is
  chosen only to bound the output into a probability-like range. Otherwise the *"elbow of
  the Pareto front"* argument at `fig:pareto` is misleading.

### 5.4 The physical reading at L871–902 is mathematically wrong

$\tanh(\exp(C_p^{\mathrm{crit}}/\eta))$ with $C_p^{\mathrm{crit}} < 0$ is **strictly
monotone increasing in $\eta$** on $(0,1]$. It is maximal at the **tip** ($\eta = 1$) and has
no interior maximum anywhere. L879–882 claims the probability *"rises over the mid-span
region where transonic shocks typically form"*, and L883–890 then argues that the tip is
unloaded by wash-out twist — which the formula contradicts, since it predicts its highest
shock probability exactly at the unloaded tip. The physical narrative was written to fit a
mid-span story the expression does not encode.

**Action:** rewrite as what the formula is — a monotone inboard-to-outboard ramp whose
steepness is set by Mach. Remove the mid-span peak claim and the wash-out paragraph, or state
explicitly that the expression fails to capture tip unloading, which is a real and reportable
limitation.

Two further points on L899–902:
- 0.6727 is weak discrimination (0.5 = chance). Drop *"notable"* — especially now that the
  v2 spatial sensor sits at F1 0.9421 on the same label.
- The *"0.6727 for a five-node formula"* framing is misleading: AUC depends only on ranking,
  and both `tanh` and `exp` are strictly monotone, so
  $\mathrm{AUC}(\tanh(\exp(z))) = \mathrm{AUC}(z)$ exactly. 0.6727 is the AUC of the bare
  ratio $C_p^{\mathrm{crit}}/\eta$ — a **two**-node expression.
- The value 0.6727 appears nowhere in `version 1/outputs/results/`. It traces only to
  `Overleaf_Paper/pipeline_brief_for_paper.md`, so it is **not reproducible from the shipped
  artefacts**. Regenerate it into a results file or mark it v1-only and unverified.
- L802's *"200 iterations"* likewise appears in no output file (`samples=50000` is confirmed
  at `symbolic_regression_surrogate_base.txt` L3, the iteration count is not). Confirm from
  the PySR run configuration or the `hall_of_fame` checkpoints before keeping it.

---

## §6. Honest limitations to state

### Measurements that must be run before submission

- **M1 — v2 inference timing (blocking for L765–770).** No v2 timing exists anywhere in
  `version 2/`. The v1 claim (*"millisecond time scales per surface point batch"*) is being
  carried onto a model 15× larger that runs 8 rounds of message passing over 2.09M edges and
  needs a KD-tree build. *"Per surface point batch"* also no longer describes the unit of
  work, which is a whole surface. **Re-time end to end and quote the measured numbers**,
  split into (a) one-off graph construction and (b) the forward pass per surface, on named
  hardware. If it cannot be measured in time, keep the qualitative comparison against
  $\mathcal{O}(10^3\text{--}10^4)$ CPU-hours and **drop the millisecond figure** rather than
  carrying it over unverified.
- **M2 — BatchNorm vs LayerNorm (blocking for L1168–1171).** *"an implementation detail with
  first-order impact on deployed accuracy"* is quantified nowhere; no such ablation exists in
  `outputs/results/` for v1 or v2. Either run it or downgrade to a methods note without the
  *"first-order impact"* claim. **Note there is a far better-supported implementation-detail
  story available for free:** the $L_{\mathrm{lb}}$ sign bug (entropy of the mean gate added
  to a minimised loss, driving collapse), fixed in commit `45cb8bc`, worth +0.0031 global
  $C_p$ and +0.0142 shock $C_p$ with the schedule held fixed. Swap the bullet.
- **M3 — shock-label positive rate on the training split.** The paper quotes 14–19% (L365,
  L644) from the v1 label. The v2 label gives 21.0% on the test set (`GATE` L14). Measure it
  on the training split, or quote the test figure and say so.
- **M4 (optional) — a $K$ ablation** ($K \in \{2,4,8\}$), if the $K=4$ justification at L511
  is to be defended rather than withdrawn. Withdraw it (§2.5) unless there is slack.
- **M5 — fix the README self-contradiction** (§1.9) so that whichever number the paper
  quotes, the repo agrees with it.

### Limitations to write into the paper

- **L1 — the v1 baseline is not on the held-out set.** v1's figures come from a 10% sample
  of all 156 test simulations; the v2 table uses 140. Putting v1 on the same footing would
  mean re-evaluating the v1 checkpoint, and v1's validation split was different, so
  *"exclude these 16"* is not even the right exclusion for it. State the protocol difference
  **and the direction of the bias**: run 4 is measured on the *stricter* set — it excludes
  the simulations its own early stopping fitted to, which v1's numbers do not — and still
  leads by +0.0223 on global $C_p$ and +0.0523 on shock. A margin that size does not turn on
  a 140-vs-156 population difference.
- **L2 — early stopping fits to test simulations.** The 16 validation simulations are drawn
  from the test split (`main_train_v2.py:180–181`). This is disclosed in §2.13 and handled by
  reporting on the remaining 140, but it must be said plainly, not buried.
- **L3 — the curriculum result rests on one run per arm.** Two-stage and from-scratch are
  otherwise matched — same data, same loss, same single-process setup, both converged on
  patience — so the comparison is sound as far as it goes. What it cannot rule out is seed
  variance: a single run per arm cannot separate a real effect from a lucky initialisation.
  Say so wherever the +0.0149 / +0.0394 result appears.
- **L4 — the model can no longer score an isolated point.** v2 requires the whole surface
  simultaneously plus a kNN graph. This is a genuine deployment regression relative to v1 and
  should be stated in the inference section, not hidden.
- **L5 — interpretability regressed.** v1 shipped a closed-form sensor (however weak, AUC
  0.6727); v2's sensor is a function of a learned 256-d latent. The v2 sensor is far more
  accurate (F1 0.9421) but far less legible. Naming this trade-off is stronger than ignoring
  it, and it sets up the future-work sentence in §5.2.
- **L6 — no cross-geometry generalisation.** Unchanged from v1 (L1181–1185) and still true:
  a single configuration, a single mesh topology.
- **L7 — friction accuracy remains the floor.** $C_{fy}$ at 0.9027 is still the weakest
  coefficient. The structural explanation at L947–974 (wall-gradient quantity, sign-changing
  projections, label noise at extreme incidence) survives v2 intact and should be **kept** —
  it is one of the strongest passages in the paper and nothing in v2 invalidates it. Only the
  numbers it cites need updating.
- **L8 — DDP.** All four reported runs were single-process, so the DDP defect fixed in
  commit `9fe5f2e` (the wrapper was built and then bypassed, so gradients were never
  all-reduced) was latent and never affected them: with `world_size == 1` the wrapper is
  never constructed. If the paper mentions multi-GPU training at all, it must say the
  reported runs were single-process.

---

## Final cross-reference sweep (do last)

After all edits, grep for and re-point:

- `\ref{eq:forward}` → now `eq:forward_cp` / `eq:forward_cf` (used at L753, implied L1062)
- `\ref{eq:srdiscovered}` → gone or appendix-only (L980, L998, L1174, L907)
- `\ref{fig:pipeline}` step numbers (L661, L921)
- `\ref{sec:sr}` (L212, L761, L915) and `\ref{sec:symgate}` (L927)
- `\ref{sec:srresults}` at L335 — the feature-engineering justification at L334–340 depends
  on the symbolic section surviving; if it is cut, that paragraph needs a different support
- every literal `305\,753` (L379, L461, L703) and `3\times 10^{5}` (L61, L177, L1131)
- every literal `4\,068\,074` (L247, L938, L978, L1151 as "four million")
- every literal `0.9506` outside the v1 baseline row
- `\ref{sec:friction}` — new label, must be defined in §2.6

---

## Found by adversarial review

Independent re-read of `main.tex` (all 1356 lines) against the document above, plus
re-derivation of every proposed number from `version 2/outputs/results/`, `version 2/data/dataset.csv`,
`version 2/src/`, and `version 1/src/`. The document is largely sound — its §3.2/§3.3/§3.4/§3.5
tables reproduce their source files exactly, and §1.9's README contradiction, §1.10's six
$\Pi_{\mathrm{norm}}$ mislabels and §1.11's sim-112 identification all verify. What follows is
what it missed.

### A. Claims in `main.tex` that become false under v2 and are covered nowhere above

#### A1. Contribution bullet 2 is entirely false under v2 — **L189–193**

> *"A \emph{Mach-gated Gumbel-Softmax routing} scheme that applies stochastic expert-selection
> noise only to transonic points ($M>0.75$), where shock-position ambiguity makes routing
> exploration beneficial…"*

§2.7 establishes there is no Mach branch anywhere in `models_v2.py`; §1.4 deletes the abstract
statement of it (L48–50); §3.8 fixes the conclusion statement of it (L1158). The **contribution
bullet** — the most-read statement of the claim, sitting in the numbered list a reviewer skims
first — is never touched. It is a headline contribution describing a mechanism that does not
exist. **Delete it.**

Knock-on the document also misses: **L181** reads *"The four main contributions are:"*. §1.3
alone replaces bullet 4 with two bullets, taking the list to five; deleting bullet 2 as well
returns it to four. Whichever way the §0 decision goes, that count word has to be re-set at the
end, and it is not in the final sweep list.

#### A2. Eq.~(`eq:labels`), the paper's definition of the shock label, is wrong for v2 — **L346–351**

`dataset.py:79` is `((Cp_true < Cp_crit) & (Mach_raw > 0.75))`. The document patches this
conjunction into the architecture figure (§2.2, L413), into the loss weight (§2.9, L630–637) and
into `tab:arch` (§3.6) — but never into **the equation all three point at**. Every
`\ref{eq:labels}` in the paper (L144, L790, L827) would then resolve to the v1 definition, and
`tab:arch`'s "Shock label" row would contradict Eq.~(`eq:labels`) three sections earlier.

```latex
\begin{equation}
  y_{\mathrm{shock}} =
    \mathbf{1}\!\left[C_{p}^{\mathrm{RANS}} < C_{p}^{\mathrm{crit}}(M)\right]
    \wedge
    \mathbf{1}\!\left[M > 0.75\right]
  \label{eq:labels}
\end{equation}
```

The prose immediately after it (**L352–357**) also describes the criterion as the single
inequality and needs the same clause.

#### A3. The physical interpretation of $p_s$ — **L486–489**

> *"…so that $p_s$ acquires the direct physical interpretation $p_s \approx P(M_{\mathrm{local}} > 1)$."*

§2.4 rewrites L475–478, L479–485 and L489–494 and steps over exactly this sentence. Under the v2
label the indicator is supervised on $M_{\mathrm{local}}>1 \wedge M_{\infty}>0.75$, so $p_s$ is
*not* $P(M_{\mathrm{local}}>1)$ — it is trained to read zero on a genuinely supersonic pocket at
$M_{\infty} \le 0.75$. `GATE` L18 (mean $p_s = 0.0317$ off-shock) is consistent with that by
construction. This is the strongest interpretability claim in `sec:indicator` and it is the one
sentence in the subsection §2.4 leaves alone.

#### A4. The abstract never acquires the GNN — **L43–48**

This is the largest gap in the document's handling of the abstract. §1.4 fixes the results
sentence (L52–61) and deletes the Mach-gating sentence (L48–50). What survives untouched is:

> *"The model, AeroSurrogate, couples a **lightweight** shock indicator network with a
> shock-gated Mixture of Experts: four smooth experts capture **regime-dependent aerodynamic
> loading**…"*

After applying §1.4 as written, the abstract is right about the scores and still wrong about the
model: it describes a two-component point-wise architecture, never mentions the spatial backbone
that is 75.4% of the parameters and — per §3.7 — *the reason v2 wins*, calls the sensor
"lightweight" in a 4.6M-parameter model, and asserts a flight-regime specialisation that §2.5
explicitly forbids claiming (*"Do not claim a flight-regime partition — none has been
measured"*). Add one clause naming the eight-layer message-passing backbone, and replace
"regime-dependent" with the measured spatial specialisation.

#### A5. Contribution bullet 1 repeats the unmeasured regime claim — **L183–188**

*"four smooth experts capture **regime-dependent** aerodynamic loading"*. §2.5 bans this wording
at L507 and supplies the measured replacement (spatial specialisation, total variation 0.3581).
The same wording appears three times — L45, L184, L507 — and only L507 is fixed.

#### A6. The friction explanation now contradicts the v2 headline — **L957–958**

§6-L7 says to keep L947–974 and update only the numbers it cites. But L957–958 explains the
friction gap by *"non-local information that **a point-wise input vector** cannot fully
represent"*. In v2 the input is an 8-hop message-passing latent — that is precisely the fix, and
§1.3's new bullet claims credit for it ($C_{fy}$ 0.8463 $\to$ 0.9027). Left as-is the paper
argues both that point-wise inputs cause the friction gap and, three pages earlier, that it no
longer has point-wise inputs. Rewrite the passage as *why a residual gap survives an 8-hop
receptive field*, which is a stronger and still true argument.

#### A7. The $C_{fy}$/$C_{fz}$ argument is falsified by the table §3.2 proposes — **L959–966**

> *"The lateral and vertical components $C_{fy}$ and $C_{fz}$ are further penalised because they
> are projections of the shear-stress vector that change sign…"*

`HELDOUT4` L19: $C_{fz} = 0.9209 > C_{fx} = 0.9172$. `V1SUB` L7: $0.8555 > 0.8512$. The ordering
is $C_{fz} > C_{fx} > C_{fy}$ in **both** v1 and v2, so the paired claim holds for $C_{fy}$
alone and is contradicted for $C_{fz}$ by the numbers in the very table the document asks to
print. Same defect at **L1067–1069** (*"$C_{fx}$ shows the cleanest relative-error map"*).
§6-L7's blanket *"keep it — one of the strongest passages"* endorses both. Restrict the
sign-change argument to $C_{fy}$, or drop the pairing.

#### A8. "312 conditions … for training **and validation**" — **L244–245**

§3.1 edits this line only for the 156$\to$140 count. But §2.13 establishes that v2's 16
validation simulations are drawn from the **test** split (`main_train_v2.py:180–181`), so under
v2 all 312 train-partition conditions are training-only, and §sec:dataset would contradict the
disclosure §2.13 asks to insert 450 lines later. **L246–247** (*"so that no flight condition
seen during training appears in the test set"*) is likewise still true of *training* and no
longer true of *model selection* — say so where the split is first described, not only in
methodology.

#### A9. Cutting `sec:sr` orphans its motivation — **L164–174, L176**

§5.2(A) enumerates what to delete and misses the introduction paragraph that opens the symbolic
gap: L164–174 (SINDy / Eureqa / PySR, ending *"has not been demonstrated"*) and **L176**'s
*"This work addresses **both gaps** simultaneously"*. With `sec:sr` gone the paper raises a gap
it never closes, and `\cite{Brunton2016}`, `\cite{Schmidt2009}`, `\cite{LaCava2021}`,
`\cite{Angelis2023}`, `\cite{Cranmer2020}` become uncited entries in the hand-rolled
`thebibliography` at **L1191**. If §2.9 also drops the Switch-Transformer citation,
`\cite{Fedus2022}` joins them (`\cite{Shazeer2017}` survives at L511).

#### A10. No related work exists for the paper's largest contribution — **L85–88, L148–162**

The introduction's surrogate citations are CNN/Kriging (`\cite{Han2018,Bhatnagar2019,Sekar2019}`)
and the shock-indicator paragraph is troubled-cell literature. §2.3 introduces an eight-layer
message-passing GNN over a kNN surface graph as the headline contribution with **zero citations
proposed anywhere in this document**. MeshGraphNets / GNS / Gilmer-style MPNN prior art is the
first thing an ECCOMAS reviewer will ask for. Relatedly, **L156–157**'s differentiator
(*"they require volumetric field data"*) needs restating: v2's sensor is still surface-only but
is no longer point-local — it is a neighbourhood operator, which narrows the stated gap.

#### A11. "(symbolic gate)" survives in the surface-fields prose — **L1023**

§4 removes *"symbolic-gate model"* from the parity caption (L1015) but the identical attribution
at L1023 (*"the surrogate prediction (symbolic gate)"*) is not in the sweep list.

#### A12. Two pages of prose interpret figures that do not exist yet — **L1038–1069**

§4 schedules the six PNGs for regeneration but treats the commentary that reads them as if it
transfers: L1038–1053 (regime-by-regime — *"the upper-wing supersonic plateau and its
terminating shock appear at the correct chord-wise and span-wise station"*) and L1055–1069
(error maps — *"the extended red patches on the fuselage"*). These are claims about specific
pixels of v1 renderings. They must be re-verified against the v2 figures after regeneration, and
one of them is already known wrong (A7, L1067–1069). Add this to §4 as a dependent task, not an
afterthought.

#### A13. The feature table is no longer the complete input — **L303–304, L308–310, L250–251**

*"Table~\ref{tab:features} lists all sixteen features"* / *"Complete feature set
$\mathbf{x}\in\mathbb{R}^{16}$"*. v2 also consumes 5-dimensional edge features. §2.3 introduces
them in the methodology; §sec:dataset still advertises 16 as complete. **L250–251**
(*"inputs and targets are standardised (z-score) with training-set statistics"*) needs the same
qualifier §2.15 adds at L751–752 — **node features only**; edge features reach the encoder raw,
in physical units (`dataset.py:110`).

#### A14. `fig:pathology`'s caption inherits A2 — **L144**

§4 keeps the figure ("still accurate as motivation"), which it is, but the caption defines the
label as *"Points with $C_p < C_p^{\mathrm{crit}}$ … define the shock label of
Eq.~(\ref{eq:labels})"* — the v1 definition, in the paper's first figure.

---

### B. Numbers that do not match the source files

#### B1. `+66.6%` is the wrong statistic, and it appears in four proposed passages

`GATE` L39–42:

```
MAE with residual    : 0.045799
MAE without residual : 0.137021
contribution         : +0.091222 (+66.58%)
```

`66.58%` is $\Delta / \mathrm{MAE}_{\text{without}}$ — the **share of the ablated error the
residual removes**. The document renders it as an increase, four times:

| where | text |
|---|---|
| §1.5 (proposed conclusion bullet) | *"raises the $C_p$ mean absolute error on shock nodes from $0.0458$ to $0.1370$, an increase of $66.6\%$"* |
| §2.5 (proposed methodology sentence) | *"raises the $C_p$ MAE on shock nodes from $0.0458$ to $0.1370$, an increase of $66.6\%$"* |
| §3.4 `tab:ablation` | `MAE($C_p$) without residual & 0.1370 \quad ($+66.6\%$)` |
| §3.8 (proposed conclusion 2) | *"worth +66.6% of the shock-node $C_p$ MAE"* |

$0.0458 \to 0.1370$ is a **+199.2%** increase, not +66.6%. Both statistics are defensible; the
sentence written is neither. Use one of:

> Removing the shock residual at inference triples the $C_p$ mean absolute error on shock nodes,
> from $0.0458$ to $0.1370$.

> The shock residual accounts for $66.6\%$ of the $C_p$ error the model would otherwise incur on
> shock nodes ($0.1370 \to 0.0458$).

The second is what the file measured. All four occurrences are inside camera-ready LaTeX.

#### B2. §1.8: $\tanh(1) = 0.7318$ is wrong

$\tanh(1) = 0.76159$. The argument around it (the raw expression is bounded away from 1, hence
the 92-knot isotonic calibrator) is correct and important; the constant is not. Note also that
the true supremum over $\eta\in(0,1]$ is $\tanh(\exp(C_p^{\mathrm{crit}}(M)))$, which is
strictly below $\tanh(1)$ — so quoting $\tanh(1)$ as *the* saturation value is loose as well as
mis-evaluated.

#### B3. The v1-baseline bias direction is contradicted by v1's own code — §3.2 caption, §2.13, §6-L1

All three places tell the reader the same thing:

> §3.2 caption: *"…it includes the simulations its own model selection saw, and is if anything
> optimistic relative to the v2 columns."*
> §2.13: *"the v1 numbers are, if anything, optimistic relative to v2's."*
> §6-L1: *"run 4 is measured on the stricter set — it excludes the simulations its own early
> stopping fitted to, which v1's numbers do not."*

**v1 did not select on test simulations.** `version 1/src/data_loader.py:79–83`:

```python
n_val   = int(len(X_train) * DATA_CONFIG.get('val_split', 0.1))
X_val   = X_train[-n_val:]
X_train = X_train[:-n_val]
```

The validation set is the last 10% of the **training** array, taken after the 312/156
simulation-level partition. v1's 156 test simulations never influenced v1 checkpoint selection.
The asymmetry runs the other way: v2 is the run whose early stopping touched the test partition,
and excluding the 16 restores parity rather than making v2 stricter than v1.

This matters because §3.2 puts the sentence **in a table caption in the paper** — a checkable
false statement about the baseline, in the one place a sceptical reviewer looks when a method
beats its predecessor by 0.022. Keep the protocol disclosure (140 sims / 36.5M points vs a 10%
sample of 156) and delete the bias direction. If a direction is wanted, the true v1 weakness is
different and worth one clause: v1's validation slice is a *contiguous tail* of the training
array, not a random simulation-level split — which does not flatter v1's test numbers.

#### B4. §1.4's replacement abstract attributes the test range to the training data

> *"Trained on RANS surface data … spanning Mach $0.30$--$0.96$ and angle of attack from
> $-15^{\circ}$ to $+12.5^{\circ}$ **on the test partition**…"*

The model is trained over the full $\pm15^{\circ}$ DoE (verified: `dataset.csv` train partition
reaches $+15.0^{\circ}$; the test partition tops out at $+12.5^{\circ}$). As written the
sentence hangs a test-set property off "Trained on". Split it: state the training envelope, then
state that the held-out partition spans $-15^{\circ}$ to $+12.5^{\circ}$.

#### Everything else in §3 verifies

Re-derived independently and confirmed exact: all 24 $R^2$ values in §3.2 against `V1SUB` L7–10 /
`HELDOUT1` L19–22 / `HELDOUT3` L19–22 / `HELDOUT4` L19–22; all 16 MAEs and the four weighted
$R^2$ in §3.3 against `HELDOUT4` L19–24; the four deltas in §3.4 ($+0.0031$, $+0.0142$,
$+0.0149$, $+0.0394$) and the $2.77\times$ ratio; §3.5's gate weights, $0.3581$,
$1.3827/1.3863$, $0.0800$ and $21.0\%$ against `GATE` L14, L22–28; §2.7's $\tau = 0.5884$;
$8N = 2\,086\,192$. Against `dataset.csv`: 312/156, three $\Pi$ values $\{1,2,4\}\times10^5$, 13
Mach numbers, `confidence_weight_simple` $\in \{0.5, 1.0\}$, test AoA range
$[-15^{\circ}, +12.5^{\circ}]$, and every row of §1.10's $\Pi$ table and §1.11's identification of
sim 112 (M=0.70, AoA=−10, $\Pi=4\times10^5$, $w_f=0.5$, in the excluded 16). §1.9's README
contradiction is real: `outputs/README.md` L157–159 does attribute F1 0.9564 / +78.6% to the
8-sim run that `gate_moefix_long.txt` reports at 0.9421 / +66.58%.

#### One number collision the document does not warn about

It flags the `2\,086\,192` collision (node count over 8 sims vs edge count of one sim) but not
this one: **`0.9421` is used for two unrelated quantities** — the v1 symbolic gate's
$R^2(C_p)$ (§1.1, `V1SUB` L12) and the v2 spatial sensor's F1 (§1.9, §2.4, §5.2, `GATE` L19).
Under option (B) or (C) of §5.2 both appear in the same paper. Spell out the metric at every
occurrence.

---

### C. LaTeX defects in the proposed replacements

#### C1. `tab:arch` (§3.6) will not fit the text block

The preamble is `{lll}` — no `p{}` column, so cells cannot wrap; the line breaks in the proposed
source are whitespace and typeset as one long line. The class sets `textwidth=16cm`
(`WCCM-ECCOMAS_2026_Full_Paper.cls` L34). Offending cells:

- `Gumbel-Softmax (\texttt{hard=False}) in training, tempered softmax at inference`
- `$256{\to}256{\to}256{\to}128{\to}1$ \;/\; $256{\to}256{\to}256{\to}128{\to}3$`
- `$10^{-3}$, 3-epoch linear warmup $+$ cosine to $10^{-6}$`
- `1 simulation graph \;/\; 200, validate every 2, patience 8 checks`
- `8 layers; msg $384{\to}256{\to}256$, upd.\ $768{\to}256{\to}256$`
- `$(\delta x,\delta y,\delta z,\|\boldsymbol{\delta}\|,\mathbf{n}_i\!\cdot\!\mathbf{n}_j)\in\mathbb{R}^{5}$`

The longest is roughly twice v1's longest cell (`LayerNorm + SiLU + Dropout(0.1)`), which is why
v1's `{lll}` worked. Make column 3 `>{\raggedright\arraybackslash}p{0.52\textwidth}`, or split
the friction-head row in two, before pasting.

#### C2. `\;` used in text mode, ~10 times

`\;` is math spacing. It appears outside `$…$` in nine `tab:arch` rows (`\;/\;`) and once in
`tab:mae` (`$C_p$ 0.9768,\; $C_{fx}$ …`). Current `amsmath` aliases `\;` to the text-safe
`\thickspace`, and `amsmath` is loaded (main.tex L4), so this will most likely compile — but it
is fragile, nothing else in `main.tex` does it, and it silently breaks if the class or a future
package redefines the alias. Use `$\,/\,$` or a plain `/`.

#### C3. The §2.2 architecture figure resizes below legibility

The picture spans $x \approx -2.1$ (input node west edge: node at 0, `text width=34mm`, inner
sep) to $x \approx +20.5$ (`outcp` at 19.0, `text width=22mm`) — about **22 cm** against a 16 cm
text block. `\resizebox{\textwidth}{!}{…}` therefore scales everything by $\approx 0.72$. The
block labels are `\scriptsize` inside `font=\small`, i.e. ~7 pt, landing near **5 pt** after the
resize. v1's figure spanned ~16 cm and needed no resize, which is why it had none. Fix by
stacking the five heads in two columns, shortening the head labels, or moving to a full-width /
landscape float — do not just wrap it and hope.

#### C4. Style deviations from the four existing tables

- All of `tab:features`, `tab:arch`, `tab:results`, `tab:pareto` end `\bottomrule` immediately
  followed by `\end{tabular}`. §3.2 puts a `\multicolumn{6}{l}{…}` note row **after**
  `\bottomrule`; §3.3 puts one between `\midrule` and `\bottomrule`. Both are legal booktabs,
  neither matches the file, and the two new tables do not match each other. Pick one.
- §2.2 TikZ: `\textbf{$\hat{C}_p$}` and `\textbf{$[\hat{C}_{fx},\hat{C}_{fy},\hat{C}_{fz}]$}`
  are no-ops — `\textbf` does not reach inside `$…$`. Use `\boldsymbol`/`\bm` or drop it.
- `violet`, `teal` (§2.2, §2.11) are fine: `xcolor` (main.tex L7) defines both in its base set.
  `\resizebox` needs `graphicx` (L3) ✓; `$(…)+(…)$` node coordinates need `calc` (L10) ✓;
  `\boldsymbol` comes with `amsmath`'s `amsbsy` ✓.

#### C5. Gaps in the final cross-reference sweep

The list at the end of the document omits:

- `\ref{eq:labels}` — L144, L350, L790, L827 (see A2)
- `\ref{tab:features}` — L303, if the edge features are added (A13)
- the three **new** labels `tab:mae`, `tab:ablation`, `tab:gate` — defined in §3.3–§3.5 but given
  a citing sentence in the body only for `tab:ablation` (§3.7). A `\label` with no `\ref` is a
  table the reader is never sent to.
- §2.5 prints the gate-weight numbers inline as a methodology table **and** §3.5 prints the same
  six numbers again as `tab:gate`. Decide which, and reference the other.

---

### D. Verdict on the abstract and the conclusions specifically

**Abstract — half-handled.** §1.4 corrects the numbers (L52–61) and removes the Mach-gating
sentence (L48–50), and those edits are right. It does not touch L43–48, so after applying this
document the abstract still describes a point-wise two-subnetwork model, never mentions the
spatial backbone, keeps "lightweight", and keeps the regime-specialisation claim §2.5 forbids
(A4, A5). Net effect: the abstract would be *correct about the scores and wrong about the
model* — the worst combination, because the scores are what makes a reviewer read the
architecture sentence carefully.

**Conclusions — better handled, two gaps.** §3.8 restructures the bullets and correctly catches
the Mach-gating at L1158.

1. §3.8 says bullet 2 (**L1163–1167**) *"survives v2 unchanged and is the only one that does"*.
   It does not quite: *"the critical-pressure criterion"* is now a conjunction with $M>0.75$
   (A2). One clause.
2. The opening sentence (**L1149–1152**) still calls the model *"a physics-aware, shock-gated
   Mixture of Experts surrogate"* with no mention of the backbone — the same omission as the
   abstract. §3.1 patches only the point count on L1151.
3. Future work (**L1181–1185**) promises to *"exploit the differentiability of the surrogate as a
   shock-aware penalty inside adjoint-based aerodynamic shape optimisation loops"*. §6-L4 states
   that v2 can no longer score an isolated point and needs the whole surface plus a kNN graph —
   which makes exactly that use-case materially harder. The document raises L4 but never connects
   it to the promise it undercuts. Either qualify the future-work item or say what changes about
   it under the graph formulation.
