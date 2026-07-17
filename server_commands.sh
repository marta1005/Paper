#!/usr/bin/env bash
# =============================================================================
# Server retraining commands — AeroSurrogate symbolic gate ablations
# Run from the project root on the GPU server.
# =============================================================================

# ── Prerequisites ─────────────────────────────────────────────────────────────
# Make sure the protected pkl exists (already created locally and committed):
#   outputs/models/shock_sensor_symbolic_protected.pkl
#
# Baseline checkpoint to warm-start from:
#   outputs/models/surrogate_best.pt  (neural gate, R²=0.9506)

# ── FASE 2a: Symbolic gate — protected formula, warm start ───────────────────
# Formula: tanh(exp(Cp_crit / max(span_norm, 0.1)))
# Fixes the near-root suppression bug in the original formula.
PAPER_EPOCHS=200 python main_train.py \
    --stages surrogate \
    --symbolic-gate outputs/models/shock_sensor_symbolic_protected.pkl \
    --gate-mode symbolic \
    --warm-start outputs/models/surrogate_best.pt

# Checkpoint saved to: outputs/models/surrogate_symbolic_best.pt
# (overwrite: the protected formula replaces the old symbolic checkpoint)
# If you want to keep both, rename first:
#   cp outputs/models/surrogate_symbolic_best.pt outputs/models/surrogate_symbolic_orig_best.pt

# ── FASE 2b: Soft-target PySR distillation ───────────────────────────────────
# Step 1: run PySR to discover formula from neural ShockIndicator soft targets.
# Requires PySR + Julia. Use 100% of training data for best formula quality.
PAPER_TRAIN_FRACTION=1.0 python distill_symbolic_soft.py
# Output: outputs/models/shock_sensor_symbolic_soft_best.pkl

# Step 2: retrain surrogate with soft-target symbolic gate (warm start from neural).
PAPER_EPOCHS=200 python main_train.py \
    --stages surrogate \
    --symbolic-gate outputs/models/shock_sensor_symbolic_soft_best.pkl \
    --gate-mode symbolic \
    --warm-start outputs/models/surrogate_best.pt

# ── Evaluation after each training ────────────────────────────────────────────
# Run evaluate_subsets.py to get correct numbers (uses PySR formula, not ShockIndicator).
# Protected formula:
python evaluate_subsets.py --skip-defaults \
    --ckpt outputs/models/surrogate_symbolic_best.pt \
    --label "Symbolic gate (protected formula, warm start)" \
    --gate-mode symbolic

# Soft-target formula (after rename if needed):
python evaluate_subsets.py --skip-defaults \
    --ckpt outputs/models/surrogate_symbolic_soft_best.pt \
    --label "Symbolic gate (soft-target PySR, warm start)" \
    --gate-mode symbolic

# ── Expected outcome ──────────────────────────────────────────────────────────
# If either variant gives R²(Cp) > 0.9421, update ablations.md and paper numbers.
# If both stay below 0.9421, publish 0.9421 as the honest symbolic gate baseline.
# The neural gate (0.9506) remains the paper's claimed result regardless.
