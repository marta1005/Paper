#!/usr/bin/env python3
"""
distill_symbolic_soft.py — Distil the neural ShockIndicator into a PySR formula
using soft (continuous) probability targets instead of hard binary shock labels.

Why: the original PySR was trained on y_shock ∈ {0,1} (hard labels).
     The neural ShockIndicator outputs smooth p_s ∈ [0,1] that correlates better
     with local flow physics (including near-root transonic shocks). Fitting PySR
     to soft targets should produce a formula that captures this nuance.

Output: outputs/models/shock_sensor_symbolic_soft_best.pkl
        (same format as shock_sensor_symbolic_surrogate_base.pkl)

Usage (server, with PySR / Julia):
    PAPER_TRAIN_FRACTION=1.0 python distill_symbolic_soft.py

Requirements: PySR, sklearn, torch, numpy
"""
import os; os.environ['PAPER_NUM_WORKERS'] = '0'
import sys, pickle, logging
import numpy as np
import torch
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent))
from config import MODEL_DIR, DATA_CONFIG, MODEL_CONFIG, SEED
from src.models import AeroSurrogate, PySRWrapper
from src.data_loader import get_dataloaders

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SR_FEATURES = ['Mach', 'AoA', 'x_norm', 'span_norm', 'nz', 'Cp_crit', 'q_dyn', 'AoA_sin', 'L_factor']
SR_IDX      = [6, 7, 14, 15, 5, 13, 9, 11, 12]   # cols in 16-feature X_derived


def get_soft_targets(fraction=0.05):
    """
    Run the trained neural ShockIndicator on training data to get soft p_s targets.
    Returns (X_sr_features, p_s_soft) as numpy arrays.
    """
    # Load neural gate checkpoint
    ckpt_path = MODEL_DIR / 'surrogate_best.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Neural gate checkpoint not found: {ckpt_path}")

    cfg   = MODEL_CONFIG['surrogate']
    model = AeroSurrogate(
        in_dim=MODEL_CONFIG['autoencoder']['input_dim'],
        num_experts=cfg['num_experts'], output_dim=cfg['output_dim'],
        indicator_hidden=cfg.get('indicator_hidden'),
        expert_hidden=cfg.get('expert_hidden'),
        shock_expert_hidden=cfg.get('shock_expert_hidden'),
    )
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.eval()
    logger.info(f"Loaded neural gate model from {ckpt_path}")

    scaler  = np.load(str(MODEL_DIR / 'scaler.npy'), allow_pickle=True).item()
    X_mean  = np.array(scaler['X_mean'], dtype=np.float32)
    X_std   = np.array(scaler['X_std'],  dtype=np.float32)

    # Load training data
    os.environ['PAPER_TRAIN_FRACTION'] = str(fraction)
    train_loader, _, _, _ = get_dataloaders(sample_fraction=fraction)
    logger.info(f"Data loaded (fraction={fraction}): {len(train_loader)} batches")

    X_raw_list, sp_list = [], []
    with torch.no_grad():
        for X_batch, _ in train_loader:
            _, sp = model.shock_indicator(X_batch)
            # Denormalize to get physical features
            X_raw = (X_batch.cpu().numpy() * X_std + X_mean)
            X_raw_list.append(X_raw)
            sp_list.append(sp.cpu().numpy().ravel())

    X_raw_all = np.vstack(X_raw_list)
    sp_all    = np.concatenate(sp_list)

    # Select SR features
    X_sr = X_raw_all[:, SR_IDX]
    logger.info(f"Soft targets: {len(sp_all):,} points  "
                f"mean={sp_all.mean():.4f}  std={sp_all.std():.4f}")
    return X_sr, sp_all


def run_pysr(X_sr, p_s_soft):
    """Run PySR to discover a symbolic formula for soft p_s targets."""
    from pysr import PySRRegressor
    model = PySRRegressor(
        niterations=100,
        binary_operators=['+', '-', '*', '/', '**'],
        unary_operators=['exp', 'tanh', 'abs', 'sqrt'],
        populations=30,
        population_size=50,
        maxsize=20,
        loss='loss(x, y) = (x - y)^2',     # MSE on soft probabilities
        verbosity=1,
        random_state=SEED,
        deterministic=True,
        parallelism='multithreading',
        extra_sympy_mappings={'maximum': 'Max'},
    )
    logger.info("Starting PySR soft-target regression ...")
    model.fit(X_sr, p_s_soft, variable_names=SR_FEATURES)
    return model


def build_pkl(pysr_model, sr_idx, X_sr_val, sp_val):
    """
    Build a pkl dict from the best PySR equation, calibrate with IsotonicRegression.
    """
    best = pysr_model.get_best()
    expr_str = str(best['sympy_format'])
    logger.info(f"Best PySR expression: {expr_str}  (complexity={best['complexity']}  MSE={best['loss']:.6f})")

    # Build PySRWrapper with a numpy lambda
    fn_lambda = pysr_model.sympy_lambda()

    wrapper = PySRWrapper(fn_lambda, expr_str=expr_str, feature_names=SR_FEATURES)
    raw_val = wrapper.predict_proba(X_sr_val)[:, 1]

    # Isotonic regression calibrator
    cal = IsotonicRegression(out_of_bounds='clip')
    cal.fit(raw_val, np.clip(sp_val, 0, 1))
    logger.info(f"Calibrator fitted  raw=[{raw_val.min():.4f},{raw_val.max():.4f}]  "
                f"cal=[{cal.predict(raw_val).min():.4f},{cal.predict(raw_val).max():.4f}]")

    return {
        'clf':         wrapper,
        'calibrator':  cal,
        'sr_features': SR_FEATURES,
        'sr_idx':      sr_idx,
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    fraction = float(os.environ.get('PAPER_TRAIN_FRACTION', 0.05))
    logger.info(f"distill_symbolic_soft.py  fraction={fraction}")

    X_sr, p_s_soft = get_soft_targets(fraction=fraction)

    # Split train/val for calibration
    n     = len(X_sr)
    n_val = max(10_000, int(0.1 * n))
    rng   = np.random.default_rng(SEED)
    val_i = rng.choice(n, n_val, replace=False)
    tr_i  = np.setdiff1d(np.arange(n), val_i)

    pysr_model = run_pysr(X_sr[tr_i], p_s_soft[tr_i])

    out_pkl = build_pkl(pysr_model, SR_IDX, X_sr[val_i], p_s_soft[val_i])

    out_path = MODEL_DIR / 'shock_sensor_symbolic_soft_best.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(out_pkl, f)
    logger.info(f"Soft-target pkl saved → {out_path}")

    # Also save pysr model for inspection
    pysr_model.save(str(MODEL_DIR / 'pysr_soft_hall_of_fame.csv'))
    logger.info("Done.")


if __name__ == '__main__':
    main()
