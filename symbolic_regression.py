#!/usr/bin/env python3
"""
Symbolic shock sensor — three modes:

  physics   (default): PySR/DecisionTree directly on CFD ground-truth labels.
                       No neural model needed.

  surrogate:           Distill the ShockIndicator of the trained AeroSurrogate.
                       Uses shock_prob (smooth 0-1) from the neural network as
                       the PySR target — much easier to fit than binary labels.
                       Recommended for getting a clean algebraic expression.

  distill:             Distill the trained VirtualShockSensor (legacy AE pipeline).

Usage:
    python symbolic_regression.py --mode surrogate --fallback   # DT from surrogate probs
    python symbolic_regression.py --mode surrogate --iterations 200   # PySR from surrogate probs
    python symbolic_regression.py --fallback                    # physics DT (fast baseline)
    python symbolic_regression.py --knn-labels                  # physics mode with K-NN labels
"""
import argparse
import logging
import sys
import numpy as np
import torch
from pathlib import Path

from config import MODEL_DIR, RESULT_DIR, MODEL_CONFIG, DATA_CONFIG, PREPROCESSING_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Feature layout (after preprocessing, 16-feature array) ───────────────────
ALL_FEATURES = [
    'x', 'y', 'z', 'nx', 'ny', 'nz',          # 0-5: geometry
    'Mach', 'AoA', 'Pi_1e5',                    # 6-8: flight conditions
    'q_dyn', 'Pi_norm', 'AoA_sin', 'L_factor', 'Cp_crit',  # 9-13: derived physics
    'x_norm', 'span_norm',                       # 14-15: geometry context
]

# Features for symbolic regression — exclude raw coords (redundant with x_norm/span_norm)
# Include nz to distinguish upper/lower surface
SR_FEATURES = ['Mach', 'AoA', 'x_norm', 'span_norm', 'nz', 'Cp_crit', 'q_dyn', 'AoA_sin', 'L_factor']
SR_IDX      = [ALL_FEATURES.index(n) for n in SR_FEATURES]


# ── Physics label construction ────────────────────────────────────────────────

def compute_Cp_crit(Mach, gamma=1.4):
    sonic_ratio = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * Mach ** 2)
    return (2.0 / (gamma * np.maximum(Mach ** 2, 1e-6))) * (sonic_ratio ** (gamma / (gamma - 1.0)) - 1.0)


def make_physics_labels(X_raw, Y_raw, margin=0.0):
    """
    Shock label: Cp_real < Cp_crit(Mach)  [point is in the supersonic bubble].
    margin > 0 filters marginal cases (points just barely below Cp_crit).
    Returns (shock_label: int array, Cp_crit array, intensity array).
    """
    g        = PREPROCESSING_CONFIG['gamma']
    Mach     = X_raw[:, 6]
    Cp_real  = Y_raw[:, 0]
    Cp_crit  = compute_Cp_crit(Mach, g)
    intensity = np.maximum(0.0, Cp_crit - Cp_real)
    shock     = (Cp_real < Cp_crit - margin).astype(int)
    pos_rate  = shock.mean() * 100
    logger.info(f"Physics labels: {shock.sum():,} shock / {len(shock):,} total  ({pos_rate:.1f}% positive)")
    return shock, Cp_crit, intensity


def make_knn_labels(X_raw, Y_raw, k=10, gradient_percentile=30, margin=0.0):
    """
    Improved labels: shock = (Cp < Cp_crit) AND (local |ΔCp| > threshold).

    Groups points by simulation (Mach, AoA rounded to 3 dp) to only compute
    K-NN within the same CFD run.  The gradient filter removes LE suction
    peaks and other low-pressure zones that are NOT shocks.

    gradient_percentile: keep only points in the top (100 - percentile)%
                         of |ΔCp| among candidate shock points.
    """
    try:
        from sklearn.neighbors import KDTree
    except ImportError:
        logger.error("scikit-learn required for K-NN labels: pip install scikit-learn")
        return make_physics_labels(X_raw, Y_raw, margin)[0], None

    base_shock, Cp_crit, intensity = make_physics_labels(X_raw, Y_raw, margin)

    xyz  = X_raw[:, 0:3].astype(np.float64)
    Cp   = Y_raw[:, 0].astype(np.float64)
    Mach = X_raw[:, 6]
    AoA  = X_raw[:, 7]

    # Simulation ID: round to 3 decimal places
    sim_key = (np.round(Mach, 3) * 1e6).astype(np.int64) + (np.round(AoA, 3) * 1e3).astype(np.int64)

    grad_mag = np.zeros(len(X_raw), dtype=np.float32)
    unique_sims = np.unique(sim_key)
    logger.info(f"K-NN gradient: {len(unique_sims)} unique simulations, k={k}")

    for sim_id in unique_sims:
        mask = sim_key == sim_id
        n    = mask.sum()
        if n < k + 2:
            continue

        pts = xyz[mask]
        cp  = Cp[mask]
        tree = KDTree(pts)
        _, idx = tree.query(pts, k=k + 1)      # (n, k+1) — first is self

        cp_nbr   = cp[idx[:, 1:]]              # (n, k) — neighbors' Cp
        delta_cp = np.abs(cp_nbr - cp[:, None])  # (n, k)
        local_g  = delta_cp.max(axis=1).astype(np.float32)
        grad_mag[mask] = local_g

    # Threshold: gradient_percentile among base_shock points
    shock_grads = grad_mag[base_shock.astype(bool)]
    if len(shock_grads) == 0:
        logger.warning("No base shock points — falling back to Cp-only labels")
        return base_shock, grad_mag

    thresh = np.percentile(shock_grads, gradient_percentile)
    logger.info(f"Gradient threshold (p{gradient_percentile}): {thresh:.4f}")

    shock = (base_shock.astype(bool) & (grad_mag > thresh)).astype(int)
    pos_rate = shock.mean() * 100
    logger.info(f"K-NN filtered labels: {shock.sum():,} shock / {len(shock):,}  ({pos_rate:.1f}% positive)")
    return shock, grad_mag


# ── Feature extraction ────────────────────────────────────────────────────────

def load_raw_data(n_samples):
    """Load preprocessed X (16 features) and raw Y without normalisation."""
    data_dir = Path(DATA_CONFIG['X_train_path']).parent

    x_path = data_dir / 'X_train_derived.npy'
    y_path = DATA_CONFIG['Y_train_path']

    if not x_path.exists():
        logger.error(f"Missing {x_path} — run preprocess_data.py first")
        sys.exit(1)

    X_full = np.load(str(x_path), mmap_mode='r')
    Y_full = np.load(str(y_path), mmap_mode='r')

    if X_full.shape[1] < 16:
        logger.error(f"X has {X_full.shape[1]} features — re-run preprocess_data.py to get 16")
        sys.exit(1)

    n = min(n_samples, len(X_full))
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(X_full), n, replace=False))
    X   = np.asarray(X_full[idx], dtype=np.float32)
    Y   = np.asarray(Y_full[idx], dtype=np.float32)
    logger.info(f"Loaded {len(X):,} samples  X={X.shape}  Y={Y.shape}")
    return X, Y


def extract_sr_features(X):
    """Extract the subset of X used for symbolic regression (unscaled)."""
    X_sr = X[:, SR_IDX]
    logger.info(f"SR features: {SR_FEATURES}  shape={X_sr.shape}")
    return X_sr


# ── Symbolic regression ───────────────────────────────────────────────────────

def feature_importance(X, y, feature_names):
    from sklearn.ensemble import RandomForestClassifier
    y_bin = (y > 0.5).astype(int)
    rf    = RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42)
    rf.fit(X, y_bin)
    imp   = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    logger.info("\n=== Feature importances (RandomForest on physics labels) ===")
    for i in order:
        bar = '█' * int(imp[i] * 50)
        logger.info(f"  {feature_names[i]:12s}: {imp[i]:.4f}  {bar}")
    return dict(zip(feature_names, imp))


def run_pysr(X, y, feature_names, n_iter=50, soft_target=False):
    try:
        from pysr import PySRRegressor
    except ImportError:
        logger.error("PySR not installed: pip install pysr")
        return None

    logger.info(f"PySR input: {len(X):,} samples  target_mean={y.mean():.3f}  soft={soft_target}")

    # For soft targets (shock_prob from neural net): add sigmoid so PySR can find
    # f(x) = sigmoid(linear_combination) — much cleaner than approximating a step function.
    unary_ops = ['exp', 'log', 'abs', 'sqrt', 'tanh']
    if soft_target:
        unary_ops.append('sigmoid')

    model = PySRRegressor(
        niterations=n_iter,
        binary_operators=['+', '-', '*', '/'],
        unary_operators=unary_ops,
        constraints={'^': (-1, 1)},
        maxsize=20,
        populations=20,
        population_size=50,
        model_selection='best',
        parsimony=0.001,
        batching=True,
        batch_size=5000,
        random_state=42,
        verbosity=1,
        tempdir='/tmp/pysr_tmp',
        delete_tempfiles=True,
    )
    model.fit(X, y.astype(np.float32), variable_names=feature_names)
    return model


def run_decision_tree(X, y, feature_names, max_depth=7):
    from sklearn.tree import DecisionTreeClassifier, export_text
    y_bin = (y > 0.5).astype(int)
    clf   = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight='balanced',
        random_state=42,
    )
    clf.fit(X, y_bin)
    rules = export_text(clf, feature_names=list(feature_names))
    return clf, rules


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_with_isotonic(scores, labels):
    """
    Fit isotonic regression to convert raw model scores to calibrated probabilities.
    Returns a fitted IsotonicRegression object.
    """
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(scores.ravel(), labels.ravel())
    logger.info("Isotonic calibration fitted.")
    return iso


def calibrate_decision_tree(clf, X, y):
    """Isotonic calibration on the decision tree's probability output."""
    from sklearn.isotonic import IsotonicRegression
    proba = clf.predict_proba(X)[:, 1]
    iso   = IsotonicRegression(out_of_bounds='clip')
    iso.fit(proba, (y > 0.5).astype(int))
    logger.info("Isotonic calibration fitted on decision tree probabilities.")
    return iso


def evaluate_labels(y_pred_binary, y_true, tag=''):
    from sklearn.metrics import classification_report, roc_auc_score
    try:
        auc = roc_auc_score(y_true, y_pred_binary)
        logger.info(f"\n=== Evaluation {tag} ===")
        logger.info(f"  AUC: {auc:.4f}")
        logger.info(classification_report(y_true, y_pred_binary, target_names=['no-shock', 'shock']))
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")


# ── Surrogate distill mode — ShockIndicator of AeroSurrogate ─────────────────

@torch.no_grad()
def extract_surrogate_shock_prob(X_raw, n_samples, device='cpu'):
    """
    Run the trained AeroSurrogate's ShockIndicator on X_raw and return
    shock_prob (continuous 0-1).  These soft labels are much better targets
    for PySR than binary 0/1 — PySR approximates a smooth function, not a step.
    """
    from src.models import AeroSurrogate
    from src.data_loader import CFDDataset
    from torch.utils.data import DataLoader

    ckpt = MODEL_DIR / 'surrogate_best.pt'
    if not ckpt.exists():
        logger.error(f"surrogate_best.pt not found in {MODEL_DIR}")
        sys.exit(1)

    scaler_path = MODEL_DIR / 'scaler.npy'
    if not scaler_path.exists():
        logger.error("scaler.npy not found — run main_train.py first")
        sys.exit(1)
    scaler = np.load(str(scaler_path), allow_pickle=True).item()

    cfg   = MODEL_CONFIG['surrogate']
    model = AeroSurrogate(
        in_dim=MODEL_CONFIG['autoencoder']['input_dim'],
        num_experts=cfg['num_experts'],
        output_dim=cfg['output_dim'],
        indicator_hidden=cfg.get('indicator_hidden'),
        expert_hidden=cfg.get('expert_hidden'),
    )
    # strict=False: checkpoints saved before mach_mean/mach_std buffers were added
    # still load correctly — those buffers are only used by the MoE gate during training,
    # not by the ShockIndicator whose output we need here.
    missing, unexpected = model.load_state_dict(torch.load(str(ckpt), map_location=device), strict=False)
    if missing:
        logger.info(f"Buffers not in checkpoint (using defaults): {missing}")
    model.eval()
    logger.info(f"Loaded AeroSurrogate from {ckpt}")

    # Dummy Y (zeros) — CFDDataset only needs Y for normalisation target, not used here
    Y_dummy = np.zeros((len(X_raw), 4), dtype=np.float32)
    ds      = CFDDataset(X_raw[:n_samples], Y_dummy[:n_samples], scaler=scaler)
    loader  = DataLoader(ds, batch_size=8192, shuffle=False, num_workers=0)

    probs = []
    for X_batch, _ in loader:
        out = model(X_batch.to(device))
        probs.append(out['shock_prob'].cpu().numpy().squeeze())

    shock_prob = np.concatenate(probs)
    logger.info(
        f"ShockIndicator output: mean={shock_prob.mean():.3f}  "
        f"std={shock_prob.std():.3f}  "
        f"p(>0.5)={( shock_prob > 0.5).mean()*100:.1f}%"
    )
    return shock_prob


# ── Distill mode (legacy — distills from trained neural sensor) ───────────────

def load_neural_models(device='cpu'):
    from src.models import ShockAutoencoder, MixtureOfExperts, VirtualShockSensor
    ae_path     = MODEL_DIR / 'autoencoder_best.pt'
    moe_path    = MODEL_DIR / 'moe_best.pt'
    sensor_path = MODEL_DIR / 'sensor_best.pt'

    missing = [str(p) for p in [ae_path, moe_path, sensor_path] if not p.exists()]
    if missing:
        logger.error(f"Missing checkpoints: {missing} — run main_train.py first")
        return None, None, None

    cfg = MODEL_CONFIG
    ae  = ShockAutoencoder(input_dim=cfg['autoencoder']['input_dim'],
                           latent_dim=cfg['autoencoder']['latent_dim'])
    ae.load_state_dict(torch.load(str(ae_path), map_location=device))
    ae.eval()

    moe = MixtureOfExperts(latent_dim=cfg['autoencoder']['latent_dim'],
                            num_experts=cfg['moe']['num_experts'],
                            expert_output_dim=cfg['moe']['expert_output_dim'],
                            output_dim=cfg['moe']['output_dim'])
    moe.load_state_dict(torch.load(str(moe_path), map_location=device))
    moe.eval()

    sensor = VirtualShockSensor(ae.encoder, moe,
                                latent_dim=cfg['autoencoder']['latent_dim'],
                                head_hidden=cfg['sensor']['head_hidden'])
    sensor.load_state_dict(torch.load(str(sensor_path), map_location=device))
    sensor.eval()

    logger.info("Neural models loaded for distillation")
    return ae, moe, sensor


@torch.no_grad()
def extract_distill_data(sensor, X_raw, Y_raw, n_samples, device, target='shock'):
    """Run the neural sensor on data to get soft labels for distillation."""
    from src.data_loader import CFDDataset
    from torch.utils.data import DataLoader

    # We need a scaler — load from disk if available, else compute locally
    scaler_path = MODEL_DIR / 'scaler.npy'
    if scaler_path.exists():
        scaler = np.load(str(scaler_path), allow_pickle=True).item()
    else:
        logger.warning("scaler.npy not found — computing local scaler (may differ from training)")
        scaler = None

    ds     = CFDDataset(X_raw[:n_samples], Y_raw[:n_samples], scaler=scaler)
    loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)

    X_out, T_out = [], []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        out = sensor(X_batch, compute_moe=False)
        X_out.append(X_batch.cpu().numpy())
        if target == 'shock':
            T_out.append(out['shock_prob'].cpu().numpy().squeeze())
        else:
            T_out.append(out['intensity'].cpu().numpy().squeeze())

    X_all = np.vstack(X_out)
    T_all = np.concatenate(T_out)
    return X_all[:, SR_IDX], T_all


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',       choices=['physics', 'surrogate', 'distill'], default='physics',
                        help='physics: PySR on CFD labels directly. '
                             'surrogate: distill ShockIndicator of AeroSurrogate (recommended for clean formula). '
                             'distill: distill from legacy VirtualShockSensor.')
    parser.add_argument('--fallback',   action='store_true',
                        help='Use Decision Tree instead of PySR (fast baseline).')
    parser.add_argument('--knn-labels', action='store_true',
                        help='In physics mode: refine shock labels with K-NN gradient filter.')
    parser.add_argument('--samples',    type=int, default=50000)
    parser.add_argument('--iterations', type=int, default=50,
                        help='PySR iterations (ignored for Decision Tree).')
    parser.add_argument('--margin',     type=float, default=0.0,
                        help='Cp margin below Cp_crit for shock label (default 0 = strict).')
    parser.add_argument('--knn-k',      type=int, default=10,
                        help='K neighbours for K-NN gradient filter.')
    parser.add_argument('--knn-percentile', type=int, default=30,
                        help='Gradient percentile threshold (default 30 = keep top 70%% of shock gradients).')
    parser.add_argument('--target',     choices=['shock', 'intensity'], default='shock',
                        help='Target variable (intensity only in distill mode).')
    parser.add_argument('--device',     default='cpu')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info(f"SYMBOLIC SHOCK SENSOR  |  mode={args.mode}  |  fallback={args.fallback}")
    logger.info("=" * 70)

    X_raw, Y_raw = load_raw_data(args.samples)

    # ── Build features and labels ────────────────────────────────────────────
    is_soft_target = False   # True when target is a continuous probability (not binary)

    if args.mode == 'surrogate':
        logger.info("\n[Surrogate distill] Extracting shock_prob from trained AeroSurrogate ShockIndicator")
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        y_label = extract_surrogate_shock_prob(X_raw, args.samples, device=device)
        X_sr    = extract_sr_features(X_raw)
        is_soft_target = True

    elif args.mode == 'physics':
        logger.info("\n[Physics mode] Computing labels directly from CFD ground truth")

        if args.knn_labels:
            logger.info("[K-NN] Computing K-NN gradient filter for improved labels...")
            y_label, grad_mag = make_knn_labels(  # noqa: F821 (defined above)
                X_raw, Y_raw, k=args.knn_k,
                gradient_percentile=args.knn_percentile, margin=args.margin
            )
        else:
            shock, _, intensity = make_physics_labels(X_raw, Y_raw, margin=args.margin)
            y_label = intensity if args.target == 'intensity' else shock
            if args.target == 'intensity':
                logger.info(f"Target: intensity (smooth)  mean={y_label.mean():.4f}  max={y_label.max():.4f}")

        X_sr = extract_sr_features(X_raw)

    else:  # distill
        logger.info("\n[Distill mode] Extracting soft labels from trained neural sensor")
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        _, _, sensor = load_neural_models(str(device))
        if sensor is None:
            sys.exit(1)
        sensor = sensor.to(device)
        X_sr, y_label = extract_distill_data(sensor, X_raw, Y_raw,
                                              args.samples, device, args.target)

    # ── Feature importance ───────────────────────────────────────────────────
    importances = feature_importance(X_sr, y_label, SR_FEATURES)

    # ── Symbolic regression ──────────────────────────────────────────────────
    eq_str  = ""
    cal_obj = None
    clf     = None

    if args.fallback:
        logger.info("\n[Decision Tree] Training symbolic classifier...")
        clf, rules = run_decision_tree(X_sr, y_label, SR_FEATURES, max_depth=5)

        # Evaluate before calibration
        y_pred_raw = clf.predict(X_sr)
        evaluate_labels(y_pred_raw, (y_label > 0.5).astype(int), tag='Decision Tree (uncalibrated)')

        # Calibrate
        cal_obj    = calibrate_decision_tree(clf, X_sr, y_label)
        y_prob_cal = cal_obj.predict(clf.predict_proba(X_sr)[:, 1])
        logger.info(f"Calibrated probability range: [{y_prob_cal.min():.3f}, {y_prob_cal.max():.3f}]")

        logger.info(f"\n{'='*60}\nDecision Tree rules\n{'='*60}\n{rules}")
        eq_str = rules

    else:
        logger.info("\n[PySR] Running symbolic regression...")
        model = run_pysr(X_sr, y_label.astype(np.float32), SR_FEATURES,
                         n_iter=args.iterations, soft_target=is_soft_target)

        if model is not None:
            eqs = model.equations_.sort_values('score', ascending=False).head(5)
            logger.info(f"\n{'='*60}\nTop PySR equations\n{'='*60}")
            for _, row in eqs.iterrows():
                logger.info(f"  complexity={row['complexity']}  score={row['score']:.4f}  {row['equation']}")
            eq_str = str(eqs[['equation', 'score', 'complexity']])

            # Calibrate PySR score with isotonic regression
            best_score = model.predict(X_sr)
            cal_obj = calibrate_with_isotonic(best_score, (y_label > 0.5).astype(int))
            y_prob_cal = cal_obj.predict(best_score)
            y_pred_cal = (y_prob_cal > 0.5).astype(int)
            evaluate_labels(y_pred_cal, (y_label > 0.5).astype(int), tag='PySR (calibrated)')

        else:
            logger.warning("PySR failed — falling back to Decision Tree")
            clf, rules = run_decision_tree(X_sr, y_label, SR_FEATURES)
            cal_obj    = calibrate_decision_tree(clf, X_sr, y_label)
            eq_str     = rules

    # ── Save results ─────────────────────────────────────────────────────────
    tag      = f"{args.mode}_{'knn' if (args.mode == 'physics' and args.knn_labels) else 'base'}"
    out_path = RESULT_DIR / f'symbolic_regression_{tag}.txt'

    with open(str(out_path), 'w') as f:
        f.write(f"SYMBOLIC SHOCK SENSOR\n")
        f.write(f"mode={args.mode}  knn_labels={args.knn_labels}  fallback={args.fallback}\n")
        f.write(f"samples={args.samples}  margin={args.margin}  SR_FEATURES={SR_FEATURES}\n")
        f.write("=" * 60 + "\n\n")
        f.write("FEATURE IMPORTANCES (RandomForest on physics labels):\n")
        for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
            f.write(f"  {name:12s}: {imp:.4f}\n")
        f.write("\nEQUATIONS / RULES:\n" + eq_str + "\n")

    logger.info(f"\nResults saved to {out_path}")

    # Save calibrated model for downstream use (e.g. surrogate gating)
    if cal_obj is not None:
        import pickle
        cal_path = MODEL_DIR / f'shock_sensor_symbolic_{tag}.pkl'
        with open(str(cal_path), 'wb') as f:
            pickle.dump({
                'clf':         clf,
                'calibrator':  cal_obj,
                'sr_features': SR_FEATURES,
                'sr_idx':      SR_IDX,
            }, f)
        logger.info(f"Calibrated sensor saved to {cal_path}")
        logger.info("Usage: obj['clf'].predict_proba(X[:,SR_IDX])[:,1] → obj['calibrator'].predict(...)")


if __name__ == '__main__':
    main()
