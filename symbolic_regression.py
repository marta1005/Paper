#!/usr/bin/env python3
"""
Symbolic regression on learned shock sensor outputs.

Usage:
    python symbolic_regression.py                   # PySR
    python symbolic_regression.py --fallback        # Decision Tree (fast)
    python symbolic_regression.py --samples 30000
    python symbolic_regression.py --target intensity
"""
import argparse
import logging
import sys
import numpy as np
import torch

from config import MODEL_DIR, RESULT_DIR, MODEL_CONFIG
from src.models import ShockAutoencoder, MixtureOfExperts, VirtualShockSensor
from src.data_loader import get_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All 14 input features (9 original + 5 derived from X only)
ALL_FEATURES = [
    'x', 'y', 'z', 'nx', 'ny', 'nz', 'Mach', 'AoA', 'Pi_1e5',
    'q_dyn', 'Pi_norm', 'AoA_sin', 'L_factor', 'Cp_crit',
]

# Physics-relevant subset for symbolic regression (exclude pure geometric coords)
SR_FEATURES = ['Mach', 'AoA', 'Pi_1e5', 'q_dyn', 'Pi_norm', 'AoA_sin', 'L_factor', 'Cp_crit']
SR_IDX      = [ALL_FEATURES.index(n) for n in SR_FEATURES]


def load_models(device='cpu'):
    ae_path     = MODEL_DIR / 'autoencoder_best.pt'
    moe_path    = MODEL_DIR / 'moe_best.pt'
    sensor_path = MODEL_DIR / 'sensor_best.pt'

    missing = [p for p in [ae_path, moe_path, sensor_path] if not p.exists()]
    if missing:
        logger.error(f"Missing: {missing} — run main_train.py first")
        return None, None, None

    cfg = MODEL_CONFIG
    ae  = ShockAutoencoder(input_dim=cfg['autoencoder']['input_dim'],
                           latent_dim=cfg['autoencoder']['latent_dim'])
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.eval()

    moe = MixtureOfExperts(latent_dim=cfg['autoencoder']['latent_dim'],
                            num_experts=cfg['moe']['num_experts'],
                            expert_output_dim=cfg['moe']['expert_output_dim'],
                            output_dim=cfg['moe']['output_dim'])
    moe.load_state_dict(torch.load(moe_path, map_location=device))
    moe.eval()

    sensor = VirtualShockSensor(ae.encoder, moe,
                                latent_dim=cfg['autoencoder']['latent_dim'],
                                head_hidden=cfg['sensor']['head_hidden'])
    sensor.load_state_dict(torch.load(sensor_path, map_location=device))
    sensor.eval()

    logger.info("Models loaded")
    return ae, moe, sensor


@torch.no_grad()
def extract_data(sensor, test_loader, n_samples, device, target='shock'):
    X_list, T_list = [], []
    collected = 0
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        out = sensor(X_batch, compute_moe=False)
        X_list.append(X_batch.cpu().numpy())
        if target == 'shock':
            T_list.append(out['shock_prob'].cpu().numpy().squeeze())
        elif target == 'intensity':
            T_list.append(out['intensity'].cpu().numpy().squeeze())
        else:
            T_list.append(out['latent'].cpu().numpy())
        collected += len(X_batch)
        if collected >= n_samples:
            break

    X_all = np.vstack(X_list)[:n_samples]
    T_all = (np.concatenate(T_list) if target != 'latent' else np.vstack(T_list))[:n_samples]

    X_phys = X_all[:, SR_IDX]
    logger.info(f"Extracted {len(X_phys):,} samples  X_phys={X_phys.shape}")
    return X_phys, T_all


def feature_importance(X, y, feature_names):
    from sklearn.ensemble import RandomForestClassifier
    y_bin = (y > 0.5).astype(int)
    rf    = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)
    rf.fit(X, y_bin)
    imp   = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    logger.info("\n=== Feature importances ===")
    for i in order:
        bar = '█' * int(imp[i] * 50)
        logger.info(f"  {feature_names[i]:12s}: {imp[i]:.4f}  {bar}")
    return dict(zip(feature_names, imp))


def run_pysr(X, y, feature_names, n_iter=40):
    try:
        from pysr import PySRRegressor
    except ImportError:
        logger.error("PySR not installed: pip install pysr")
        return None
    model = PySRRegressor(
        niterations=n_iter,
        binary_operators=['+', '-', '*', '/', '**'],
        unary_operators=['exp', 'log', 'abs', 'sqrt', 'tanh'],
        maxsize=20, populations=15, population_size=33,
        model_selection='best', parsimony=0.003, random_state=42, verbosity=1,
    )
    model.fit(X, y, variable_names=feature_names)
    return model


def run_decision_tree(X, y, feature_names, max_depth=4):
    from sklearn.tree import DecisionTreeClassifier, export_text
    y_bin = (y > 0.5).astype(int)
    clf   = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, y_bin)
    rules = export_text(clf, feature_names=feature_names)
    return clf, rules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fallback',   action='store_true')
    parser.add_argument('--samples',    type=int, default=20000)
    parser.add_argument('--target',     choices=['shock', 'intensity', 'latent'], default='shock')
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--device',     default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ae, moe, sensor = load_models(str(device))
    if sensor is None:
        sys.exit(1)
    sensor = sensor.to(device)

    scaler_path = MODEL_DIR / 'scaler.npy'
    saved_scaler = np.load(str(scaler_path), allow_pickle=True).item() if scaler_path.exists() else None
    if saved_scaler is None:
        logger.warning("scaler.npy not found — using locally recomputed scaler (may differ from training)")
    _, _, test_loader, _ = get_dataloaders(sample_fraction=0.01, scaler=saved_scaler)
    X_phys, y_target = extract_data(sensor, test_loader, args.samples, device, args.target)

    if args.target == 'latent':
        from sklearn.decomposition import PCA
        pca      = PCA(n_components=1)
        y_target = pca.fit_transform(y_target).squeeze()
        logger.info(f"Latent PC1 variance explained: {pca.explained_variance_ratio_[0]:.3f}")

    importances = feature_importance(X_phys, y_target, SR_FEATURES)

    eq_str = ""
    if args.fallback:
        _, rules = run_decision_tree(X_phys, y_target, SR_FEATURES)
        print(f"\n{'='*60}\nDecision Tree rules for: {args.target}\n{'='*60}\n{rules}")
        eq_str = rules
    else:
        model = run_pysr(X_phys, y_target, SR_FEATURES, n_iter=args.iterations)
        if model is not None:
            eqs = model.equations_.sort_values('score', ascending=False).head(5)
            print(f"\n{'='*60}\nPySR equations for: {args.target}\n{'='*60}")
            for _, row in eqs.iterrows():
                print(f"  complexity={row['complexity']}  score={row['score']:.4f}  {row['equation']}")
            eq_str = str(eqs[['equation', 'score', 'complexity']])
        else:
            _, rules = run_decision_tree(X_phys, y_target, SR_FEATURES)
            print(rules)
            eq_str = rules

    out_path = RESULT_DIR / f'symbolic_regression_{args.target}.txt'
    with open(out_path, 'w') as f:
        f.write(f"SYMBOLIC REGRESSION — target: {args.target}\n{'='*60}\n\n")
        f.write("FEATURE IMPORTANCES:\n")
        for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
            f.write(f"  {name:12s}: {imp:.4f}\n")
        f.write("\nEQUATIONS:\n" + eq_str)
    logger.info(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
