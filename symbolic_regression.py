#!/usr/bin/env python3
"""
Regresión simbólica post-entrenamiento para extraer ecuaciones interpretables.

Usa PySR (si disponible) o Decision Tree como fallback.

Flujo:
  1. Carga AE + Sensor entrenados
  2. Extrae X_physics y shock_prob = Sensor(X) en N puntos del test set
  3. Corre PySR: busca f(X_physics) ≈ shock_prob
  4. Imprime las ecuaciones encontradas en formato LaTeX y Python
  5. Guarda resultados en outputs/results/

Uso:
  python symbolic_regression.py                   # PySR completo
  python symbolic_regression.py --fallback        # Decision Tree (rápido)
  python symbolic_regression.py --samples 30000   # ajustar N puntos
  python symbolic_regression.py --target latent   # ecuación para espacio latente
"""

import argparse
import logging
import sys
import numpy as np
import torch
from pathlib import Path

from config import MODEL_DIR, RESULT_DIR, DERIVED_FEATURE_INDICES, MODEL_CONFIG, TRAINING_CONFIG
from src.models import ShockAutoencoder, MixtureOfExperts, VirtualShockSensor
from src.data_loader import get_dataloaders

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Nombres de features disponibles para la regresión simbólica
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    # Originales (0-8)
    'Mach', 'AoA', 'Pi', 'x', 'y', 'z', 'nx', 'ny', 'nz',
    # Derivados (9-18)
    'M_local', 'grad_p', 'cp_loss', 'shock_ind', 'Cf_mag',
    'q_dyn', 'Pi_norm', 'AoA_norm', 'grad_cf', 'L_factor',
]

# Features más relevantes físicamente para un sensor de choque
# (subconjunto para SR — menos features = ecuaciones más simples)
PHYSICS_FEATURES = ['M_local', 'grad_p', 'cp_loss', 'shock_ind', 'Mach', 'AoA', 'Pi']
PHYSICS_IDX = [FEATURE_NAMES.index(n) for n in PHYSICS_FEATURES]


def load_models(device='cpu'):
    """Carga AE + MoE + Sensor entrenados."""
    ae_path     = MODEL_DIR / 'autoencoder_best.pt'
    moe_path    = MODEL_DIR / 'moe_best.pt'
    sensor_path = MODEL_DIR / 'sensor_best.pt'

    missing = [p for p in [ae_path, moe_path, sensor_path] if not p.exists()]
    if missing:
        logger.error(f"Modelos no encontrados: {missing}")
        logger.error("Ejecuta primero: python main_train.py")
        return None, None, None

    ae = ShockAutoencoder(input_dim=19, latent_dim=MODEL_CONFIG['autoencoder']['latent_dim'])
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.eval()

    moe = MixtureOfExperts(
        latent_dim=MODEL_CONFIG['autoencoder']['latent_dim'],
        num_experts=MODEL_CONFIG['moe']['num_experts'],
        expert_output_dim=MODEL_CONFIG['moe']['expert_output_dim'],
    )
    moe.load_state_dict(torch.load(moe_path, map_location=device))
    moe.eval()

    sensor = VirtualShockSensor(ae.encoder, moe, latent_dim=MODEL_CONFIG['autoencoder']['latent_dim'])
    sensor.load_state_dict(torch.load(sensor_path, map_location=device))
    sensor.eval()

    logger.info("✓ Modelos cargados: AE + MoE + Sensor")
    return ae, moe, sensor


@torch.no_grad()
def extract_features_and_targets(sensor, test_loader, n_samples, device, target='shock'):
    """
    Extrae X_physics y el target del Sensor para N puntos del test set.

    target:
        'shock'     → shock_prob (float 0-1)
        'intensity' → intensity de choque (float ≥0)
        'latent_0'  → primera dimensión del espacio latente
        'latent_pca'→ componente PCA 0 del espacio latente (más informativa)
    """
    X_list, T_list, Z_list = [], [], []

    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        out     = sensor(X_batch, compute_moe=False)

        X_list.append(X_batch.cpu().numpy())
        Z_list.append(out['latent'].cpu().numpy())

        if target == 'shock':
            T_list.append(out['shock_prob'].cpu().numpy().squeeze())
        elif target == 'intensity':
            T_list.append(out['intensity'].cpu().numpy().squeeze())
        else:
            T_list.append(out['latent'].cpu().numpy())  # guardar latente completo

        if sum(len(t) for t in T_list) >= n_samples:
            break

    X_all = np.vstack(X_list)[:n_samples]
    Z_all = np.vstack(Z_list)[:n_samples]

    if target in ('shock', 'intensity'):
        T_all = np.concatenate(T_list)[:n_samples]
    else:
        T_all = np.vstack(T_list)[:n_samples]

    # Seleccionar solo las features físicas relevantes
    X_phys = X_all[:, PHYSICS_IDX]

    logger.info(f"Extraídas {len(X_phys):,} muestras")
    logger.info(f"  X_physics shape: {X_phys.shape}  features: {PHYSICS_FEATURES}")

    if target in ('shock', 'intensity'):
        logger.info(f"  Target '{target}': mean={T_all.mean():.4f}, std={T_all.std():.4f}")
    else:
        logger.info(f"  Target latent shape: {T_all.shape}")

    return X_phys, T_all, Z_all


def run_pysr(X, y, feature_names, max_eqns=10, n_iter=40):
    """
    Corre PySR para encontrar f(X) ≈ y.
    Devuelve el dataframe de ecuaciones ordenado por complejidad/score.
    """
    try:
        from pysr import PySRRegressor
    except ImportError:
        logger.error("PySR no instalado. Ejecuta: pip install pysr")
        logger.error("También necesita Julia: https://julialang.org/downloads/")
        return None

    logger.info(f"Iniciando PySR ({n_iter} iteraciones, hasta {max_eqns} ecuaciones)...")

    model = PySRRegressor(
        niterations=n_iter,
        binary_operators=['+', '-', '*', '/', '**'],
        unary_operators=['exp', 'log', 'abs', 'sqrt', 'tanh'],
        maxsize=20,
        populations=15,
        population_size=33,
        model_selection='best',
        verbosity=1,
        random_state=42,
        # Penalizar complejidad para favorecer ecuaciones cortas
        parsimony=0.0032,
    )

    model.fit(X, y, variable_names=feature_names)

    return model


def run_fallback_tree(X, y, feature_names, max_depth=4):
    """
    Fallback: Decision Tree + extracción de reglas como 'pseudo-ecuación'.
    Rápido, interpretable, util para el paper como baseline.
    """
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.tree import export_text
    from sklearn.preprocessing import Binarizer

    logger.info(f"Ejecutando Decision Tree (depth={max_depth}) como fallback...")

    # Clasificación si y es binario, regresión si es continuo
    is_binary = np.unique(y).size <= 2 or (y.min() >= 0 and y.max() <= 1
                                           and np.unique(y).size > 10)

    if is_binary:
        y_bin = (y > 0.5).astype(int)
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        clf.fit(X, y_bin)
        rules = export_text(clf, feature_names=feature_names)
    else:
        clf = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        clf.fit(X, y)
        rules = export_text(clf, feature_names=feature_names)

    return clf, rules


def feature_importance_analysis(X, y, feature_names):
    """
    Análisis de importancia de features con Random Forest.
    Sirve como ranking previo a SR para seleccionar features relevantes.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import label_binarize

    logger.info("Calculando importancia de features (Random Forest)...")

    y_bin = (y > 0.5).astype(int)
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)
    rf.fit(X, y_bin)

    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]

    logger.info("\n=== Feature Importance para shock_prob ===")
    for i in order:
        bar = '█' * int(importances[i] * 50)
        logger.info(f"  {feature_names[i]:15s}: {importances[i]:.4f}  {bar}")

    return dict(zip(feature_names, importances))


def print_pysr_equations(model, target_name):
    """Imprime las ecuaciones PySR en formato LaTeX y Python."""
    print("\n" + "="*70)
    print(f"ECUACIONES PySR para: {target_name}")
    print("="*70)

    eqs = model.equations_
    print(f"\nTop ecuaciones (ordenadas por score):\n")
    for i, row in eqs.sort_values('score', ascending=False).head(5).iterrows():
        print(f"  Complejidad {row['complexity']:2d} | Score {row['score']:.4f}")
        print(f"  Python: {row['equation']}")
        try:
            print(f"  LaTeX:  {model.latex(index=i)}")
        except Exception:
            pass
        print()

    print("Mejor ecuación (balance complejidad/precision):")
    best = model.get_best()
    print(f"  {best['equation']}")
    try:
        print(f"  LaTeX: {model.latex()}")
    except Exception:
        pass
    print("="*70 + "\n")


def save_results(equations_str, importances, target, out_dir=RESULT_DIR):
    """Guarda el reporte en outputs/results/."""
    out_path = out_dir / f'symbolic_regression_{target}.txt'
    with open(out_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"SYMBOLIC REGRESSION RESULTS — target: {target}\n")
        f.write("="*70 + "\n\n")
        f.write("FEATURE IMPORTANCES:\n")
        for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
            f.write(f"  {name:15s}: {imp:.4f}\n")
        f.write("\nEQUATIONS:\n")
        f.write(equations_str)
    logger.info(f"✓ Resultados guardados en {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Regresión simbólica post-entrenamiento')
    parser.add_argument('--fallback', action='store_true',
                        help='Usar Decision Tree en lugar de PySR')
    parser.add_argument('--samples', type=int, default=20000,
                        help='Nº de puntos para la SR (default: 20000)')
    parser.add_argument('--target', choices=['shock', 'intensity', 'latent'],
                        default='shock', help='Qué predicción modelar simbólicamente')
    parser.add_argument('--iterations', type=int, default=40,
                        help='Iteraciones de PySR (default: 40, más = mejor)')
    parser.add_argument('--device', default='cpu', help='cuda o cpu')
    args = parser.parse_args()

    device = torch.device(args.device
                          if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # ── 1. Cargar modelos ────────────────────────────────────────────────────
    ae, moe, sensor = load_models(device=str(device))
    if sensor is None:
        sys.exit(1)

    # ── 2. Extraer features + targets ───────────────────────────────────────
    logger.info("Cargando datos de test (1% muestra para SR)...")
    _, _, test_loader, _ = get_dataloaders(sample_fraction=0.01)

    X_phys, y_target, Z_all = extract_features_and_targets(
        sensor, test_loader,
        n_samples=args.samples,
        device=device,
        target=args.target,
    )

    if args.target == 'latent':
        # Usar PCA para seleccionar la componente más informativa
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        y_target = pca.fit_transform(Z_all).squeeze()
        logger.info(f"Latent PCA-0 varianza explicada: {pca.explained_variance_ratio_[0]:.3f}")

    # ── 3. Importancia de features ───────────────────────────────────────────
    importances = feature_importance_analysis(X_phys, y_target, PHYSICS_FEATURES)

    # ── 4. Regresión simbólica ───────────────────────────────────────────────
    equations_str = ""

    if args.fallback:
        clf, rules = run_fallback_tree(X_phys, y_target,
                                       feature_names=PHYSICS_FEATURES)
        print("\n" + "="*70)
        print(f"REGLAS Decision Tree para: {args.target}")
        print("="*70)
        print(rules)
        equations_str = rules

    else:
        pysr_model = run_pysr(
            X_phys, y_target,
            feature_names=PHYSICS_FEATURES,
            n_iter=args.iterations,
        )
        if pysr_model is not None:
            print_pysr_equations(pysr_model, args.target)
            try:
                equations_str = str(pysr_model.equations_[['equation', 'score', 'complexity']])
            except Exception:
                equations_str = str(pysr_model.get_best())
        else:
            logger.info("Fallback automático a Decision Tree...")
            clf, rules = run_fallback_tree(X_phys, y_target,
                                           feature_names=PHYSICS_FEATURES)
            print(rules)
            equations_str = rules

    # ── 5. Guardar ───────────────────────────────────────────────────────────
    save_results(equations_str, importances, args.target)

    logger.info("\n✓ Regresión simbólica completada.")
    logger.info("  Para el paper: copia las ecuaciones de outputs/results/")


if __name__ == '__main__':
    main()
