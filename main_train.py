#!/usr/bin/env python3
"""
Training pipeline.

Stages:
  surrogate   — Shock-Gated Surrogate (ShockIndicator + MoE, no AE needed)
  ae          — Autoencoder
  moe         — Mixture of Experts (requires ae checkpoint)
  sensor      — Virtual Shock Sensor (requires ae + moe checkpoints)

Usage:
    python main_train.py --stages surrogate          # main model (recommended)
    python main_train.py --stages ae moe sensor      # legacy AE pipeline
    python main_train.py --stages ae                 # only autoencoder
    python main_train.py --stages moe                # MoE only (loads saved AE)
    python main_train.py --stages sensor             # Sensor only (loads saved AE + MoE)

Env vars:
    PAPER_TRAIN_FRACTION   fraction of train data to use  (default 0.05)
    PAPER_EPOCHS           epochs per stage               (default 20)
    PAPER_BATCH_SIZE       batch size                     (default 256)
    PAPER_NUM_WORKERS      dataloader workers             (default 0)
"""
import argparse
import logging
import sys

import torch
import numpy as np

from config import SEED, DEVICE, DATA_CONFIG, OUTPUT_DIR, LOGGING_CONFIG, MODEL_DIR, MODEL_CONFIG
from src.data_loader import get_dataloaders
from src.models import ShockAutoencoder, MixtureOfExperts
from src.training import AETrainer, MOETrainer, SensorTrainer, SurrogateTrainer
from src.evaluation import ModelEvaluator, VisualizationTools, save_evaluation_report

logging.basicConfig(
    level=LOGGING_CONFIG['log_level'],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=(
        [logging.StreamHandler(), logging.FileHandler(LOGGING_CONFIG['log_file'])]
        if LOGGING_CONFIG['save_log']
        else [logging.StreamHandler()]
    ),
)
logger = logging.getLogger(__name__)

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')


def load_ae(device):
    cfg = MODEL_CONFIG['autoencoder']
    ae  = ShockAutoencoder(input_dim=cfg['input_dim'], latent_dim=cfg['latent_dim'],
                           batch_norm=cfg['batch_norm'], dropout=cfg['dropout']).to(device)
    path = MODEL_DIR / 'autoencoder_best.pt'
    if not path.exists():
        logger.error(f"AE checkpoint not found: {path} — run with --stages ae first")
        sys.exit(1)
    ae.load_state_dict(torch.load(path, map_location=device))
    ae.eval()
    logger.info(f"Loaded AE from {path}")
    return ae


def load_moe(device):
    cfg = MODEL_CONFIG
    moe = MixtureOfExperts(
        latent_dim=cfg['autoencoder']['latent_dim'],
        num_experts=cfg['moe']['num_experts'],
        expert_output_dim=cfg['moe']['expert_output_dim'],
        output_dim=cfg['moe']['output_dim'],
    ).to(device)
    path = MODEL_DIR / 'moe_best.pt'
    if not path.exists():
        logger.error(f"MoE checkpoint not found: {path} — run with --stages moe first")
        sys.exit(1)
    moe.load_state_dict(torch.load(path, map_location=device))
    moe.eval()
    logger.info(f"Loaded MoE from {path}")
    return moe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stages', nargs='+',
        choices=['ae', 'moe', 'sensor', 'surrogate'],
        default=['ae', 'moe', 'sensor'],
        help='Stages to run (default: ae moe sensor). Use surrogate for the new shock-gated model.'
    )
    parser.add_argument(
        '--symbolic-gate', default=None, metavar='PKL',
        help='Path to symbolic sensor pkl. When set with --stages surrogate, '
             'freezes ShockIndicator and trains MoE with symbolic gates.'
    )
    args = parser.parse_args()
    stages = set(args.stages)

    device = torch.device('cuda' if torch.cuda.is_available() and DEVICE == 'cuda' else 'cpu')

    logger.info("=" * 70)
    logger.info(f"SHOCK DETECTION PIPELINE  |  stages: {sorted(stages)}  |  device: {device}")
    logger.info("=" * 70)

    frac = DATA_CONFIG.get('train_sample_fraction', 0.05)
    logger.info(f"\n[DATA] Loading {100*frac:.1f}% of training data...")
    train_loader, val_loader, test_loader, scaler = get_dataloaders(sample_fraction=frac)
    logger.info(f"Batches — train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")

    # Save scaler so inference scripts use the same normalization as training
    np.save(str(MODEL_DIR / 'scaler.npy'), scaler)
    logger.info(f"Scaler saved to {MODEL_DIR / 'scaler.npy'}")

    ae_model  = None
    moe_model = None

    # ── Autoencoder ────────────────────────────────────────────────────────────
    if 'ae' in stages:
        logger.info("\n[AE] Training Autoencoder  (16 -> 128 -> 64 -> 32 -> 64 -> 128 -> 16)")
        ae_trainer = AETrainer(device=device)
        ae_model   = ae_trainer.train(train_loader, val_loader)
        logger.info(f"AE best val loss: {ae_trainer.best_val_loss:.6f}")

        try:
            ae_eval_obj = ModelEvaluator(ae_model, device=device, is_autoencoder=True)
            ae_eval     = ae_eval_obj.evaluate(test_loader, return_predictions=True)
            ae_eval_obj.log_metrics(ae_eval['metrics'])
            save_evaluation_report(ae_eval['metrics'], ae_eval, model_name='autoencoder')
            viz = VisualizationTools()
            viz.plot_losses(ae_trainer.loss_history['train'], ae_trainer.loss_history['val'],
                            save_path=OUTPUT_DIR / 'plots' / 'ae_losses.png')
            if 'y_true' in ae_eval and 'y_pred' in ae_eval:
                viz.plot_reconstruction_error(ae_eval['y_true'], ae_eval['y_pred'],
                                              save_path=OUTPUT_DIR / 'plots' / 'reconstruction_error.png')
            if 'z' in ae_eval:
                n = min(100_000, len(ae_eval['z']))
                idx = np.random.default_rng(42).choice(len(ae_eval['z']), n, replace=False)
                viz.plot_latent_space(ae_eval['z'][idx], X_raw=ae_eval['y_true'][idx],
                                      save_path=OUTPUT_DIR / 'plots' / 'latent_space.png')
        except Exception as e:
            logger.warning(f"AE eval/viz failed (non-critical): {e}")
    elif 'moe' in stages or 'sensor' in stages:
        ae_model = load_ae(device)

    # ── Mixture of Experts ─────────────────────────────────────────────────────
    if 'moe' in stages:
        logger.info("\n[MoE] Training Mixture of Experts  (4 experts, latent=32 -> output=4)")
        moe_trainer = MOETrainer(encoder=ae_model.encoder, device=device)
        moe_model   = moe_trainer.train(train_loader, val_loader)
        logger.info(f"MoE best val loss: {moe_trainer.best_val_loss:.6f}")

        try:
            viz = VisualizationTools()
            viz.plot_losses(moe_trainer.loss_history['train'], moe_trainer.loss_history['val'],
                            save_path=OUTPUT_DIR / 'plots' / 'moe_losses.png')
        except Exception as e:
            logger.warning(f"MoE viz failed (non-critical): {e}")
    elif 'sensor' in stages:
        moe_model = load_moe(device)

    # ── Shock Sensor ───────────────────────────────────────────────────────────
    if 'sensor' in stages:
        logger.info("\n[Sensor] Training Virtual Shock Sensor  (CFD ground-truth labels: Cp<Cp_crit, Cfx<0)")
        sensor_trainer = SensorTrainer(
            encoder=ae_model.encoder,
            moe=moe_model,
            scaler=scaler,
            device=device,
        )
        sensor_trainer.train(train_loader, val_loader)
        logger.info(f"Sensor best val loss: {sensor_trainer.best_val_loss:.6f}")

        try:
            viz = VisualizationTools()
            viz.plot_losses(sensor_trainer.loss_history['train'], sensor_trainer.loss_history['val'],
                            save_path=OUTPUT_DIR / 'plots' / 'sensor_losses.png')
        except Exception as e:
            logger.warning(f"Sensor viz failed (non-critical): {e}")

    # ── Shock-Gated Surrogate ──────────────────────────────────────────────────
    if 'surrogate' in stages:
        symbolic_gate = args.symbolic_gate
        if symbolic_gate:
            logger.info(f"\n[Surrogate] Training MoE with symbolic gate: {symbolic_gate}")
            model_name = 'surrogate_symbolic'
        else:
            logger.info("\n[Surrogate] Training Shock-Gated Surrogate  (ShockIndicator + MoE, no AE)")
            model_name = 'surrogate'

        surrogate_trainer = SurrogateTrainer(
            scaler=scaler, device=device,
            symbolic_sensor_path=symbolic_gate,
            save_name=f'{model_name}_best.pt',
        )
        surrogate_trainer.train(train_loader, val_loader)
        logger.info(f"Surrogate best val loss: {surrogate_trainer.best_val_loss:.6f}")

        try:
            viz = VisualizationTools()
            viz.plot_losses(surrogate_trainer.loss_history['train_mse'],
                            surrogate_trainer.loss_history['val'],
                            save_path=OUTPUT_DIR / 'plots' / f'{model_name}_losses.png')
            surr_eval_obj = ModelEvaluator(surrogate_trainer.model, device=device, is_autoencoder=False)
            surr_eval     = surr_eval_obj.evaluate(test_loader, return_predictions=True)
            surr_eval_obj.log_metrics(surr_eval['metrics'])
            save_evaluation_report(surr_eval['metrics'], surr_eval, model_name=model_name)
            if 'y_true' in surr_eval and 'y_pred' in surr_eval:
                viz.plot_predictions_vs_truth(surr_eval['y_true'], surr_eval['y_pred'],
                                              save_path=OUTPUT_DIR / 'plots' / f'{model_name}_predictions.png')
        except Exception as e:
            logger.warning(f"Surrogate eval/viz failed (non-critical): {e}")

    logger.info("\n" + "=" * 70)
    logger.info(f"DONE  |  Results in: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
