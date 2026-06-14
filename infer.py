#!/usr/bin/env python3
"""
Inference: load trained AE + MoE + Sensor and run analysis.

Usage:
    python infer.py --analyze
    python infer.py --analyze --samples 500 --device cuda
"""
import torch
import numpy as np
import logging
from pathlib import Path

from config import MODEL_DIR, DEVICE, MODEL_CONFIG
from src.models import ShockAutoencoder, MixtureOfExperts, VirtualShockSensor
from src.data_loader import get_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trained_models(device='cpu'):
    ae_path     = MODEL_DIR / 'autoencoder_best.pt'
    moe_path    = MODEL_DIR / 'moe_best.pt'
    sensor_path = MODEL_DIR / 'sensor_best.pt'

    cfg        = MODEL_CONFIG
    latent_dim = cfg['autoencoder']['latent_dim']
    input_dim  = cfg['autoencoder']['input_dim']

    if not ae_path.exists():
        logger.error(f"AE not found: {ae_path} — run main_train.py first")
        return None, None, None

    ae = ShockAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.eval()
    logger.info(f"AE loaded")

    moe = None
    if moe_path.exists():
        moe = MixtureOfExperts(
            latent_dim=latent_dim,
            num_experts=cfg['moe']['num_experts'],
            expert_output_dim=cfg['moe']['expert_output_dim'],
            output_dim=cfg['moe']['output_dim'],
        )
        moe.load_state_dict(torch.load(moe_path, map_location=device))
        moe.eval()
        logger.info(f"MoE loaded")

    sensor = None
    if sensor_path.exists() and moe is not None:
        sensor = VirtualShockSensor(ae.encoder, moe, latent_dim=latent_dim,
                                    head_hidden=cfg['sensor']['head_hidden'])
        sensor.load_state_dict(torch.load(sensor_path, map_location=device))
        sensor.eval()
        logger.info(f"Sensor loaded")

    return ae, moe, sensor


@torch.no_grad()
def infer_batch(ae, sensor, x_batch, device='cpu'):
    x_batch = x_batch.to(device)
    x_recon, z = ae(x_batch)
    result = {
        'x_reconstructed':     x_recon.cpu().numpy(),
        'latent_code':         z.cpu().numpy(),
        'reconstruction_error': ((x_batch - x_recon).pow(2).mean(dim=1).cpu().numpy()),
    }
    if sensor is not None:
        out = sensor(x_batch, compute_moe=False)
        result['shock_prob']     = out['shock_prob'].cpu().numpy().squeeze()
        result['intensity']      = out['intensity'].cpu().numpy().squeeze()
        result['separation_prob'] = out['separation_prob'].cpu().numpy().squeeze()
    return result


def analyze_results(test_loader, num_samples=1000, device='cpu'):
    ae, moe, sensor = load_trained_models(device=device)
    if ae is None:
        return
    ae = ae.to(device)
    if sensor:
        sensor = sensor.to(device)

    latent_codes, recon_errors, shock_probs = [], [], []
    collected = 0
    for X_batch, _ in test_loader:
        pred = infer_batch(ae, sensor, X_batch, device=device)
        latent_codes.append(pred['latent_code'])
        recon_errors.append(pred['reconstruction_error'])
        if 'shock_prob' in pred:
            shock_probs.append(pred['shock_prob'])
        collected += len(pred['reconstruction_error'])
        if collected >= num_samples:
            break

    latent_codes = np.vstack(latent_codes)
    recon_errors = np.concatenate(recon_errors)

    logger.info("\n" + "=" * 60)
    logger.info("INFERENCE ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Samples: {len(recon_errors):,}")
    logger.info(f"Latent [{latent_codes.shape[1]}-dim]  mean(first 5): {latent_codes.mean(0)[:5]}")
    logger.info(f"Reconstruction error  mean={recon_errors.mean():.6f}  p95={np.percentile(recon_errors,95):.6f}")
    if shock_probs:
        sp = np.concatenate(shock_probs)
        logger.info(f"Shock probability  mean={sp.mean():.4f}  shock_ratio={(sp>0.5).mean()*100:.1f}%")
    logger.info("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--device',  default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.analyze:
        _, _, test_loader, _ = get_dataloaders(sample_fraction=0.01)
        analyze_results(test_loader, num_samples=args.samples, device=str(device))
    else:
        logger.info("Usage: python infer.py --analyze [--samples N] [--device cuda]")
