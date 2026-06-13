#!/usr/bin/env python3
"""
Script de inferencia: carga AE + MoE + Sensor entrenados y genera predicciones.

Uso:
  python infer.py --analyze              # análisis rápido del test set
  python infer.py --analyze --samples 500
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
    """
    Carga AE + MoE + Sensor entrenados.
    Devuelve (ae, moe, sensor) — cualquiera puede ser None si no existe el checkpoint.
    """
    latent_dim   = MODEL_CONFIG['autoencoder']['latent_dim']
    num_experts  = MODEL_CONFIG['moe']['num_experts']
    expert_out   = MODEL_CONFIG['moe']['expert_output_dim']

    ae_path     = MODEL_DIR / 'autoencoder_best.pt'
    moe_path    = MODEL_DIR / 'moe_best.pt'
    sensor_path = MODEL_DIR / 'sensor_best.pt'

    ae = moe = sensor = None

    if ae_path.exists():
        ae = ShockAutoencoder(input_dim=19, latent_dim=latent_dim)
        ae.load_state_dict(torch.load(ae_path, map_location=device))
        ae.eval()
        logger.info(f"✓ AE cargado ({ae_path.name})")
    else:
        logger.warning(f"AE no encontrado: {ae_path}  →  ejecuta main_train.py")
        return None, None, None

    if moe_path.exists():
        moe = MixtureOfExperts(
            latent_dim=latent_dim,
            num_experts=num_experts,
            expert_output_dim=expert_out,
            output_dim=MODEL_CONFIG['moe']['output_dim'],
        )
        moe.load_state_dict(torch.load(moe_path, map_location=device))
        moe.eval()
        logger.info(f"✓ MoE cargado ({moe_path.name})")
    else:
        logger.warning(f"MoE no encontrado: {moe_path}")

    if sensor_path.exists() and moe is not None:
        sensor = VirtualShockSensor(ae.encoder, moe, latent_dim=latent_dim)
        sensor.load_state_dict(torch.load(sensor_path, map_location=device))
        sensor.eval()
        logger.info(f"✓ Sensor cargado ({sensor_path.name})")
    else:
        logger.warning(f"Sensor no encontrado o MoE faltante: {sensor_path}")

    return ae, moe, sensor


@torch.no_grad()
def infer_batch(ae, sensor, x_batch, device='cpu'):
    """
    Inferencia completa en un batch.

    Returns dict:
        x_reconstructed     [batch, 19]
        latent_code         [batch, 32]
        reconstruction_error [batch]
        shock_prob          [batch]  (solo si sensor disponible)
        intensity           [batch]  (solo si sensor disponible)
        separation_prob     [batch]  (solo si sensor disponible)
    """
    x_batch = x_batch.to(device)
    x_recon, z = ae(x_batch)

    result = {
        'x_reconstructed':      x_recon.cpu().numpy(),
        'latent_code':           z.cpu().numpy(),
        'reconstruction_error':  ((x_batch.cpu() - x_recon.cpu()).pow(2).mean(dim=1).numpy()),
    }

    if sensor is not None:
        out = sensor(x_batch, compute_moe=False)
        result['shock_prob']      = out['shock_prob'].cpu().numpy().squeeze()
        result['intensity']       = out['intensity'].cpu().numpy().squeeze()
        result['separation_prob'] = out['separation_prob'].cpu().numpy().squeeze()

    return result


def analyze_results(test_loader, num_samples=1000, device='cpu'):
    """Análisis rápido de resultados con los modelos entrenados."""
    ae, moe, sensor = load_trained_models(device=device)
    if ae is None:
        return

    ae = ae.to(device)
    if sensor:
        sensor = sensor.to(device)

    latent_codes, recon_errors, shock_probs = [], [], []

    for i, (X_batch, _) in enumerate(test_loader):
        pred = infer_batch(ae, sensor, X_batch, device=device)
        latent_codes.append(pred['latent_code'])
        recon_errors.append(pred['reconstruction_error'])
        if 'shock_prob' in pred:
            shock_probs.append(pred['shock_prob'])
        if sum(len(e) for e in recon_errors) >= num_samples:
            break

    latent_codes = np.vstack(latent_codes)
    recon_errors = np.concatenate(recon_errors)

    logger.info("\n" + "="*60)
    logger.info("INFERENCE ANALYSIS")
    logger.info("="*60)
    logger.info(f"Samples analyzed: {len(recon_errors):,}")
    logger.info(f"\nLatent space [{latent_codes.shape[1]}-dim]:")
    logger.info(f"  Mean (first 5):  {latent_codes.mean(axis=0)[:5]}")
    logger.info(f"  Std  (first 5):  {latent_codes.std(axis=0)[:5]}")
    logger.info(f"\nReconstruction error:")
    logger.info(f"  Mean: {recon_errors.mean():.6f}")
    logger.info(f"  Std:  {recon_errors.std():.6f}")
    logger.info(f"  95th: {np.percentile(recon_errors, 95):.6f}")

    if shock_probs:
        sp = np.concatenate(shock_probs)
        logger.info(f"\nShock probability:")
        logger.info(f"  Mean:        {sp.mean():.4f}")
        logger.info(f"  Shock ratio: {(sp > 0.5).mean()*100:.1f}%")

    logger.info("="*60 + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--analyze', action='store_true', help='Run quick analysis')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device
                          if torch.cuda.is_available() else 'cpu')

    if args.analyze:
        _, _, test_loader, _ = get_dataloaders(sample_fraction=0.01)
        analyze_results(test_loader, num_samples=args.samples, device=str(device))
    else:
        logger.info("Uso: python infer.py --analyze [--samples N] [--device cuda]")
