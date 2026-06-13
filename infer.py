#!/usr/bin/env python3
"""
Script de inferencia: usar modelo entrenado para predicciones
"""
import torch
import numpy as np
import logging
from pathlib import Path

from config import MODEL_DIR, DEVICE
from src.models import ShockAutoencoder, MixtureOfExperts, VirtualShockSensor
from src.data_loader import get_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trained_models(device='cpu'):
    """Carga modelos entrenados"""
    
    ae_path = MODEL_DIR / 'autoencoder_best.pt'
    
    if not ae_path.exists():
        logger.error(f"Model not found at {ae_path}")
        logger.error("Run main_train.py first to train the model")
        return None, None
    
    # Cargar autoencoder
    ae = ShockAutoencoder(input_dim=19, latent_dim=32)
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.eval()
    
    logger.info(f"✓ Loaded autoencoder from {ae_path}")
    
    return ae


def infer_batch(ae, x_batch, device='cpu'):
    """
    Realiza inferencia en un batch
    
    Args:
        ae: modelo entrenado
        x_batch: batch de datos [batch_size, 19]
        device: dispositivo (cpu/cuda)
    
    Returns:
        predictions: dict con predicciones
    """
    ae = ae.to(device)
    x_batch = x_batch.to(device)
    
    with torch.no_grad():
        x_recon, z = ae(x_batch)
    
    return {
        'x_reconstructed': x_recon.cpu().numpy(),
        'latent_code': z.cpu().numpy(),
        'reconstruction_error': ((x_batch.cpu().numpy() - x_recon.cpu().numpy()) ** 2).mean(axis=1),
    }


def analyze_results(test_loader, num_samples=100):
    """Análisis rápido de resultados"""
    
    logger.info("\n" + "="*80)
    logger.info("QUICK ANALYSIS")
    logger.info("="*80)
    
    ae = load_trained_models(device=DEVICE)
    if ae is None:
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() and DEVICE == 'cuda' else 'cpu')
    ae = ae.to(device)
    
    # Procesar batches
    latent_codes = []
    reconstruction_errors = []
    
    for i, (X_batch, Y_batch) in enumerate(test_loader):
        if i >= num_samples // len(X_batch):  # Limitar a num_samples
            break
        
        pred = infer_batch(ae, X_batch, device=device)
        latent_codes.append(pred['latent_code'])
        reconstruction_errors.append(pred['reconstruction_error'])
    
    # Analizar
    latent_codes = np.vstack(latent_codes)
    reconstruction_errors = np.concatenate(reconstruction_errors)
    
    logger.info(f"\nLatent space analysis (first {len(latent_codes)} samples):")
    logger.info(f"  Shape: {latent_codes.shape}")
    logger.info(f"  Mean: {latent_codes.mean(axis=0)[:5]}... (mostrando primeras 5 dims)")
    logger.info(f"  Std:  {latent_codes.std(axis=0)[:5]}...")
    
    logger.info(f"\nReconstruction error:")
    logger.info(f"  Mean: {reconstruction_errors.mean():.6f}")
    logger.info(f"  Std:  {reconstruction_errors.std():.6f}")
    logger.info(f"  Min:  {reconstruction_errors.min():.6f}")
    logger.info(f"  Max:  {reconstruction_errors.max():.6f}")
    logger.info(f"  95%:  {np.percentile(reconstruction_errors, 95):.6f}")
    
    logger.info("="*80 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--analyze', action='store_true', help='Run quick analysis')
    parser.add_argument('--samples', type=int, default=100, help='Num samples to analyze')
    
    args = parser.parse_args()
    
    if args.analyze:
        _, _, test_loader, _ = get_dataloaders(sample_fraction=0.01)
        analyze_results(test_loader, num_samples=args.samples)
    else:
        logger.info("Usage: python infer.py --analyze [--samples N]")
