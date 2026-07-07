#!/usr/bin/env python3
"""
Fine-tune ShockIndicator to minimize Cp prediction error.

Freezes MoE gate + experts; only ShockIndicator weights are updated.
Gradient path: MSE(Cp_pred, Y_true) → gate[frozen] → shock_prob → ShockIndicator.

After this, run PySR on the new shock_prob to get a better symbolic sensor:
    python symbolic_regression.py --mode surrogate --ckpt outputs/models/surrogate_ft.pt --samples 200000 --iterations 100

Usage:
    python finetune_sensor.py --epochs 25 --lr 1e-4
"""
import os; os.environ['PAPER_NUM_WORKERS'] = '0'
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from config import MODEL_DIR, MODEL_CONFIG, DEVICE
from src.models import AeroSurrogate
from src.data_loader import get_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-in',  default=str(MODEL_DIR / 'surrogate_best.pt'))
    parser.add_argument('--ckpt-out', default=str(MODEL_DIR / 'surrogate_ft.pt'))
    parser.add_argument('--epochs',   type=int,   default=25)
    parser.add_argument('--lr',       type=float, default=1e-4)
    parser.add_argument('--fraction', type=float, default=0.15,
                        help='Fraction of training data to use (default 0.15)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and DEVICE == 'cuda' else 'cpu')
    logger.info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    cfg = MODEL_CONFIG['surrogate']
    model = AeroSurrogate(
        in_dim=MODEL_CONFIG['autoencoder']['input_dim'],
        num_experts=cfg['num_experts'],
        output_dim=cfg['output_dim'],
        indicator_hidden=cfg.get('indicator_hidden'),
        expert_hidden=cfg.get('expert_hidden'),
    )
    missing, _ = model.load_state_dict(
        torch.load(args.ckpt_in, map_location=device), strict=False
    )
    if missing:
        logger.info(f"Buffers not in checkpoint (using defaults): {missing}")
    model = model.to(device)

    # ── Freeze everything except ShockIndicator ───────────────────────────────
    freeze(model.moe.gate)
    freeze(model.moe.experts)
    for p in model.shock_indicator.parameters():
        p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {n_trainable:,} / {n_total:,} params  (ShockIndicator only)")

    # ── Data — use the saved scaler so normalization matches the trained model ──
    scaler_path = MODEL_DIR / 'scaler.npy'
    if not scaler_path.exists():
        raise FileNotFoundError(f"scaler.npy not found in {MODEL_DIR}")
    saved_scaler = np.load(str(scaler_path), allow_pickle=True).item()
    logger.info(f"Loaded scaler from {scaler_path}")

    logger.info(f"Loading {args.fraction*100:.0f}% of training data...")
    train_loader, val_loader, _, _ = get_dataloaders(
        sample_fraction=args.fraction, scaler=saved_scaler
    )

    # Keep gate Mach buffers consistent with the saved scaler
    model.moe.mach_mean.fill_(float(saved_scaler['X_mean'][6]))
    model.moe.mach_std.fill_(float(saved_scaler['X_std'][6]))

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val, best_state = float('inf'), None

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        # Keep frozen modules in eval mode so BN running stats don't shift
        model.moe.gate.eval()
        for expert in model.moe.experts:
            expert.eval()

        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            out  = model(X_batch)
            loss = criterion(out['pred'], Y_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.shock_indicator.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                out  = model(X_batch)
                val_loss += criterion(out['pred'], Y_batch).item()
        val_loss /= len(val_loader)

        scheduler.step()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.5f}  val={val_loss:.5f}  best={best_val:.5f}")

    # ── Save best ─────────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    torch.save(best_state, args.ckpt_out)
    logger.info(f"\nFine-tuned model saved → {args.ckpt_out}")
    logger.info(f"Best val MSE: {best_val:.5f}")
    logger.info("\nNext step:")
    logger.info(f"  python symbolic_regression.py --mode surrogate --ckpt {args.ckpt_out} --samples 200000 --iterations 100")


if __name__ == '__main__':
    main()
