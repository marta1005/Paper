#!/usr/bin/env python3
"""
main_train_v2.py — Entry point for AeroSurrogate v2 training.

Single-GPU:
  python main_train_v2.py

Multi-GPU (DDP, 4× H100):
  torchrun --nproc_per_node=4 main_train_v2.py

Env vars:
  PAPER_EPOCHS=50         number of epochs
  PAPER_NUM_WORKERS=4     DataLoader workers
  V2_KNN_K=8              kNN k (must match build_knn_cache.py)
"""
import os
# Must be set before importing torch — prevents fragmentation OOM on large graphs
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
import logging
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

# Make sure imports resolve from this directory
sys.path.insert(0, str(Path(__file__).parent))

from config import (DATA_DIR, CACHE_DIR, MODEL_DIR, RESULT_DIR,
                    MODEL_CONFIG, TRAINING_CONFIG, PREPROCESSING_CONFIG,
                    LOGGING_CONFIG, SEED, KNN_K, N_P, N_TRAIN, N_TEST)
from src.preprocessing import CFDPreprocessor
from src.models_v2 import AeroSurrogatev2
from src.dataset import SimulationDataset, load_sim_weights, collate_single
from src.training import Surrogatev2Trainer


# ── reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── DDP setup ────────────────────────────────────────────────────────────────

def init_ddp():
    if 'RANK' not in os.environ:
        return 0, 1, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dist.init_process_group('nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    return rank, world_size, device


def cleanup_ddp(world_size):
    if world_size > 1:
        dist.destroy_process_group()


# ── scaler / preprocessor ─────────────────────────────────────────────────────

def fit_or_load_scaler(preprocessor, is_main):
    scaler_path = MODEL_DIR / 'scaler_v2.npy'
    spatial_path = MODEL_DIR / 'spatial_stats_v2.npy'

    if scaler_path.exists() and spatial_path.exists():
        scaler       = np.load(str(scaler_path),  allow_pickle=True).item()
        spatial_stats = np.load(str(spatial_path), allow_pickle=True).item()
        preprocessor.spatial_stats = spatial_stats
        logging.getLogger(__name__).info("Loaded existing scaler from disk.")
        return scaler

    if not is_main:
        # Wait for main process to build the scaler
        dist.barrier()
        scaler        = np.load(str(scaler_path),  allow_pickle=True).item()
        spatial_stats = np.load(str(spatial_path), allow_pickle=True).item()
        preprocessor.spatial_stats = spatial_stats
        return scaler

    logging.getLogger(__name__).info("Fitting scaler from training data (this may take a few minutes)...")

    X_mmap = np.load(str(DATA_DIR / 'X_train.npy'), mmap_mode='r')
    Y_mmap = np.load(str(DATA_DIR / 'Ytrain.npy'),  mmap_mode='r')

    # Fit spatial stats from all training coords (use stride to save memory)
    stride = 100
    X_sub  = np.asarray(X_mmap[::stride], dtype=np.float32)
    spatial_stats = preprocessor.fit_spatial(X_sub)

    # Sample a fraction to compute feature-wise mean/std
    rng    = np.random.default_rng(SEED)
    n_samp = min(2_000_000, len(X_mmap))
    idx    = np.sort(rng.choice(len(X_mmap), n_samp, replace=False))
    X_samp = preprocessor.compute_derived_features(np.asarray(X_mmap[idx], dtype=np.float32))
    Y_samp = np.asarray(Y_mmap[idx], dtype=np.float32)

    X_mean = X_samp.mean(axis=0).astype(np.float32)
    X_std  = X_samp.std(axis=0).clip(min=1e-6).astype(np.float32)
    Y_mean = Y_samp.mean(axis=0).astype(np.float32)
    Y_std  = Y_samp.std(axis=0).clip(min=1e-6).astype(np.float32)

    scaler = {'X_mean': X_mean, 'X_std': X_std, 'Y_mean': Y_mean, 'Y_std': Y_std}
    np.save(str(scaler_path),  scaler)
    np.save(str(spatial_path), spatial_stats)
    logging.getLogger(__name__).info(f"Scaler saved to {scaler_path}")

    if 'RANK' in os.environ:
        dist.barrier()

    return scaler


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=None, metavar='PT',
                        help='Resume from checkpoint')
    parser.add_argument('--save-name', default='surrogate_v2_best.pt')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze the GNN backbone and train only the heads '
                             '(shock indicator, MoE, friction). Use with --resume '
                             'to fine-tune the heads from a trained checkpoint.')
    parser.add_argument('--val-sims',  type=int, default=None,
                        help='Number of sims to use for validation (default: all test sims)')
    args = parser.parse_args()

    rank, world_size, device = init_ddp()
    is_main = (rank == 0)
    set_seed(SEED + rank)

    # Logging — only main process logs to console
    log_level = logging.INFO if is_main else logging.WARNING
    logging.basicConfig(level=log_level,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')
    logger = logging.getLogger(__name__)

    # Persist the log. Without this the whole training curve lives only in
    # whatever captured stderr, and is lost once the job's output is gone.
    # One file per --save-name, so runs never overwrite each other's history.
    if is_main:
        log_path = LOGGING_CONFIG['log_file'].with_name(
            f"training_{Path(args.save_name).stem}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode='w')
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s',
                                          datefmt='%H:%M:%S'))
        logging.getLogger().addHandler(fh)
        logger.info(f"Logging to {log_path}")

    if is_main:
        logger.info(f"AeroSurrogate v2 — rank {rank}/{world_size} — device {device}")

    # ── preprocessor + scaler
    preprocessor = CFDPreprocessor()
    scaler       = fit_or_load_scaler(preprocessor, is_main)

    # ── datasets
    train_weights = load_sim_weights(DATA_DIR / 'dataset.csv', 'train')
    test_weights  = load_sim_weights(DATA_DIR / 'dataset.csv', 'test')

    train_ds = SimulationDataset('train', CACHE_DIR, DATA_DIR,
                                  train_weights, preprocessor, scaler, k=KNN_K)
    test_ds  = SimulationDataset('test',  CACHE_DIR, DATA_DIR,
                                  test_weights,  preprocessor, scaler, k=KNN_K)

    # Pick val sims (fixed subset of test set)
    n_val = args.val_sims or TRAINING_CONFIG['val_sims']
    rng   = np.random.default_rng(SEED)
    val_sim_indices = sorted(rng.choice(N_TEST, n_val, replace=False).tolist())
    if is_main:
        logger.info(f"Train sims: {N_TRAIN}, Val sims: {len(val_sim_indices)} "
                    f"from test set, kNN k={KNN_K}")

    # ── model
    cfg = {
        'model':    MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'dirs':     {'model_dir': str(MODEL_DIR)},
    }
    model = AeroSurrogatev2(MODEL_CONFIG).to(device)

    if args.resume:
        sd = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(sd, strict=False)
        logger.info(f"Resumed from {args.resume}")

    if args.freeze_backbone:
        # Must happen before DDP wraps the model and before the optimizer is
        # built, so the frozen params stay out of the reducer and the optimizer.
        for p in model.backbone.parameters():
            p.requires_grad = False
        # Gradient checkpointing only pays off when the backbone needs a
        # backward pass; with it frozen the recompute is pure overhead.
        model.backbone.use_checkpoint = False
        n_frozen = sum(p.numel() for p in model.backbone.parameters())
        if is_main:
            logger.info(f"Backbone frozen ({n_frozen:,} params); training heads only")

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        logger.info(f"Parameters: {n_params:,}")

    # ── train
    raw_model = model.module if world_size > 1 else model
    trainer   = Surrogatev2Trainer(
        raw_model, cfg, scaler, device,
        save_name=args.save_name,
        rank=rank, world_size=world_size,
    )

    best_r2 = trainer.train(
        train_ds, test_ds, val_sim_indices,
        validate_every=TRAINING_CONFIG['validate_every'],
    )

    if is_main:
        logger.info(f"Training complete. Best R²(Cp) = {best_r2:.4f}")

    cleanup_ddp(world_size)


if __name__ == '__main__':
    main()
