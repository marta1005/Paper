#!/usr/bin/env python3
"""
Offline preprocessing: compute derived features and save to NPY.
Run once before main_train.py.

New in this version:
  - Fits spatial stats (x/y bounds) on the full training set first
  - Produces 16 features instead of 14 (adds x_norm, span_norm)
  - Saves spatial_stats.npy for use in inference and symbolic regression

Usage:
    python preprocess_data.py
"""
import numpy as np
import logging
from pathlib import Path
from src.preprocessing import CFDPreprocessor
from config import DATA_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_and_save():
    data_dir = Path(DATA_CONFIG['X_train_path']).parent

    logger.info("=" * 70)
    logger.info("OFFLINE PREPROCESSING  (9 raw -> 16 derived features)")
    logger.info("=" * 70)

    logger.info("\n[0/3] Loading full X_train to fit spatial stats...")
    X_train_raw = np.asarray(np.load(DATA_CONFIG['X_train_path'], mmap_mode='r')[:], dtype=np.float32)
    logger.info(f"  Raw X_train shape: {X_train_raw.shape}")

    prep = CFDPreprocessor()
    spatial_stats = prep.fit_spatial(X_train_raw)

    stats_path = data_dir / 'spatial_stats.npy'
    np.save(str(stats_path), spatial_stats)
    logger.info(f"  Spatial stats saved to {stats_path}")

    logger.info("\n[1/3] Processing TRAIN data...")
    X_train_derived = prep.compute_derived_features(X_train_raw)
    out = data_dir / "X_train_derived.npy"
    np.save(str(out), X_train_derived)
    logger.info(f"  Saved {out}  shape={X_train_derived.shape}")
    del X_train_raw, X_train_derived

    logger.info("\n[2/3] Processing TEST data (chunked)...")
    X_test_full = np.load(DATA_CONFIG['X_test_path'], mmap_mode='r')
    chunks, chunk_size = [], 1_000_000
    n_chunks = (len(X_test_full) - 1) // chunk_size + 1
    for i in range(0, len(X_test_full), chunk_size):
        chunk = np.asarray(X_test_full[i:i + chunk_size], dtype=np.float32)
        chunks.append(prep.compute_derived_features(chunk))
        logger.info(f"  Chunk {i // chunk_size + 1}/{n_chunks}")
    X_test_derived = np.vstack(chunks)
    out = data_dir / "X_test_derived.npy"
    np.save(str(out), X_test_derived)
    logger.info(f"  Saved {out}  shape={X_test_derived.shape}")

    logger.info("\n[3/3] Verifying feature counts...")
    for label, path in [("train", data_dir / "X_train_derived.npy"),
                        ("test",  data_dir / "X_test_derived.npy")]:
        shape = np.load(str(path), mmap_mode='r').shape
        assert shape[1] == 16, f"Expected 16 features, got {shape[1]} in {path}"
        logger.info(f"  {label}: {shape}  ✓")

    logger.info("\nDone: 9 features -> 16 features")
    logger.info("  Cols 0-8:  original [x, y, z, nx, ny, nz, Mach, AoA, Pi]")
    logger.info("  Cols 9-13: physics scalars [q_dyn, Pi_norm, AoA_sin, L_factor, Cp_crit]")
    logger.info("  Cols 14-15: geometry [x_norm, span_norm]")


if __name__ == '__main__':
    preprocess_and_save()
