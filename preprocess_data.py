#!/usr/bin/env python3
"""
Offline preprocessing: compute derived features and save to NPY.
Run once before main_train.py.

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
    logger.info("=" * 70)
    logger.info("OFFLINE PREPROCESSING")
    logger.info("=" * 70)

    prep = CFDPreprocessor()

    logger.info("\n[1/2] Processing TRAIN data...")
    X_train = np.asarray(np.load(DATA_CONFIG['X_train_path'], mmap_mode='r')[:], dtype=np.float32)
    logger.info(f"  Original shape: {X_train.shape}")
    X_train_derived = prep.compute_derived_features(X_train)
    out = DATA_CONFIG['X_train_path'].parent / "X_train_derived.npy"
    np.save(out, X_train_derived)
    logger.info(f"  Saved {out}  shape={X_train_derived.shape}")

    logger.info("\n[2/2] Processing TEST data (chunked)...")
    X_test_full = np.load(DATA_CONFIG['X_test_path'], mmap_mode='r')
    chunks, chunk_size = [], 1_000_000
    for i in range(0, len(X_test_full), chunk_size):
        chunk = np.asarray(X_test_full[i:i + chunk_size], dtype=np.float32)
        chunks.append(prep.compute_derived_features(chunk))
        logger.info(f"  Chunk {i//chunk_size + 1}/{(len(X_test_full) - 1)//chunk_size + 1}")
    X_test_derived = np.vstack(chunks)
    out = DATA_CONFIG['X_test_path'].parent / "X_test_derived.npy"
    np.save(out, X_test_derived)
    logger.info(f"  Saved {out}  shape={X_test_derived.shape}")

    logger.info("\nDone: 9 features -> 14 features (+ 5 derived from X only)")


if __name__ == '__main__':
    preprocess_and_save()
