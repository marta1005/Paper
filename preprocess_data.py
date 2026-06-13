#!/usr/bin/env python3
"""
Preprocesamiento offline: calcula derivadas y guarda en NPY
Ejecutar una sola vez: python preprocess_data.py
"""
import numpy as np
import logging
from pathlib import Path
from src.preprocessing import CFDPreprocessor
from config import DATA_CONFIG, PREPROCESSING_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_and_save():
    """
    Carga datos originales, calcula derivadas, guarda versiones preprocesadas
    """
    
    logger.info("="*80)
    logger.info("PREPROCESSING DATA OFFLINE")
    logger.info("="*80)
    
    preprocessor = CFDPreprocessor()
    
    # ============ TRAIN ============
    logger.info("\n[1/2] Processing TRAIN data...")
    X_train = np.load(DATA_CONFIG['X_train_path'], mmap_mode='r')
    Y_train = np.load(DATA_CONFIG['Y_train_path'], mmap_mode='r')
    
    logger.info(f"  Original shapes: X={X_train.shape}, Y={Y_train.shape}")
    
    # Convertir a memoria normal
    X_train = np.asarray(X_train[:], dtype=np.float32)
    Y_train = np.asarray(Y_train[:], dtype=np.float32)
    
    logger.info(f"  Computing derived features for train...")
    X_train_derived = preprocessor.compute_derived_features(X_train, Y_train)
    
    # Guardar
    train_derived_path = DATA_CONFIG['X_train_path'].parent / "X_train_derived.npy"
    np.save(train_derived_path, X_train_derived)
    logger.info(f"  ✓ Saved to {train_derived_path}")
    logger.info(f"  Shape: {X_train_derived.shape}")
    
    # ============ TEST ============
    logger.info("\n[2/2] Processing TEST data...")
    X_test = np.load(DATA_CONFIG['X_test_path'], mmap_mode='r')
    Y_test = np.load(DATA_CONFIG['Y_test_path'], mmap_mode='r')
    
    logger.info(f"  Original shapes: X={X_test.shape}, Y={Y_test.shape}")
    
    # Convertir a memoria normal (en chunks si es muy grande)
    logger.info(f"  Loading test data in chunks...")
    chunk_size = 1000000
    X_test_derived_list = []
    
    for i in range(0, len(X_test), chunk_size):
        end = min(i + chunk_size, len(X_test))
        X_chunk = np.asarray(X_test[i:end], dtype=np.float32)
        Y_chunk = np.asarray(Y_test[i:end], dtype=np.float32)
        
        logger.info(f"  Processing chunk {i//chunk_size + 1}/{(len(X_test)-1)//chunk_size + 1}")
        X_derived_chunk = preprocessor.compute_derived_features(X_chunk, Y_chunk)
        X_test_derived_list.append(X_derived_chunk)
    
    X_test_derived = np.vstack(X_test_derived_list)
    
    # Guardar
    test_derived_path = DATA_CONFIG['X_test_path'].parent / "X_test_derived.npy"
    np.save(test_derived_path, X_test_derived)
    logger.info(f"  ✓ Saved to {test_derived_path}")
    logger.info(f"  Shape: {X_test_derived.shape}")
    
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING COMPLETED")
    logger.info("="*80)
    logger.info(f"Train: 9 features → {X_train_derived.shape[1]} features")
    logger.info(f"Test:  9 features → {X_test_derived.shape[1]} features")
    logger.info("="*80)


if __name__ == '__main__':
    preprocess_and_save()
