#!/usr/bin/env python3
"""
MAIN SCRIPT: Entrenamiento completo de arquitectura de shock detection
Shock Detection via Physics-Informed Machine Learning

Uso:
    python main_train.py
"""
import logging
import sys
from pathlib import Path

# Imports
import torch
import numpy as np

from config import (
    SEED, DEVICE, TRAINING_CONFIG, DATA_CONFIG, OUTPUT_DIR,
    LOGGING_CONFIG
)
from src.data_loader import get_dataloaders
from src.preprocessing import CFDPreprocessor
from src.training import AETrainer, MOETrainer, SensorTrainer
from src.evaluation import ModelEvaluator, VisualizationTools, save_evaluation_report

# ============ SETUP LOGGING ============
logging.basicConfig(
    level=LOGGING_CONFIG['log_level'],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGGING_CONFIG['log_file'])
    ] if LOGGING_CONFIG['save_log'] else [logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Set random seed
torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    """
    Pipeline completo:
    1. Carga datos
    2. Preprocesamiento (variables derivadas)
    3. Entrena Autoencoder
    4. Entrena Mixture of Experts
    5. Entrena Sensor Virtual
    6. Evaluación completa
    """
    
    logger.info("="*80)
    logger.info("SHOCK DETECTION PIPELINE")
    logger.info("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and DEVICE == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    # ============ STAGE 1: DATA LOADING ============
    logger.info("\n[STAGE 1] Loading data...")
    
    sample_fraction = DATA_CONFIG.get('train_sample_fraction', 0.1)
    logger.info(f"Using {100*sample_fraction:.1f}% of training data")
    
    try:
        train_loader, val_loader, test_loader, scaler = get_dataloaders(
            sample_fraction=sample_fraction
        )
        logger.info(f"✓ Data loaded successfully")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        logger.info(f"  Test batches: {len(test_loader)}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False
    
    # ============ STAGE 2: PREPROCESSING ============
    logger.info("\n[STAGE 2] Preprocessing (computing derived features)...")
    
    try:
        preprocessor = CFDPreprocessor()
        logger.info("✓ Preprocessor initialized")
        logger.info("  Derived features will be computed on-the-fly during training")
    except Exception as e:
        logger.error(f"Failed to initialize preprocessor: {e}")
        return False
    
    # ============ STAGE 3: AUTOENCODER TRAINING ============
    logger.info("\n[STAGE 3] Training Autoencoder...")
    logger.info(f"  Architecture: 19 -> 128 -> 64 -> 32 (latent) -> 64 -> 128 -> 19")
    logger.info(f"  Training for {TRAINING_CONFIG['num_epochs']} epochs")
    
    try:
        ae_trainer = AETrainer(device=device)
        ae_model = ae_trainer.train(train_loader, val_loader, num_epochs=TRAINING_CONFIG['num_epochs'])
        logger.info("✓ Autoencoder training completed")
        logger.info(f"  Best validation loss: {ae_trainer.best_val_loss:.6f}")
    except Exception as e:
        logger.error(f"Failed to train autoencoder: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============ STAGE 4: MIXTURE OF EXPERTS TRAINING ============
    logger.info("\n[STAGE 4] Training Mixture of Experts...")
    logger.info(f"  Num experts: 4 (Adherent | Transonic | Shock | Separated)")
    
    try:
        encoder = ae_model.encoder
        moe_trainer = MOETrainer(encoder=encoder, device=device)
        moe_trainer.train(train_loader, val_loader, num_epochs=TRAINING_CONFIG['num_epochs']//2)
        moe_model = moe_trainer.model
        logger.info("✓ MoE training completed")
    except Exception as e:
        logger.error(f"Failed to train MoE: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============ STAGE 5: EVALUATION ============
    logger.info("\n[STAGE 5] Evaluation...")
    
    try:
        evaluator = ModelEvaluator(ae_model, device=device)
        eval_result = evaluator.evaluate(test_loader, return_predictions=True)
        
        metrics = eval_result['metrics']
        evaluator.log_metrics(metrics)
        
        # Guardar reporte
        save_evaluation_report(metrics, eval_result, model_name='autoencoder')
        
        logger.info("✓ Evaluation completed")
    except Exception as e:
        logger.error(f"Failed evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============ STAGE 6: VISUALIZATION ============
    logger.info("\n[STAGE 6] Generating visualizations...")
    
    try:
        viz = VisualizationTools()
        
        # Loss curves
        viz.plot_losses(
            ae_trainer.loss_history['train'],
            ae_trainer.loss_history['val'],
            save_path=OUTPUT_DIR / 'plots' / 'ae_losses.png'
        )
        
        # Predictions vs truth
        if 'y_true' in eval_result and 'y_pred' in eval_result:
            viz.plot_predictions_vs_truth(
                eval_result['y_true'][:10000],  # Subsample para visualizar
                eval_result['y_pred'][:10000],
                save_path=OUTPUT_DIR / 'plots' / 'predictions_vs_truth.png'
            )
        
        # Latent space
        if 'z' in eval_result:
            viz.plot_latent_space(
                eval_result['z'][:10000],
                save_path=OUTPUT_DIR / 'plots' / 'latent_space.png'
            )
        
        # Reconstruction error
        if 'y_true' in eval_result and 'y_pred' in eval_result:
            viz.plot_reconstruction_error(
                eval_result['y_true'],
                eval_result['y_pred'],
                save_path=OUTPUT_DIR / 'plots' / 'reconstruction_error.png'
            )
        
        logger.info("✓ Visualizations saved")
    except Exception as e:
        logger.warning(f"Visualization failed (non-critical): {e}")
    
    # ============ SUMMARY ============
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info(f"  - Models: {OUTPUT_DIR}/models/")
    logger.info(f"  - Results: {OUTPUT_DIR}/results/")
    logger.info(f"  - Plots: {OUTPUT_DIR}/plots/")
    logger.info("="*80 + "\n")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
