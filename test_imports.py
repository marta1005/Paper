#!/usr/bin/env python3
"""
Test rápido: verifica que todo se carga correctamente sin errores
"""
import sys
print("Testing imports...")

try:
    print("  ✓ torch", end="")
    import torch
    print(f" (v{torch.__version__})")
    
    print("  ✓ numpy", end="")
    import numpy as np
    print(f" (v{np.__version__})")
    
    print("  ✓ config")
    from config import DATA_CONFIG, TRAINING_CONFIG, MODEL_DIR
    
    print("  ✓ data_loader")
    from src.data_loader import load_data_with_sampling
    
    print("  ✓ preprocessing")
    from src.preprocessing import CFDPreprocessor
    
    print("  ✓ models")
    from src.models import ShockAutoencoder, MixtureOfExperts, VirtualShockSensor
    
    print("  ✓ training")
    from src.training import AETrainer
    
    print("  ✓ evaluation")
    from src.evaluation import ModelEvaluator
    
    print("\n✓ All imports successful!\n")
    
    # Quick test
    print("Testing model instantiation...")
    ae = ShockAutoencoder(input_dim=19, latent_dim=32)
    print(f"  ✓ Autoencoder: {sum(p.numel() for p in ae.parameters()):,} parameters")
    
    moe = MixtureOfExperts(latent_dim=32, num_experts=4, expert_output_dim=16)
    print(f"  ✓ MoE: {sum(p.numel() for p in moe.parameters()):,} parameters")
    
    sensor = VirtualShockSensor(ae.encoder, moe)
    print(f"  ✓ Sensor: initialized")
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(4, 19)
    with torch.no_grad():
        x_recon, z = ae(x)
        print(f"  ✓ AE forward: x({x.shape}) → z({z.shape}) → x_recon({x_recon.shape})")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED - Ready to train!")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
