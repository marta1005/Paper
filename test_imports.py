#!/usr/bin/env python3
"""
Correctness tests: shapes, no NaN, no data leakage.
Run: python test_imports.py
"""
import sys
import inspect
import numpy as np
import torch


def test_no_leakage():
    from src.preprocessing import CFDPreprocessor
    sig = inspect.signature(CFDPreprocessor.compute_derived_features)
    params = list(sig.parameters.keys())
    assert params == ['self', 'X'], f"compute_derived_features must take only X, got {params}"
    print("  [PASS] No leakage: preprocessing signature is (self, X) — no Y")


def test_preprocessing():
    from src.preprocessing import CFDPreprocessor
    X = np.random.rand(200, 9).astype(np.float32)
    X[:, 0] = np.random.uniform(0.3, 0.9, 200)
    X[:, 1] = np.random.uniform(-15, 15, 200)
    X[:, 2] = np.random.uniform(0.5, 2.0, 200)
    out = CFDPreprocessor().compute_derived_features(X)
    assert out.shape == (200, 14), f"Expected (200,14), got {out.shape}"
    assert not np.isnan(out).any(), "NaN in derived features"
    assert not np.isinf(out).any(), "Inf in derived features"
    print("  [PASS] Preprocessing: 9 -> 14 features, no NaN/Inf")


def test_ae_bottleneck():
    from config import MODEL_CONFIG
    from src.models import ShockAutoencoder
    cfg    = MODEL_CONFIG['autoencoder']
    in_d   = cfg['input_dim']
    lat_d  = cfg['latent_dim']
    ae     = ShockAutoencoder(input_dim=in_d, latent_dim=lat_d)
    x      = torch.randn(32, in_d)
    x_rec, z = ae(x)
    assert x_rec.shape == (32, in_d),  f"Recon shape: {x_rec.shape}"
    assert z.shape     == (32, lat_d), f"Latent shape: {z.shape}"
    assert not torch.isnan(x_rec).any()
    assert 'l1_lambda' in cfg, "Missing l1_lambda in config"
    print(f"  [PASS] AE: {in_d} -> {lat_d} -> {in_d}  (L1 sparsity on latent)")


def test_moe_forward():
    from config import MODEL_CONFIG
    from src.models import MixtureOfExperts
    cfg    = MODEL_CONFIG
    lat_d  = cfg['autoencoder']['latent_dim']
    out_d  = cfg['moe']['output_dim']
    n_exp  = cfg['moe']['num_experts']
    moe    = MixtureOfExperts(latent_dim=lat_d, num_experts=n_exp,
                               expert_output_dim=cfg['moe']['expert_output_dim'],
                               output_dim=out_d)
    z      = torch.randn(32, lat_d)
    y, gates = moe(z)
    assert y.shape     == (32, out_d), f"MoE output: {y.shape}"
    assert gates.shape == (32, n_exp), f"Gates: {gates.shape}"
    assert not torch.isnan(y).any()
    assert torch.allclose(gates.sum(1), torch.ones(32), atol=1e-5)
    print(f"  [PASS] MoE: {lat_d}-dim latent -> {n_exp} experts -> Y={out_d}")


def test_sensor_forward():
    from config import MODEL_CONFIG
    from src.models import ShockAutoencoder, MixtureOfExperts, VirtualShockSensor
    cfg    = MODEL_CONFIG
    in_d   = cfg['autoencoder']['input_dim']
    lat_d  = cfg['autoencoder']['latent_dim']
    ae     = ShockAutoencoder(input_dim=in_d, latent_dim=lat_d)
    moe    = MixtureOfExperts(latent_dim=lat_d, num_experts=cfg['moe']['num_experts'],
                               expert_output_dim=cfg['moe']['expert_output_dim'],
                               output_dim=cfg['moe']['output_dim'])
    sensor = VirtualShockSensor(ae.encoder, moe, latent_dim=lat_d,
                                head_hidden=cfg['sensor']['head_hidden'])
    x      = torch.randn(32, in_d)
    out    = sensor(x, compute_moe=True)
    for key in ('shock_prob', 'intensity', 'separation_prob', 'latent', 'moe_output'):
        assert key in out, f"Missing key: {key}"
    assert (out['shock_prob'] >= 0).all() and (out['shock_prob'] <= 1).all()
    assert (out['separation_prob'] >= 0).all() and (out['separation_prob'] <= 1).all()
    assert (out['intensity'] >= 0).all()
    assert not torch.isnan(out['shock_prob']).any()
    print("  [PASS] Sensor outputs shock_prob, intensity, separation_prob in valid ranges")


def test_scaler_nan_safe():
    from src.data_loader import CFDDataset
    X = np.random.rand(100, 14).astype(np.float32)
    Y = np.random.rand(100, 4).astype(np.float32)
    X[0, 0] = np.nan
    X[1, 1] = np.inf
    ds = CFDDataset(X, Y)
    x0, _ = ds[0]
    assert not torch.isnan(x0).any(), "NaN survived into dataset"
    assert not torch.isinf(x0).any(), "Inf survived into dataset"
    print("  [PASS] Dataset: NaN/Inf sanitized before normalization")


if __name__ == '__main__':
    print("Running tests...\n")
    try:
        test_no_leakage()
        test_preprocessing()
        test_ae_bottleneck()
        test_moe_forward()
        test_sensor_forward()
        test_scaler_nan_safe()
        print("\nAll tests passed.")
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
