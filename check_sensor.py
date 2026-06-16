#!/usr/bin/env python3
import os; os.environ['PAPER_NUM_WORKERS'] = '0'
import torch, numpy as np
from config import MODEL_DIR, MODEL_CONFIG
from src.models import ShockAutoencoder, MixtureOfExperts, VirtualShockSensor
from src.data_loader import get_dataloaders

cfg = MODEL_CONFIG
ae = ShockAutoencoder(input_dim=cfg['autoencoder']['input_dim'], latent_dim=cfg['autoencoder']['latent_dim'])
ae.load_state_dict(torch.load(MODEL_DIR / 'autoencoder_best.pt', map_location='cpu'))
moe = MixtureOfExperts(latent_dim=cfg['autoencoder']['latent_dim'], num_experts=cfg['moe']['num_experts'],
                       expert_output_dim=cfg['moe']['expert_output_dim'], output_dim=cfg['moe']['output_dim'])
moe.load_state_dict(torch.load(MODEL_DIR / 'moe_best.pt', map_location='cpu'))
sensor = VirtualShockSensor(ae.encoder, moe, latent_dim=cfg['autoencoder']['latent_dim'],
                            head_hidden=cfg['sensor']['head_hidden'])
sensor.load_state_dict(torch.load(MODEL_DIR / 'sensor_best.pt', map_location='cpu'))
sensor.eval()

# Load the training scaler so normalization matches what the model was trained on
scaler_path = MODEL_DIR / 'scaler.npy'
saved_scaler = np.load(str(scaler_path), allow_pickle=True).item() if scaler_path.exists() else None
if saved_scaler is None:
    print("WARNING: scaler.npy not found — using locally recomputed scaler (may differ from training)")

_, _, test_loader, _ = get_dataloaders(sample_fraction=0.01, scaler=saved_scaler)
probs, seps = [], []
with torch.no_grad():
    for i, (X_batch, _) in enumerate(test_loader):
        out = sensor(X_batch)
        probs.append(out['shock_prob'].cpu().numpy().squeeze())
        seps.append(out['separation_prob'].cpu().numpy().squeeze())
        if i >= 5:
            break

p = np.concatenate(probs)
s = np.concatenate(seps)
print(f"shock_prob:  min={p.min():.4f}  max={p.max():.4f}  mean={p.mean():.4f}  std={p.std():.4f}  >0.5: {(p>0.5).mean()*100:.1f}%")
print(f"sep_prob:    min={s.min():.4f}  max={s.max():.4f}  mean={s.mean():.4f}  std={s.std():.4f}  >0.5: {(s>0.5).mean()*100:.1f}%")
