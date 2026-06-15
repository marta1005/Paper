import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from pathlib import Path
from config import TRAINING_CONFIG, MODEL_DIR, MODEL_CONFIG
from src.models import ShockAutoencoder, MixtureOfExperts, VirtualShockSensor

logger = logging.getLogger(__name__)


class AETrainer:
    def __init__(self, device='cpu'):
        self.device = device
        cfg = MODEL_CONFIG['autoencoder']
        self.model = ShockAutoencoder(
            input_dim=cfg['input_dim'],
            latent_dim=cfg['latent_dim'],
            batch_norm=cfg['batch_norm'],
            dropout=cfg['dropout'],
        ).to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay'],
        )
        self.criterion    = nn.MSELoss()
        self.l1_lambda    = MODEL_CONFIG['autoencoder']['l1_lambda']
        self.loss_history = {'train': [], 'val': []}
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader):
        self.model.train()
        total = 0.0
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(self.device)
            x_recon, z = self.model(X_batch)
            loss = self.criterion(x_recon, X_batch) + self.l1_lambda * z.abs().mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total += loss.item()
        avg = total / len(train_loader)
        self.loss_history['train'].append(avg)
        return avg

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total = 0.0
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(self.device)
            x_recon, z = self.model(X_batch)
            total += (self.criterion(x_recon, X_batch) + self.l1_lambda * z.abs().mean()).item()
        avg = total / len(val_loader)
        self.loss_history['val'].append(avg)
        if avg < self.best_val_loss:
            self.best_val_loss = avg
            self.save_model()
        return avg

    def train(self, train_loader, val_loader, num_epochs=None):
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG['num_epochs']
        logger.info(f"AE training for {num_epochs} epochs")
        for epoch in range(num_epochs):
            tr = self.train_epoch(train_loader)
            if (epoch + 1) % TRAINING_CONFIG['validate_every'] == 0:
                vl = self.validate(val_loader)
                logger.info(f"Epoch {epoch+1}/{num_epochs} | train={tr:.6f} val={vl:.6f} best={self.best_val_loss:.6f}")
        logger.info("AE training done")
        self.load_model()
        return self.model

    def save_model(self, name='autoencoder_best.pt'):
        torch.save(self.model.state_dict(), MODEL_DIR / name)

    def load_model(self, name='autoencoder_best.pt'):
        p = MODEL_DIR / name
        if p.exists():
            self.model.load_state_dict(torch.load(p, map_location=self.device))


class MOETrainer:
    def __init__(self, encoder, device='cpu'):
        self.device  = device
        self.encoder = encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        cfg = MODEL_CONFIG
        self.model = MixtureOfExperts(
            latent_dim=cfg['autoencoder']['latent_dim'],
            num_experts=cfg['moe']['num_experts'],
            expert_output_dim=cfg['moe']['expert_output_dim'],
            output_dim=cfg['moe']['output_dim'],
        ).to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'] * 0.5,
            weight_decay=TRAINING_CONFIG['weight_decay'],
        )
        self.criterion    = nn.MSELoss()
        self.loss_history = {'train': [], 'val': []}
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader):
        self.model.train()
        self.encoder.eval()
        total = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            with torch.no_grad():
                z = self.encoder(X_batch)
            pred, _ = self.model(z)
            loss = self.criterion(pred, Y_batch)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total += loss.item()
        avg = total / len(train_loader)
        self.loss_history['train'].append(avg)
        return avg

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total = 0.0
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            z = self.encoder(X_batch)
            pred, _ = self.model(z)
            total += self.criterion(pred, Y_batch).item()
        avg = total / len(val_loader)
        self.loss_history['val'].append(avg)
        if avg < self.best_val_loss:
            self.best_val_loss = avg
            self.save_model()
        return avg

    def train(self, train_loader, val_loader, num_epochs=None):
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG['num_epochs']
        logger.info(f"MoE training for {num_epochs} epochs")
        for epoch in range(num_epochs):
            tr = self.train_epoch(train_loader)
            if (epoch + 1) % TRAINING_CONFIG['validate_every'] == 0:
                vl = self.validate(val_loader)
                logger.info(f"Epoch {epoch+1}/{num_epochs} | train={tr:.6f} val={vl:.6f} best={self.best_val_loss:.6f}")
        logger.info("MoE training done")
        self.load_model()
        return self.model

    def save_model(self, name='moe_best.pt'):
        torch.save(self.model.state_dict(), MODEL_DIR / name)

    def load_model(self, name='moe_best.pt'):
        p = MODEL_DIR / name
        if p.exists():
            self.model.load_state_dict(torch.load(p, map_location=self.device))


class SensorTrainer:
    """
    Trains the shock sensor using CFD ground-truth Y to build labels.

    Shock label: Cp drops sharply (below Cp_crit threshold derived from Mach)
                 AND Cfx changes sign or drops — both conditions from real CFD data.
    Intensity:   magnitude of Cp below Cp_crit (how far into supersonic the local flow is).
    Separation:  Cfx < 0 (reversed flow — attached to shock-induced separation).

    At inference the sensor uses only X (no Y). The symbolic regression on X-features
    post-training extracts the discovered shock condition as a closed-form equation.
    """

    def __init__(self, encoder, moe, scaler, device='cpu'):
        self.device = device
        self.scaler = scaler

        cfg = MODEL_CONFIG
        self.sensor = VirtualShockSensor(
            encoder,
            moe,
            latent_dim=cfg['autoencoder']['latent_dim'],
            head_hidden=cfg['sensor']['head_hidden'],
        ).to(device)

        for p in self.sensor.encoder.parameters():
            p.requires_grad = False
        for p in self.sensor.moe.parameters():
            p.requires_grad = False

        head_params = (
            list(self.sensor.shock_head.parameters()) +
            list(self.sensor.intensity_head.parameters()) +
            list(self.sensor.sep_head.parameters())
        )
        self.optimizer = optim.Adam(
            head_params,
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay'],
        )

        # Shock labels: ~14% positive → pos_weight = 86/14 ≈ 6.0 to avoid collapse to all-zero
        # Sep labels:   ~37% positive → pos_weight = 63/37 ≈ 1.7 (mild imbalance)
        self.criterion_bce_shock = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([6.0]).to(device)
        )
        self.criterion_bce_sep = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1.7]).to(device)
        )
        self.criterion_mse = nn.MSELoss()
        self.loss_history  = {'train': [], 'val': []}
        self.best_val_loss = float('inf')

        self._X_mean = torch.tensor(scaler['X_mean'])
        self._X_std  = torch.tensor(scaler['X_std'])
        self._Y_mean = torch.tensor(scaler['Y_mean'])
        self._Y_std  = torch.tensor(scaler['Y_std'])

    def _make_labels(self, X_batch, Y_batch):
        """
        Build shock/separation labels from CFD ground truth Y.

        Y columns (normalised): [Cp, Cfx, Cfy, Cfz]
        X col 6 = Mach (normalised), col 13 = Cp_crit (normalised, derived feature)

        Shock: Cp_real < Cp_crit  (local flow went supersonic — physically meaningful)
        Intensity: max(0, Cp_crit - Cp_real)  (how far below critical pressure)
        Separation: Cfx_real < 0  (reversed boundary layer flow)
        """
        dev    = X_batch.device
        Y_mean = self._Y_mean.to(dev)
        Y_std  = self._Y_std.to(dev)
        X_mean = self._X_mean.to(dev)
        X_std  = self._X_std.to(dev)

        # Denormalise Cp and Cfx
        Cp_real  = Y_batch[:, 0] * Y_std[0] + Y_mean[0]
        Cfx_real = Y_batch[:, 1] * Y_std[1] + Y_mean[1]

        # Cp_crit from derived features (col 13, already in X_batch normalised)
        Cp_crit = X_batch[:, 13] * X_std[13] + X_mean[13]

        shock_label = (Cp_real < Cp_crit).float().unsqueeze(1)
        intensity   = torch.relu(Cp_crit - Cp_real).unsqueeze(1)
        sep_label   = (Cfx_real < 0.0).float().unsqueeze(1)

        return shock_label, intensity, sep_label

    def _forward_heads(self, X_batch):
        with torch.no_grad():
            z = self.sensor.encoder(X_batch)
        s_in        = self.sensor._sensor_input(X_batch, z)
        shock_logit = self.sensor.shock_head(s_in)
        intensity   = torch.relu(self.sensor.intensity_head(s_in))
        sep_logit   = self.sensor.sep_head(s_in)
        return shock_logit, intensity, sep_logit

    def train_epoch(self, train_loader):
        self.sensor.shock_head.train()
        self.sensor.intensity_head.train()
        self.sensor.sep_head.train()
        self.sensor.encoder.eval()
        self.sensor.moe.eval()

        total = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)

            with torch.no_grad():
                shock_label, intensity_label, sep_label = self._make_labels(X_batch, Y_batch)

            shock_logit, intensity, sep_logit = self._forward_heads(X_batch)

            loss = (
                self.criterion_bce_shock(shock_logit, shock_label) +
                self.criterion_mse(intensity,          intensity_label) +
                self.criterion_bce_sep(sep_logit,      sep_label)
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.sensor.shock_head.parameters()) +
                list(self.sensor.intensity_head.parameters()) +
                list(self.sensor.sep_head.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()
            total += loss.item()

        avg = total / len(train_loader)
        self.loss_history['train'].append(avg)
        return avg

    @torch.no_grad()
    def validate(self, val_loader):
        self.sensor.eval()
        total = 0.0
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            shock_label, intensity_label, sep_label = self._make_labels(X_batch, Y_batch)
            shock_logit, intensity, sep_logit = self._forward_heads(X_batch)
            loss = (
                self.criterion_bce_shock(shock_logit, shock_label) +
                self.criterion_mse(intensity,          intensity_label) +
                self.criterion_bce_sep(sep_logit,      sep_label)
            )
            total += loss.item()
        avg = total / len(val_loader)
        self.loss_history['val'].append(avg)
        if avg < self.best_val_loss:
            self.best_val_loss = avg
            self.save_model()
        return avg

    def train(self, train_loader, val_loader, num_epochs=None):
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG['num_epochs']
        logger.info(f"Sensor training for {num_epochs} epochs")
        for epoch in range(num_epochs):
            tr = self.train_epoch(train_loader)
            if (epoch + 1) % TRAINING_CONFIG['validate_every'] == 0:
                vl = self.validate(val_loader)
                logger.info(f"Epoch {epoch+1}/{num_epochs} | train={tr:.6f} val={vl:.6f} best={self.best_val_loss:.6f}")
        logger.info("Sensor training done")
        self.load_model()
        return self.sensor

    def save_model(self, name='sensor_best.pt'):
        torch.save(self.sensor.state_dict(), MODEL_DIR / name)

    def load_model(self, name='sensor_best.pt'):
        p = MODEL_DIR / name
        if p.exists():
            self.sensor.load_state_dict(torch.load(p, map_location=self.device))
