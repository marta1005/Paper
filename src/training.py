"""
Training loop: autoencoder + MoE + sensor
"""
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from pathlib import Path
from config import TRAINING_CONFIG, MODEL_DIR, MODEL_CONFIG, DERIVED_FEATURE_INDICES
from src.models import (
    ShockAutoencoder, MixtureOfExperts, VirtualShockSensor, ReconstructionLoss
)

logger = logging.getLogger(__name__)


class AETrainer:
    """Entrena el autoencoder"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model = ShockAutoencoder(
            input_dim=19,
            latent_dim=MODEL_CONFIG['autoencoder']['latent_dim'],
            batch_norm=MODEL_CONFIG['autoencoder']['batch_norm'],
            dropout=MODEL_CONFIG['autoencoder']['dropout'],
        ).to(device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        
        self.criterion = ReconstructionLoss(weight_high_gradient=10.0)
        self.loss_history = {'train': [], 'val': []}
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            
            # Forward
            x_recon, z = self.model(X_batch)
            
            # Loss (usar gradiente de presión como métrica de importancia)
            pressure_gradient = torch.abs(X_batch[:, 2:3])
            loss = self.criterion(X_batch, x_recon, gradient_metric=pressure_gradient)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.loss_history['train'].append(avg_loss)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(self.device)
            
            x_recon, z = self.model(X_batch)
            # Calcular gradiente de presión de forma robusta
            pressure_gradient = torch.abs(X_batch[:, 2:3])
            loss = self.criterion(X_batch, x_recon, gradient_metric=pressure_gradient)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.loss_history['val'].append(avg_loss)
        
        # Early stopping
        if avg_loss < self.best_val_loss - TRAINING_CONFIG['early_stopping_delta']:
            self.best_val_loss = avg_loss
            self.patience_counter = 0
            self.save_model()
        else:
            self.patience_counter += 1
        
        return avg_loss
    
    def train(self, train_loader, val_loader, num_epochs=None):
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG['num_epochs']
        
        logger.info(f"Starting AE training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            
            if (epoch + 1) % TRAINING_CONFIG['validate_every'] == 0:
                val_loss = self.validate(val_loader)
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Patience: {self.patience_counter}"
                )
                
                if self.patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info("AE training completed")
        self.load_model()  # Load best model
        return self.model
    
    def save_model(self, name='autoencoder_best.pt'):
        path = MODEL_DIR / name
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved AE to {path}")
    
    def load_model(self, name='autoencoder_best.pt'):
        path = MODEL_DIR / name
        if path.exists():
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Loaded AE from {path}")


class MOETrainer:
    """Entrena el Mixture of Experts"""

    def __init__(self, encoder, device='cpu'):
        self.device = device
        self.encoder = encoder
        self.encoder.eval()

        latent_dim = MODEL_CONFIG['autoencoder']['latent_dim']

        self.model = MixtureOfExperts(
            latent_dim=latent_dim,
            num_experts=MODEL_CONFIG['moe']['num_experts'],
            expert_output_dim=MODEL_CONFIG['moe']['expert_output_dim'],
            output_dim=MODEL_CONFIG['moe']['output_dim'],
        ).to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'] * 0.5,
            weight_decay=TRAINING_CONFIG['weight_decay']
        )

        self.criterion = nn.MSELoss()
        self.loss_history = {'train': [], 'val': []}
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self._latent_dim = latent_dim

        self._idx_mach    = DERIVED_FEATURE_INDICES['M_local']
        self._idx_shock   = DERIVED_FEATURE_INDICES['shock_indicator']
        self._idx_sep     = DERIVED_FEATURE_INDICES['Cf_mag']

    def _encode_latent(self, X_batch):
        """Codifica X → z y verifica que la dimensión coincida con el MoE."""
        z = self.encoder(X_batch)
        if z.shape[-1] != self._latent_dim:
            raise RuntimeError(
                f"Dim mismatch: encoder devuelve z.shape={z.shape}, "
                f"pero MoE espera latent_dim={self._latent_dim}. "
                f"Asegúrate de que MixtureOfExperts(latent_dim=32) coincide con el AE."
            )
        return z

    def _get_physical_indicators(self, X_batch):
        shock_indicator = X_batch[:, self._idx_shock:self._idx_shock + 1]
        separation_risk = X_batch[:, self._idx_sep:self._idx_sep + 1]
        mach_local      = X_batch[:, self._idx_mach:self._idx_mach + 1]
        return shock_indicator, separation_risk, mach_local

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)

            with torch.no_grad():
                z = self._encode_latent(X_batch)

            shock_indicator, separation_risk, mach_local = self._get_physical_indicators(X_batch)

            moe_output, gate_weights = self.model(
                z, shock_indicator, separation_risk, mach_local
            )

            loss = self.criterion(moe_output, Y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        self.loss_history['train'].append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)

            z = self._encode_latent(X_batch)

            shock_indicator, separation_risk, mach_local = self._get_physical_indicators(X_batch)

            moe_output, _ = self.model(z, shock_indicator, separation_risk, mach_local)
            loss = self.criterion(moe_output, Y_batch)

            total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        self.loss_history['val'].append(avg_loss)

        # Early stopping + checkpointing
        if avg_loss < self.best_val_loss - TRAINING_CONFIG['early_stopping_delta']:
            self.best_val_loss = avg_loss
            self.patience_counter = 0
            self.save_model()
        else:
            self.patience_counter += 1

        return avg_loss

    def train(self, train_loader, val_loader, num_epochs=None):
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG['num_epochs'] // 2

        logger.info(f"Starting MoE training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)

            if (epoch + 1) % TRAINING_CONFIG['validate_every'] == 0:
                val_loss = self.validate(val_loader)
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Patience: {self.patience_counter}"
                )

                if self.patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
                    logger.info(f"Early stopping MoE at epoch {epoch+1}")
                    break

        logger.info("MoE training completed")
        self.load_model()  # Restaurar mejor modelo
        return self.model

    def save_model(self, name='moe_best.pt'):
        path = MODEL_DIR / name
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved MoE to {path}")

    def load_model(self, name='moe_best.pt'):
        path = MODEL_DIR / name
        if path.exists():
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Loaded MoE from {path}")


class SensorTrainer:
    """
    Entrena el Virtual Shock Sensor (heads de clasificación/regresión).
    Usa pseudo-labels derivados del shock_indicator de cada batch.
    Encoder y MoE permanecen congelados.
    """

    # Umbral para binarizar shock_indicator → etiqueta de choque
    SHOCK_THRESHOLD = 0.5

    def __init__(self, encoder, moe, device='cpu'):
        self.device = device
        self.encoder = encoder
        self.moe = moe

        latent_dim = MODEL_CONFIG['autoencoder']['latent_dim']
        self.sensor = VirtualShockSensor(encoder, moe, latent_dim=latent_dim).to(device)

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.moe.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(
            list(self.sensor.shock_head.parameters()) +
            list(self.sensor.intensity_head.parameters()) +
            list(self.sensor.separation_head.parameters()),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )

        self.criterion_shock = nn.BCELoss()
        self.criterion_intensity = nn.MSELoss()
        self.loss_history = {'train': [], 'val': []}
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        self._idx_shock = DERIVED_FEATURE_INDICES['shock_indicator']  # 12
        self._idx_cf    = DERIVED_FEATURE_INDICES['Cf_mag']           # 13

    def _make_pseudo_labels(self, X_batch):
        """
        Genera pseudo-labels a partir del shock_indicator normalizado.
        - shock_label: binario (shock_indicator > umbral normalizado)
        - intensity: valor continuo de shock_indicator (no negativo)
        """
        shock_raw = X_batch[:, self._idx_shock:self._idx_shock + 1]

        # Normalizar dentro del batch para comparación robusta
        s_mean = shock_raw.mean()
        s_std  = shock_raw.std() + 1e-8
        shock_norm = (shock_raw - s_mean) / s_std

        shock_label = (shock_norm > self.SHOCK_THRESHOLD).float()
        intensity   = torch.relu(shock_norm)  # intensidad ≥ 0

        return shock_label, intensity

    def train_epoch(self, train_loader):
        self.sensor.train()
        total_loss = 0

        for X_batch, _ in train_loader:
            X_batch = X_batch.to(self.device)

            shock_label, intensity = self._make_pseudo_labels(X_batch)

            output = self.sensor(X_batch, compute_moe=False)

            loss_shock     = self.criterion_shock(output['shock_prob'], shock_label)
            loss_intensity = self.criterion_intensity(output['intensity'], intensity)
            loss_total = (
                TRAINING_CONFIG.get('loss_shock_weight', 1.0) * loss_shock +
                0.5 * loss_intensity
            )

            self.optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.sensor.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss_total.item()

        avg_loss = total_loss / len(train_loader)
        self.loss_history['train'].append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader):
        self.sensor.eval()
        total_loss = 0

        for X_batch, _ in val_loader:
            X_batch = X_batch.to(self.device)

            shock_label, intensity = self._make_pseudo_labels(X_batch)

            output = self.sensor(X_batch, compute_moe=False)

            loss_shock     = self.criterion_shock(output['shock_prob'], shock_label)
            loss_intensity = self.criterion_intensity(output['intensity'], intensity)
            total_loss += (loss_shock + 0.5 * loss_intensity).item()

        avg_loss = total_loss / len(val_loader)
        self.loss_history['val'].append(avg_loss)

        if avg_loss < self.best_val_loss - TRAINING_CONFIG['early_stopping_delta']:
            self.best_val_loss = avg_loss
            self.patience_counter = 0
            self.save_model()
        else:
            self.patience_counter += 1

        return avg_loss

    def train(self, train_loader, val_loader, num_epochs=None):
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG['num_epochs']

        logger.info(f"Starting Sensor training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)

            if (epoch + 1) % TRAINING_CONFIG['validate_every'] == 0:
                val_loss = self.validate(val_loader)
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Patience: {self.patience_counter}"
                )

                if self.patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
                    logger.info(f"Early stopping Sensor at epoch {epoch+1}")
                    break

        logger.info("Sensor training completed")
        self.load_model()
        return self.sensor

    def save_model(self, name='sensor_best.pt'):
        path = MODEL_DIR / name
        torch.save(self.sensor.state_dict(), path)
        logger.info(f"Saved Sensor to {path}")

    def load_model(self, name='sensor_best.pt'):
        path = MODEL_DIR / name
        if path.exists():
            self.sensor.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Loaded Sensor from {path}")


if __name__ == '__main__':
    print("Training module loaded successfully")
