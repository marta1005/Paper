"""
Training loop: autoencoder + MoE + sensor
"""
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from pathlib import Path
from config import TRAINING_CONFIG, MODEL_DIR
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
            pressure_gradient = torch.abs(torch.gradient(X_batch[:, 2])[0])
            loss = self.criterion(X_batch, x_recon, gradient_metric=pressure_gradient.unsqueeze(1))
            
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
            pressure_gradient = torch.abs(torch.gradient(X_batch[:, 2])[0])
            loss = self.criterion(X_batch, x_recon, gradient_metric=pressure_gradient.unsqueeze(1))
            
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
        self.encoder.eval()  # Congelar encoder
        
        self.model = MixtureOfExperts(
            latent_dim=MODEL_CONFIG['moe']['expert_output_dim'],
            num_experts=MODEL_CONFIG['moe']['num_experts'],
            expert_output_dim=MODEL_CONFIG['moe']['expert_output_dim'],
        ).to(device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'] * 0.5,
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        
        self.criterion = nn.MSELoss()
        self.loss_history = {'train': [], 'val': []}
        self.patience_counter = 0
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            
            with torch.no_grad():
                z = self.encoder(X_batch)
            
            # Indicadores físicos (derivados de X)
            shock_indicator = X_batch[:, 9:10]  # Derived feature
            separation_risk = X_batch[:, 10:11]
            mach_local = X_batch[:, 8:9]
            
            # Forward MoE
            moe_output, gate_weights = self.model(
                z, shock_indicator, separation_risk, mach_local
            )
            
            # Loss: predecir Y (aerodinámica)
            loss = self.criterion(moe_output, Y_batch[:, :moe_output.shape[1]])
            
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
            
            z = self.encoder(X_batch)
            
            shock_indicator = X_batch[:, 9:10]
            separation_risk = X_batch[:, 10:11]
            mach_local = X_batch[:, 8:9]
            
            moe_output, _ = self.model(z, shock_indicator, separation_risk, mach_local)
            loss = self.criterion(moe_output, Y_batch[:, :moe_output.shape[1]])
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.loss_history['val'].append(avg_loss)
        
        return avg_loss
    
    def train(self, train_loader, val_loader, num_epochs=None):
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG['num_epochs'] // 2  # Menos épocas para MoE
        
        logger.info(f"Starting MoE training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            
            if (epoch + 1) % TRAINING_CONFIG['validate_every'] == 0:
                val_loss = self.validate(val_loader)
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f}"
                )


class SensorTrainer:
    """Entrena el sensor virtual (heads especializados)"""
    
    def __init__(self, encoder, moe, device='cpu'):
        self.device = device
        self.encoder = encoder
        self.moe = moe
        
        self.sensor = VirtualShockSensor(encoder, moe, latent_dim=32).to(device)
        
        # Congelar encoder y MoE
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.moe.parameters():
            param.requires_grad = False
        
        self.optimizer = optim.Adam(
            [
                {'params': self.sensor.shock_head.parameters()},
                {'params': self.sensor.intensity_head.parameters()},
                {'params': self.sensor.separation_head.parameters()},
            ],
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        
        self.criterion_shock = nn.BCELoss()  # Clasificación
        self.criterion_intensity = nn.MSELoss()  # Regresión
        self.loss_history = {'train': [], 'val': []}
    
    def train_epoch(self, train_loader, pseudo_labels):
        """
        Args:
            pseudo_labels: dict con 'shock', 'intensity', 'separation'
        """
        self.sensor.train()
        total_loss = 0
        
        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            
            # Obtener pseudo-labels para este batch
            batch_shock = pseudo_labels['shock'][batch_idx * len(X_batch):(batch_idx + 1) * len(X_batch)]
            batch_shock = torch.tensor(batch_shock, dtype=torch.float32).to(self.device)
            
            # Forward
            output = self.sensor(X_batch, compute_moe=False)
            
            # Loss
            loss_shock = self.criterion_shock(output['shock_prob'], batch_shock.unsqueeze(1))
            loss_total = loss_shock * TRAINING_CONFIG.get('loss_shock_weight', 1.0)
            
            self.optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.sensor.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss_total.item()
        
        avg_loss = total_loss / len(train_loader)
        self.loss_history['train'].append(avg_loss)
        return avg_loss


# Config for compatibility
MODEL_CONFIG = {
    'autoencoder': {
        'latent_dim': 32,
        'batch_norm': True,
        'dropout': 0.1,
    },
    'moe': {
        'expert_output_dim': 16,
        'num_experts': 4,
    },
}


if __name__ == '__main__':
    print("Training module loaded successfully")
