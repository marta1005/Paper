import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from pathlib import Path
from config import TRAINING_CONFIG, MODEL_DIR, MODEL_CONFIG
from src.models import ShockAutoencoder, MixtureOfExperts, VirtualShockSensor, AeroSurrogate

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


class SurrogateTrainer:
    """
    Trains AeroSurrogate end-to-end with:
        L = L_aero + λ_shock * L_shock

    L_aero:  MSE on [Cp, Cfx, Cfy, Cfz]  — main task
    L_shock: BCE on ShockIndicator output  — physics supervision
             (labels from Cp_real < Cp_crit, only available during training via Y)

    At inference only X is needed; Y is never used.
    """

    def __init__(self, scaler, device='cpu', symbolic_sensor_path=None, save_name='surrogate_best.pt'):
        self.device = device
        cfg = MODEL_CONFIG['surrogate']

        self.model = AeroSurrogate(
            in_dim=MODEL_CONFIG['autoencoder']['input_dim'],
            num_experts=cfg['num_experts'],
            output_dim=cfg['output_dim'],
            indicator_hidden=cfg.get('indicator_hidden'),
            expert_hidden=cfg.get('expert_hidden'),
        ).to(device)

        # Symbolic gate mode: freeze ShockIndicator, train MoE only
        self.symbolic_sensor = None
        if symbolic_sensor_path:
            import pickle
            with open(symbolic_sensor_path, 'rb') as f:
                self.symbolic_sensor = pickle.load(f)
            if self.symbolic_sensor.get('clf') is None:
                raise ValueError("Symbolic sensor pkl missing 'clf' — re-run symbolic_regression.py --fallback")
            for p in self.model.shock_indicator.parameters():
                p.requires_grad_(False)
            logger.info(f"Symbolic gate loaded from {symbolic_sensor_path} — ShockIndicator frozen")

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay'],
        )

        self.criterion_mse       = nn.MSELoss()
        self.criterion_bce_shock = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([cfg.get('shock_pos_weight', 5.0)]).to(device)
        )
        self.shock_weight        = cfg.get('shock_weight', 0.1)
        self.load_balance_weight = cfg.get('load_balance_weight', 0.01)
        self.num_experts         = cfg.get('num_experts', 4)
        self.shock_mse_weight    = cfg.get('shock_mse_weight', 10.0)
        self.tau_start           = cfg.get('gumbel_tau_start', 1.0)
        self.tau_end             = cfg.get('gumbel_tau_end', 0.1)

        self._X_mean = torch.tensor(scaler['X_mean'], dtype=torch.float32)
        self._X_std  = torch.tensor(scaler['X_std'],  dtype=torch.float32)
        self._Y_mean = torch.tensor(scaler['Y_mean'], dtype=torch.float32)
        self._Y_std  = torch.tensor(scaler['Y_std'],  dtype=torch.float32)

        self.save_name     = save_name
        self.loss_history  = {'train': [], 'val': []}
        self.best_val_loss = float('inf')

    def _make_shock_label(self, X_batch, Y_batch):
        dev     = X_batch.device
        Cp_real = Y_batch[:, 0] * self._Y_std[0].to(dev) + self._Y_mean[0].to(dev)
        Cp_crit = X_batch[:, 13] * self._X_std[13].to(dev) + self._X_mean[13].to(dev)
        return (Cp_real < Cp_crit).float().unsqueeze(1)

    def _load_balance_loss(self, gate_weights):
        # Switch-Transformer style: penalises one expert absorbing all traffic.
        # gate_weights: [B, E] already softmax'd → mean over batch gives E fractions.
        # Loss = E * sum(mean_i^2); equals 1.0 when perfectly balanced.
        mean_gates = gate_weights.mean(0)          # [E]
        return self.num_experts * (mean_gates * mean_gates).sum()

    def _update_tau(self, epoch, num_epochs):
        """Exponentially anneal Gumbel-Softmax gate temperature from tau_start → tau_end."""
        progress = epoch / max(num_epochs - 1, 1)
        tau = self.tau_start * (self.tau_end / self.tau_start) ** progress
        self.model.moe.tau.fill_(tau)
        return tau

    def _loss(self, out, Y_batch, shock_label):
        # Shock-weighted MSE: errors at the shock front are penalised shock_mse_weight× more
        w            = 1.0 + self.shock_mse_weight * shock_label   # [B, 1]
        mse_weighted = (w * (out['pred'] - Y_batch) ** 2).mean()
        return (
            mse_weighted +
            self.shock_weight * self.criterion_bce_shock(out['shock_logit'], shock_label) +
            self.load_balance_weight * self._load_balance_loss(out['gate_weights'])
        )

    def _loss_symbolic(self, pred, gates, Y_batch, shock_label=None):
        """Loss when ShockIndicator is frozen — no BCE, only weighted MSE + load balance."""
        if shock_label is not None:
            w   = 1.0 + self.shock_mse_weight * shock_label
            mse = (w * (pred - Y_batch) ** 2).mean()
        else:
            mse = self.criterion_mse(pred, Y_batch)
        return mse + self.load_balance_weight * self._load_balance_loss(gates)

    def _symbolic_shock_prob(self, X_batch):
        """Compute shock probability from the frozen symbolic DT sensor."""
        import numpy as np
        dev   = X_batch.device
        X_raw = (X_batch.cpu() * self._X_std + self._X_mean).numpy()
        sr_idx = self.symbolic_sensor['sr_idx']
        proba  = self.symbolic_sensor['clf'].predict_proba(X_raw[:, sr_idx])[:, 1]
        sp_cal = self.symbolic_sensor['calibrator'].predict(proba).astype(np.float32)
        return torch.from_numpy(sp_cal[:, None]).to(dev)

    def _forward(self, X_batch):
        """Forward pass: use symbolic gate if loaded, else neural ShockIndicator."""
        if self.symbolic_sensor is not None:
            shock_prob = self._symbolic_shock_prob(X_batch)
            pred, gates = self.model.moe(X_batch, shock_prob)
            return {'pred': pred, 'shock_prob': shock_prob, 'gate_weights': gates}
        return self.model(X_batch)

    def train_epoch(self, train_loader):
        self.model.train()
        total = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch     = X_batch.to(self.device)
            Y_batch     = Y_batch.to(self.device)
            shock_label = self._make_shock_label(X_batch, Y_batch)

            out = self._forward(X_batch)

            if self.symbolic_sensor is not None:
                loss = self._loss_symbolic(out['pred'], out['gate_weights'], Y_batch, shock_label)
            else:
                loss = self._loss(out, Y_batch, shock_label)

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
            out     = self._forward(X_batch)
            # Plain MSE for checkpoint metric — tracks Cp quality directly,
            # independent of shock weight or load balance
            total += self.criterion_mse(out['pred'], Y_batch).item()
        avg = total / len(val_loader)
        self.loss_history['val'].append(avg)
        if avg < self.best_val_loss:
            self.best_val_loss = avg
            self.save_model(self.save_name)
        return avg

    def train(self, train_loader, val_loader, num_epochs=None):
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG['num_epochs']
        logger.info(
            f"Surrogate training for {num_epochs} epochs | "
            f"τ: {self.tau_start:.1f}→{self.tau_end:.2f} | "
            f"shock_mse_w={self.shock_mse_weight}"
        )
        for epoch in range(num_epochs):
            tau = self._update_tau(epoch, num_epochs)
            tr  = self.train_epoch(train_loader)
            if (epoch + 1) % TRAINING_CONFIG['validate_every'] == 0:
                vl = self.validate(val_loader)
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | τ={tau:.3f} | "
                    f"train={tr:.6f} val={vl:.6f} best={self.best_val_loss:.6f}"
                )
        logger.info("Surrogate training done")
        self.load_model()
        return self.model

    def save_model(self, name='surrogate_best.pt'):
        torch.save(self.model.state_dict(), MODEL_DIR / name)

    def load_model(self, name='surrogate_best.pt'):
        p = MODEL_DIR / name
        if p.exists():
            self.model.load_state_dict(torch.load(p, map_location=self.device))
