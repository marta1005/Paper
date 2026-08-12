"""
training.py — Training loop for AeroSurrogate v2.

Supports single-GPU and DDP (DistributedDataParallel).
Each step processes one full simulation (260k nodes).
"""
import os
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Loss functions ────────────────────────────────────────────────────────────

def friction_loss(cf_pred, cf_true_norm, y_std_cf, y_mean_cf):
    """
    Vector friction loss: magnitude (log-MSE) + direction (cosine).
    cf_pred      [N, 3]  normalised prediction
    cf_true_norm [N, 3]  normalised target
    """
    # Denormalise to physical for direction/magnitude decomposition
    cf_pred_phys = cf_pred * y_std_cf + y_mean_cf     # [N, 3]
    cf_true_phys = cf_true_norm * y_std_cf + y_mean_cf

    mag_pred = cf_pred_phys.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [N,1]
    mag_true = cf_true_phys.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [N,1]

    L_mag = F.mse_loss(torch.log(mag_pred), torch.log(mag_true))

    cos_sim = torch.cosine_similarity(cf_pred_phys, cf_true_phys, dim=-1)  # [N]
    # Weight direction loss by true magnitude (near-zero vectors → ambiguous direction)
    w_dir   = (mag_true.squeeze(1) / mag_true.squeeze(1).mean()).clamp(max=5.0)
    L_dir   = (w_dir * (1.0 - cos_sim)).mean()

    return L_mag + L_dir


import torch.nn.functional as F


def compute_loss(output, batch, cfg, scaler, device):
    """
    Full multi-task loss for one simulation.
    Returns (total_loss, loss_dict).
    """
    moe_cfg  = cfg['moe']
    fric_cfg = cfg['friction']

    Y_norm     = batch['y'].to(device)          # [N, 4]
    y_shock    = batch['y_shock'].to(device)    # [N]
    node_w     = batch['shock_node_w'].to(device)   # [N]
    sim_weight = batch['weight']

    cp_pred    = output['cp_pred']              # [N, 1]
    cf_pred    = output['cf_pred']              # [N, 3]
    gate_w     = output['gate_weights']         # [N, num_experts]
    shock_logit= output['shock_logit']          # [N, 1]

    Y_std  = torch.tensor(scaler['Y_std'],  dtype=torch.float32, device=device)
    Y_mean = torch.tensor(scaler['Y_mean'], dtype=torch.float32, device=device)

    # 1. Cp loss (shock-weighted MSE, × sim confidence weight)
    cp_true  = Y_norm[:, 0:1]
    cp_resid = (cp_pred - cp_true) ** 2                    # [N, 1]
    L_cp     = (node_w.unsqueeze(1) * cp_resid).mean() * sim_weight

    # 2. Shock sensor BCE
    pos_weight = torch.tensor(moe_cfg['shock_pos_weight'], device=device)
    L_shock    = F.binary_cross_entropy_with_logits(
        shock_logit.squeeze(1), y_shock, pos_weight=pos_weight
    )

    # 3. Load-balancing: encourage uniform usage across experts.
    # L_lb is the entropy of the mean gate: maximal (log num_experts) when all
    # experts are used equally, 0 when the gate collapses onto one expert.
    # It is SUBTRACTED from the total below so that minimising the loss
    # maximises the entropy. Adding it collapses the MoE onto a single expert.
    mean_gate = gate_w.mean(dim=0)                          # [num_experts]
    L_lb      = (mean_gate * torch.log(mean_gate + 1e-8)).sum().neg()   # entropy

    # 4. Friction loss (magnitude + direction)
    cf_true_norm  = Y_norm[:, 1:4]
    y_std_cf  = Y_std[1:4]
    y_mean_cf = Y_mean[1:4]
    L_fric = friction_loss(cf_pred, cf_true_norm, y_std_cf, y_mean_cf) * sim_weight

    total = (L_cp
             + moe_cfg['shock_weight']        * L_shock
             - moe_cfg['load_balance_weight'] * L_lb
             + fric_cfg['loss_weight']        * L_fric)

    return total, {
        'loss': total.item(),
        'L_cp': L_cp.item(),
        'L_shock': L_shock.item(),
        'L_lb': L_lb.item(),
        'L_fric': L_fric.item(),
    }


# ── Trainer ───────────────────────────────────────────────────────────────────

class Surrogatev2Trainer:
    def __init__(self, model, cfg, scaler, device,
                 save_name='surrogate_v2_best.pt',
                 rank=0, world_size=1):
        self.model      = model
        self.cfg        = cfg
        self.scaler     = scaler
        self.device     = device
        self.save_name  = save_name
        self.rank       = rank
        self.world_size = world_size
        self.is_main    = (rank == 0)

        train_cfg = cfg['training']
        # Only optimise trainable params, so --freeze-backbone genuinely keeps
        # the backbone out of AdamW (and out of its optimiser state).
        self.backbone_frozen = not any(p.requires_grad
                                       for p in model.backbone.parameters())
        self.optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay'],
        )

        # Cosine annealing with linear warmup
        self.num_epochs   = train_cfg['num_epochs']
        self.warmup_steps = train_cfg['warmup_epochs']
        self.scheduler    = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs - self.warmup_steps,
            eta_min=1e-6,
        )

        self.grad_clip = train_cfg['grad_clip']
        self.best_r2   = -float('inf')
        self.patience  = train_cfg['early_stopping_patience']
        self.patience_cnt = 0

        # Gumbel temperature annealing
        moe_cfg  = cfg['model']['moe']
        self.tau_start = moe_cfg['gumbel_tau_start']
        self.tau_end   = moe_cfg['gumbel_tau_end']

    def _set_tau(self, epoch):
        frac = min(epoch / max(self.num_epochs - 1, 1), 1.0)
        tau  = self.tau_start + frac * (self.tau_end - self.tau_start)
        self.model.moe.tau.fill_(tau)

    def _warmup_lr(self, epoch):
        if epoch < self.warmup_steps:
            factor = (epoch + 1) / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.cfg['training']['learning_rate'] * factor

    def train_epoch(self, loader, epoch):
        self.model.train()
        if self.backbone_frozen:
            # Keep dropout/norm in the frozen backbone deterministic — its
            # features no longer adapt, so training noise there is pure noise.
            self.model.backbone.eval()
        self._set_tau(epoch)
        self._warmup_lr(epoch)

        totals = {'loss': 0, 'L_cp': 0, 'L_shock': 0, 'L_lb': 0, 'L_fric': 0}
        n_sims = 0

        for batch in loader:
            x          = batch['x'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            edge_attr  = batch['edge_attr'].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x, edge_index, edge_attr)
            loss, ld = compute_loss(output, batch, self.cfg['model'], self.scaler, self.device)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            for k in totals:
                totals[k] += ld.get(k, 0)
            n_sims += 1

        if epoch >= self.warmup_steps:
            self.scheduler.step()

        return {k: v / max(n_sims, 1) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self, val_dataset, val_sim_indices):
        """Quick validation on a subset of simulations."""
        self.model.eval()
        all_cp_true, all_cp_pred = [], []
        all_cf_true, all_cf_pred = [], []

        Y_std  = np.array(self.scaler['Y_std'],  dtype=np.float32)
        Y_mean = np.array(self.scaler['Y_mean'], dtype=np.float32)

        for idx in val_sim_indices:
            batch = val_dataset[idx]
            x          = batch['x'].unsqueeze(0).to(self.device) if batch['x'].dim() == 2 else batch['x'].to(self.device)
            x          = batch['x'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            edge_attr  = batch['edge_attr'].to(self.device)

            out    = self.model(x, edge_index, edge_attr)
            cp_n   = out['cp_pred'].squeeze(1).cpu().numpy()
            cf_n   = out['cf_pred'].cpu().numpy()

            cp_true_n = batch['y'][:, 0].numpy()
            cf_true_n = batch['y'][:, 1:4].numpy()

            # Denormalise
            cp_pred = cp_n * Y_std[0] + Y_mean[0]
            cp_true = cp_true_n * Y_std[0] + Y_mean[0]
            cf_pred = cf_n * Y_std[1:4] + Y_mean[1:4]
            cf_true = cf_true_n * Y_std[1:4] + Y_mean[1:4]

            all_cp_true.append(cp_true)
            all_cp_pred.append(cp_pred)
            all_cf_true.append(cf_true)
            all_cf_pred.append(cf_pred)

        cp_true = np.concatenate(all_cp_true)
        cp_pred = np.concatenate(all_cp_pred)
        cf_true = np.vstack(all_cf_true)
        cf_pred = np.vstack(all_cf_pred)

        def r2(y, yp):
            ss_res = np.sum((y - yp) ** 2, axis=0)
            ss_tot = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
            return 1.0 - ss_res / (ss_tot + 1e-12)

        r2_cp = float(r2(cp_true, cp_pred))
        r2_cf = r2(cf_true, cf_pred)
        return r2_cp, r2_cf

    def train(self, train_dataset, val_dataset, val_sim_indices,
              validate_every=2):
        from torch.utils.data import DataLoader
        from .dataset import collate_single

        sampler = (DistributedSampler(train_dataset, shuffle=True)
                   if self.world_size > 1 else RandomSampler(train_dataset))
        loader  = DataLoader(train_dataset, batch_size=1, sampler=sampler,
                             collate_fn=collate_single, num_workers=0, pin_memory=False)

        model_dir = Path(self.cfg['dirs']['model_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.num_epochs):
            if self.world_size > 1:
                sampler.set_epoch(epoch)

            t0   = time.time()
            ld   = self.train_epoch(loader, epoch)
            dt   = time.time() - t0

            if self.is_main:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch+1:3d}/{self.num_epochs}  "
                    f"loss={ld['loss']:.4f}  L_cp={ld['L_cp']:.4f}  "
                    f"L_shock={ld['L_shock']:.4f}  L_lb={ld['L_lb']:.4f}  "
                    f"L_fric={ld['L_fric']:.4f}  lr={lr:.2e}  t={dt:.0f}s"
                )

            if self.is_main and (epoch + 1) % validate_every == 0:
                r2_cp, r2_cf = self.validate(val_dataset, val_sim_indices)
                logger.info(
                    f"  Val  R²(Cp)={r2_cp:.4f}  "
                    f"R²(Cfx)={r2_cf[0]:.4f}  R²(Cfy)={r2_cf[1]:.4f}  R²(Cfz)={r2_cf[2]:.4f}"
                )

                if r2_cp > self.best_r2:
                    self.best_r2     = r2_cp
                    self.patience_cnt = 0
                    save_path = model_dir / self.save_name
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"  → Saved best model (R²(Cp)={r2_cp:.4f}) to {save_path}")
                else:
                    self.patience_cnt += 1
                    if self.patience_cnt >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

        return self.best_r2
