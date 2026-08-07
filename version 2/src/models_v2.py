"""
models_v2.py — AeroSurrogate v2: spatial GNN backbone + ShockGatedMoE head.

Architecture:
  1. NodeEncoder:     16 → 128 → 256   (d=256 latent per node)
  2. EdgeEncoder:     5  → 64  → 128   (edge_dim=128)
  3. GNNLayer × 8:   message passing (h_src + edge_enc) → mean+max agg → h_i' + residual
  4. ShockIndicator:  d → 64 → 32 → 1 → sigmoid  (supervisado con BCE)
  5. ShockGatedMoE:   gate [h_i ‖ p_s] → 4 logits; 4 smooth experts + shock_expert; Cp only
  6. FrictionHead:    h_i → (log_mag, dir_norm) → Cf_vec [3]

Output: y_pred = [Cp, Cfx, Cfy, Cfz]  (compatible con protocolo de evaluación de v1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# ── helpers ─────────────────────────────────────────────────────────────────

def _mlp(dims, dropout=0.1, final_activation=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = (i == len(dims) - 2)
        if not is_last or final_activation:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def scatter_mean(src, idx, n):
    """Mean aggregation without PyG: src [E, d], idx [E] → [N, d]."""
    d   = src.size(1)
    out = torch.zeros(n, d, dtype=src.dtype, device=src.device)
    cnt = torch.zeros(n, 1, dtype=src.dtype, device=src.device)
    out.scatter_add_(0, idx.unsqueeze(1).expand(-1, d), src)
    cnt.scatter_add_(0, idx.unsqueeze(1), torch.ones(len(idx), 1, dtype=src.dtype, device=src.device))
    return out / cnt.clamp(min=1.0)


def scatter_max(src, idx, n):
    """Max aggregation without PyG: src [E, d], idx [E] → [N, d]."""
    d   = src.size(1)
    out = torch.full((n, d), float('-inf'), dtype=src.dtype, device=src.device)
    out.scatter_reduce_(0, idx.unsqueeze(1).expand(-1, d), src, reduce='amax', include_self=True)
    out = torch.where(out == float('-inf'), torch.zeros_like(out), out)
    return out


# ── GNN building blocks ──────────────────────────────────────────────────────

class GNNLayer(nn.Module):
    """
    One round of message passing (GraphSAGE-style with edge features).
    msg  = MLP([h_src ‖ edge_enc])         [E, d]
    agg  = mean(msg) ‖ max(msg)            [N, 2d]
    h'   = MLP([h_i ‖ agg])               [N, d]
    out  = LayerNorm(h' + h_i)            (residual)
    """
    def __init__(self, d, edge_enc_dim, dropout=0.1):
        super().__init__()
        self.msg_mlp = _mlp([d + edge_enc_dim, d, d], dropout=dropout)
        self.upd_mlp = _mlp([3 * d, d, d],            dropout=dropout)
        self.norm    = nn.LayerNorm(d)

    def forward(self, h, edge_index, edge_enc):
        src, dst = edge_index                         # each [E]
        n        = h.size(0)
        msgs     = self.msg_mlp(torch.cat([h[src], edge_enc], dim=-1))   # [E, d]
        mean_agg = scatter_mean(msgs, dst, n)         # [N, d]
        max_agg  = scatter_max(msgs,  dst, n)         # [N, d]
        h_new    = self.upd_mlp(torch.cat([h, mean_agg, max_agg], dim=-1))  # [N, d]
        return self.norm(h_new + h)


class GNNBackbone(nn.Module):
    """
    Spatial backbone: maps (node_feats, edge_index, edge_attr) → node latents h [N, d].

    node_feats  [N, 16]  (derived features from preprocessing)
    edge_index  [2, E]
    edge_attr   [E, 5]   (dx, dy, dz, dist, n_dot)
    """
    def __init__(self, node_in=16, edge_in=5, node_enc_dim=128,
                 d=256, edge_enc_dim=128, n_layers=8, dropout=0.1,
                 use_checkpoint=True):
        super().__init__()
        self.node_encoder    = _mlp([node_in, node_enc_dim, d], dropout=dropout)
        self.edge_encoder    = _mlp([edge_in, 64, edge_enc_dim], dropout=dropout)
        self.layers          = nn.ModuleList([
            GNNLayer(d, edge_enc_dim, dropout) for _ in range(n_layers)
        ])
        self.use_checkpoint  = use_checkpoint

    def forward(self, x, edge_index, edge_attr):
        h        = self.node_encoder(x)                  # [N, d]
        edge_enc = self.edge_encoder(edge_attr)           # [E, edge_enc_dim]
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # Re-compute activations on backward instead of storing them.
                # Reduces peak memory from O(n_layers) to O(1) at cost of one
                # extra forward pass per layer during backward.
                h = grad_checkpoint(layer, h, edge_index, edge_enc, use_reentrant=False)
            else:
                h = layer(h, edge_index, edge_enc)
        return h                                          # [N, d]


# ── ShockIndicator (spatial) ─────────────────────────────────────────────────

class ShockIndicatorSpatial(nn.Module):
    """
    ShockIndicator that operates on the spatial latent h_i (not raw features).
    Now implicitly aware of neighborhood context via the GNN backbone.
    Supervised by BCE: y_shock = 1[Cp_RANS < Cp_crit(M)] AND Mach > 0.75.
    """
    def __init__(self, d=256, hidden=None, dropout=0.1):
        super().__init__()
        if hidden is None:
            hidden = [128, 64, 32]
        self.network = _mlp([d] + hidden + [1], dropout=dropout)

    def forward(self, h):
        logit = self.network(h)
        return logit, torch.sigmoid(logit)


# ── ShockGatedMoE (adapted for latent input, Cp only) ────────────────────────

class ShockGatedMoEv2(nn.Module):
    """
    MoE head operating on GNN latent h_i. Predicts Cp only.
    Friction is handled by the dedicated FrictionHead.

    Gate: [h_i ‖ p_s] → 4 logits (Gumbel-Softmax during training).
    Experts: 4 smooth MLPs h_i → Cp.
    Shock residual: p_s · shock_expert(h_i) → Cp correction.
    """
    def __init__(self, d=256, num_experts=4, expert_hidden=None,
                 shock_expert_hidden=None, dropout=0.1):
        super().__init__()
        if expert_hidden is None:
            expert_hidden = [256, 256, 128]
        if shock_expert_hidden is None:
            shock_expert_hidden = [128, 128]

        self.gate = nn.Sequential(
            nn.Linear(d + 1, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 64),    nn.LayerNorm(64),  nn.SiLU(),
            nn.Linear(64, num_experts),
        )
        self.register_buffer('tau', torch.ones(1))

        self.experts = nn.ModuleList([
            _mlp([d] + expert_hidden + [1], dropout=dropout)
            for _ in range(num_experts)
        ])
        self.shock_expert = _mlp([d] + shock_expert_hidden + [1], dropout=dropout)

    def forward(self, h, shock_prob):
        gate_logits  = self.gate(torch.cat([h, shock_prob], dim=1))   # [N, 4]
        tau          = float(self.tau)
        if self.training:
            gates = F.gumbel_softmax(gate_logits, tau=tau, hard=False, dim=-1)
        else:
            gates = F.softmax(gate_logits / tau, dim=-1)

        expert_stack = torch.stack([e(h) for e in self.experts], dim=1)  # [N, 4, 1]
        smooth_out   = (gates.unsqueeze(-1) * expert_stack).sum(dim=1)   # [N, 1]
        output       = smooth_out + shock_prob * self.shock_expert(h)     # [N, 1]
        return output, gates


# ── FrictionHead ─────────────────────────────────────────────────────────────

class FrictionHead(nn.Module):
    """
    Dedicated head for friction vector [Cfx, Cfy, Cfz].
    Predicts (magnitude, unit direction) and reconstructs the vector.
    This decouples direction learning from magnitude learning, which is
    important because friction magnitude spans several orders of magnitude.
    """
    def __init__(self, d=256, hidden=None, dropout=0.1):
        super().__init__()
        if hidden is None:
            hidden = [256, 256, 128]
        self.mag_head = _mlp([d] + hidden + [1],  dropout=dropout)  # → log-magnitude
        self.dir_head = _mlp([d] + hidden + [3],  dropout=dropout)  # → unnorm direction

    def forward(self, h):
        log_mag  = self.mag_head(h)                          # [N, 1]
        mag      = torch.exp(log_mag.clamp(min=-10, max=5)) # [N, 1]
        dir_raw  = self.dir_head(h)                          # [N, 3]
        dir_norm = F.normalize(dir_raw, dim=-1, eps=1e-8)   # [N, 3]
        return mag * dir_norm                                 # [N, 3]


# ── Full AeroSurrogate v2 ─────────────────────────────────────────────────────

class AeroSurrogatev2(nn.Module):
    """
    Full model: GNNBackbone → ShockIndicatorSpatial → ShockGatedMoEv2 + FrictionHead.

    Forward returns dict:
      pred         [N, 4]  = [Cp, Cfx, Cfy, Cfz]  (compatible with v1 evaluation)
      shock_logit  [N, 1]
      shock_prob   [N, 1]
      gate_weights [N, E]
      cp_pred      [N, 1]
      cf_pred      [N, 3]
    """
    def __init__(self, cfg):
        super().__init__()
        d            = cfg['latent_dim']
        edge_enc_dim = cfg['edge_enc_dim']
        moe_cfg      = cfg['moe']
        fric_cfg     = cfg['friction']

        self.backbone = GNNBackbone(
            node_in=cfg['node_in_dim'],
            edge_in=cfg['edge_in_dim'],
            node_enc_dim=cfg['node_enc_dim'],
            d=d,
            edge_enc_dim=edge_enc_dim,
            n_layers=cfg['n_gnn_layers'],
        )
        self.shock_indicator = ShockIndicatorSpatial(
            d=d, hidden=moe_cfg['indicator_hidden'],
        )
        self.moe = ShockGatedMoEv2(
            d=d,
            num_experts=moe_cfg['num_experts'],
            expert_hidden=moe_cfg['expert_hidden'],
            shock_expert_hidden=moe_cfg['shock_expert_hidden'],
        )
        self.friction_head = FrictionHead(
            d=d, hidden=fric_cfg['hidden'],
        )

    def forward(self, x, edge_index, edge_attr):
        h              = self.backbone(x, edge_index, edge_attr)
        shock_logit, shock_prob = self.shock_indicator(h)
        cp_pred, gates = self.moe(h, shock_prob)
        cf_pred        = self.friction_head(h)
        pred           = torch.cat([cp_pred, cf_pred], dim=-1)   # [N, 4]
        return {
            'pred':         pred,
            'cp_pred':      cp_pred,
            'cf_pred':      cf_pred,
            'shock_logit':  shock_logit,
            'shock_prob':   shock_prob,
            'gate_weights': gates,
            'latent':       h,
        }
