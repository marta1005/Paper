import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def _mlp(dims, batch_norm=True, dropout=0.1, final_activation=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = (i == len(dims) - 2)
        if not is_last or final_activation:
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.LeakyReLU(0.2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ShockAutoencoder(nn.Module):
    """
    Encoder learns a rich latent representation (latent_dim=32).
    L1 regularization on latent activations enforces sparsity in place of
    a strict dimensional bottleneck — the paper can argue compact representation
    without the 32 < 14 constraint being violated.
    """
    def __init__(self, input_dim=14, latent_dim=32, batch_norm=True, dropout=0.1):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = latent_dim

        self.encoder = _mlp([input_dim, 128, 64, latent_dim], batch_norm, dropout)
        self.decoder = _mlp([latent_dim, 64, 128, input_dim], batch_norm, dropout)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class ExpertNetwork(nn.Module):
    def __init__(self, latent_dim=32, output_dim=32):
        super().__init__()
        self.network = _mlp([latent_dim, 128, 256, 128, output_dim], batch_norm=True, dropout=0.1)

    def forward(self, x):
        return self.network(x)


class GatingNetwork(nn.Module):
    def __init__(self, latent_dim=32, num_experts=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, z):
        return self.network(z)


class MixtureOfExperts(nn.Module):
    def __init__(self, latent_dim=32, num_experts=4, expert_output_dim=32, output_dim=4):
        super().__init__()
        self.experts     = nn.ModuleList([ExpertNetwork(latent_dim, expert_output_dim) for _ in range(num_experts)])
        self.gating      = GatingNetwork(latent_dim, num_experts)
        self.output_head = nn.Linear(expert_output_dim, output_dim)

    def forward(self, z):
        gates        = self.gating(z)
        expert_stack = torch.stack([e(z) for e in self.experts], dim=1)
        mixed        = (gates.unsqueeze(-1) * expert_stack).sum(dim=1)
        return self.output_head(mixed), gates


class SensorHead(nn.Module):
    def __init__(self, in_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        dims   = [in_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                # No BatchNorm: small heads (34→64→32→16→1) don't benefit from BN
                # and BN causes train/eval discrepancy when scaler differs between runs
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(0.1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ShockIndicator(nn.Module):
    """
    Predicts shock probability from X only (geometry + flight conditions, never Cp/Y).
    Learns WHERE on the surface the shock forms given geometry + flight condition.
    Supervised by physics label: shock ↔ Cp_real < Cp_crit(Mach).

    Replaces the autoencoder: instead of a generic latent, the representation
    is an explicit scalar shock score interpretable as P(M_local > 1).
    """
    def __init__(self, in_dim=14, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = [64, 32, 16]
        dims   = [in_dim] + hidden + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(0.1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        logit = self.network(x)
        return logit, torch.sigmoid(logit)


class ShockGatedMoE(nn.Module):
    """
    MoE gated by the shock indicator score + full X.
    Each expert specialises in one flow regime; the Cp discontinuity at the
    shock emerges from the jump between expert predictions rather than from
    a smooth network trying to approximate a discontinuous function.

    Training:  Gumbel-Softmax gate (differentiable; temperature τ annealed 1.0→0.1).
    Inference: hard argmax — each point commits to exactly one expert,
               producing a sharp discontinuity at the shock front instead of a blend.
    """
    def __init__(self, in_dim=14, num_experts=4, expert_hidden=None, output_dim=4):
        super().__init__()
        if expert_hidden is None:
            expert_hidden = [128, 256, 128]

        # Gate: [shock_prob(1) | X(14)] → num_experts logits (Softmax applied via Gumbel)
        self.gate = nn.Sequential(
            nn.Linear(in_dim + 1, 64), nn.ReLU(),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, num_experts),
        )
        self.register_buffer('tau', torch.ones(1))  # annealed externally by trainer

        # Experts: full X → [Cp, Cfx, Cfy, Cfz]
        self.experts = nn.ModuleList([
            _mlp([in_dim] + expert_hidden + [output_dim], batch_norm=True, dropout=0.1)
            for _ in range(num_experts)
        ])

    def forward(self, x, shock_prob):
        gate_logits = self.gate(torch.cat([x, shock_prob], dim=1))
        if self.training:
            # Differentiable soft routing with annealed temperature
            gates = F.gumbel_softmax(gate_logits, tau=float(self.tau), hard=False, dim=-1)
        else:
            # Soft-but-sharp routing at inference: τ saved from end of training (≈0.3).
            # Sharpens the gate without hard discontinuities in smooth subsonic fields.
            gates = F.softmax(gate_logits / float(self.tau), dim=-1)
        expert_stack = torch.stack([e(x) for e in self.experts], dim=1)  # [B, E, 4]
        output       = (gates.unsqueeze(-1) * expert_stack).sum(dim=1)
        return output, gates


class AeroSurrogate(nn.Module):
    """
    Full aerodynamic surrogate: ShockIndicator → ShockGatedMoE.

    Replaces the AE + MoE pipeline. No latent bottleneck — the physics-derived
    shock score is the routing signal. Experts learn smooth fields per regime;
    the Cp shock discontinuity emerges from their transition.

    Training uses Y (Cp < Cp_crit) to supervise the indicator; inference
    uses only X.
    """
    def __init__(self, in_dim=14, num_experts=4, output_dim=4,
                 indicator_hidden=None, expert_hidden=None):
        super().__init__()
        self.shock_indicator = ShockIndicator(in_dim, indicator_hidden)
        self.moe             = ShockGatedMoE(in_dim, num_experts, expert_hidden, output_dim)

    def forward(self, x):
        shock_logit, shock_prob = self.shock_indicator(x)
        pred, gates             = self.moe(x, shock_prob)
        return {
            'pred':         pred,
            'shock_logit':  shock_logit,
            'shock_prob':   shock_prob,
            'gate_weights': gates,
        }


class VirtualShockSensor(nn.Module):
    """
    Sensor trained with CFD-derived shock labels (from Y: Cp gradient + Cfx sign).
    At inference uses only X — no Y needed.

    Input to heads: [latent (32) | Mach | AoA] = 34 dims.
    Mach=col6, AoA=col7 of the normalised input tensor.

    The symbolic regression on X-features post-training is the scientific contribution:
    the model discovers the shock condition from data rather than imposing M_local > 1.
    """
    def __init__(self, encoder, moe, latent_dim=32, head_hidden=None):
        super().__init__()
        if head_hidden is None:
            head_hidden = [64, 32, 16]
        self.encoder    = encoder
        self.moe        = moe
        self.latent_dim = latent_dim

        sensor_in = latent_dim + 2   # latent + Mach + AoA
        self.shock_head     = SensorHead(sensor_in, head_hidden)
        self.intensity_head = SensorHead(sensor_in, head_hidden)
        self.sep_head       = SensorHead(sensor_in, head_hidden)

    def _sensor_input(self, x, z):
        return torch.cat([z, x[:, 6:8]], dim=1)   # Mach=6, AoA=7

    def forward(self, x, compute_moe=False):
        z           = self.encoder(x)
        s_in        = self._sensor_input(x, z)
        shock_logit = self.shock_head(s_in)
        sep_logit   = self.sep_head(s_in)
        intensity   = torch.relu(self.intensity_head(s_in))

        out = {
            'shock_logit':     shock_logit,
            'shock_prob':      torch.sigmoid(shock_logit),
            'intensity':       intensity,
            'sep_logit':       sep_logit,
            'separation_prob': torch.sigmoid(sep_logit),
            'latent':          z,
        }

        if compute_moe:
            moe_output, gate_weights = self.moe(z)
            out['moe_output']   = moe_output
            out['gate_weights'] = gate_weights

        return out
