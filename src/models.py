import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def _mlp(dims, batch_norm=True, dropout=0.1, final_activation=False):
    # batch_norm param kept for API compatibility; now uses LayerNorm+SiLU (no train/eval gap)
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
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.SiLU())
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
    def __init__(self, in_dim=14, num_experts=4, expert_hidden=None, output_dim=4,
                 shock_expert_hidden=None, disable_shock_expert=False):
        super().__init__()
        if expert_hidden is None:
            expert_hidden = [128, 256, 128]
        if shock_expert_hidden is None:
            shock_expert_hidden = [128, 128]
        self.disable_shock_expert = disable_shock_expert

        # Gate: [shock_prob(1) | X(14)] → num_experts logits (Softmax applied via Gumbel)
        self.gate = nn.Sequential(
            nn.Linear(in_dim + 1, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, 32),         nn.LayerNorm(32), nn.SiLU(),
            nn.Linear(32, num_experts),
        )
        self.register_buffer('tau', torch.ones(1))       # annealed externally by trainer
        self.register_buffer('mach_mean', torch.zeros(1))  # set from scaler by trainer
        self.register_buffer('mach_std',  torch.ones(1))

        # Smooth experts: full X → [Cp, Cfx, Cfy, Cfz]
        self.experts = nn.ModuleList([
            _mlp([in_dim] + expert_hidden + [output_dim], dropout=0.1)
            for _ in range(num_experts)
        ])

        # Shock-gated residual expert: explicit discontinuity term, scaled by shock_prob
        self.shock_expert = _mlp([in_dim] + shock_expert_hidden + [output_dim], dropout=0.1)

    def forward(self, x, shock_prob):
        gate_logits = self.gate(torch.cat([x, shock_prob], dim=1))
        if self.training:
            # Mach-gated Gumbel: stochastic routing only for transonic points (Mach > 0.75).
            # Subsonic points use standard softmax — no noise on smooth Cp fields.
            Mach_real    = x[:, 6:7] * self.mach_std + self.mach_mean   # [B, 1]
            is_transonic = (Mach_real > 0.75)                            # [B, 1] bool
            tau = float(self.tau)
            gates_gumbel = F.gumbel_softmax(gate_logits, tau=tau, hard=False, dim=-1)
            gates_soft   = F.softmax(gate_logits / tau, dim=-1)
            gates        = torch.where(is_transonic, gates_gumbel, gates_soft)
        else:
            # Soft-but-sharp at inference: deterministic, τ≈0.3 saved from end of training.
            gates = F.softmax(gate_logits / float(self.tau), dim=-1)
        expert_stack = torch.stack([e(x) for e in self.experts], dim=1)  # [B, E, 4]
        smooth_out   = (gates.unsqueeze(-1) * expert_stack).sum(dim=1)
        # Shock-gated residual: shock_prob in [0,1] gates the explicit discontinuity term
        if self.disable_shock_expert:
            output = smooth_out
        else:
            output = smooth_out + shock_prob * self.shock_expert(x)
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
                 indicator_hidden=None, expert_hidden=None, shock_expert_hidden=None,
                 disable_shock_expert=False):
        super().__init__()
        self.shock_indicator = ShockIndicator(in_dim, indicator_hidden)
        self.moe             = ShockGatedMoE(in_dim, num_experts, expert_hidden, output_dim,
                                             shock_expert_hidden,
                                             disable_shock_expert=disable_shock_expert)

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


class PySRWrapper:
    """
    Wraps a PySR equation as an sklearn-compatible predict_proba interface.
    Pure numpy+sympy at inference — no Julia/PySR required on the server.

    Defined here (not in symbolic_regression.py) so pickle can resolve the
    class path as 'src.models.PySRWrapper' without importing PySR.
    """
    def __init__(self, fn, expr_str, feature_names):
        self._fn           = fn
        self.expr_str      = expr_str
        self.feature_names = list(feature_names)

    def predict_proba(self, X):
        import numpy as np
        args = [X[:, i] for i in range(X.shape[1])]
        raw  = np.asarray(self._fn(*args), dtype=np.float64).ravel()
        raw  = np.clip(raw, 0.0, None)
        return np.column_stack([np.zeros(len(raw)), raw])

    def __getstate__(self):
        return {'expr_str': self.expr_str, 'feature_names': self.feature_names}

    def __setstate__(self, state):
        import numpy as _np
        self.expr_str      = state['expr_str']
        self.feature_names = state['feature_names']
        _ns = {fn: getattr(_np, fn)
               for fn in ['exp', 'log', 'sqrt', 'sin', 'cos', 'tan',
                          'tanh', 'sinh', 'cosh', 'abs', 'sign']}
        _ns['maximum'] = _np.maximum   # needed for span_norm-protected formula
        _expr  = self.expr_str
        _names = self.feature_names
        def _fn(*args):
            local = dict(zip(_names, args))
            with _np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                result = eval(_expr, _ns, local)   # noqa: S307
            return _np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
        self._fn = _fn
