"""
Modelos: Autoencoder + Mixture of Experts + Sensor Virtual
"""
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ShockAutoencoder(nn.Module):
    """
    Autoencoder para capturar física de choques en espacio latente
    """
    
    def __init__(self, input_dim=19, latent_dim=32, batch_norm=True, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # ENCODER
        encoder_layers = []
        dims = [input_dim, 128, 64, 32, latent_dim]
        
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            if batch_norm and i < len(dims) - 2:
                encoder_layers.append(nn.BatchNorm1d(dims[i+1]))
            if i < len(dims) - 2:
                encoder_layers.append(nn.LeakyReLU(0.2))
                if dropout > 0:
                    encoder_layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # DECODER (espejo del encoder)
        decoder_layers = []
        dims_dec = [latent_dim, 32, 64, 128, input_dim]
        
        for i in range(len(dims_dec) - 1):
            decoder_layers.append(nn.Linear(dims_dec[i], dims_dec[i+1]))
            if batch_norm and i < len(dims_dec) - 2:
                decoder_layers.append(nn.BatchNorm1d(dims_dec[i+1]))
            if i < len(dims_dec) - 2:
                decoder_layers.append(nn.LeakyReLU(0.2))
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class ExpertNetwork(nn.Module):
    """
    Red especializada en un régimen físico
    """
    
    def __init__(self, input_dim=32, output_dim=16, hidden_dims=[64, 128, 64]):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(0.1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class GatingNetwork(nn.Module):
    """
    Red de gating que asigna puntos a expertos
    Entrada: espacio latente + indicadores físicos
    """
    
    def __init__(self, latent_dim=32, num_experts=4, num_physical_features=3):
        super().__init__()
        
        input_dim = latent_dim + num_physical_features
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, z, physical_features):
        """
        Args:
            z: latent space [batch, latent_dim]
            physical_features: [batch, num_physical_features]
                - shock_indicator, separation_risk, mach_local
        """
        x = torch.cat([z, physical_features], dim=1)
        return self.network(x)


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts para múltiples regímenes físicos.
    Los expertos producen features intermedias (expert_output_dim);
    output_head las proyecta a coeficientes aerodinámicos (output_dim=4).
    """

    def __init__(self, latent_dim=32, num_experts=4, expert_output_dim=16, output_dim=4):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.expert_output_dim = expert_output_dim
        self.output_dim = output_dim

        self.experts = nn.ModuleList([
            ExpertNetwork(latent_dim, expert_output_dim)
            for _ in range(num_experts)
        ])

        self.gating = GatingNetwork(latent_dim, num_experts, num_physical_features=3)
        self.output_head = nn.Linear(expert_output_dim, output_dim)

    def forward(self, z, shock_indicator, separation_risk, mach_local):
        """
        Args:
            z: espacio latente [batch, latent_dim]
            shock_indicator: [batch, 1]
            separation_risk: [batch, 1]
            mach_local: [batch, 1]

        Returns:
            output: [batch, output_dim]  — Cp, Cfx, Cfy, Cfz
            gate_weights: [batch, num_experts]
        """
        physical_features = torch.cat([
            shock_indicator, separation_risk, mach_local
        ], dim=1)

        gate_weights = self.gating(z, physical_features)

        expert_outputs = [expert(z) for expert in self.experts]
        expert_stack = torch.stack(expert_outputs, dim=1)

        mixed = (gate_weights.unsqueeze(-1) * expert_stack).sum(dim=1)
        output = self.output_head(mixed)

        return output, gate_weights


class ShockClassificationHead(nn.Module):
    """
    Head para clasificación de choque
    """
    
    def __init__(self, input_dim=32, hidden_dims=[16, 8]):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
        
        layers.append(nn.Sigmoid())  # Probabilidad
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class IntensityRegressionHead(nn.Module):
    """
    Head para regresión de intensidad de choque
    """
    
    def __init__(self, input_dim=32, hidden_dims=[16, 8]):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.relu(self.network(x))  # Intensidad ≥ 0


class VirtualShockSensor(nn.Module):
    """
    Sensor virtual que integra autoencoder + MoE + heads especializados
    """
    
    def __init__(self, encoder, moe, latent_dim=32):
        super().__init__()
        
        self.encoder = encoder
        self.moe = moe
        
        # Heads de predicción
        self.shock_head = ShockClassificationHead(input_dim=latent_dim)
        self.intensity_head = IntensityRegressionHead(input_dim=latent_dim)
        self.separation_head = ShockClassificationHead(input_dim=latent_dim)
    
    def forward(self, x, compute_moe=False):
        """
        Args:
            x: input features [batch, 19]
               - 0-8: original features
               - 9-18: derived features
            compute_moe: bool, incluir MoE en forward
        
        Returns:
            {
                'shock_prob': [batch, 1],
                'intensity': [batch, 1],
                'separation_prob': [batch, 1],
                'latent': [batch, latent_dim],
                'moe_output': [batch, 4] (Cp,Cfx,Cfy,Cfz si compute_moe=True),
                'gate_weights': [batch, num_experts] (si compute_moe=True),
            }
        """
        # Codificar
        z = self.encoder(x)
        
        # Predicciones desde latent space
        shock_prob = self.shock_head(z)
        intensity = self.intensity_head(z)
        separation_prob = self.separation_head(z)
        
        output = {
            'shock_prob': shock_prob,
            'intensity': intensity,
            'separation_prob': separation_prob,
            'latent': z,
        }
        
        # MoE (opcional, requiere indicadores físicos)
        if compute_moe:
            # Índices según preprocessing.py (X_derived columnas 9-18)
            # 9=M_local, 10=grad_p, 11=cp_loss, 12=shock_indicator, 13=Cf_mag
            shock_indicator = x[:, 12:13]   # shock_indicator
            separation_risk = x[:, 13:14]   # Cf_mag: proxy de riesgo de separación
            mach_local      = x[:, 9:10]    # M_local
            
            moe_output, gate_weights = self.moe(
                z, shock_indicator, separation_risk, mach_local
            )
            
            output['moe_output'] = moe_output
            output['gate_weights'] = gate_weights
        
        return output


class ReconstructionLoss(nn.Module):
    """
    Loss mejorada para autoencoder que penaliza más en regiones críticas
    """
    
    def __init__(self, weight_high_gradient=10.0):
        super().__init__()
        self.weight_high_gradient = weight_high_gradient
    
    def forward(self, x_real, x_reconstructed, gradient_metric=None):
        """
        Args:
            x_real: valores reales [batch, features]
            x_reconstructed: reconstruidos
            gradient_metric: [batch, 1] indicador de importancia (ej: gradiente de presión)
                Si None, usa MSE estándar
        """
        base_loss = (x_real - x_reconstructed) ** 2
        
        if gradient_metric is not None:
            # Normalizar gradient metric de forma robusta
            grad_mean = gradient_metric.mean()
            grad_std = gradient_metric.std() + 1e-8
            grad_norm = (gradient_metric - grad_mean) / grad_std
            # Clip para evitar valores extremos
            grad_norm = torch.clamp(grad_norm, -3.0, 3.0)
            
            # Amplificar peso en regiones de alto gradiente (siempre positivo)
            weight = 1.0 + self.weight_high_gradient * torch.relu(grad_norm)
            weight = torch.clamp(weight, min=0.1, max=100.0)  # Prevenir extremos
            weighted_loss = weight * base_loss
            
            return weighted_loss.mean()
        else:
            return base_loss.mean()


if __name__ == '__main__':
    device = 'cpu'
    
    # Test de modelos
    batch_size = 32
    input_dim = 19
    latent_dim = 32
    
    # Autoencoder
    ae = ShockAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    ae = ae.to(device)
    
    # MoE
    moe = MixtureOfExperts(latent_dim=latent_dim, num_experts=4, expert_output_dim=16)
    moe = moe.to(device)
    
    # Sensor
    sensor = VirtualShockSensor(ae.encoder, moe, latent_dim=latent_dim)
    sensor = sensor.to(device)
    
    # Forward pass
    x = torch.randn(batch_size, input_dim).to(device)
    
    x_recon, z = ae(x)
    print(f"AE: x.shape={x.shape} -> z.shape={z.shape} -> x_recon.shape={x_recon.shape}")
    
    output = sensor(x, compute_moe=True)
    print(f"Sensor output keys: {output.keys()}")
    print(f"  shock_prob.shape: {output['shock_prob'].shape}")
    print(f"  intensity.shape: {output['intensity'].shape}")
    print(f"  latent.shape: {output['latent'].shape}")
    if 'gate_weights' in output:
        print(f"  gate_weights.shape: {output['gate_weights'].shape}")
