import torch

from cp_shock_project.features.fourier import FourierFeatureEncoder
from cp_shock_project.graph.message_passing import LocalGraphEncoder


def test_message_passing_shape():
    x = torch.randn(6, 9)
    nx = torch.randn(6, 3, 9)
    dist = torch.rand(6, 3)
    fourier = FourierFeatureEncoder(num_frequencies=2)
    phi = fourier(x)
    nphi = fourier(nx.reshape(-1, 9)).reshape(6, 3, -1)
    enc = LocalGraphEncoder(node_dim=9 + fourier.output_dim, hidden_dim=16, out_dim=12)
    out = enc(x, phi, nx, nphi, dist)
    assert out.shape == (6, 12)
