import torch

from cp_shock_project.features.fourier import FourierFeatureEncoder


def test_fourier_feature_shape():
    enc = FourierFeatureEncoder(input_dim=9, num_frequencies=4)
    x = torch.randn(5, 9)
    y = enc(x)
    assert y.shape == (5, enc.output_dim)
