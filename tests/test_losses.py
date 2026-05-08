import torch

from cp_shock_project.training.losses import cp_loss_components, gradient_aware_loss


def test_losses_and_gradient_loss():
    true = torch.tensor([[0.0], [1.0], [2.0]])
    pred = true.clone()
    nidx = torch.tensor([[1], [0], [1]])
    dist = torch.ones(3, 1)
    assert gradient_aware_loss(pred, true, nidx, dist).item() == 0.0
    losses = cp_loss_components(
        {"Cp_pred": pred, "delta_Cp": torch.zeros_like(pred), "chi": torch.ones_like(pred)},
        {"Cp": true, "oracle_shock_score": torch.ones_like(pred)},
        lambdas={"global": 1.0, "shock": 1.0, "residual": 1.0},
    )
    assert losses["total"].item() == 0.0
