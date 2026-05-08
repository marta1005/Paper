from __future__ import annotations

import torch
import torch.nn.functional as F


def gradient_aware_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_distances: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compare Cp jumps across kNN edges on a full tensor of points."""
    pred = pred.reshape(-1)
    true = true.reshape(-1)
    nidx = neighbor_indices.long()
    valid = (nidx >= 0) & torch.isfinite(neighbor_distances)
    safe = torch.where(valid, nidx, torch.zeros_like(nidx))
    dp = pred[safe] - pred.unsqueeze(1)
    dt = true[safe] - true.unsqueeze(1)
    err = torch.abs((dp - dt) / (neighbor_distances + eps))
    err = torch.where(valid, err, torch.zeros_like(err))
    return err.sum() / valid.sum().clamp_min(1)


def cp_loss_components(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    lambdas: dict[str, float] | None = None,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compute configurable Cp losses."""
    lambdas = lambdas or {}
    pred = outputs["Cp_pred"]
    true = batch["Cp"]
    chi = outputs.get("chi", batch.get("symbolic_chi"))
    oracle = batch.get("oracle_shock_score")
    delta = outputs.get("delta_Cp")
    losses: dict[str, torch.Tensor] = {}
    losses["global_mse"] = F.mse_loss(pred, true)
    losses["global_mae"] = F.l1_loss(pred, true)
    losses["huber"] = F.huber_loss(pred, true)
    if chi is not None:
        weight = 1.0 + float(lambdas.get("sensor_weight", 1.0)) * chi
        losses["weighted_mse"] = torch.mean(weight * (pred - true) ** 2)
    else:
        losses["weighted_mse"] = losses["global_mse"]
    if oracle is not None:
        losses["shock_mse"] = torch.sum(oracle * (pred - true) ** 2) / (torch.sum(oracle) + eps)
        non = 1.0 - oracle
        losses["nonshock_mse"] = torch.sum(non * (pred - true) ** 2) / (torch.sum(non) + eps)
    else:
        losses["shock_mse"] = losses["global_mse"]
        losses["nonshock_mse"] = losses["global_mse"]
    if delta is not None and chi is not None:
        losses["residual_reg"] = torch.mean((1.0 - chi) * delta**2)
    else:
        losses["residual_reg"] = torch.zeros((), dtype=pred.dtype, device=pred.device)
    total = (
        float(lambdas.get("global", 1.0)) * losses["global_mse"]
        + float(lambdas.get("mae", 0.0)) * losses["global_mae"]
        + float(lambdas.get("weighted", 0.0)) * losses["weighted_mse"]
        + float(lambdas.get("shock", 0.0)) * losses["shock_mse"]
        + float(lambdas.get("nonshock", 0.0)) * losses["nonshock_mse"]
        + float(lambdas.get("residual", 0.0)) * losses["residual_reg"]
    )
    losses["total"] = total
    return losses
