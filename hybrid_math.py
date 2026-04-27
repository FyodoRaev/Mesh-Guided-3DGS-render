from __future__ import annotations

import torch


def compute_depth_gate(
    gs_depth: torch.Tensor,
    mesh_depth: torch.Tensor,
    mesh_mask: torch.Tensor,
    eps: float,
    band: float,
) -> torch.Tensor:
    margin = mesh_depth - gs_depth - eps
    hard = (margin >= 0.0).to(gs_depth.dtype)
    return torch.where(mesh_mask.bool(), hard, torch.ones_like(hard))


def compose_hybrid(
    gs_rgb: torch.Tensor,
    gs_alpha: torch.Tensor,
    mesh_rgb: torch.Tensor,
    mesh_mask: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    mesh = torch.where(mesh_mask.bool().expand_as(mesh_rgb), mesh_rgb, torch.zeros_like(mesh_rgb))
    return (gs_rgb * gate + (1.0 - gs_alpha * gate) * mesh).clamp(0.0, 1.0)


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = ((pred - gt) ** 2).mean()
    return float(-10.0 * torch.log10(mse.clamp_min(1e-8)))
