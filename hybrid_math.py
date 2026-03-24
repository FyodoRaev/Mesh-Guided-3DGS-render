from __future__ import annotations

import torch


def compute_depth_gate(
    gs_depth: torch.Tensor,
    mesh_depth: torch.Tensor,
    mesh_mask: torch.Tensor,
    beta: float,
    eps: float,
) -> torch.Tensor:
    gate_on_mesh = torch.sigmoid(beta * (mesh_depth - gs_depth - eps))
    return torch.where(mesh_mask.bool(), gate_on_mesh, torch.ones_like(gate_on_mesh))


def compose_hybrid(
    gs_rgb: torch.Tensor,
    gs_alpha: torch.Tensor,
    mesh_rgb: torch.Tensor,
    mesh_mask: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    support = torch.where(mesh_mask.bool().expand_as(mesh_rgb), mesh_rgb, torch.zeros_like(mesh_rgb))
    hybrid = gs_rgb * gate + (1.0 - gs_alpha * gate) * support
    return hybrid.clamp(0.0, 1.0)


def residual_weight_from_mesh_error(
    gt_rgb: torch.Tensor,
    mesh_rgb: torch.Tensor,
    mesh_mask: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    mesh_err = (mesh_rgb - gt_rgb).abs().mean(dim=-1, keepdim=True)
    w_mesh = (mesh_err / residual_scale).clamp(0.0, 1.0)
    return torch.where(mesh_mask.bool(), w_mesh, torch.ones_like(w_mesh)).detach()


def weighted_l1(pred: torch.Tensor, gt: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (weight * (pred - gt).abs()).sum() / (3.0 * weight.sum().clamp_min(1e-8))


def psnr(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor | None = None) -> float:
    err = (pred - gt) ** 2
    if valid is not None:
        valid = valid.float()
        err = err * valid
        mse = err.sum() / (3.0 * valid.sum()).clamp_min(1.0)
    else:
        mse = err.mean()
    return float(-10.0 * torch.log10(mse.clamp_min(1e-8)))


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    return (x * m).sum() / m.sum().clamp_min(1.0)


def _gradient_mag(x: torch.Tensor) -> torch.Tensor:
    dx = torch.zeros_like(x[..., :1])
    dy = torch.zeros_like(x[..., :1])
    dx[:, :, 1:, :] = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=-1, keepdim=True)
    dy[:, 1:, :, :] = (x[:, 1:, :, :] - x[:, :-1, :, :]).abs().mean(dim=-1, keepdim=True)
    return dx + dy


def _masked_quantile(v: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    vv = v[mask.bool()]
    if vv.numel() == 0:
        return torch.tensor(0.0, dtype=v.dtype, device=v.device)
    return torch.quantile(vv, float(q))


def selective_metrics(
    gt: torch.Tensor,
    mesh_rgb: torch.Tensor,
    hybrid_rgb: torch.Tensor,
    gs_alpha: torch.Tensor,
    gate: torch.Tensor,
    mesh_mask: torch.Tensor,
    good_q: float = 0.35,
    bad_q: float = 0.85,
) -> dict[str, float]:
    mesh_err = (mesh_rgb - gt).abs().mean(dim=-1, keepdim=True)
    hybrid_err = (hybrid_rgb - gt).abs().mean(dim=-1, keepdim=True)

    mask = mesh_mask.bool()
    th_good = _masked_quantile(mesh_err, mask, good_q)
    th_bad = _masked_quantile(mesh_err, mask, bad_q)
    good_region = mask & (mesh_err <= th_good)
    bad_region = mask & (mesh_err >= th_bad)

    repair_gain = _masked_mean((mesh_err - hybrid_err).clamp_min(0.0), bad_region)
    preserve_damage = _masked_mean((hybrid_err - mesh_err).clamp_min(0.0), good_region)

    alpha_eff = gs_alpha * gate
    leakage_good = _masked_mean(alpha_eff, good_region)

    grad_gt = _gradient_mag(gt)
    grad_mesh = _gradient_mag(mesh_rgb)
    grad_hybrid = _gradient_mag(hybrid_rgb)
    grad_mesh_err = (grad_mesh - grad_gt).abs()
    grad_hybrid_err = (grad_hybrid - grad_gt).abs()
    blur_regression_good = _masked_mean((grad_hybrid_err - grad_mesh_err).clamp_min(0.0), good_region)

    return {
        "repair_gain": float(repair_gain.item()),
        "preserve_damage": float(preserve_damage.item()),
        "selectivity_score": float((repair_gain - preserve_damage).item()),
        "leakage_good": float(leakage_good.item()),
        "blur_regression_good": float(blur_regression_good.item()),
        "good_region_ratio": float(good_region.float().mean().item()),
        "bad_region_ratio": float(bad_region.float().mean().item()),
    }
