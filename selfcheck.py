from __future__ import annotations

import torch

from hybrid_math import compose_hybrid, compute_depth_gate, residual_weight_from_mesh_error, weighted_l1


def main():
    gs_depth = torch.tensor([[[[1.0]], [[2.0]]]])
    mesh_depth = torch.tensor([[[[2.0]], [[1.0]]]])
    mask = torch.ones_like(gs_depth, dtype=torch.bool)
    gate = compute_depth_gate(gs_depth, mesh_depth, mask, beta=100.0, eps=0.0)
    assert gate[0, 0, 0, 0] > 0.99
    assert gate[0, 1, 0, 0] < 0.01

    M = torch.tensor([[[[0.8, 0.4, 0.2]]]])
    a = torch.tensor([[[[0.25]]]])
    gt = M.clone()

    # If gs_rgb is premultiplied (as in gsplat), this must reconstruct M exactly.
    h1 = compose_hybrid(
        torch.zeros_like(M),
        torch.zeros_like(a),
        M,
        torch.ones_like(a, dtype=torch.bool),
        torch.ones_like(a),
    )
    h2 = compose_hybrid(
        a * M,
        a,
        M,
        torch.ones_like(a, dtype=torch.bool),
        torch.ones_like(a),
    )
    assert torch.allclose(h1, gt, atol=1e-6)
    assert torch.allclose(h2, gt, atol=1e-6)

    w_good = residual_weight_from_mesh_error(gt, gt, torch.ones_like(a, dtype=torch.bool), 0.05)
    w_bad = residual_weight_from_mesh_error(gt, torch.zeros_like(gt), torch.ones_like(a, dtype=torch.bool), 0.05)
    assert w_good.item() == 0.0
    assert w_bad.item() == 1.0

    l = weighted_l1(torch.ones((1, 1, 1, 3)), torch.zeros((1, 1, 1, 3)), torch.ones((1, 1, 1, 1)))
    assert abs(l.item() - 1.0) < 1e-6

    print("[OK] distillate selfcheck passed")


if __name__ == "__main__":
    main()
