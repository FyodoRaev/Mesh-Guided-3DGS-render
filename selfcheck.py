from __future__ import annotations

import torch

from hybrid_math import compose_hybrid, compute_depth_gate, mesh_rgb_error


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

    mesh_good = mesh_rgb_error(gt, gt, torch.ones_like(a, dtype=torch.bool))
    mesh_bad = mesh_rgb_error(gt, torch.zeros_like(gt), torch.ones_like(a, dtype=torch.bool))
    assert mesh_good.item() == 0.0
    assert mesh_bad.item() > 0.0

    print("[OK] distillate selfcheck passed")


if __name__ == "__main__":
    main()
