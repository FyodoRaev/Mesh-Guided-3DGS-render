from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


def _save_rgb(path: Path, x: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = x.detach().cpu().float().numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    imageio.imwrite(path, (arr * 255.0).astype(np.uint8))


def _mono_to_rgb(x: torch.Tensor, vmin: float = 0.0, vmax: float = 1.0):
    y = (x - vmin) / max(vmax - vmin, 1e-8)
    return y.clamp(0.0, 1.0).repeat(1, 1, 3)


def save_visual_pack(out: dict, out_dir: Path, image_idx: int):
    """
    Minimal diagnostics focused on selective metrics interpretation.
    Required keys: gt, mesh_rgb, hybrid, weight, gate, mesh_mask
    Shape convention: [B,H,W,C], B>=1
    """
    gt = out["gt"][0]
    mesh = out["mesh_rgb"][0]
    hybrid = out["hybrid"][0]
    weight = out["weight"][0]
    gate = out["gate"][0]
    mesh_mask = out["mesh_mask"][0].float()

    mesh_err = (mesh - gt).abs().mean(dim=-1, keepdim=True)
    hybrid_err = (hybrid - gt).abs().mean(dim=-1, keepdim=True)

    better_mag = (mesh_err - hybrid_err).clamp_min(0.0) * mesh_mask
    worse_mag = (hybrid_err - mesh_err).clamp_min(0.0) * mesh_mask
    max_delta = float(torch.quantile(torch.cat([better_mag.flatten(), worse_mag.flatten()]), 0.98).item())
    max_delta = max(max_delta, 1e-3)

    images = {
        "gt": gt,
        "mesh": mesh,
        "hybrid": hybrid,
        "weight": _mono_to_rgb(weight),
        "gate": _mono_to_rgb(gate),
        "better_than_mesh": _mono_to_rgb(better_mag, 0.0, max_delta),
        "worse_than_mesh": _mono_to_rgb(worse_mag, 0.0, max_delta),
    }

    stem = f"{image_idx:04d}"
    for name, img in images.items():
        _save_rgb(out_dir / f"{stem}_{name}.png", img)
