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


def save_visual_pack(out: dict, out_dir: Path, image_idx: int):
    mesh = out["mesh_rgb"][0]
    gs = out["gs_rgb"][0]
    hybrid = out["hybrid"][0]

    # One compact compare strip: mesh | gs | hybrid
    compare = torch.cat([mesh, gs, hybrid], dim=1)

    stem = f"{image_idx:04d}"
    _save_rgb(out_dir / f"{stem}_mesh.png", mesh)
    _save_rgb(out_dir / f"{stem}_gs.png", gs)
    _save_rgb(out_dir / f"{stem}_hybrid.png", hybrid)
    _save_rgb(out_dir / f"{stem}_compare.png", compare)
