from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def _rgb(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    while x.ndim > 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 2:
        x = x[..., None]
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def save_rgb(path: Path, image: torch.Tensor | np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, (_rgb(image) * 255.0).astype(np.uint8))


def save_visual_pack(out: dict, out_dir: Path, image_idx: int):
    mesh = out["mesh_rgb"][0]
    gs = out["gs_rgb"][0]
    hybrid = out["hybrid"][0]
    gt = out["gt"][0]
    compare = torch.cat([out["gt"][0], mesh, gs, hybrid], dim=1)
    stem = f"{image_idx:04d}"
    save_rgb(out_dir / f"{stem}_mesh.png", mesh)
    save_rgb(out_dir / f"{stem}_hybrid.png", hybrid)
    save_rgb(out_dir / f"{stem}_gt.png", gt)
    save_rgb(out_dir / f"{stem}_compare.png", compare)
