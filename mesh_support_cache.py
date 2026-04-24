from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class MeshSupportCache:
    def __init__(self, scene_dir: str | Path, support_dir: str | Path | None = None):
        scene_dir = Path(scene_dir)
        self.support_dir = Path(support_dir) if support_dir else scene_dir / "mesh_support"
        self.files = {p.stem: p for p in sorted(self.support_dir.glob("*.npz"))} if self.support_dir.is_dir() else {}

    @property
    def available(self) -> bool:
        return bool(self.files)

    def has_frame(self, frame_name: str) -> bool:
        return Path(frame_name).stem in self.files

    def load_tensors(self, frame_name: str, device: str | torch.device, width: int, height: int):
        path = self.files.get(Path(frame_name).stem)
        if path is None:
            raise KeyError(f"missing mesh support for {frame_name}")
        with np.load(path, allow_pickle=False) as data:
            rgb = data["rgb"]
            depth = data["depth"]
            mask = data["mask"]
        if rgb.shape[:2] != (height, width):
            raise RuntimeError(f"{frame_name}: cached shape {rgb.shape[:2]} != {(height, width)}")
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).to(device).float() / 255.0
        depth_t = torch.from_numpy(np.ascontiguousarray(depth)).to(device).float()[..., None]
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).to(device).bool()[..., None]
        return rgb_t, depth_t, mask_t
