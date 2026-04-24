from __future__ import annotations

from pathlib import Path

import nvdiffrast.torch as dr
import torch


class MeshRenderer:
    def __init__(self, mesh_obj: str, device: str = "cuda"):
        self.device = torch.device(device)
        cache_path = Path(mesh_obj).with_suffix(Path(mesh_obj).suffix + ".meshcache.pt")
        if not cache_path.exists():
            raise FileNotFoundError(f"missing mesh cache: {cache_path}")

        data = torch.load(cache_path, map_location=self.device)
        self.verts = data["verts"].float().contiguous()
        self.faces = data["faces_idx"].int().contiguous()
        self.verts_uvs = data["verts_uvs"].float().contiguous()
        self.faces_uvs = data["faces_uvs"].int().contiguous()
        self.tex = data["tex"].float()[None].contiguous()
        self.glctx = dr.RasterizeGLContext()

    def _clip(self, K: torch.Tensor, c2w: torch.Tensor, width: int, height: int) -> torch.Tensor:
        w2c = torch.linalg.inv(c2w)
        xyz = (self.verts @ w2c[:3, :3].T) + w2c[:3, 3]
        z = xyz[:, 2].clamp_min(1e-6)
        u = K[0, 0] * xyz[:, 0] / z + K[0, 2]
        v = K[1, 1] * xyz[:, 1] / z + K[1, 2]
        x = (2.0 * u / float(width) - 1.0) * z
        y = (1.0 - 2.0 * v / float(height)) * z
        return torch.stack([x, y, z - 0.01, z], dim=-1)[None].contiguous()

    @torch.no_grad()
    def render(self, K: torch.Tensor, c2w: torch.Tensor, width: int, height: int):
        pos = self._clip(K, c2w, width, height)
        rast, _ = dr.rasterize(self.glctx, pos, self.faces, resolution=[height, width])
        mask = rast[0, ..., 3] > 0

        uv, _ = dr.interpolate(self.verts_uvs[None], rast, self.faces_uvs)
        rgb = dr.texture(self.tex, uv, filter_mode="linear", boundary_mode="clamp")[0]
        rgb = torch.where(mask[..., None], rgb, torch.zeros_like(rgb))

        w2c = torch.linalg.inv(c2w)
        z = ((self.verts @ w2c[:3, :3].T) + w2c[:3, 3])[:, 2:3].contiguous()
        depth, _ = dr.interpolate(z[None].contiguous(), rast, self.faces)
        depth = torch.where(mask[..., None], depth[0], torch.full_like(depth[0], float("inf")))
        return rgb.clamp(0.0, 1.0), depth.float(), mask[..., None]
