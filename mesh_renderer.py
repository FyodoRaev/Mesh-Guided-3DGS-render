from __future__ import annotations

from pathlib import Path

import nvdiffrast.torch as dr
from pytorch3d.io import load_obj
import torch


class MeshRenderer:
    def __init__(self, mesh_obj: str, device: str = "cuda"):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("MeshRenderer requires CUDA; CPU is unsupported")
        if not torch.cuda.is_available():
            raise RuntimeError("MeshRenderer requires CUDA; torch.cuda.is_available() is false")

        obj_path = Path(mesh_obj)
        verts, faces, aux = load_obj(str(obj_path), load_textures=True, device=self.device)
        if aux.verts_uvs is None or faces.textures_idx is None:
            raise RuntimeError(f"{obj_path}: OBJ must contain texture UVs")
        if not aux.texture_images:
            raise RuntimeError(f"{obj_path}: OBJ/MTL must reference a texture image")
        if len(aux.texture_images) != 1:
            raise RuntimeError(f"{obj_path}: only single-texture OBJ meshes are supported")

        self.verts = verts.to(self.device).float().contiguous()
        self.faces = faces.verts_idx.to(self.device).int().contiguous()
        self.verts_uvs = aux.verts_uvs.to(self.device).float().contiguous()
        self.verts_uvs[:, 1] = 1.0 - self.verts_uvs[:, 1]
        self.faces_uvs = faces.textures_idx.to(self.device).int().contiguous()
        self.tex = next(iter(aux.texture_images.values())).to(self.device).float()[None].contiguous()
        self._require_cuda(
            verts=self.verts,
            faces=self.faces,
            verts_uvs=self.verts_uvs,
            faces_uvs=self.faces_uvs,
            tex=self.tex,
        )
        self.glctx = dr.RasterizeGLContext()

    def _require_cuda(self, **tensors: torch.Tensor) -> None:
        for name, tensor in tensors.items():
            if not tensor.is_cuda:
                raise RuntimeError(f"{name} must be CUDA, got {tensor.device}")
            if self.device.index is not None and tensor.device.index != self.device.index:
                raise RuntimeError(f"{name} must be on {self.device}, got {tensor.device}")

    def _clip(self, K: torch.Tensor, c2w: torch.Tensor, width: int, height: int) -> torch.Tensor:
        self._require_cuda(K=K, c2w=c2w)
        w2c = torch.linalg.inv(c2w)
        xyz = (self.verts @ w2c[:3, :3].T) + w2c[:3, 3]
        z = xyz[:, 2].clamp_min(1e-6)
        u = K[0, 0] * xyz[:, 0] / z + K[0, 2]
        v = K[1, 1] * xyz[:, 1] / z + K[1, 2]
        x = (2.0 * u / float(width) - 1.0) * z
        y = (2.0 * v / float(height) - 1.0) * z
        return torch.stack([x, y, z - 0.01, z], dim=-1)[None].contiguous()

    @torch.no_grad()
    def render(self, K: torch.Tensor, c2w: torch.Tensor, width: int, height: int):
        pos = self._clip(K, c2w, width, height)
        self._require_cuda(pos=pos)
        rast, _ = dr.rasterize(self.glctx, pos, self.faces, resolution=[height, width])
        mask = rast[0, ..., 3] > 0

        uv, _ = dr.interpolate(self.verts_uvs[None], rast, self.faces_uvs)
        self._require_cuda(rast=rast, mask=mask, uv=uv)
        rgb = dr.texture(self.tex, uv, filter_mode="linear", boundary_mode="clamp")[0]
        rgb = torch.where(mask[..., None], rgb, torch.zeros_like(rgb))

        w2c = torch.linalg.inv(c2w)
        z = ((self.verts @ w2c[:3, :3].T) + w2c[:3, 3])[:, 2:3].contiguous()
        depth, _ = dr.interpolate(z[None].contiguous(), rast, self.faces)
        depth = torch.where(mask[..., None], depth[0], torch.full_like(depth[0], float("inf")))
        self._require_cuda(rgb=rgb, depth=depth)
        return rgb.clamp(0.0, 1.0), depth.float(), mask[..., None]
