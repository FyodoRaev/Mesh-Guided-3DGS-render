from __future__ import annotations

import torch
from pytorch3d.io import load_obj
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.mesh.textures import TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection


class MeshRenderer:
    def __init__(self, mesh_obj: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.mesh = self._load(mesh_obj)
        self.verts = self.mesh.verts_packed()
        self.faces = self.mesh.faces_packed()
        self.num_faces = int(self.faces.shape[0])
        self._rasterizers: dict[tuple[int, int], MeshRasterizer] = {}

    def _load(self, mesh_obj: str) -> Meshes:
        verts, faces, aux = load_obj(mesh_obj, device=self.device, load_textures=True, create_texture_atlas=False)
        if aux.verts_uvs is None or faces.textures_idx is None:
            raise RuntimeError("mesh has no UV map")
        tex_dict = aux.texture_images or {}
        if len(tex_dict) != 1:
            raise RuntimeError(f"expected exactly one texture image, got {len(tex_dict)}")

        tex = next(iter(tex_dict.values()))[..., :3].to(self.device)
        tex = tex.float() / 255.0 if tex.dtype == torch.uint8 else tex.float()
        textures = TexturesUV(
            maps=tex[None],
            faces_uvs=faces.textures_idx.to(self.device)[None],
            verts_uvs=aux.verts_uvs.to(self.device)[None],
            sampling_mode="bilinear",
            align_corners=True,
        )
        return Meshes(verts=[verts.to(self.device)], faces=[faces.verts_idx.to(self.device)], textures=textures).to(
            self.device
        )

    def _rasterizer(self, height: int, width: int, cameras) -> MeshRasterizer:
        key = (int(height), int(width))
        if key not in self._rasterizers:
            self._rasterizers[key] = MeshRasterizer(
                cameras=cameras,
                raster_settings=RasterizationSettings(
                    image_size=key,
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    perspective_correct=True,
                ),
            )
        self._rasterizers[key].cameras = cameras
        return self._rasterizers[key]

    @torch.no_grad()
    def render(self, K: torch.Tensor, c2w: torch.Tensor, width: int, height: int):
        w2c = torch.linalg.inv(c2w)
        R, t = w2c[:3, :3], w2c[:3, 3]
        cameras = cameras_from_opencv_projection(
            R=R[None],
            tvec=t[None],
            camera_matrix=K[None],
            image_size=torch.tensor([[height, width]], dtype=torch.float32, device=self.device),
        )

        fragments = self._rasterizer(height, width, cameras)(self.mesh)
        mask = fragments.pix_to_face[0, ..., 0] >= 0
        rgb = self.mesh.sample_textures(fragments)[0, ..., 0, :3]
        rgb = torch.where(mask[..., None], rgb, torch.zeros_like(rgb))

        verts_cam = (R @ self.verts.T).T + t[None, :]
        pix_cam = interpolate_face_attributes(
            fragments.pix_to_face,
            fragments.bary_coords,
            verts_cam[self.faces],
        )[0, ..., 0, :]
        depth = torch.where(mask[..., None], pix_cam[..., 2:3], torch.full_like(pix_cam[..., 2:3], float("inf")))
        return rgb.clamp(0.0, 1.0), depth.float(), mask[..., None].bool()
