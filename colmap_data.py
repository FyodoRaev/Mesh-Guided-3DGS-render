from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float32,
    )


def parse_K(model: str, params: list[float]) -> np.ndarray:
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        fx, fy = f, f
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
    elif model in ("SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = params[:3]
        fx, fy = f, f
    elif model in ("OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = params[:4]
    else:
        raise ValueError(f"unsupported camera model: {model}")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


@dataclass
class Frame:
    name: str
    image_path: Path
    K: np.ndarray
    c2w: np.ndarray


class ColmapScene:
    def __init__(self, scene_dir: str, test_every: int = 8):
        self.scene_dir = Path(scene_dir)
        self.sparse_dir = self.scene_dir / "sparse" / "0"
        self.images_dir = self.scene_dir / "images"
        self.frames = self._load_frames()
        self.points, self.points_rgb = self._load_points()
        self.scene_scale = self._estimate_scene_scale()

        idx = np.arange(len(self.frames))
        self.val_idx = idx[::test_every]
        self.train_idx = np.setdiff1d(idx, self.val_idx)

    def _load_frames(self) -> list[Frame]:
        cams: dict[int, np.ndarray] = {}
        with open(self.sparse_dir / "cameras.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                t = line.split()
                cam_id = int(t[0])
                model = t[1]
                params = list(map(float, t[4:]))
                cams[cam_id] = parse_K(model, params)

        valid_lines = []
        with open(self.sparse_dir / "images.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                valid_lines.append(line)

        frames = []
        for i in range(0, len(valid_lines), 2):
            t = valid_lines[i].split()
            qvec = np.array(list(map(float, t[1:5])), dtype=np.float32)
            tvec = np.array(list(map(float, t[5:8])), dtype=np.float32)
            cam_id = int(t[8])
            name = " ".join(t[9:])

            R = qvec2rotmat(qvec)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ tvec

            frames.append(
                Frame(
                    name=name,
                    image_path=self.images_dir / name,
                    K=cams[cam_id],
                    c2w=c2w,
                )
            )

        frames.sort(key=lambda x: x.name)
        return frames

    def _load_points(self):
        xyz = []
        rgb = []
        p = self.sparse_dir / "points3D.txt"
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                t = line.split()
                xyz.append([float(t[1]), float(t[2]), float(t[3])])
                rgb.append([float(t[4]), float(t[5]), float(t[6])])

        if len(xyz) == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
        return np.asarray(xyz, np.float32), np.asarray(rgb, np.float32)

    def _estimate_scene_scale(self) -> float:
        if len(self.points) == 0:
            return 1.0
        c = np.median(self.points, axis=0)
        d = np.linalg.norm(self.points - c[None, :], axis=1)
        return float(np.percentile(d, 90) + 1e-6)


class SceneDataset(torch.utils.data.Dataset):
    def __init__(self, scene: ColmapScene, split: str):
        self.scene = scene
        if split == "train":
            self.indices = scene.train_idx
        elif split == "val":
            self.indices = scene.val_idx
        else:
            raise ValueError(split)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        f = self.scene.frames[int(self.indices[idx])]
        img = imageio.imread(f.image_path)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        if img.shape[-1] > 3:
            img = img[..., :3]
        return {
            "image": torch.from_numpy(img.copy()),
            "camtoworld": torch.from_numpy(f.c2w.copy()),
            "K": torch.from_numpy(f.K.copy()),
        }
