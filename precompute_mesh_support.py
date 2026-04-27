from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import tqdm

from colmap_data import ColmapScene, SceneDataset
from diagnostics import save_json, save_rgb
from mesh_renderer import MeshRenderer



def validate(frame_name: str, rgb: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor, width: int, height: int):
    if tuple(rgb.shape) != (height, width, 3):
        raise RuntimeError(f"{frame_name}: rgb shape {tuple(rgb.shape)}")
    if tuple(depth.shape) != (height, width, 1):
        raise RuntimeError(f"{frame_name}: depth shape {tuple(depth.shape)}")
    if tuple(mask.shape) != (height, width, 1):
        raise RuntimeError(f"{frame_name}: mask shape {tuple(mask.shape)}")
    if not torch.isfinite(rgb).all():
        raise RuntimeError(f"{frame_name}: non-finite rgb")
    if bool(mask.any()) and not torch.isfinite(depth[mask]).all():
        raise RuntimeError(f"{frame_name}: non-finite depth under mask")


def save_npz(path: Path, rgb: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        rgb=(rgb.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8),
        depth=depth.detach().cpu().numpy()[..., 0].astype(np.float32),
        mask=mask.detach().cpu().numpy()[..., 0].astype(np.uint8),
    )



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--mesh_obj", required=True)
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--split", default="all", choices=["all", "train", "val"])
    ap.add_argument("--test_every", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check_images", type=int, default=4)
    ap.add_argument("--save_renders", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    scene = ColmapScene(args.scene_dir, test_every=args.test_every)
    dataset = SceneDataset(scene, split=args.split)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.scene_dir) / "mesh_support"
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(dataset)))[0:]

    renderer = MeshRenderer(args.mesh_obj, device="cuda")
    t0 = time.perf_counter()

    for local_idx in tqdm.tqdm(indices):
        frame_idx = int(dataset.indices[local_idx])
        frame = scene.frames[frame_idx]
        out_path = out_dir / f"{Path(frame.name).stem}.npz"
        if out_path.exists() and not args.overwrite:
            continue

        K = torch.from_numpy(frame.K).cuda().float()
        c2w = torch.from_numpy(frame.c2w).cuda().float()
        rgb, depth, mask = renderer.render(K, c2w, frame.width, frame.height)
        validate(frame.name, rgb, depth, mask, frame.width, frame.height)
        save_npz(out_path, rgb, depth, mask)
        if args.save_renders:
            save_rgb(out_dir / "renders" / f"{Path(frame.name).stem}.png", rgb)


    summary = {
        "scene_dir": str(Path(args.scene_dir).resolve()),
        "mesh_obj": str(Path(args.mesh_obj).resolve()),
        "output_dir": str(out_dir.resolve()),
        "frames": len(indices),
        "seconds": round(time.perf_counter() - t0, 3),
    }
    save_json(out_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
