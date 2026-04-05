from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import tqdm

from colmap_data import ColmapScene, SceneDataset
from diagnostics import save_visual_pack
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import remove
from hybrid_math import compute_depth_gate, compose_hybrid, psnr
from mesh_renderer import MeshRenderer

SH_DEGREE = 3
SH_STEP_INTERVAL = 1000
INIT_OPACITY = 0.1
DEPTH_GATE_BETA = 200.0
DEPTH_GATE_EPS = 1e-4


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0


def load_means(path: str) -> np.ndarray:
    data = torch.load(path, map_location="cpu")
    means = None
    if isinstance(data, dict):
        if "means" in data:
            means = data["means"]
        elif "splats" in data and isinstance(data["splats"], dict) and "means" in data["splats"]:
            means = data["splats"]["means"]
    if means is None:
        raise RuntimeError(f"Could not find means in checkpoint: {path}")
    if isinstance(means, torch.nn.Parameter):
        means = means.detach()
    means = means.float().cpu().numpy().astype(np.float32)
    if means.ndim != 2 or means.shape[1] != 3:
        raise RuntimeError(f"Unexpected means shape in checkpoint: {means.shape}")
    return means


def sample_tie_points(scene: ColmapScene, n: int, seed: int) -> np.ndarray:
    if len(scene.points) == 0:
        raise RuntimeError("No COLMAP tie-points found")
    pts = scene.points.astype(np.float32)
    if len(pts) <= n:
        return pts
    rng = np.random.default_rng(seed)
    return pts[rng.choice(len(pts), size=n, replace=False)]


def nearest_scene_rgb(scene: ColmapScene, points: np.ndarray) -> np.ndarray:
    if len(scene.points) == 0:
        return np.full((len(points), 3), 0.5, dtype=np.float32)

    src_xyz = scene.points.astype(np.float32)
    src_rgb = (scene.points_rgb / 255.0).astype(np.float32)
    out = np.empty((len(points), 3), dtype=np.float32)

    chunk = 2048
    for start in range(0, len(points), chunk):
        end = min(start + chunk, len(points))
        diff = points[start:end, None, :] - src_xyz[None, :, :]
        idx = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        out[start:end] = src_rgb[idx]
    return out


def init_splats(scene: ColmapScene, args, device: str):
    if args.init_means_ckpt:
        points = load_means(args.init_means_ckpt)
        source = "pretrain_means_ckpt"
    else:
        points = sample_tie_points(scene, args.init_points, args.seed)
        source = "tie_points"

    rgbs = nearest_scene_rgb(scene, points)
    n = len(points)
    base_scale = max(scene.scene_scale / 80.0, 1e-4)

    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.from_numpy(points).to(device).float()),
            "scales": torch.nn.Parameter(torch.full((n, 3), math.log(base_scale), device=device)),
            "quats": torch.nn.Parameter(torch.randn((n, 4), device=device)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), INIT_OPACITY, device=device))),
            "sh0": torch.nn.Parameter(rgb_to_sh(torch.from_numpy(rgbs).to(device).float()).unsqueeze(1)),
            "shN": torch.nn.Parameter(torch.zeros((n, (SH_DEGREE + 1) ** 2 - 1, 3), device=device)),
        }
    )

    info = {"source": source, "num_points": int(n)}
    if args.init_means_ckpt:
        info["ckpt_path"] = str(args.init_means_ckpt)
    print(f"[init] source={source} num_gs={n}")
    return params, info


def make_optimizers(splats: torch.nn.ParameterDict, scene_scale: float):
    lrs = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20.0,
    }
    opts = {}
    for name, lr in lrs.items():
        opts[name] = torch.optim.Adam([{"params": splats[name], "lr": lr, "name": name}], eps=1e-15)
    return opts


class HybridTrainer:
    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")

        self.args = args
        self.device = "cuda"
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.scene = ColmapScene(args.scene_dir, test_every=args.test_every)
        self.trainset = SceneDataset(self.scene, split="train")
        self.valset = SceneDataset(self.scene, split="val")
        self.mesh = MeshRenderer(args.mesh_obj, device=self.device)

        self.out = Path(args.result_dir)
        self.ckpt_dir = self.out / "ckpts"
        self.stats_dir = self.out / "stats"
        self.vis_dir = self.out / "vis"
        for d in (self.ckpt_dir, self.stats_dir, self.vis_dir):
            d.mkdir(parents=True, exist_ok=True)
        save_json(self.out / "config.json", vars(args))

        self.splats, init_info = init_splats(self.scene, args, self.device)
        save_json(self.stats_dir / "init_info.json", init_info)

        self.optimizers = make_optimizers(self.splats, self.scene.scene_scale)
        self.strategy = DefaultStrategy(verbose=False)
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.state = self.strategy.initialize_state(scene_scale=self.scene.scene_scale)

        self.means_sched = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["means"], gamma=0.01 ** (1.0 / args.max_steps)
        )

    def _rasterize_gs(self, c2w: torch.Tensor, K: torch.Tensor, W: int, H: int, step: int | None):
        sh_degree = SH_DEGREE if step is None else min(step // SH_STEP_INTERVAL, SH_DEGREE)
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
        rc, ra, info = rasterization(
            means=self.splats["means"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=colors,
            viewmats=torch.linalg.inv(c2w),
            Ks=K,
            width=W,
            height=H,
            render_mode="RGB+ED",
            sh_degree=sh_degree,
            packed=False,
            sparse_grad=False,
            near_plane=0.01,
            far_plane=1e10,
        )
        return rc[..., :3].clamp(0.0, 1.0), rc[..., 3:4], ra[..., :1], info

    def _render_mesh(self, c2w: torch.Tensor, K: torch.Tensor, W: int, H: int):
        rgbs, depths, masks = [], [], []
        for i in range(c2w.shape[0]):
            rgb, depth, mask = self.mesh.render(K[i], c2w[i], W, H)
            rgbs.append(rgb)
            depths.append(depth)
            masks.append(mask)
        return torch.stack(rgbs, 0), torch.stack(depths, 0), torch.stack(masks, 0)

    def forward(self, batch: dict, step: int | None):
        gt = batch["image"].to(self.device).float() / 255.0
        c2w = batch["camtoworld"].to(self.device).float()
        K = batch["K"].to(self.device).float()
        H, W = gt.shape[1], gt.shape[2]

        gs_rgb, gs_depth, gs_alpha, info = self._rasterize_gs(c2w, K, W, H, step)
        mesh_rgb, mesh_depth, mesh_mask = self._render_mesh(c2w, K, W, H)
        gate = compute_depth_gate(gs_depth, mesh_depth, mesh_mask, DEPTH_GATE_BETA, DEPTH_GATE_EPS)
        hybrid = compose_hybrid(gs_rgb, gs_alpha, mesh_rgb, mesh_mask, gate)

        return {
            "gt": gt,
            "mesh_rgb": mesh_rgb,
            "gs_rgb": gs_rgb,
            "hybrid": hybrid,
            "info": info,
        }

    def train(self):
        loader = torch.utils.data.DataLoader(self.trainset, batch_size=1, shuffle=True, num_workers=0)
        it = iter(loader)

        for step in tqdm.trange(self.args.max_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            out = self.forward(batch, step)
            self.strategy.step_pre_backward(self.splats, self.optimizers, self.state, step, out["info"])

            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)
            loss = (out["hybrid"] - out["gt"]).abs().mean()
            loss.backward()
            for opt in self.optimizers.values():
                opt.step()
            self.means_sched.step()

            self.strategy.step_post_backward(self.splats, self.optimizers, self.state, step, out["info"], packed=False)

            if self.args.max_gs > 0 and len(self.splats["means"]) > self.args.max_gs:
                n = len(self.splats["means"])
                keep = torch.topk(torch.sigmoid(self.splats["opacities"].detach()), k=self.args.max_gs, largest=True).indices
                keep_mask = torch.zeros(n, dtype=torch.bool, device=self.splats["opacities"].device)
                keep_mask[keep] = True
                remove(params=self.splats, optimizers=self.optimizers, state=self.state, mask=~keep_mask)

            step1 = step + 1
            if step1 % self.args.save_every == 0 or step1 == self.args.max_steps:
                self.save_ckpt(step1)
            if step1 % self.args.eval_every == 0 or step1 == self.args.max_steps:
                self.eval(step1)

    @torch.no_grad()
    def eval(self, step: int):
        loader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=0)
        mesh_psnr_vals, gs_psnr_vals, hybrid_psnr_vals = [], [], []

        vis_step_dir = self.vis_dir / f"step_{step:06d}"
        vis_step_dir.mkdir(parents=True, exist_ok=True)

        for i, batch in enumerate(loader):
            out = self.forward(batch, None)
            mesh_psnr_vals.append(psnr(out["mesh_rgb"], out["gt"]))
            gs_psnr_vals.append(psnr(out["gs_rgb"], out["gt"]))
            hybrid_psnr_vals.append(psnr(out["hybrid"], out["gt"]))
            if i < self.args.save_vis_images:
                save_visual_pack(out, vis_step_dir, i)

        stats = {
            "step": int(step),
            "num_gs": int(len(self.splats["means"])),
            "mesh_psnr": float(np.mean(mesh_psnr_vals)) if mesh_psnr_vals else 0.0,
            "gs_psnr": float(np.mean(gs_psnr_vals)) if gs_psnr_vals else 0.0,
            "hybrid_psnr": float(np.mean(hybrid_psnr_vals)) if hybrid_psnr_vals else 0.0,
        }
        save_json(self.stats_dir / f"eval_{step:06d}.json", stats)
        print(
            f"[eval {step}] mesh={stats['mesh_psnr']:.3f} gs={stats['gs_psnr']:.3f} hybrid={stats['hybrid_psnr']:.3f}"
        )

    def save_ckpt(self, step: int):
        torch.save(
            {"step": int(step), "splats": {k: v.detach().cpu() for k, v in self.splats.items()}},
            self.ckpt_dir / f"ckpt_{step:06d}.pt",
        )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", type=str, required=True)
    ap.add_argument("--mesh_obj", type=str, required=True)
    ap.add_argument("--result_dir", type=str, default="runs/hybrid_from_pretrain")
    ap.add_argument("--init_means_ckpt", type=str, default="")
    ap.add_argument("--init_points", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--save_vis_images", type=int, default=4)
    ap.add_argument("--max_gs", type=int, default=0)
    ap.add_argument("--test_every", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    HybridTrainer(parse_args()).train()


if __name__ == "__main__":
    main()
