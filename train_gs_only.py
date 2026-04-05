from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import tqdm

from colmap_data import ColmapScene, SceneDataset
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import remove
from hybrid_math import psnr


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0


def init_splats(scene: ColmapScene, args, device: str):
    points_np = scene.points
    rgb_np = scene.points_rgb / 255.0
    if len(points_np) == 0:
        raise RuntimeError("No COLMAP tie-points found: random init disabled by design")

    if len(points_np) > args.init_points:
        idx = np.random.choice(len(points_np), size=args.init_points, replace=False)
        points_np = points_np[idx]
        rgb_np = rgb_np[idx]

    points = torch.from_numpy(points_np).float()
    rgbs = torch.from_numpy(rgb_np).float()

    n = points.shape[0]
    base_scale = max(scene.scene_scale / 80.0, 1e-4)
    params = {
        "means": torch.nn.Parameter(points.to(device)),
        "scales": torch.nn.Parameter(torch.full((n, 3), math.log(base_scale), device=device)),
        "quats": torch.nn.Parameter(torch.randn((n, 4), device=device)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), args.init_opa, device=device))),
        "sh0": torch.nn.Parameter(rgb_to_sh(rgbs.to(device)).unsqueeze(1)),
        "shN": torch.nn.Parameter(torch.zeros((n, (args.sh_degree + 1) ** 2 - 1, 3), device=device)),
    }
    print(f"[init] source=tie_points num_gs={n}")
    return torch.nn.ParameterDict(params)


def make_optimizers(splats, scene_scale: float, batch_size: int, means_lr_mult: float = 1.0):
    lrs = {
        "means": 1.6e-4 * scene_scale * means_lr_mult,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20.0,
    }
    sq_bs = math.sqrt(batch_size)
    opts = {}
    for k, lr in lrs.items():
        opts[k] = torch.optim.Adam(
            [{"params": splats[k], "lr": lr * sq_bs, "name": k}],
            eps=1e-15 / sq_bs,
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
    return opts


def _save_rgb(path: Path, x: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = x.detach().cpu().float().numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    imageio.imwrite(path, (arr * 255.0).astype(np.uint8))


class Trainer:
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

        self.splats = init_splats(self.scene, args, self.device)
        self.optimizers = make_optimizers(self.splats, self.scene.scene_scale, args.batch_size, args.means_lr_mult)

        self.strategy = DefaultStrategy(verbose=False)
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.state = self.strategy.initialize_state(scene_scale=self.scene.scene_scale)

        self.means_sched = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["means"], gamma=0.01 ** (1.0 / args.max_steps)
        )

        self.out = Path(args.result_dir)
        self.ckpt_dir = self.out / "ckpts"
        self.vis_dir = self.out / "vis"
        for d in (self.ckpt_dir, self.vis_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _forward(self, batch, step: int | None):
        gt = batch["image"].to(self.device).float() / 255.0
        c2w = batch["camtoworld"].to(self.device).float()
        K = batch["K"].to(self.device).float()

        H, W = gt.shape[1], gt.shape[2]
        sh_degree = self.args.sh_degree if step is None else min(step // self.args.sh_step_interval, self.args.sh_degree)
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
        rc, _, info = rasterization(
            means=self.splats["means"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=colors,
            viewmats=torch.linalg.inv(c2w),
            Ks=K,
            width=W,
            height=H,
            render_mode="RGB",
            sh_degree=sh_degree,
            packed=False,
            sparse_grad=False,
            near_plane=0.01,
            far_plane=1e10,
        )

        gs_rgb = rc[..., :3].clamp(0.0, 1.0)
        return {"gt": gt, "gs_rgb": gs_rgb, "info": info}

    def train(self):
        loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            persistent_workers=self.args.num_workers > 0,
        )
        it = iter(loader)

        pbar = tqdm.trange(self.args.max_steps)
        for step in pbar:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            out = self._forward(batch, step)
            self.strategy.step_pre_backward(self.splats, self.optimizers, self.state, step, out["info"])

            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)

            loss = (out["gs_rgb"] - out["gt"]).abs().mean()
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
            pbar.set_description(f"step={step1} loss={loss.item():.4f} gs={len(self.splats['means'])}")
            if step1 % self.args.save_every == 0 or step1 == self.args.max_steps:
                self.save_ckpt(step1)
            if step1 % self.args.eval_every == 0 or step1 == self.args.max_steps:
                self.eval(step1)

    @torch.no_grad()
    def eval(self, step: int):
        loader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=0)
        psnrs = []

        vis_step_dir = self.vis_dir / f"step_{step:06d}"
        vis_step_dir.mkdir(parents=True, exist_ok=True)

        for i, batch in enumerate(loader):
            out = self._forward(batch, None)
            psnrs.append(psnr(out["gs_rgb"], out["gt"]))
            if i < self.args.save_vis_images:
                _save_rgb(vis_step_dir / f"{i:04d}_gt.png", out["gt"][0])
                _save_rgb(vis_step_dir / f"{i:04d}_gs.png", out["gs_rgb"][0])

        print(f"[eval {step}] psnr={float(np.mean(psnrs)):.4f}")

    def save_ckpt(self, step: int):
        torch.save(
            {"step": step, "splats": {k: v.detach().cpu() for k, v in self.splats.items()}},
            self.ckpt_dir / f"ckpt_{step:06d}.pt",
        )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", type=str, required=True)
    ap.add_argument("--result_dir", type=str, default="results/gs_only")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--test_every", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init_points", type=int, default=100000)
    ap.add_argument("--init_opa", type=float, default=0.1)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--sh_step_interval", type=int, default=1000)
    ap.add_argument("--means_lr_mult", type=float, default=1.0)
    ap.add_argument("--max_gs", type=int, default=0)
    ap.add_argument("--save_vis_images", type=int, default=4)
    return ap.parse_args()


def main():
    Trainer(parse_args()).train()


if __name__ == "__main__":
    main()
