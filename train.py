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
from hybrid_math import (
    compute_depth_gate,
    compose_hybrid,
    psnr,
    residual_weight_from_mesh_error,
    selective_metrics,
    weighted_l1,
)
from mesh_renderer import MeshRenderer


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def init_splats(scene: ColmapScene, args, device: str):
    use_random = bool(args.ignore_tie_points) or len(scene.points) == 0
    if use_random:
        points = torch.rand((args.init_points, 3), dtype=torch.float32)
        points = (points * 2.0 - 1.0) * scene.scene_scale
        rgbs = torch.full((args.init_points, 3), 0.5, dtype=torch.float32)
    else:
        points_np = scene.points
        rgb_np = scene.points_rgb / 255.0
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
    print(f"[init] source={'random' if use_random else 'tie_points'} num_gs={n}")
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
        self.mesh = MeshRenderer(args.mesh_obj, device=self.device)

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
        self.stats_dir = self.out / "stats"
        self.vis_dir = self.out / "vis"
        self.train_vis_dir = self.out / "train_vis"
        self.gs_trace_dir = self.out / "gs_trace"
        for d in (self.ckpt_dir, self.stats_dir, self.vis_dir, self.train_vis_dir, self.gs_trace_dir):
            d.mkdir(parents=True, exist_ok=True)
        save_json(self.out / "config.json", vars(args))

        self.geom_warmup_steps = 600
        self.warmup_weight_floor = 0.05
        self.gs_trace_warmup_every = 50
        self.gs_trace_post_every = 100
        self.gs_trace_post_steps = 800
        self._save_gs_trace(step=0)
    def _save_gs_trace(self, step: int):
        torch.save({"step": int(step), "means": self.splats["means"].detach().cpu()}, self.gs_trace_dir / f"means_{step:06d}.pt")

    def _maybe_save_gs_trace(self, step: int):
        if step <= self.geom_warmup_steps:
            every = self.gs_trace_warmup_every
        elif step <= self.geom_warmup_steps + self.gs_trace_post_steps:
            every = self.gs_trace_post_every
        else:
            return
        if step % every == 0:
            self._save_gs_trace(step)

    def _rasterize_gs(self, c2w, K, W, H, sh_degree):
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

    def _render_mesh_batch(self, c2w, K, W, H):
        rgbs, depths, masks = [], [], []
        for i in range(c2w.shape[0]):
            rgb, depth, mask = self.mesh.render(K[i], c2w[i], W, H)
            rgbs.append(rgb)
            depths.append(depth)
            masks.append(mask)
        return torch.stack(rgbs, 0), torch.stack(depths, 0), torch.stack(masks, 0)

    def _forward(self, batch, step: int | None):
        gt = batch["image"].to(self.device).float() / 255.0
        c2w = batch["camtoworld"].to(self.device).float()
        K = batch["K"].to(self.device).float()

        H, W = gt.shape[1], gt.shape[2]
        sh_degree = self.args.sh_degree if step is None else min(step // self.args.sh_step_interval, self.args.sh_degree)

        gs_rgb, gs_depth, gs_alpha, info = self._rasterize_gs(c2w, K, W, H, sh_degree)
        mesh_rgb, mesh_depth, mesh_mask = self._render_mesh_batch(c2w, K, W, H)

        gate = compute_depth_gate(gs_depth, mesh_depth, mesh_mask, self.args.depth_gate_beta, self.args.depth_gate_eps)
        hybrid = compose_hybrid(gs_rgb, gs_alpha, mesh_rgb, mesh_mask, gate)
        weight = residual_weight_from_mesh_error(gt, mesh_rgb, mesh_mask, self.args.residual_scale)
        if step is not None and step < self.geom_warmup_steps and self.warmup_weight_floor > 0.0:
            weight = torch.where(weight >= self.warmup_weight_floor, weight, torch.zeros_like(weight)).detach()

        return {
            "gt": gt,
            "gs_rgb": gs_rgb,
            "gs_alpha": gs_alpha,
            "mesh_rgb": mesh_rgb,
            "hybrid": hybrid,
            "weight": weight,
            "gate": gate,
            "mesh_mask": mesh_mask,
            "info": info,
        }

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
            warmup = step < self.geom_warmup_steps
            if step == 0:
                print(f"[phase] warmup_steps={self.geom_warmup_steps} warmup_weight_floor={self.warmup_weight_floor}")
            if step == self.geom_warmup_steps:
                print("[phase] warmup finished -> enabling full GS optimization + strategy")

            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            out = self._forward(batch, step)
            if not warmup:
                self.strategy.step_pre_backward(self.splats, self.optimizers, self.state, step, out["info"])

            active_names = {"means"} if warmup else set(self.splats.keys())
            for name, param in self.splats.items():
                param.requires_grad_(name in active_names)
            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)

            loss = weighted_l1(out["hybrid"], out["gt"], out["weight"])
            loss.backward()

            for name, opt in self.optimizers.items():
                if name in active_names:
                    opt.step()
            self.means_sched.step()

            if not warmup:
                self.strategy.step_post_backward(
                    self.splats, self.optimizers, self.state, step, out["info"], packed=False
                )

            if (not warmup) and self.args.max_gs > 0 and len(self.splats["means"]) > self.args.max_gs:
                n = len(self.splats["means"])
                keep = torch.topk(torch.sigmoid(self.splats["opacities"].detach()), k=self.args.max_gs, largest=True).indices
                keep_mask = torch.zeros(n, dtype=torch.bool, device=self.splats["opacities"].device)
                keep_mask[keep] = True
                remove(params=self.splats, optimizers=self.optimizers, state=self.state, mask=~keep_mask)

            with torch.no_grad():
                mesh_psnr = psnr(out["mesh_rgb"], out["gt"])
                hybrid_psnr = psnr(out["hybrid"], out["gt"])

            step1 = step + 1
            self._maybe_save_gs_trace(step1)
            phase = "warmup" if warmup else "full"
            pbar.set_description(
                f"step={step1} phase={phase} loss={loss.item():.4f} mesh_psnr={mesh_psnr:.2f} hybrid_psnr={hybrid_psnr:.2f} gs={len(self.splats['means'])}"
            )

            if self.args.save_train_vis_every > 0 and (step == 0 or step1 % self.args.save_train_vis_every == 0):
                save_visual_pack(out, self.train_vis_dir / f"step_{step1:06d}", image_idx=0)
            if step1 % self.args.save_every == 0 or step1 == self.args.max_steps:
                self.save_ckpt(step1)
            if step1 % self.args.eval_every == 0 or step1 == self.args.max_steps:
                self.eval(step1)

    @torch.no_grad()
    def eval(self, step: int):
        loader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=0)
        mesh_vals, hybrid_vals, gs_vals = [], [], []
        sel_acc: dict[str, list[float]] = {}

        vis_step_dir = self.vis_dir / f"step_{step:06d}"
        vis_step_dir.mkdir(parents=True, exist_ok=True)

        for i, batch in enumerate(loader):
            out = self._forward(batch, None)
            mesh_vals.append(psnr(out["mesh_rgb"], out["gt"]))
            hybrid_vals.append(psnr(out["hybrid"], out["gt"]))
            gs_vals.append(psnr(out["gs_rgb"], out["gt"]))

            sel = selective_metrics(
                gt=out["gt"],
                mesh_rgb=out["mesh_rgb"],
                hybrid_rgb=out["hybrid"],
                gs_alpha=out["gs_alpha"],
                gate=out["gate"],
                mesh_mask=out["mesh_mask"],
            )
            for k, v in sel.items():
                sel_acc.setdefault(k, []).append(float(v))

            if i < self.args.save_vis_images:
                save_visual_pack(out, vis_step_dir, image_idx=i)

        stats = {
            "step": step,
            "mesh_psnr": float(np.mean(mesh_vals)),
            "gs_psnr": float(np.mean(gs_vals)),
            "hybrid_psnr": float(np.mean(hybrid_vals)),
            "num_gs": int(len(self.splats["means"])),
            "save_vis_images": int(self.args.save_vis_images),
        }
        for k, vals in sel_acc.items():
            stats[k] = float(np.mean(vals))
        save_json(self.stats_dir / f"step_{step:06d}.json", stats)
        print(
            f"[eval {step}] mesh={stats['mesh_psnr']:.3f} gs={stats['gs_psnr']:.3f} hybrid={stats['hybrid_psnr']:.3f} "
            f"select={stats['selectivity_score']:.4f} leak={stats['leakage_good']:.4f} blur={stats['blur_regression_good']:.4f}"
        )

    def save_ckpt(self, step: int):
        torch.save(
            {"step": step, "splats": {k: v.detach().cpu() for k, v in self.splats.items()}},
            self.ckpt_dir / f"ckpt_{step:06d}.pt",
        )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", type=str, required=True)
    ap.add_argument("--mesh_obj", type=str, required=True)
    ap.add_argument("--result_dir", type=str, default="results/distillate")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--test_every", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init_points", type=int, default=100000)
    ap.add_argument("--init_opa", type=float, default=0.1)
    ap.add_argument("--ignore_tie_points", type=int, default=0)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--sh_step_interval", type=int, default=1000)
    ap.add_argument("--depth_gate_beta", type=float, default=200.0)
    ap.add_argument("--depth_gate_eps", type=float, default=1e-4)
    ap.add_argument("--residual_scale", type=float, default=0.05)
    ap.add_argument("--means_lr_mult", type=float, default=1.0)
    ap.add_argument("--max_gs", type=int, default=0)
    ap.add_argument("--save_vis_images", type=int, default=4)
    ap.add_argument("--save_train_vis_every", type=int, default=1000)
    return ap.parse_args()


def main():
    Trainer(parse_args()).train()


if __name__ == "__main__":
    main()
