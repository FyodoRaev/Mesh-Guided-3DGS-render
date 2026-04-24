from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import tqdm

from colmap_data import ColmapScene, SceneDataset
from diagnostics import save_json, save_visual_pack
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from hybrid_math import compose_hybrid, compute_depth_gate, psnr
from mesh_renderer import MeshRenderer
from mesh_support_cache import MeshSupportCache

SH_DEGREE = 3
SH_STEP_INTERVAL = 1000
ZNEAR = 0.01
ZFAR = 1e10


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / 0.28209479177387814


def load_means(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            key = "means" if "means" in data else "points"
            return np.asarray(data[key], dtype=np.float32)
    data = torch.load(path, map_location="cpu")
    means = data.get("means") if isinstance(data, dict) else None
    if means is None and isinstance(data, dict) and isinstance(data.get("splats"), dict):
        means = data["splats"].get("means")
    if means is None:
        raise RuntimeError(f"could not find means in {path}")
    return means.detach().cpu().float().numpy().astype(np.float32)


def sample_tie_points(scene: ColmapScene, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(scene.points) == 0:
        raise RuntimeError("No COLMAP tie-points found")
    rng = np.random.default_rng(seed)
    points = scene.points.astype(np.float32)
    colors = (scene.points_rgb / 255.0).astype(np.float32)
    if n <= 0 or n >= len(points):
        return points, colors
    idx = rng.choice(len(points), size=min(n, len(points)), replace=False)
    return points[idx], colors[idx]


def nearest_scene_rgb(scene: ColmapScene, points: np.ndarray) -> np.ndarray:
    if len(scene.points) == 0:
        return np.full((len(points), 3), 0.5, dtype=np.float32)
    xyz = scene.points.astype(np.float32)
    rgb = (scene.points_rgb / 255.0).astype(np.float32)
    out = np.empty((len(points), 3), dtype=np.float32)
    for start in range(0, len(points), 2048):
        end = min(start + 2048, len(points))
        idx = np.argmin(((points[start:end, None] - xyz[None]) ** 2).sum(axis=2), axis=1)
        out[start:end] = rgb[idx]
    return out


def init_splats(scene: ColmapScene, args, device: str):
    if args.init_means_ckpt:
        points = load_means(args.init_means_ckpt)
        colors = nearest_scene_rgb(scene, points)
        source = "means_ckpt"
    else:
        points, colors = sample_tie_points(scene, args.init_points, args.seed)
        source = "tie_points"

    n = len(points)
    scale = max(scene.scene_scale / 80.0, 1e-4)
    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.from_numpy(points).to(device).float()),
            "scales": torch.nn.Parameter(torch.full((n, 3), math.log(scale), device=device)),
            "quats": torch.nn.Parameter(torch.randn((n, 4), device=device)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), args.init_opacity, device=device))),
            "sh0": torch.nn.Parameter(rgb_to_sh(torch.from_numpy(colors).to(device).float()).unsqueeze(1)),
            "shN": torch.nn.Parameter(torch.zeros((n, (SH_DEGREE + 1) ** 2 - 1, 3), device=device)),
        }
    )
    print(f"[init] source={source} num_gs={n}")
    return splats, {"source": source, "num_gs": int(n)}


def make_optimizers(splats: torch.nn.ParameterDict, scene_scale: float):
    lrs = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20.0,
    }
    return {
        name: torch.optim.Adam([{"params": splats[name], "lr": lr, "name": name}], eps=1e-15)
        for name, lr in lrs.items()
    }


def frame_names_from_batch(batch: dict) -> list[str]:
    names = batch["frame_name"]
    return [names] if isinstance(names, str) else [str(x) for x in names]


def projection_from_opencv(K: torch.Tensor, width: int, height: int) -> torch.Tensor:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    P = torch.zeros((4, 4), dtype=K.dtype, device=K.device)
    P[0, 0] = 2.0 * fx / width
    P[1, 1] = 2.0 * fy / height
    P[0, 2] = 2.0 * cx / width - 1.0
    P[1, 2] = 2.0 * cy / height - 1.0
    P[2, 2] = ZFAR / (ZFAR - ZNEAR)
    P[2, 3] = -(ZFAR * ZNEAR) / (ZFAR - ZNEAR)
    P[3, 2] = 1.0
    return P.transpose(0, 1).contiguous()


def fov_from_K(K: torch.Tensor, width: int, height: int) -> tuple[float, float]:
    tanfovx = 0.5 * float(width) / float(K[0, 0].detach().cpu())
    tanfovy = 0.5 * float(height) / float(K[1, 1].detach().cpu())
    return tanfovx, tanfovy


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

        self.out = Path(args.result_dir)
        self.ckpt_dir = self.out / "ckpts"
        self.vis_dir = self.out / "vis"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        save_json(self.out / "config.json", vars(args))

        self.cache = MeshSupportCache(args.scene_dir, args.mesh_support_dir or None)
        cached = sum(self.cache.has_frame(f.name) for f in self.scene.frames)
        self.use_live_mesh = args.force_live_mesh or cached != len(self.scene.frames)
        self.mesh = MeshRenderer(args.mesh_obj, self.device) if self.use_live_mesh else None
        print(f"[mesh] source={'live' if self.use_live_mesh else 'cache'} cached={cached}/{len(self.scene.frames)}")

        self.splats, init_info = init_splats(self.scene, args, self.device)
        save_json(self.out / "init.json", init_info)
        self.optimizers = make_optimizers(self.splats, self.scene.scene_scale)
        self.means_sched = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["means"], gamma=0.01 ** (1.0 / args.max_steps)
        )
        self.metrics = self.out / "metrics.jsonl"

    def _rasterize_gs(self, c2w: torch.Tensor, K: torch.Tensor, width: int, height: int, step: int | None):
        rgbs, depths, alphas = [], [], []
        sh_degree = SH_DEGREE if step is None else min(step // SH_STEP_INTERVAL, SH_DEGREE)
        shs = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)[:, : (sh_degree + 1) ** 2]
        scales = torch.exp(self.splats["scales"])
        rotations = torch.nn.functional.normalize(self.splats["quats"], dim=-1)
        opacities = torch.sigmoid(self.splats["opacities"])[..., None]
        means2d = torch.zeros_like(self.splats["means"], requires_grad=True)

        for i in range(c2w.shape[0]):
            w2c = torch.linalg.inv(c2w[i])
            view = w2c.transpose(0, 1).contiguous()
            proj = projection_from_opencv(K[i], width, height)
            tanfovx, tanfovy = fov_from_K(K[i], width, height)
            settings = GaussianRasterizationSettings(
                image_height=height,
                image_width=width,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=torch.zeros(3, device=self.device),
                scale_modifier=1.0,
                viewmatrix=view,
                projmatrix=(view @ proj).contiguous(),
                sh_degree=sh_degree,
                campos=c2w[i, :3, 3].contiguous(),
                prefiltered=False,
                debug=False,
            )
            rgb, depth, _norm, alpha, radii, _extra = GaussianRasterizer(settings)(
                means3D=self.splats["means"],
                means2D=means2d,
                shs=shs,
                colors_precomp=None,
                opacities=opacities,
                scales=scales,
                rotations=rotations,
                cov3Ds_precomp=None,
            )
            rgbs.append(rgb.permute(1, 2, 0).clamp(0.0, 1.0))
            depths.append(depth.permute(1, 2, 0))
            alphas.append(alpha.permute(1, 2, 0).clamp(0.0, 1.0))
        return torch.stack(rgbs, 0), torch.stack(depths, 0), torch.stack(alphas, 0), {"radii": radii}

    def _render_mesh(self, c2w: torch.Tensor, K: torch.Tensor, width: int, height: int, names: list[str]):
        rgbs, depths, masks = [], [], []
        for i, name in enumerate(names):
            if not self.use_live_mesh and self.cache.has_frame(name):
                rgb, depth, mask = self.cache.load_tensors(name, self.device, width, height)
            else:
                rgb, depth, mask = self.mesh.render(K[i], c2w[i], width, height)
            rgbs.append(rgb)
            depths.append(depth)
            masks.append(mask)
        return torch.stack(rgbs, 0), torch.stack(depths, 0), torch.stack(masks, 0)

    def forward(self, batch: dict, step: int | None):
        gt = batch["image"].to(self.device).float() / 255.0
        c2w = batch["camtoworld"].to(self.device).float()
        K = batch["K"].to(self.device).float()
        height, width = gt.shape[1], gt.shape[2]

        gs_rgb, gs_depth, gs_alpha, info = self._rasterize_gs(c2w, K, width, height, step)
        mesh_rgb, mesh_depth, mesh_mask = self._render_mesh(c2w, K, width, height, frame_names_from_batch(batch))
        gate = compute_depth_gate(gs_depth, mesh_depth, mesh_mask, eps=self.args.gate_eps, band=self.args.gate_band)
        hybrid = compose_hybrid(gs_rgb, gs_alpha, mesh_rgb, mesh_mask, gate)
        return {
            "gt": gt,
            "mesh_rgb": mesh_rgb,
            "mesh_depth": mesh_depth,
            "mesh_mask": mesh_mask,
            "gs_rgb": gs_rgb,
            "gs_depth": gs_depth,
            "gs_alpha": gs_alpha,
            "gate": gate,
            "hybrid": hybrid,
            "info": info,
        }

    def prune_to_max_gs(self):
        if self.args.max_gs <= 0 or len(self.splats["means"]) <= self.args.max_gs:
            return
        keep = torch.topk(torch.sigmoid(self.splats["opacities"].detach()), k=self.args.max_gs).indices
        for name, value in list(self.splats.items()):
            self.splats[name] = torch.nn.Parameter(value.detach()[keep].clone())
        self.optimizers = make_optimizers(self.splats, self.scene.scene_scale)
        self.means_sched = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["means"], gamma=0.01 ** (1.0 / self.args.max_steps)
        )

    def train(self):
        loader = torch.utils.data.DataLoader(self.trainset, batch_size=1, shuffle=True, num_workers=0)
        iterator = iter(loader)
        for step in tqdm.trange(self.args.max_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            out = self.forward(batch, step)
            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)
            loss = (out["hybrid"] - out["gt"]).abs().mean()
            loss.backward()
            for opt in self.optimizers.values():
                opt.step()
            self.means_sched.step()
            self.prune_to_max_gs()

            step1 = step + 1
            append_jsonl(
                self.metrics,
                {"event": "train", "step": step1, "loss": float(loss.item()), "num_gs": len(self.splats["means"])},
            )
            if (self.args.save_every > 0 and step1 % self.args.save_every == 0) or step1 == self.args.max_steps:
                self.save_ckpt(step1)
            if (self.args.eval_every > 0 and step1 % self.args.eval_every == 0) or step1 == self.args.max_steps:
                self.eval(step1)

    @torch.no_grad()
    def eval(self, step: int):
        loader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=0)
        mesh_vals, gs_vals, hybrid_vals = [], [], []
        vis_dir = self.vis_dir / f"step_{step:06d}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        for i, batch in enumerate(loader):
            out = self.forward(batch, None)
            mesh_vals.append(psnr(out["mesh_rgb"], out["gt"]))
            gs_vals.append(psnr(out["gs_rgb"], out["gt"]))
            hybrid_vals.append(psnr(out["hybrid"], out["gt"]))
            if i < self.args.save_vis_images:
                save_visual_pack(out, vis_dir, i)
        stats = {
            "event": "eval",
            "step": step,
            "mesh_psnr": float(np.mean(mesh_vals)) if mesh_vals else 0.0,
            "gs_psnr": float(np.mean(gs_vals)) if gs_vals else 0.0,
            "hybrid_psnr": float(np.mean(hybrid_vals)) if hybrid_vals else 0.0,
        }
        append_jsonl(self.metrics, stats)
        print(f"[eval {step}] mesh={stats['mesh_psnr']:.3f} gs={stats['gs_psnr']:.3f} hybrid={stats['hybrid_psnr']:.3f}")

    def save_ckpt(self, step: int):
        torch.save(
            {"step": int(step), "splats": {k: v.detach().cpu() for k, v in self.splats.items()}},
            self.ckpt_dir / f"ckpt_{step:06d}.pt",
        )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--mesh_obj", required=True)
    ap.add_argument("--result_dir", default="runs/hybrid_simple")
    ap.add_argument("--mesh_support_dir", default="")
    ap.add_argument("--force_live_mesh", action="store_true")
    ap.add_argument("--init_means_ckpt", default="")
    ap.add_argument("--init_points", type=int, default=0)
    ap.add_argument("--init_opacity", type=float, default=0.1)
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--save_vis_images", type=int, default=4)
    ap.add_argument("--max_gs", type=int, default=0)
    ap.add_argument("--test_every", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gate_eps", type=float, default=1e-4)
    ap.add_argument("--gate_band", type=float, default=0.02)
    return ap.parse_args()


def main():
    HybridTrainer(parse_args()).train()


if __name__ == "__main__":
    main()
