from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_step(path: Path) -> int:
    m = re.search(r"(ckpt|means)_(\d+)\.pt$", path.name)
    if not m:
        raise ValueError(f"Bad snapshot name: {path}")
    return int(m.group(2))


def list_snapshots(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = list(path.glob("means_*.pt")) + list(path.glob("ckpt_*.pt"))
    files = sorted(files, key=parse_step)
    if not files:
        raise RuntimeError(f"No means_*.pt or ckpt_*.pt in {path}")
    return files


def load_means(path: Path) -> np.ndarray:
    data = torch.load(path, map_location="cpu")
    if "means" in data:
        means = data["means"]
    else:
        means = data["splats"]["means"]
    return means.detach().cpu().numpy()


def nearest_step(paths: list[Path], target: int) -> Path:
    return min(paths, key=lambda p: abs(parse_step(p) - target))


def first_after_step(paths: list[Path], target: int) -> Path | None:
    for p in paths:
        if parse_step(p) > target:
            return p
    return None


def pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    _, _, vh = np.linalg.svd(xc, full_matrices=False)
    basis = vh[:2].T
    proj = xc @ basis
    return proj, mu[0], basis


def project(x: np.ndarray, mu: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (x - mu[None, :]) @ basis


def detect_births(prev_pts: np.ndarray, cur_pts: np.ndarray, eps: float) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(prev_pts)
        d, _ = tree.query(cur_pts, k=1)
        return d > eps
    except Exception:
        births = np.zeros((len(cur_pts),), dtype=bool)
        if len(cur_pts) > len(prev_pts):
            births[-(len(cur_pts) - len(prev_pts)) :] = True
        return births


def save_txt(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_path", type=str, required=True, help="dir with means_*.pt or ckpt_*.pt")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--warmup_step", type=int, default=600)
    ap.add_argument("--after_step", type=int, default=-1)
    ap.add_argument("--sample_arrows", type=int, default=2500)
    ap.add_argument("--birth_eps_rel", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    trace_path = Path(args.trace_path)
    paths = list_snapshots(trace_path)

    p0 = paths[0]
    pw = nearest_step(paths, args.warmup_step)
    if args.after_step > 0:
        pa = nearest_step(paths, args.after_step)
    else:
        pa = first_after_step(paths, parse_step(pw))
        if pa is None:
            raise RuntimeError("Need at least one snapshot after warmup_step")

    m0 = load_means(p0)
    mw = load_means(pw)
    ma = load_means(pa)

    all_pts = np.concatenate([m0, mw, ma], axis=0)
    bb_min = all_pts.min(axis=0)
    bb_max = all_pts.max(axis=0)
    diag = float(np.linalg.norm(bb_max - bb_min))
    eps = max(1e-8, args.birth_eps_rel * diag)

    births = detect_births(mw, ma, eps=eps)

    n_common = min(len(m0), len(mw))
    idx = np.arange(n_common)
    if len(idx) > args.sample_arrows:
        idx = rng.choice(idx, size=args.sample_arrows, replace=False)

    _, mu, basis = pca2(np.concatenate([m0[:n_common], mw[:n_common]], axis=0))
    p0_2d = project(m0[:n_common], mu, basis)
    pw_2d = project(mw[:n_common], mu, basis)
    pa_2d = project(ma, mu, basis)

    out_dir = Path(args.out_dir) if args.out_dir else (trace_path.parent / "gs_evolution")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.scatter(p0_2d[:, 0], p0_2d[:, 1], s=1, alpha=0.15, label=f"start ({parse_step(p0)})")
    ax.scatter(pw_2d[:, 0], pw_2d[:, 1], s=1, alpha=0.2, label=f"warmup_end ({parse_step(pw)})")
    du = pw_2d[idx] - p0_2d[idx]
    ax.quiver(
        p0_2d[idx, 0],
        p0_2d[idx, 1],
        du[:, 0],
        du[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0012,
        alpha=0.35,
    )
    ax.set_title("Warmup GS motion (PCA projection)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "warmup_motion.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.scatter(pa_2d[~births, 0], pa_2d[~births, 1], s=1, alpha=0.08, label="existing")
    if births.any():
        ax.scatter(pa_2d[births, 0], pa_2d[births, 1], s=6, alpha=0.55, label="new after warmup")
    ax.set_title(f"New GS after warmup: warmup={parse_step(pw)}, after={parse_step(pa)}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "births_after_warmup.png", dpi=180)
    plt.close(fig)

    lines = [
        f"snapshot_start={p0.name} step={parse_step(p0)} count={len(m0)}",
        f"snapshot_warmup={pw.name} step={parse_step(pw)} count={len(mw)}",
        f"snapshot_after={pa.name} step={parse_step(pa)} count={len(ma)}",
        f"birth_eps_abs={eps:.8f}",
        f"new_after_warmup={int(births.sum())}",
        f"new_after_warmup_ratio={float(births.mean()):.6f}",
        f"artifacts: warmup_motion.png, births_after_warmup.png",
    ]
    save_txt(out_dir / "summary.txt", lines)
    print("[ok] wrote", out_dir)


if __name__ == "__main__":
    main()
