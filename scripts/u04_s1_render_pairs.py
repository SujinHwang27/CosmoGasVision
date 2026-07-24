"""[U-04] Stage-1 R8 (A8): per-physics pair figures (S3) + occupancy stats (S4).

Spec SS2(d): S3 = one figure per physics — central slab of the input delta_F
channel, the ray-mask channel, and the target x, side by side — under
``experiments/unet-inversion/artifacts/stage1/pair_p{1..4}.png`` (PI eyeball
at the Stage-1 gate). S4 (reported, not gated) = per-crop ray occupancy at
n_rays in {64, 1024} + zero-ray rejection rate -> ``s4_occupancy.json``.

Slab definition (recorded in the figure title): mean over the 8 central
split-axis planes (local axis-0 indices 28..35) for all three panels — thick
enough that both transverse ray lines and split-axis ray dots are visible.

Run:  PYTHONPATH=. .venv/bin/python -u scripts/u04_s1_render_pairs.py
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.sightline_rasterizer import DELTA_F_SCALE  # noqa: E402
from src.data.unet_pair_dataset import (  # noqa: E402
    UNetPairDataset,
    build_physics_source,
)

REPO = Path(__file__).resolve().parents[1]
STAGE1 = REPO / "experiments/unet-inversion/artifacts/stage1"
CUBES = {
    1: REPO / "experiments/nerf/artifacts/d75_rescore/cubes/truth_real_192.npy",
    2: STAGE1 / "cubes/truth_real_192_p2.npy",
    3: STAGE1 / "cubes/truth_real_192_p3.npy",
    4: STAGE1 / "cubes/truth_real_192_p4.npy",
}
SEED = 42
N_OCC_CROPS = 100          # per physics per ray count (S4)
N_REJ_DRAWS = 400          # production-config draws for the rejection rate
SLAB = slice(28, 36)       # central split-axis slab (8 planes)


def render_figure(ds: UNetPairDataset, pid: int, out_png: Path) -> None:
    inp, tgt, spec, _ = ds.example(0)
    df_slab = inp[0][SLAB].mean(axis=0)
    mask_slab = inp[1][SLAB].mean(axis=0)
    x_slab = tgt[0][SLAB].mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (df_slab, f"input ch0: {DELTA_F_SCALE}*delta_F", "magma", None),
        (mask_slab, "input ch1: ray mask (slab mean)", "gray", (0, 1)),
        (x_slab, "target x = log10 rho/<rho>", "viridis", None),
    ]
    for ax, (img, title, cmap, clim) in zip(axes, panels):
        im = ax.imshow(img.T, origin="lower", cmap=cmap,
                       vmin=None if clim is None else clim[0],
                       vmax=None if clim is None else clim[1])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("local axis 1")
        ax.set_ylabel("local axis 2")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"[U-04] Stage-1 pair, P{pid} z=0.3 — corner {spec.corner.tolist()}, "
        f"{len(spec.ray_indices)} rays (of {spec.n_rays_available} "
        f"intersecting), slab = mean local axis-0 [28,36)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")


def occupancy_stats(ds: UNetPairDataset) -> dict:
    out = {}
    for n_rays in (64, 1024):
        ds.n_rays_fixed = n_rays
        occ, taken = [], []
        for i in range(N_OCC_CROPS):
            inp, _, spec, _ = ds.example(i)
            occ.append(float(inp[1].mean()))
            taken.append(len(spec.ray_indices))
        occ = np.array(occ)
        out[f"n_rays_{n_rays}"] = {
            "n_crops": N_OCC_CROPS,
            "occupancy_mean": float(occ.mean()),
            "occupancy_std": float(occ.std()),
            "occupancy_min": float(occ.min()),
            "occupancy_max": float(occ.max()),
            "rays_taken_mean": float(np.mean(taken)),
            "rays_taken_min": int(np.min(taken)),
            "note_shortfall": "rays_taken < n_rays when fewer rays "
                              "intersect the crop than requested",
        }
    # zero-ray rejection under the PRODUCTION config (log-uniform draw)
    ds.n_rays_fixed = None
    s = ds.samplers[0]
    d0, r0 = s.n_corner_draws, s.n_zero_ray_rejections
    rng = np.random.default_rng([SEED, 999_999])
    for _ in range(N_REJ_DRAWS):
        s.sample_spec(rng)
    out["zero_ray_rejection"] = {
        "n_draws": s.n_corner_draws - d0,
        "n_rejections": s.n_zero_ray_rejections - r0,
        "rate": (s.n_zero_ray_rejections - r0) / (s.n_corner_draws - d0),
        "pool": "all 16384 sightlines (expected intersecting per 64^3 crop "
                "~ 16384/9 ~ 1820 -> rejections expected ~0)",
    }
    return out


def main() -> None:
    s4 = {
        "rung": "R8 — S3 figures + S4 occupancy/rejection stats (A8)",
        "spec": "u04_stage1_ratification.md SS2(d), commit 58ac831",
        "config": {
            "seed": SEED, "crop_size": 64, "n_grid": 192,
            "delta_f_scale": DELTA_F_SCALE,
            "occupancy_crops_per_setting": N_OCC_CROPS,
            "rejection_draws": N_REJ_DRAWS,
        },
        "per_physics": {},
        "cross_physics_note": "occupancy/rejection stats are IDENTICAL "
                              "across P1-P4 by construction: the four los "
                              "files carry byte-identical ray geometry "
                              "(iaxis/x/y/zaxis/pos_axis verified equal "
                              "this session; axis counts 5439/5452/5493) "
                              "and corners are (seed,index)-deterministic; "
                              "only delta_F values differ per physics.",
    }
    for pid in (1, 2, 3, 4):
        src = build_physics_source(
            str(REPO / "Sherwood"), pid, str(CUBES[pid]),
            provider_seed=SEED,
        )
        ds = UNetPairDataset([src], length=10_000, seed=SEED,
                             n_rays_fixed=1024, augment=False)
        render_figure(ds, pid, STAGE1 / f"pair_p{pid}.png")
        s4["per_physics"][f"P{pid}"] = occupancy_stats(ds)
        del src, ds
        gc.collect()
    out = STAGE1 / "s4_occupancy.json"
    out.write_text(json.dumps(s4, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
