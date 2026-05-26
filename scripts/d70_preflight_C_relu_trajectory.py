"""
D70 Pre-flight C: training-time dead-fraction trajectory under (gamma) loss at
lr=1e-4 (the d69 FAIL_SINKING cell, most pathological case) on the current
IGMNeRF body.

Spec (per D70 Rev 5 dispatch brief, 2026-05-25):
- 3 seeds [0, 1, 2], 500 training steps each.
- Loss: log10-MSE on Sherwood overdensity with +1e-3 floor (PRETRAIN_LOG_EPS).
- Optimizer: AdamW lr=1e-4 (matches d69 FAIL_SINKING cell; NO warmup-cosine to
  keep the probe minimal -- training-time stability of dead_frac under the
  worst observed (gamma) regime is what we care about).
- Sampling matches pipeline._pretrain_sample_voxels defaults: microbatch=1024,
  crops_per_step=4, crop_size=48 (P1 z=0.3 n_grid=768).
- Measurement (separate, no-grad) at checkpoints t in {0, 100, 250, 500}:
  pass 4096 random uniform coords in [0,1]^3 through the model with forward
  hooks on the 8 Linear sites (layers1[0..3] + layers2[0..3]); compute
  dead_frac = mean(preact <= 0) per site.
- Per-seed asymmetric drift = max_site |dead_frac(t=500) - 0.5|.
- PASS = >= 2/3 seeds with drift >= 0.05 (5 percentage points).

Output: cloud_runs/d70_preflight_C_relu_trajectory/result.json + traj.png.

NOT git-tracked, NOT DVC-tracked (consistent with pre-flight A/B convention).
CPU-only, ~30 min budget.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.nerf import IGMNeRF  # noqa: E402

# --- Config -----------------------------------------------------------------
SEEDS = [0, 1, 2]
N_STEPS = 500
CHECKPOINTS = [0, 100, 250, 500]
N_MEASURE_COORDS = 4096
HIDDEN_DIM = 256
NUM_LAYERS = 8
L_PE = 10
LR = 1e-4
MICROBATCH = 1024
CROPS_PER_STEP = 4
CROP_SIZE = 48
PRETRAIN_LOG_EPS = 1.0e-3
DEAD_DRIFT_THRESHOLD_PP = 0.05  # 5 percentage points

OUT_DIR = REPO_ROOT / "cloud_runs" / "d70_preflight_C_relu_trajectory"
OUT_JSON = OUT_DIR / "result.json"
OUT_PNG = OUT_DIR / "traj.png"

RHO_CACHE = (
    REPO_ROOT / "Sherwood" / ".rho_field_cache" / "rho_field_p1_z0.300_n768.npy"
)

SITE_NAMES = [f"layers1[{i}]" for i in range(4)] + [f"layers2[{i}]" for i in range(4)]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT)
        ).decode().strip()
    except Exception:
        return "unknown"


def _sample_voxels(rho_field, microbatch, n_crops, crop_size, generator, device):
    """Mirror of pipeline._pretrain_sample_voxels (canonical training-step
    construction). Verbatim copy of the sampling math; kept local to avoid the
    full pipeline import surface.
    """
    n_grid = rho_field.shape[0]
    corners = torch.randint(0, n_grid, (n_crops, 3), generator=generator, device=device)
    L = crop_size
    offset = torch.arange(L, device=device)

    crops = torch.empty((n_crops, L, L, L), dtype=rho_field.dtype, device=device)
    for c in range(n_crops):
        i0 = int(corners[c, 0].item())
        j0 = int(corners[c, 1].item())
        k0 = int(corners[c, 2].item())
        if (i0 + L) <= n_grid and (j0 + L) <= n_grid and (k0 + L) <= n_grid:
            crops[c] = rho_field[i0:i0+L, j0:j0+L, k0:k0+L]
        else:
            ii = (i0 + offset) % n_grid
            jj = (j0 + offset) % n_grid
            kk = (k0 + offset) % n_grid
            crops[c] = rho_field[ii[:, None, None], jj[None, :, None], kk[None, None, :]]

    per_crop = max(1, microbatch // n_crops)
    total = per_crop * n_crops
    vi = torch.randint(0, L, (n_crops, per_crop), generator=generator, device=device)
    vj = torch.randint(0, L, (n_crops, per_crop), generator=generator, device=device)
    vk = torch.randint(0, L, (n_crops, per_crop), generator=generator, device=device)
    crop_idx = torch.arange(n_crops, device=device).unsqueeze(1).expand(n_crops, per_crop)
    rho_truth = crops[crop_idx, vi, vj, vk].reshape(total)
    coords_int = torch.stack([vi, vj, vk], dim=-1).reshape(total, 3).to(torch.float32)
    coords_unit = (coords_int + 0.5) / float(L)
    return coords_unit[:microbatch], rho_truth[:microbatch].to(torch.float32)


def _pretrain_loss(rho_theta, rho_truth, eps=PRETRAIN_LOG_EPS):
    diff = torch.log10(rho_theta + eps) - torch.log10(rho_truth + eps)
    return (diff * diff).mean()


def measure_dead_fracs(model, device):
    """Run a separate measurement-only forward pass on 4096 uniform coords
    in [0,1]^3 and return per-site dead-fraction (preact <= 0)."""
    model.eval()
    site_modules = list(model.layers1) + list(model.layers2)
    assert len(site_modules) == 8

    captured = [None] * 8
    hooks = []
    for i, m in enumerate(site_modules):
        def make_hook(idx):
            def hook(_mod, _inp, out):
                captured[idx] = out.detach()
            return hook
        hooks.append(m.register_forward_hook(make_hook(i)))

    coords = torch.rand(1, N_MEASURE_COORDS, 3, device=device)
    with torch.no_grad():
        _ = model(coords)
    for h in hooks:
        h.remove()

    dead = []
    for t in captured:
        flat = t.reshape(-1)
        dead.append(float((flat <= 0).float().mean().item()))
    model.train()
    return dead


def run_one_seed(seed: int, rho_field, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = IGMNeRF(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, L=L_PE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Per-seed sampling generator (CPU - matches pipeline pattern)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 10_000)  # offset to decouple from torch.manual_seed scope

    trajectory = {}  # step -> list of 8 dead-fractions
    losses_at_chkpt = {}

    # t=0 measurement before any training step
    trajectory[0] = measure_dead_fracs(model, device)

    # Pre-step-0 loss for sanity (~ order-of-magnitude check vs d69 L_pre_step0 = 0.28)
    with torch.no_grad():
        coords0, rho0 = _sample_voxels(
            rho_field, MICROBATCH, CROPS_PER_STEP, CROP_SIZE, gen, device,
        )
        out0 = model(coords0.unsqueeze(0))
        loss0 = float(_pretrain_loss(out0[0, :, 0], rho0).item())
    losses_at_chkpt[0] = loss0

    for step in range(1, N_STEPS + 1):
        coords, rho_truth = _sample_voxels(
            rho_field, MICROBATCH, CROPS_PER_STEP, CROP_SIZE, gen, device,
        )
        out = model(coords.unsqueeze(0))
        rho_theta = out[0, :, 0]
        loss = _pretrain_loss(rho_theta, rho_truth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in CHECKPOINTS:
            trajectory[step] = measure_dead_fracs(model, device)
            losses_at_chkpt[step] = float(loss.item())

    return trajectory, losses_at_chkpt


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    t_start = time.time()
    print(f"[d70_preflight_C] loading rho_field from {RHO_CACHE}")
    if not RHO_CACHE.exists():
        raise FileNotFoundError(f"missing P1 rho-field cache: {RHO_CACHE}")
    rho_np = np.load(RHO_CACHE)
    rho_field = torch.from_numpy(rho_np).to(torch.float32).to(device)
    print(f"[d70_preflight_C] rho_field shape={tuple(rho_field.shape)} "
          f"mean={float(rho_field.mean()):.4f}")

    per_seed_results = {}
    per_seed_losses = {}

    for seed in SEEDS:
        print(f"[d70_preflight_C] starting seed={seed} ({N_STEPS} steps, "
              f"lr={LR}, mb={MICROBATCH}, crops={CROPS_PER_STEP}, crop={CROP_SIZE})")
        t0 = time.time()
        traj, losses = run_one_seed(seed, rho_field, device)
        per_seed_results[seed] = traj
        per_seed_losses[seed] = losses
        elapsed = time.time() - t0
        print(f"[d70_preflight_C] seed={seed} done in {elapsed:.1f}s; "
              f"losses at ckpts {losses}")

    # Per-seed asymmetric drift = max_site |dead_frac(500) - 0.5|
    drifts = []
    for seed in SEEDS:
        dead_500 = per_seed_results[seed][500]
        drift = max(abs(d - 0.5) for d in dead_500)
        drifts.append(drift)

    n_above = sum(1 for d in drifts if d >= DEAD_DRIFT_THRESHOLD_PP)
    verdict = "PASS" if n_above >= 2 else "FAIL"

    out = {
        "spec": (
            "training-time dead-fraction trajectory under (gamma) lr=1e-4 on "
            "current IGMNeRF, P1, 3 seeds x 500 steps"
        ),
        "seeds": SEEDS,
        "checkpoints": CHECKPOINTS,
        "n_measure_coords": N_MEASURE_COORDS,
        "lr": LR,
        "microbatch": MICROBATCH,
        "crops_per_step": CROPS_PER_STEP,
        "crop_size": CROP_SIZE,
        "pretrain_log_eps": PRETRAIN_LOG_EPS,
        "dead_drift_threshold_pp": DEAD_DRIFT_THRESHOLD_PP,
        "site_names": SITE_NAMES,
        "dead_frac_per_seed_per_checkpoint_per_site": {
            f"seed_{s}": [per_seed_results[s][t] for t in CHECKPOINTS]
            for s in SEEDS
        },
        "loss_per_seed_per_checkpoint": {
            f"seed_{s}": per_seed_losses[s] for s in SEEDS
        },
        "asymmetric_drift_per_seed": drifts,
        "n_seeds_above_5pp": n_above,
        "pre_flight_C_verdict": verdict,
        "d69_L_pre_step0_reference": 0.2769606411457062,
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_total_seconds": round(time.time() - t_start, 1),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[d70_preflight_C] wrote {OUT_JSON}")

    # 3-panel PNG: per-seed dead_frac trajectories at all 8 sites
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for ax_i, seed in enumerate(SEEDS):
        ax = axes[ax_i]
        for s in range(8):
            ys = [per_seed_results[seed][t][s] for t in CHECKPOINTS]
            ax.plot(CHECKPOINTS, ys, marker="o", color=colors[s],
                    label=SITE_NAMES[s], lw=1.4)
        ax.axhline(0.5, color="gray", lw=1.0, ls="--", label="symmetric-null 0.5")
        ax.axhline(0.5 + DEAD_DRIFT_THRESHOLD_PP, color="red", lw=0.8, ls=":")
        ax.axhline(0.5 - DEAD_DRIFT_THRESHOLD_PP, color="red", lw=0.8, ls=":")
        ax.set_xlabel("training step")
        if ax_i == 0:
            ax.set_ylabel("dead-fraction (preact <= 0)")
        ax.set_title(f"seed={seed}  drift={drifts[ax_i]:.3f}")
        ax.set_ylim(0.30, 0.70)
        if ax_i == 2:
            ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(
        f"D70 pre-flight C: training-time ReLU dead-fraction under (gamma) "
        f"lr=1e-4; n_above_5pp={n_above}/3 -> {verdict}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[d70_preflight_C] wrote {OUT_PNG}")
    print(f"[d70_preflight_C] VERDICT: {verdict} "
          f"(drifts={drifts}, n_above={n_above}/3)")


if __name__ == "__main__":
    main()
