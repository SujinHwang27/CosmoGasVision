"""
D70 Pre-flight D: operative-cause swap-back test (M0-PASS-conditional).

Spec (per D70 Rev 5.1 amendment block S1, Stage 1a precondition)
----------------------------------------------------------------
At matched seed, instantiate BOTH (1b) skip-rich-mlp AND current-ReLU body
variants; train each under (γ) loss at lr=1e-4 for ≥500 steps on P1 z=0.3.
Compare Var_ratio trajectories.

PASS criteria:
  - (1b) Var_ratio(500) > current Var_ratio(500) by ≥ 2 σ_within in ≥ 2/3 seeds.
  - AND swap-back: current arch at same seed shows the pathology
    (Var_ratio(500) ≤ Var_ratio(0)).

Pre-condition gate:
  - **This harness MUST NOT EXECUTE until Stage 1a M0 lands a PASS verdict.**
    Execution path is gated by an explicit ``--i-am-post-M0-pass`` flag; if
    the flag is absent, the harness exits with code 3 and prints the gate
    rationale. This is the M0-PASS-conditional binding from S1.

Isolation logic
---------------
If (1b) escapes the dying-ReLU pathology AND current arch fails at matched
seed/data, the body architecture is structurally implicated as the operative
axis — not the activation, not the data, not the optimizer.

CPU pre-flight (P1 z=0.3, n_grid=64) per design §1.6 line 356 contract.

Out-of-scope (DO NOT execute under this dispatch)
-------------------------------------------------
This file lands the spec and the harness skeleton. Execution is performed in
the post-M0 sprint; this dispatch only ships the code so it can be reviewed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.nerf import IGMNeRF  # noqa: E402

# --- Config -----------------------------------------------------------------
SEEDS = [0, 1, 2]
N_STEPS = 500
N_MEASURE_COORDS = 4096
HIDDEN_DIM = 256
NUM_LAYERS = 8
L_PE = 10
LR = 1e-4
MICROBATCH = 1024
CROPS_PER_STEP = 4
CROP_SIZE = 48          # P1 z=0.3 CPU pre-flight
PRETRAIN_LOG_EPS = 1.0e-3
SIGMA_WITHIN_MULTIPLIER = 2.0

OUT_DIR = REPO_ROOT / "cloud_runs" / "d70_preflight_D_operative_cause_swap"
OUT_JSON = OUT_DIR / "result.json"

# P1 z=0.3 rho-field cache (small CPU-friendly grid).
RHO_CACHE = (
    REPO_ROOT / "Sherwood" / ".rho_field_cache" / "rho_field_p1_z0.300_n64.npy"
)


def _sample_voxels(rho_field, microbatch, n_crops, crop_size, generator, device):
    """Mirror of pipeline._pretrain_sample_voxels."""
    n_grid = rho_field.shape[0]
    corners = torch.randint(0, n_grid, (n_crops, 3), generator=generator, device=device)
    L = crop_size
    offset = torch.arange(L, device=device)

    crops = torch.empty((n_crops, L, L, L), dtype=rho_field.dtype, device=device)
    for c in range(n_crops):
        i0 = int(corners[c, 0].item())
        j0 = int(corners[c, 1].item())
        k0 = int(corners[c, 2].item())
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


def _measure_var_ratio(model, device, gen):
    """Var_ratio metric: ratio of post-Softplus density variance over a fixed
    measurement-coord set, to a baseline variance (computed at t=0).

    See design §2.2 (amended) for the canonical Var_ratio definition.
    Implementation note: caller provides the measurement gen + tracks ratios.
    """
    model.eval()
    coords = torch.rand(1, N_MEASURE_COORDS, 3, generator=gen, device=device)
    with torch.no_grad():
        out = model(coords)
    rho = out[0, :, 0]
    model.train()
    return float(rho.var().item())


def run_one_seed(seed, body_arch, rho_field, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = IGMNeRF(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, L=L_PE,
                    body_arch=body_arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    sample_gen = torch.Generator(device=device)
    sample_gen.manual_seed(seed + 10_000)

    # Measurement generator uses an independent fixed seed so both arches
    # see identical measurement coords (controls for σ_redraw confound).
    measure_gen = torch.Generator(device=device)
    measure_gen.manual_seed(seed + 90_000)
    var0 = _measure_var_ratio(model, device, measure_gen)

    for step in range(1, N_STEPS + 1):
        coords, rho_truth = _sample_voxels(
            rho_field, MICROBATCH, CROPS_PER_STEP, CROP_SIZE, sample_gen, device,
        )
        out = model(coords.unsqueeze(0))
        rho_theta = out[0, :, 0]
        loss = _pretrain_loss(rho_theta, rho_truth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    measure_gen_final = torch.Generator(device=device)
    measure_gen_final.manual_seed(seed + 90_000)  # same coords as t=0
    var500 = _measure_var_ratio(model, device, measure_gen_final)
    ratio = var500 / max(var0, 1e-30)
    return {"var0": var0, "var500": var500, "var_ratio": ratio}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--i-am-post-M0-pass", action="store_true",
        help="Required gate: this harness only executes after Stage 1a M0 PASS.",
    )
    args = parser.parse_args()
    if not args.i_am_post_M0_pass:
        print("[d70_preflight_D] GATE: M0-PASS-conditional execution.")
        print("[d70_preflight_D] Re-run with --i-am-post-M0-pass once Stage 1a M0 lands PASS.")
        return 3

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    if not RHO_CACHE.exists():
        raise FileNotFoundError(f"missing P1 z=0.3 n=64 rho-field cache: {RHO_CACHE}")
    rho_np = np.load(RHO_CACHE)
    rho_field = torch.from_numpy(rho_np).to(torch.float32).to(device)

    results = {"current": {}, "skip-rich-mlp": {}}
    t0 = time.time()
    for seed in SEEDS:
        for arch in ("current", "skip-rich-mlp"):
            print(f"[d70_preflight_D] seed={seed} arch={arch}")
            r = run_one_seed(seed, arch, rho_field, device)
            results[arch][seed] = r
            print(f"  -> var0={r['var0']:.4e}  var500={r['var500']:.4e}  ratio={r['var_ratio']:.3f}")

    # PASS criteria
    ratios_current = [results["current"][s]["var_ratio"] for s in SEEDS]
    ratios_skip = [results["skip-rich-mlp"][s]["var_ratio"] for s in SEEDS]

    sigma_within = float(np.std(ratios_current))  # within-arch noise estimate
    diffs = [rs - rc for rs, rc in zip(ratios_skip, ratios_current)]
    n_above = sum(1 for d in diffs if d > SIGMA_WITHIN_MULTIPLIER * sigma_within)
    swap_back_pathology = all(
        results["current"][s]["var500"] <= results["current"][s]["var0"]
        for s in SEEDS
    )

    verdict_skip_dominates = n_above >= 2  # ≥ 2/3 seeds
    verdict = "PASS" if (verdict_skip_dominates and swap_back_pathology) else "FAIL"

    out = {
        "seeds": SEEDS,
        "n_steps": N_STEPS,
        "lr": LR,
        "results": results,
        "sigma_within": sigma_within,
        "n_seeds_skip_dominates": n_above,
        "swap_back_pathology": swap_back_pathology,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[d70_preflight_D] VERDICT={verdict}; wrote {OUT_JSON}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
