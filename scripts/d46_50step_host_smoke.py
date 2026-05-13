"""[D-46] 50-step P-mixed host smoke on real Sherwood data.

Sprint-6 prerequisite (1/3) — runs the [D-46] joint-physics conditional
MLP for 50 steps on the pooled 4-physics dataset, then evaluates the 7
smoke gates pre-registered in LEDGER \xa73 [D-46] Smoke-gate spec. Writes
``experiments/nerf/artifacts/eval/d46_smoke/<run_id>_gates.json``.

Defense-panel review (prereq 2/3) consumes this JSON. Juno sbatch
submission (prereq 3/3) is gated on panel PASS.

The 7 gates (per [D-46] entry):
  1. No NaN / Inf
  2. loss(50) / loss(10) < 0.85               (descent, [D-19])
  3. mean_F in [0.5, 0.99] AND |mean_F - 1| > 1e-3
                                              (trivial-collapse backstop)
  4. tau_amp in [0.5, 2.0]                    (amplitude not exploded)
  5. Density spread >= 1.45 per physics       (anti-collapse, [D-41]/[D-42])
  6. X_HI spread >= 6e-5 per physics          (asymmetric anti-collapse)
  7. Embedding pairwise L2 max > 0.1          (non-degeneracy, [D-46] D-new)

Run:
    PYTHONPATH=. python -u scripts/d46_50step_host_smoke.py [--n_rays 1024]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Make repo root importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from experiments.nerf.pipeline import (  # noqa: E402
    IGMNeRF,
    build_lr_lambda,
    load_dataset,
    parse_args,
    set_global_seed,
)
from src.models.nerf import volume_render_physics  # noqa: E402

OUT_DIR = Path(_REPO) / "experiments/nerf/artifacts/eval/d46_smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Gate thresholds per LEDGER \xa73 [D-46] Smoke-gate spec
GATE2_LOSS_RATIO_MAX = 0.85
GATE3_MEAN_F_LO = 0.5
GATE3_MEAN_F_HI = 0.99
GATE3_MEAN_F_DELTA_MIN = 1e-3
GATE4_TAU_AMP_LO = 0.5
GATE4_TAU_AMP_HI = 2.0
GATE5_DENSITY_SPREAD_MIN = 1.45
GATE6_XHI_SPREAD_MIN = 6e-5
GATE7_EMBED_L2_MIN = 0.1

# Step at which to record the "early-loss" reference for the ratio gate
LOSS_REF_STEP = 10
# Per-physics post-train eval sample size for gates 3-6
POST_EVAL_RAYS_PER_PHYSICS = 256


def _make_microbatch_indices(
    pool_by_physics: list[torch.Tensor],
    per_physics_quota: int,
    rng: torch.Generator,
) -> torch.Tensor:
    chunks = []
    for p in range(4):
        pool = pool_by_physics[p]
        perm = torch.randperm(pool.numel(), generator=rng)[:per_physics_quota]
        chunks.append(pool[perm])
    return torch.cat(chunks, dim=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rays", type=int, default=1024,
                    help="Pooled across 4 physics by the joint loader.")
    ap.add_argument("--microbatch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--data_root", type=str, default="Sherwood")
    ap.add_argument("--hidden_dim", type=int, default=256,
                    help="Production scale = 256. Smaller for memory-tight hosts.")
    args = ap.parse_args()

    run_id = f"d46_smoke_{int(time.time())}"
    print(f"[d46_50step] run_id={run_id}", flush=True)

    # Build pipeline args so we reuse parse_args's validation + defaults.
    argv = [
        "--n_rays", str(args.n_rays),
        "--physics", "1",  # logging hint only when --use_physics_embedding
        "--seed", str(args.seed),
        "--microbatch", str(args.microbatch),
        "--max_steps", str(args.steps),
        "--warmup_steps", "5",
        "--use_physics_embedding",
        "--lambda_inv", "0.0",
        "--data_root", args.data_root,
    ]
    pargs = parse_args(argv)
    if pargs.accum_steps is None:
        pargs.accum_steps = max(1, math.ceil(pargs.n_rays / pargs.microbatch))
    print(
        f"[d46_50step] args: n_rays={pargs.n_rays} microbatch={pargs.microbatch} "
        f"accum_steps={pargs.accum_steps} steps={pargs.max_steps} "
        f"hidden_dim={args.hidden_dim} data_root={args.data_root}",
        flush=True,
    )

    # 1) Dataset
    set_global_seed(args.seed)
    print("[d46_50step] loading dataset ...", flush=True)
    t_load_start = time.perf_counter()
    (
        coords,
        vel_axis,
        tau_gt_profile,
        mask_no_dla_profile,
        box_max,
        v_pec_grad_profile,
        physics_id_per_ray,
    ) = load_dataset(pargs)
    print(
        f"[d46_50step] dataset loaded in {time.perf_counter()-t_load_start:.1f}s — "
        f"coords {tuple(coords.shape)} physics histogram "
        f"{torch.bincount(physics_id_per_ray, minlength=4).tolist()}",
        flush=True,
    )
    assert (torch.bincount(physics_id_per_ray, minlength=4) > 0).all(), (
        "Pooled dataset must contain all 4 physics buckets."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[d46_50step] device={device}", flush=True)

    # 2) Model
    torch.manual_seed(args.seed)
    model = IGMNeRF(
        hidden_dim=args.hidden_dim, num_layers=8, L=10,
        use_physics_embedding=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[d46_50step] model params: {n_params}", flush=True)

    log_tau_amp = nn.Parameter(torch.tensor(0.0, device=device))
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [log_tau_amp],
        lr=pargs.lr_max, betas=(0.9, 0.999), weight_decay=1e-6,
    )

    # 3) Training loop with per-step interleaver (mirrors pipeline.py)
    d46_rng = torch.Generator().manual_seed(args.seed + 4646)
    pool_by_physics = [
        torch.nonzero(physics_id_per_ray == p_idx, as_tuple=True)[0]
        for p_idx in range(4)
    ]
    per_physics_quota = pargs.microbatch // 4

    coords_dev = coords.to(device)
    vel_axis_dev = vel_axis.to(device)
    tau_gt_dev = tau_gt_profile.to(device)
    physics_id_dev = physics_id_per_ray.to(device)

    loss_history: list[float] = []
    has_nan = False

    t_train_start = time.perf_counter()
    for step in range(args.steps):
        idx_mb = _make_microbatch_indices(
            pool_by_physics, per_physics_quota, d46_rng,
        ).to(device)
        pid_mb = physics_id_dev[idx_mb]

        tau_amp = torch.exp(log_tau_amp)
        tau_pred_mb = volume_render_physics(
            model, coords_dev[idx_mb], vel_axis=vel_axis_dev,
            tau_amp=tau_amp, physics_id=pid_mb,
        )
        tau_gt_mb = tau_gt_dev[idx_mb].clamp_max(10.0)
        tau_pred_eff = tau_pred_mb.clamp_max(10.0)
        loss = ((torch.log1p(tau_pred_eff) - torch.log1p(tau_gt_mb)) ** 2).mean()

        if not torch.isfinite(loss):
            has_nan = True
            print(
                f"[d46_50step] step {step+1}: NaN/Inf in loss "
                f"-> gate 1 FAIL — aborting",
                flush=True,
            )
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Detect NaN in grads (extra gate-1 protection)
        for name, par in model.named_parameters():
            if par.grad is not None and not torch.isfinite(par.grad).all():
                has_nan = True
                print(
                    f"[d46_50step] step {step+1}: NaN/Inf in grad of "
                    f"{name} -> gate 1 FAIL — aborting",
                    flush=True,
                )
                break
        if has_nan:
            break
        optimizer.step()
        loss_history.append(loss.item())
        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"[d46_50step] step {step+1}/{args.steps}: "
                f"loss={loss.item():.6f} tau_amp={tau_amp.item():.4f}",
                flush=True,
            )
    t_train_elapsed = time.perf_counter() - t_train_start
    print(f"[d46_50step] training done in {t_train_elapsed:.1f}s", flush=True)

    # 4) Gate 1 first; if it failed mid-training, emit a minimal result.
    if has_nan:
        result = {
            "run_id": run_id,
            "spec": "[D-46] 50-step P-mixed host smoke",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "steps_completed": len(loss_history),
            "loss_history": loss_history,
            "gates": {"gate_1_no_nan_inf": {"pass": False}},
            "overall_pass": False,
        }
        out_path = OUT_DIR / f"{run_id}_gates.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(f"[d46_50step] gate 1 FAIL — artifact at {out_path}", flush=True)
        return 1

    # 5) Post-train evaluation: per-physics field statistics on fixed sample
    print("[d46_50step] post-train evaluation ...", flush=True)
    model.eval()
    final_tau_amp = float(torch.exp(log_tau_amp).item())

    eval_rng = torch.Generator().manual_seed(args.seed + 9090)
    mean_F_per_physics: list[float] = []
    density_spread_per_physics: list[float] = []
    xhi_spread_per_physics: list[float] = []
    density_min_per_physics: list[float] = []
    density_max_per_physics: list[float] = []
    xhi_min_per_physics: list[float] = []
    xhi_max_per_physics: list[float] = []

    with torch.no_grad():
        for p_idx in range(4):
            pool = pool_by_physics[p_idx]
            n_eval = min(POST_EVAL_RAYS_PER_PHYSICS, pool.numel())
            sel = torch.randperm(pool.numel(), generator=eval_rng)[:n_eval]
            idx = pool[sel].to(device)
            pid = torch.full((idx.numel(),), p_idx, dtype=torch.long, device=device)

            # Volume-render for mean_F per physics
            tau_pred = volume_render_physics(
                model, coords_dev[idx], vel_axis=vel_axis_dev,
                tau_amp=torch.exp(log_tau_amp), physics_id=pid,
            )
            F_pred = torch.exp(-tau_pred.clamp_max(50.0))
            mean_F_per_physics.append(float(F_pred.mean().item()))

            # Field-level forward for density / X_HI spread
            fields = model(coords_dev[idx], physics_id=pid)
            rho_field = fields[..., 0]
            xhi_field = fields[..., 2]

            rho_min = float(rho_field.min().item())
            rho_max = float(rho_field.max().item())
            xhi_min = float(xhi_field.min().item())
            xhi_max = float(xhi_field.max().item())

            density_spread_per_physics.append(rho_max - rho_min)
            xhi_spread_per_physics.append(xhi_max - xhi_min)
            density_min_per_physics.append(rho_min)
            density_max_per_physics.append(rho_max)
            xhi_min_per_physics.append(xhi_min)
            xhi_max_per_physics.append(xhi_max)

    # 6) Gate computation
    # Gate 1: no NaN/Inf (already PASS if we reached here)
    gate1 = {"pass": True}

    # Gate 2: loss ratio
    if len(loss_history) >= LOSS_REF_STEP:
        loss_ref = loss_history[LOSS_REF_STEP - 1]
        loss_end = loss_history[-1]
        loss_ratio = loss_end / max(loss_ref, 1e-12)
    else:
        loss_ref = float("nan")
        loss_end = float("nan")
        loss_ratio = float("nan")
    gate2 = {
        "pass": bool(loss_ratio < GATE2_LOSS_RATIO_MAX) if math.isfinite(loss_ratio) else False,
        "threshold": f"loss({args.steps})/loss({LOSS_REF_STEP}) < {GATE2_LOSS_RATIO_MAX}",
        "observed_ratio": loss_ratio,
        "observed_loss_ref": loss_ref,
        "observed_loss_end": loss_end,
    }

    # Gate 3: mean_F window (per-physics + overall checks)
    mean_F_overall = float(np.mean(mean_F_per_physics))
    gate3_in_window = all(
        GATE3_MEAN_F_LO <= m <= GATE3_MEAN_F_HI for m in mean_F_per_physics
    )
    gate3_not_collapsed = all(
        abs(m - 1.0) > GATE3_MEAN_F_DELTA_MIN for m in mean_F_per_physics
    )
    gate3 = {
        "pass": bool(gate3_in_window and gate3_not_collapsed),
        "threshold": (
            f"per-physics mean_F in [{GATE3_MEAN_F_LO}, {GATE3_MEAN_F_HI}] "
            f"AND |mean_F - 1| > {GATE3_MEAN_F_DELTA_MIN}"
        ),
        "observed_per_physics": mean_F_per_physics,
        "observed_overall": mean_F_overall,
    }

    # Gate 4: tau_amp
    gate4 = {
        "pass": bool(GATE4_TAU_AMP_LO <= final_tau_amp <= GATE4_TAU_AMP_HI),
        "threshold": f"tau_amp in [{GATE4_TAU_AMP_LO}, {GATE4_TAU_AMP_HI}]",
        "observed": final_tau_amp,
    }

    # Gate 5: density spread per physics
    gate5 = {
        "pass": all(s >= GATE5_DENSITY_SPREAD_MIN for s in density_spread_per_physics),
        "threshold": f"density_spread >= {GATE5_DENSITY_SPREAD_MIN} per physics",
        "observed_spread_per_physics": density_spread_per_physics,
        "observed_min_per_physics": density_min_per_physics,
        "observed_max_per_physics": density_max_per_physics,
    }

    # Gate 6: X_HI spread per physics
    gate6 = {
        "pass": all(s >= GATE6_XHI_SPREAD_MIN for s in xhi_spread_per_physics),
        "threshold": f"X_HI spread >= {GATE6_XHI_SPREAD_MIN} per physics",
        "observed_spread_per_physics": xhi_spread_per_physics,
        "observed_min_per_physics": xhi_min_per_physics,
        "observed_max_per_physics": xhi_max_per_physics,
    }

    # Gate 7: embedding non-degeneracy
    emb = model.physics_embedding.weight.detach().cpu().numpy()  # (4, 16)
    pairwise = []
    for i in range(4):
        for j in range(i + 1, 4):
            d = float(np.linalg.norm(emb[i] - emb[j]))
            pairwise.append({"p_i": i, "p_j": j, "l2": d})
    max_dist = max(pd["l2"] for pd in pairwise) if pairwise else 0.0
    gate7 = {
        "pass": bool(max_dist > GATE7_EMBED_L2_MIN),
        "threshold": f"max pairwise L2 > {GATE7_EMBED_L2_MIN}",
        "observed_max": max_dist,
        "observed_pairwise": pairwise,
    }

    gates = {
        "gate_1_no_nan_inf": gate1,
        "gate_2_loss_descent": gate2,
        "gate_3_mean_F_window": gate3,
        "gate_4_tau_amp_window": gate4,
        "gate_5_density_spread": gate5,
        "gate_6_xhi_spread": gate6,
        "gate_7_embedding_nondegeneracy": gate7,
    }
    all_pass = all(g["pass"] for g in gates.values())

    result = {
        "run_id": run_id,
        "spec": "[D-46] 50-step P-mixed host smoke per LEDGER \xa73 [D-46] Smoke-gate spec",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "steps_completed": args.steps,
        "training_seconds": t_train_elapsed,
        "model_params": n_params,
        "hidden_dim": args.hidden_dim,
        "n_rays_pooled": args.n_rays,
        "microbatch": args.microbatch,
        "seed": args.seed,
        "loss_history": loss_history,
        "final_tau_amp": final_tau_amp,
        "gates": gates,
        "overall_pass": all_pass,
    }

    out_path = OUT_DIR / f"{run_id}_gates.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)

    print("\n[d46_50step] gate summary:")
    for name, g in gates.items():
        verdict = "PASS" if g["pass"] else "FAIL"
        print(f"  {name}: {verdict}")
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"  Artifact: {out_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
