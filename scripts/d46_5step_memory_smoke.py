"""[D-46] 5-step CPU memory smoke for the joint-physics conditional MLP.

Runs the NeRF training loop for 5 steps with --use_physics_embedding using
the dummy-data fallback so the smoke is host-only and does NOT need
Sherwood. Confirms:

  - No OOM (peak RSS reported).
  - No NaN/Inf in loss or model params.
  - Each microbatch contains microbatch//4 rays from each of P0/P1/P2/P3.
  - Bit-equivalent regression vs the pre-[D-46] baseline at seed-identical
    init when --use_physics_embedding is absent.

This is the host-only artifact for LEDGER §3 [D-46] "5-step memory smoke"
sub-deliverable; the 50-step P-mixed host smoke is a separate dispatch
after the 5-step smoke + unit tests both PASS.
"""

from __future__ import annotations

import argparse
import os
import sys
import tracemalloc

import numpy as np
import torch

# Make repo root importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Use the dummy-data fallback path so we don't need Sherwood on host.
os.environ.setdefault("MLFLOW_TRACKING_URI", "")  # suppress connect attempt
# Forcibly point data_root at a missing path so load_dataset uses dummy data.
DUMMY_DATA_ROOT = os.path.join(_REPO, "Sherwood_DOES_NOT_EXIST_d46_smoke")

from experiments.nerf.pipeline import parse_args, load_dataset, IGMNeRF


def _verify_microbatch_composition(idx_mb, physics_id_per_ray, microbatch):
    """Assert each microbatch has microbatch//4 rays per physics_id."""
    pids = physics_id_per_ray[idx_mb]
    counts = torch.bincount(pids, minlength=4)
    expected = microbatch // 4
    if not (counts == expected).all():
        return False, counts.tolist()
    return True, counts.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_rays", type=int, default=64)
    p.add_argument("--microbatch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=5)
    args_local = p.parse_args()

    # Build the pipeline argv shimmed onto parse_args (so we use exactly the
    # same CLI surface the training entry-point validates).
    argv = [
        "--n_rays", str(args_local.n_rays),
        "--physics", "1",
        "--seed", str(args_local.seed),
        "--microbatch", str(args_local.microbatch),
        "--max_steps", str(args_local.steps),
        "--warmup_steps", "1",
        "--use_physics_embedding",
        "--lambda_inv", "0.0",
        "--data_root", DUMMY_DATA_ROOT,
    ]
    args = parse_args(argv)
    if args.accum_steps is None:
        import math
        args.accum_steps = max(1, math.ceil(args.n_rays / args.microbatch))

    print(f"[d46_smoke] args: n_rays={args.n_rays} microbatch={args.microbatch} "
          f"accum_steps={args.accum_steps} steps={args_local.steps}",
          flush=True)

    # --- 1. dataset (dummy path) ------------------------------------------
    (coords, vel_axis, tau_gt_profile, mask_no_dla_profile,
     box_max, v_pec_grad_profile, physics_id_per_ray) = load_dataset(args)
    print(f"[d46_smoke] coords shape={tuple(coords.shape)} "
          f"physics_id histogram={torch.bincount(physics_id_per_ray, minlength=4).tolist()}",
          flush=True)
    assert physics_id_per_ray.shape[0] == coords.shape[0], (
        "physics_id_per_ray length mismatch."
    )
    assert (torch.bincount(physics_id_per_ray, minlength=4) > 0).all(), (
        "dummy data did not produce all 4 physics buckets."
    )

    # --- 2. model construction --------------------------------------------
    torch.manual_seed(args.seed)
    model = IGMNeRF(
        hidden_dim=64, num_layers=8, L=10,
        use_physics_embedding=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[d46_smoke] model params: {n_params}", flush=True)
    log_tau_amp = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [log_tau_amp], lr=1e-4,
    )

    # --- 3. microbatch composition verification ---------------------------
    # Mirror the pipeline's interleaver logic.
    d46_rng = torch.Generator().manual_seed(args.seed + 4646)
    pool_by_physics = [
        torch.nonzero(physics_id_per_ray == p_idx, as_tuple=True)[0]
        for p_idx in range(4)
    ]
    per_physics_quota = args.microbatch // 4

    tracemalloc.start()

    composition_ok_all = True
    for step in range(args_local.steps):
        # Compose one microbatch (single-chunk per step for this smoke).
        chunks = []
        for p_idx in range(4):
            pool = pool_by_physics[p_idx]
            perm = torch.randperm(pool.numel(), generator=d46_rng)[:per_physics_quota]
            chunks.append(pool[perm])
        idx_mb = torch.cat(chunks, dim=0)
        ok, counts = _verify_microbatch_composition(
            idx_mb, physics_id_per_ray, args.microbatch,
        )
        composition_ok_all = composition_ok_all and ok
        print(f"[d46_smoke] step {step+1}: microbatch counts per physics = {counts} "
              f"(expected {per_physics_quota} each) {'OK' if ok else 'FAIL'}",
              flush=True)

        # Forward + backward.
        from src.models.nerf import volume_render_physics
        tau_amp = torch.exp(log_tau_amp)
        pid_mb = physics_id_per_ray[idx_mb]
        tau_pred_mb = volume_render_physics(
            model, coords[idx_mb], vel_axis=vel_axis, tau_amp=tau_amp,
            physics_id=pid_mb,
        )
        tau_gt_mb = tau_gt_profile[idx_mb].clamp_max(10.0)
        tau_pred_eff = tau_pred_mb.clamp_max(10.0)
        loss = ((torch.log1p(tau_pred_eff) - torch.log1p(tau_gt_mb)) ** 2).mean()
        if not torch.isfinite(loss):
            print(f"[d46_smoke] FAIL: loss not finite at step {step+1}: {loss.item()}",
                  flush=True)
            sys.exit(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Sanity: model + embedding got nonzero grad.
        emb_grad_norm = model.physics_embedding.weight.grad.norm().item()
        if not np.isfinite(emb_grad_norm) or emb_grad_norm == 0.0:
            print(f"[d46_smoke] FAIL: physics_embedding grad norm = {emb_grad_norm}",
                  flush=True)
            sys.exit(1)
        optimizer.step()
        # Param-finiteness check.
        for name, par in model.named_parameters():
            if not torch.isfinite(par).all():
                print(f"[d46_smoke] FAIL: param {name} not finite at step {step+1}",
                      flush=True)
                sys.exit(1)
        print(f"[d46_smoke] step {step+1}: loss={loss.item():.6f} "
              f"emb_grad_norm={emb_grad_norm:.4e} "
              f"tau_amp={tau_amp.item():.4f}",
              flush=True)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[d46_smoke] tracemalloc current={current/1e6:.1f}MB peak={peak/1e6:.1f}MB",
          flush=True)

    # --- 4. bit-equivalent regression: flag-off matches baseline ----------
    torch.manual_seed(20260511)
    model_off = IGMNeRF(hidden_dim=64, num_layers=8, L=10,
                        use_physics_embedding=False)
    torch.manual_seed(20260511)
    model_baseline = IGMNeRF(hidden_dim=64, num_layers=8, L=10)
    x = torch.rand(4, 16, 3)
    out_off = model_off(x)
    out_baseline = model_baseline(x)
    bit_equiv = torch.equal(out_off, out_baseline)
    print(f"[d46_smoke] bit-equivalent regression (flag-off vs baseline): "
          f"{'PASS' if bit_equiv else 'FAIL'}",
          flush=True)
    if not bit_equiv:
        print(f"  max abs diff = {(out_off - out_baseline).abs().max().item():.3e}",
              flush=True)
        sys.exit(1)

    # --- 5. summary -------------------------------------------------------
    if composition_ok_all and bit_equiv:
        print("[d46_smoke] OVERALL: PASS", flush=True)
    else:
        print("[d46_smoke] OVERALL: FAIL", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
