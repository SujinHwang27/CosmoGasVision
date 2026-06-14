#!/usr/bin/env python
"""[D-73] (1d') §S host-side 3D-xi consumer (PCV gate (ii)(b)).

Loads a copied-out ``step_*.pt`` voxel-grid checkpoint, reconstructs the linear
rho/<rho> cube at G=192, downsamples the n768 P1 z=0.3 truth rho cube to 192^3 by
mean-pooling (CIC-consistent, conserves <rho>), and scores the true [D-13] 3D
density-density cross-correlation xi_pearson(r) against the 0.6 [D-36]-provenance
gate.

This is the consumer end of the producer-consumer seam: pipeline.py writes
``model_state['log_rho_grid']`` -> this script loads it -> compute_xi_pearson.
Run AFTER the sbatch copy-out has preserved step_*.pt to the host.

Resolution match (PRE-COMMITTED, amendment-6 §S): downsample TRUTH 768->192 by
mean-pooling. Do NOT upsample the grid to 768 (the grid carries no structure
above its 192 Nyquist; upsampling would fabricate high-k and bias xi).

[D-36] / PROBE-6 provenance disclosure (mandatory on every xi citation):
  estimator = Stark+2015; the 0.6 threshold is a PROJECT-SIDE ADOPTION per [D-36],
  NOT a Stark+2015-quoted value. NO classical anchor until A4' lands.

Usage:
    python scripts/d73_xi_host_consumer.py \
        --checkpoint cloud_runs/<RUN_TAG>/checkpoints/step_050000.pt \
        --truth-rho Sherwood/.rho_field_cache/rho_field_p1_z0.300_n768.npy \
        --grid-size 192 \
        --box-kpc-h 60000 \
        --out cloud_runs/<RUN_TAG>/eval/xi_3d.json

Exit codes:
    0  scored successfully (PASS or FAIL on the 0.6 gate, both reported)
    2  checkpoint missing / unloadable
    3  truth rho cube missing
    4  log_rho_grid not in checkpoint state / shape mismatch
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

# Repo-root import contract: run with PYTHONPATH=. from the repo root.
from src.models.voxel_grid_field import VoxelGridField
from src.analysis.cross_corr import compute_xi_pearson


def _mean_pool_downsample(cube: np.ndarray, target: int) -> np.ndarray:
    """Mean-pool a (N,N,N) cube to (target,target,target). N must be an
    integer multiple of target (768 -> 192 is 4x). Conserves <rho>."""
    N = cube.shape[0]
    if cube.ndim != 3 or len(set(cube.shape)) != 1:
        raise ValueError(f"truth cube must be cubic 3D; got {cube.shape}")
    if N % target != 0:
        raise ValueError(
            f"truth N={N} not an integer multiple of target={target}; "
            "mean-pool requires an integer block factor"
        )
    f = N // target
    # Reshape into blocks and average over the three intra-block axes.
    pooled = cube.reshape(target, f, target, f, target, f).mean(axis=(1, 3, 5))
    return pooled.astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True,
                    help="Path to the copied-out step_*.pt voxel-grid checkpoint.")
    ap.add_argument("--truth-rho", required=True,
                    help="Path to the n768 P1 z=0.3 truth rho_field .npy cache.")
    ap.add_argument("--grid-size", type=int, default=192,
                    help="G of the voxel grid (must match the checkpoint).")
    ap.add_argument("--box-kpc-h", type=float, default=60000.0,
                    help="Comoving box length in kpc/h (Sherwood = 60000).")
    ap.add_argument("--out", required=True,
                    help="Path to write the xi_3d.json result.")
    args = ap.parse_args()

    # --- Producer-consumer seam 1: checkpoint must exist + load. ---
    if not os.path.isfile(args.checkpoint):
        print(f"FATAL: checkpoint not found at {args.checkpoint}", file=sys.stderr)
        return 2
    try:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - loud, explicit failure per PCV
        print(f"FATAL: torch.load failed on {args.checkpoint}: {exc}",
              file=sys.stderr)
        return 2

    model_state = state.get("model_state", state)
    if "log_rho_grid" not in model_state:
        print("FATAL: 'log_rho_grid' absent from checkpoint model_state; keys="
              f"{list(model_state.keys())[:8]}", file=sys.stderr)
        return 4

    # --- Reconstruct the VoxelGridField and load the trained grid. ---
    # init_xhi here is irrelevant to the density cube (we only read log_rho_grid),
    # but we instantiate with the dispatch's pinned value for a clean load.
    model = VoxelGridField(grid_size=args.grid_size, init_noise_std=0.0,
                           init_xhi=0.2)
    try:
        model.load_state_dict(model_state)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: load_state_dict failed (shape/G mismatch?): {exc}",
              file=sys.stderr)
        return 4
    model.eval()

    log_rho_grid = model.log_rho_grid.detach()  # (G,G,G), stores log10(rho/<rho>+1e-3)
    if tuple(log_rho_grid.shape) != (args.grid_size,) * 3:
        print(f"FATAL: log_rho_grid shape {tuple(log_rho_grid.shape)} != "
              f"({args.grid_size},)*3", file=sys.stderr)
        return 4

    # --- §S: dump log_rho_grid -> linear rho/<rho> cube at G. ---
    with torch.no_grad():
        rho_pred = VoxelGridField.density_log_to_linear(log_rho_grid).cpu().numpy()
    rho_pred = rho_pred.astype(np.float64)
    print(f"[xi-consumer] pred cube G={args.grid_size}: mean={rho_pred.mean():.4f} "
          f"std={rho_pred.std():.4f} min={rho_pred.min():.4f} max={rho_pred.max():.4f}")

    # --- Producer-consumer seam 2: truth rho cube must exist. ---
    if not os.path.isfile(args.truth_rho):
        print(f"FATAL: truth rho cube not found at {args.truth_rho}",
              file=sys.stderr)
        return 3
    rho_truth_768 = np.load(args.truth_rho)
    print(f"[xi-consumer] truth cube N={rho_truth_768.shape[0]}: "
          f"mean={rho_truth_768.mean():.4f}")

    # --- §S resolution match: downsample TRUTH 768 -> G by mean-pool. ---
    rho_truth = _mean_pool_downsample(rho_truth_768, args.grid_size)
    print(f"[xi-consumer] truth downsampled to {rho_truth.shape}: "
          f"mean={rho_truth.mean():.4f} (mean conserved by pooling)")

    # --- Run the true [D-13] 3D xi_pearson on matched (G,G,G) cubes. ---
    # r bins straddling the 2 h^-1 Mpc gate point; fine enough to read xi(2 Mpc/h).
    r_bins = np.linspace(0.0, 10.0, 21)  # 0.5 Mpc/h bins, 0..10 Mpc/h
    r_centers, xi = compute_xi_pearson(
        rho_pred=rho_pred,
        rho_truth=rho_truth,
        box_kpc_h=args.box_kpc_h,
        r_bins=r_bins,
    )

    # Read xi at the [D-13] gate point r = 2 h^-1 Mpc (nearest bin center).
    gate_r = 2.0
    i_gate = int(np.argmin(np.abs(r_centers - gate_r)))
    xi_at_gate = float(xi[i_gate])
    gate_pass = bool(np.isfinite(xi_at_gate) and xi_at_gate > 0.6)

    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "grid_size": args.grid_size,
        "box_kpc_h": args.box_kpc_h,
        "truth_rho": os.path.abspath(args.truth_rho),
        "resolution_match": "truth_768_meanpool_to_%d" % args.grid_size,
        "r_centers_mpc_h": r_centers.tolist(),
        "xi": [None if not np.isfinite(v) else float(v) for v in xi],
        "gate_r_mpc_h": gate_r,
        "xi_at_gate_r": xi_at_gate,
        "gate_threshold": 0.6,
        "gate_pass": gate_pass,
        "provenance": (
            "estimator=Stark+2015; 0.6 threshold = project-side adoption per "
            "[D-36], NOT Stark+2015-quoted; NO classical anchor until A4' lands"
        ),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)

    verdict = "PASS" if gate_pass else "FAIL"
    print(f"[xi-consumer] xi_3d(r={gate_r} Mpc/h) = {xi_at_gate:.4f} "
          f"vs 0.6 gate -> {verdict}")
    print(f"[xi-consumer] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
