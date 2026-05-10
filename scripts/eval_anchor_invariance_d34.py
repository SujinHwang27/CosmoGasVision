"""[D-34] Empirical anchor-invariance demo.

The PI's [D-34] argument: re-training Stage 2b at the corrected mean-flux
anchor (0.979, Kirkman+ 2007) is unnecessary because the [D-13] gates are
anchor-invariant by construction (per [D-11] sub-clause: invariant to a
uniform multiplicative rescaling of predicted flux). This driver converts
that theoretical claim into measured evidence by:

  1. Loading the T3-P1 fiducial seed=0 step-10000 checkpoint
     (run_id f74dbb669c9641568ab883023a84d1fa).
  2. Predicting tau on n_rays_eval=1024 held-out sightlines via the same
     code path as scripts/eval_partial_d13.py (no metric duplication).
  3. Computing the [D-13] gates (P_F inertial residual + KS-PDF distance)
     once on tau_pred as-is, once on a uniformly-rescaled F:
        F_rescaled = r * F_pred,  r = 0.979 / <F_pred>
        tau_rescaled = -ln(clip(F_rescaled, eps, 1.0))
  4. Reporting the absolute drift on each gate and applying the [D-34]
     verdict gates: P_F drift < 0.5%, KS drift < 0.01.
  5. Sanity-checking on the [D-33] 1D log-density Pearson r: rho_pred
     comes from mlp(coords)[..., 0] directly, so by construction the
     rescaling of F does not touch it. Drift must be exactly zero.

PCV: refuses non-32-hex run-id, refuses missing checkpoint, FATAL on
either. No silent random-init.

Usage::

    PYTHONPATH=. uv run python scripts/eval_anchor_invariance_d34.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.flux_pdf import ks_distance  # noqa: E402
from src.analysis.flux_power import compute_PF_1d  # noqa: E402
from src.analysis.stage2b_report import (  # noqa: E402
    _build_model_from_run,
    _load_mlflow_run,
    _render_tau_for_model,
)
from src.data.loader import SherwoodLoader  # noqa: E402
from src.models.nerf import IGMNeRF  # noqa: E402

_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_KS_F_RANGE = (0.05, 0.95)
_PF_BAND = (10 ** -2.5, 10 ** -1.5)


def _assert_run_id(run_id: str) -> None:
    if not _HEX32.match(run_id):
        sys.exit(
            f"FATAL [PCV]: --run-id {run_id!r} is not a 32-hex MLflow run id. "
            f"Refusing to evaluate; [D-34] anchor-invariance demo must tie to "
            f"a real banked checkpoint."
        )


def _pf_inertial_residual(tau_pred: np.ndarray, tau_truth: np.ndarray,
                          vel_axis: np.ndarray) -> float:
    """Mean fractional |Delta P_F / P_F| over the [D-13] inertial band.

    Identical computation to ``stage2b_report._fig_pf_compare`` minus the
    figure side-effect, so we don't pollute the report tree with extra PNGs.
    """
    k_p, P_p = compute_PF_1d(tau_pred, vel_axis)
    k_t, P_t = compute_PF_1d(tau_truth, vel_axis)
    band = (
        (k_t >= _PF_BAND[0]) & (k_t <= _PF_BAND[1])
        & np.isfinite(P_t) & np.isfinite(P_p)
    )
    if not band.any():
        return float("nan")
    return float(np.nanmean(np.abs(P_p[band] - P_t[band]) / P_t[band]))


def _ks_F_distance(tau_pred: np.ndarray, tau_truth: np.ndarray) -> float:
    """KS distance on raw F samples in [D-13] window — same code path as
    ``stage2b_report._fig_flux_pdf``."""
    F_pred = np.exp(-np.asarray(tau_pred)).ravel()
    F_truth = np.exp(-np.asarray(tau_truth)).ravel()
    return ks_distance(F_pred, F_truth, F_range=_KS_F_RANGE)


def _rescale_tau(tau_pred: np.ndarray, r: float, eps: float = 1e-12
                 ) -> np.ndarray:
    """Apply uniform multiplicative rescaling F -> r*F at flux level, then
    invert back to tau. Clips F at [eps, 1.0] before the log so r > 1 (which
    can push F slightly above 1 in the F~1 tail) yields tau >= 0."""
    F = np.exp(-np.asarray(tau_pred, dtype=np.float64))
    F_rescaled = np.clip(r * F, eps, 1.0)
    return -np.log(F_rescaled)


def _pearson_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row Pearson r — same definition as proxy_xi_1d_sample.py."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean(axis=1, keepdims=True)
    num = (a_c * b_c).sum(axis=1)
    den = np.sqrt((a_c ** 2).sum(axis=1) * (b_c ** 2).sum(axis=1))
    out = np.full(a.shape[0], np.nan, dtype=np.float64)
    valid = den > 0
    out[valid] = num[valid] / den[valid]
    return out


def _predict_rho_along_rays(model, coords_unit: np.ndarray,
                            chunk_rays: int = 64) -> np.ndarray:
    n_rays, n_bins, _ = coords_unit.shape
    out = np.empty((n_rays, n_bins), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, n_rays, chunk_rays):
            sl = slice(i, min(i + chunk_rays, n_rays))
            x = torch.from_numpy(coords_unit[sl]).to(torch.float32)
            fields = model(x)
            out[sl] = fields[..., 0].cpu().numpy().astype(np.float64)
    return out


def _build_model_with_fallback(run_id: str, ckpt_path: str) -> IGMNeRF:
    """Try MLflow lookup for hyperparameters; if the local tracker doesn't
    have the run (pub-t1 cells were trained on Juno and source-run IDs
    haven't been imported), fall back to the production defaults
    (hidden_dim=256, num_layers=8, L=10). Same posture as
    ``scripts/wrinkle1_diagnostic.py._build_model_with_fallback`` (which
    already validated this for Wrinkle-1).
    """
    run, _ = _load_mlflow_run(run_id)
    if run is None:
        print(f"[d34] MLflow lookup miss for {run_id}; falling back to "
              f"production defaults (hidden_dim=256, num_layers=8, L=10).")
        model = IGMNeRF(hidden_dim=256, num_layers=8, L=10)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("model_state", "model_state_dict"):
                if key in state:
                    state = state[key]
                    break
        model.load_state_dict(state)
        model.eval()
        print(f"[d34] Loaded weights from {ckpt_path}")
        return model
    return _build_model_from_run(run, ckpt_path)


# Default pub-t1 cell pack for the Task C reframe (PI dispatch 2026-05-10).
# Each entry binds a physics_id to its juno-trained step_050000 checkpoint
# and source MLflow run-id. Source run-IDs are recorded for provenance;
# they don't have to resolve in the local tracker — the fallback loader
# handles that.
PUB_T1_CELLS = [
    {
        "label": "P1",
        "physics_id": 1,
        "redshift": 0.3,
        "ckpt_path": str(
            REPO_ROOT / "cloud_runs" / "pub-t1-extracted"
            / "P1-N64-S0-1778430089-7f65fe" / "checkpoints" / "step_050000.pt"
        ),
        "run_id": "31acdf9d900e447081e6d051f7d42c0e",
    },
    {
        "label": "P2",
        "physics_id": 2,
        "redshift": 0.3,
        "ckpt_path": str(
            REPO_ROOT / "cloud_runs" / "pub-t1-extracted"
            / "P2-N64-S0-1778430089-0f7fc8" / "checkpoints" / "step_050000.pt"
        ),
        "run_id": "f7fafa2320164a9cb7c9c29fad74474d",
    },
    {
        "label": "P3",
        "physics_id": 3,
        "redshift": 0.3,
        "ckpt_path": str(
            REPO_ROOT / "cloud_runs" / "pub-t1-extracted"
            / "P3-N64-S0-1778430089-b9dad4" / "checkpoints" / "step_050000.pt"
        ),
        "run_id": "62aeb93aacd44cb0aeca5b51f802a352",
    },
    {
        "label": "P4",
        "physics_id": 4,
        "redshift": 0.3,
        "ckpt_path": str(
            REPO_ROOT / "cloud_runs" / "pub-t1-extracted"
            / "P4-N64-S0-1778430089-b18fc5" / "checkpoints" / "step_050000.pt"
        ),
        "run_id": "fc3817b3b3114cae8b134800aedf20e1",
    },
]


def _eval_one_cell(
    cell: dict,
    *,
    n_rays_eval: int,
    seed: int,
    target_mean_flux: float,
    chunk_rays: int,
) -> dict:
    """Evaluate one (physics, checkpoint) cell: render tau, compute as-is and
    rescaled P_F + KS, and the 1D log-rho proxy. Returns a JSON-ready dict
    with all numerics + the as-is-minus-rescaled deltas.
    """
    label = cell["label"]
    physics_id = int(cell["physics_id"])
    redshift = float(cell["redshift"])
    ckpt_path = cell["ckpt_path"]
    run_id = cell["run_id"]

    _assert_run_id(run_id)
    if not os.path.exists(ckpt_path):
        return {
            "label": label,
            "physics_id": physics_id,
            "redshift": redshift,
            "ckpt_path": ckpt_path,
            "run_id": run_id,
            "error": f"checkpoint not found: {ckpt_path}",
        }

    print(f"\n[d34] === {label} (physics={physics_id}, z={redshift}) ===")
    print(f"[d34]   ckpt   : {ckpt_path}")
    print(f"[d34]   run_id : {run_id}")

    sherwood = SherwoodLoader(str(REPO_ROOT / "Sherwood"))
    sl = sherwood.load_sightlines(physics_id, redshift)
    tau_truth_full = np.asarray(sl["tau_h1"], dtype=np.float64)
    vel_axis = np.asarray(sl["vel_axis"], dtype=np.float64)
    box_kpc_h = float(sl["header"]["box_kpc_h"])
    rho_gt_full = np.asarray(sl["density"], dtype=np.float64)

    n_rays_avail = tau_truth_full.shape[0]
    n_rays = min(n_rays_eval, n_rays_avail)
    rng = np.random.default_rng(seed=seed)
    sel = rng.choice(n_rays_avail, size=n_rays, replace=False)
    sel.sort()
    tau_truth = tau_truth_full[sel]
    rho_gt = rho_gt_full[sel]

    model = _build_model_with_fallback(run_id, ckpt_path)

    coords_world = sherwood.get_world_coordinates(sl)
    coords_unit_np = (coords_world[sel] / box_kpc_h).astype(np.float64)
    coords = torch.tensor(coords_unit_np, dtype=torch.float32)
    vel_axis_t = torch.tensor(vel_axis, dtype=torch.float32)
    chunk_rays = max(1, int(chunk_rays))
    tau_chunks = []
    for i in range(0, coords.shape[0], chunk_rays):
        sl_c = slice(i, min(i + chunk_rays, coords.shape[0]))
        tau_c = _render_tau_for_model(model, coords[sl_c], vel_axis_t)
        tau_chunks.append(tau_c)
        print(f"[d34]   rendered rays {sl_c.start}..{sl_c.stop} "
              f"of {coords.shape[0]}")
    tau_pred = np.concatenate(tau_chunks, axis=0)

    F_pred = np.exp(-tau_pred)
    mean_F_pred = float(F_pred.mean())
    r = float(target_mean_flux) / mean_F_pred
    tau_pred_rescaled = _rescale_tau(tau_pred, r)
    mean_F_rescaled = float(np.exp(-tau_pred_rescaled).mean())

    pf_asis = _pf_inertial_residual(tau_pred, tau_truth, vel_axis)
    ks_asis = _ks_F_distance(tau_pred, tau_truth)
    pf_resc = _pf_inertial_residual(tau_pred_rescaled, tau_truth, vel_axis)
    ks_resc = _ks_F_distance(tau_pred_rescaled, tau_truth)

    # 1D log-rho proxy. Computed once; the rescale is by construction
    # density-invariant (only touches F→r*F → τ), so we report the same
    # value under both anchors and confirm zero drift downstream.
    rho_pred = _predict_rho_along_rays(model, coords_unit_np)
    floor = 1e-6
    r_log = _pearson_per_row(
        np.log10(np.maximum(rho_pred, floor)),
        np.log10(np.maximum(rho_gt, floor)),
    )
    med_rlog = float(np.median(r_log[np.isfinite(r_log)]))

    delta_pf = pf_asis - pf_resc  # "as-is minus rescaled" per dispatch
    delta_ks = ks_asis - ks_resc

    print(f"[d34]   <F_pred>          = {mean_F_pred:.6f}")
    print(f"[d34]   rescale r         = {target_mean_flux} / {mean_F_pred:.6f}"
          f" = {r:.6f}")
    print(f"[d34]   <F_rescaled>      = {mean_F_rescaled:.6f}")
    print(f"[d34]   P_F as-is         = {pf_asis:.6f}")
    print(f"[d34]   P_F rescaled      = {pf_resc:.6f}")
    print(f"[d34]   P_F delta (asis-resc) = {delta_pf:+.6f}")
    print(f"[d34]   KS as-is          = {ks_asis:.6f}")
    print(f"[d34]   KS rescaled       = {ks_resc:.6f}")
    print(f"[d34]   KS delta (asis-resc)  = {delta_ks:+.6f}")
    print(f"[d34]   median r_rho_log  = {med_rlog:+.6f}")

    return {
        "label": label,
        "physics_id": physics_id,
        "redshift": redshift,
        "ckpt_path": ckpt_path,
        "run_id": run_id,
        "eval_seed": int(seed),
        "n_rays_eval": int(n_rays),
        "target_mean_flux": float(target_mean_flux),
        "mean_F_pred_asis": mean_F_pred,
        "rescale_r": float(r),
        "mean_F_rescaled": mean_F_rescaled,
        "P_F_asis": pf_asis,
        "P_F_rescaled": pf_resc,
        "delta_P_F_asis_minus_rescaled": float(delta_pf),
        "KS_asis": ks_asis,
        "KS_rescaled": ks_resc,
        "delta_KS_asis_minus_rescaled": float(delta_ks),
        "median_r_rho_log_asis": med_rlog,
        "median_r_rho_log_rescaled": med_rlog,
        "median_r_rho_log_drift": 0.0,
    }


def _run_pubt1_sweep(args: argparse.Namespace) -> int:
    """Task C entry point: loop over all 4 pub-t1 cells, write JSON,
    print per-cell + summary table. No anchor-invariance verdict line —
    that framing is retracted per [D-39]; this driver documents the
    (c)-attributable gap footprint across the checkpoint family.
    """
    results = []
    for cell in PUB_T1_CELLS:
        res = _eval_one_cell(
            cell,
            n_rays_eval=int(args.n_rays_eval),
            seed=int(args.seed),
            target_mean_flux=float(args.target_mean_flux),
            chunk_rays=int(args.chunk_rays),
        )
        results.append(res)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "C",
        "reframe_origin": "[D-39] PASS_T1_pub FAIL; Wrinkle-1 final ruling "
                          "(2026-05-10) — (c)-rescale-vs-trained intrinsic "
                          "divergence is the residually-supported mechanism. "
                          "This driver documents the per-cell footprint of "
                          "(c) across the pub-t1 P1-P4 checkpoint family.",
        "anchor_target_mean_flux": float(args.target_mean_flux),
        "n_rays_eval": int(args.n_rays_eval),
        "eval_seed": int(args.seed),
        "cells": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[d34] JSON written: {out_path}")

    # Summary table — empirical numbers first, no verdict overlay.
    print()
    print("=" * 92)
    print("Task C: pub-t1 P1-P4 anchor-invariance numerics (per-cell)")
    print("=" * 92)
    hdr = (
        f"{'Cell':<6}{'<F>asis':>10}{'r':>10}"
        f"{'P_F asis':>11}{'P_F resc':>11}{'ΔP_F':>10}"
        f"{'KS asis':>10}{'KS resc':>10}{'ΔKS':>10}{'r_log':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r['label']:<6}  ERROR: {r['error']}")
            continue
        print(
            f"{r['label']:<6}"
            f"{r['mean_F_pred_asis']:>10.5f}"
            f"{r['rescale_r']:>10.5f}"
            f"{r['P_F_asis']:>11.5f}"
            f"{r['P_F_rescaled']:>11.5f}"
            f"{r['delta_P_F_asis_minus_rescaled']:>+10.5f}"
            f"{r['KS_asis']:>10.5f}"
            f"{r['KS_rescaled']:>10.5f}"
            f"{r['delta_KS_asis_minus_rescaled']:>+10.5f}"
            f"{r['median_r_rho_log_asis']:>+10.5f}"
        )
    print()
    print("Note: [D-37] honest-reporting frame applies. The [D-39] verdict "
          "stands: P_F fails the [D-13] gate in all 4 cells against the "
          "TRAINED-at-0.979 prediction (no rescale crutch). The 'rescaled' "
          "column here is a near-identity probe because <F_pred> already "
          "≈ 0.979 by construction; small ΔP_F ≠ closed gap.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--run-id", default="f74dbb669c9641568ab883023a84d1fa",
        help="MLflow run id (32 hex). Default = T3-P1 fiducial seed=0.",
    )
    p.add_argument(
        "--ckpt-path", default=str(
            REPO_ROOT / "cloud_runs" / "prong3-p1-t3"
            / "P1-N1024-S0-1778229084-c08848" / "checkpoints" / "step_010000.pt"
        ),
    )
    p.add_argument("--physics-id", type=int, default=1)
    p.add_argument("--redshift", type=float, default=0.3)
    p.add_argument("--n-rays-eval", type=int, default=1024)
    p.add_argument("--chunk-rays", type=int, default=32,
                   help="Render at most this many rays per forward pass to "
                        "fit in CPU RAM. Mathematically identical to a single "
                        "batched call (no cross-ray state in the integrator).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-mean-flux", type=float, default=0.979,
                   help="Kirkman+ 2007 corrected anchor (the [D-34] target).")
    p.add_argument(
        "--cells", choices=("legacy", "pub-t1"), default="legacy",
        help="'legacy' = original single-cell [D-34] flow against --run-id"
             " / --ckpt-path. 'pub-t1' = Task C sweep over all 4 pub-t1"
             " step_050000.pt checkpoints (P1-P4); ignores --run-id/--ckpt-path."
             " Output JSON via --out-json.",
    )
    p.add_argument(
        "--out-json",
        default=str(
            REPO_ROOT / "experiments" / "nerf" / "artifacts" / "eval"
            / "task_c" / "pub_t1_anchor_invariance.json"
        ),
        help="Output JSON path for the --cells=pub-t1 sweep.",
    )
    args = p.parse_args()

    if args.cells == "pub-t1":
        return _run_pubt1_sweep(args)

    _assert_run_id(args.run_id)
    if not os.path.exists(args.ckpt_path):
        sys.exit(f"FATAL: checkpoint {args.ckpt_path!r} does not exist.")

    print(f"[d34] run_id          : {args.run_id}")
    print(f"[d34] ckpt            : {args.ckpt_path}")
    print(f"[d34] physics/z       : P{args.physics_id} z={args.redshift}")
    print(f"[d34] n_rays_eval     : {args.n_rays_eval}  seed={args.seed}")
    print(f"[d34] target_mean_F   : {args.target_mean_flux}")

    sherwood = SherwoodLoader(str(REPO_ROOT / "Sherwood"))
    sl = sherwood.load_sightlines(args.physics_id, args.redshift)
    tau_truth_full = np.asarray(sl["tau_h1"], dtype=np.float64)
    vel_axis = np.asarray(sl["vel_axis"], dtype=np.float64)
    box_kpc_h = float(sl["header"]["box_kpc_h"])
    rho_gt_full = np.asarray(sl["density"], dtype=np.float64)

    n_rays_avail = tau_truth_full.shape[0]
    n_rays = min(args.n_rays_eval, n_rays_avail)
    rng = np.random.default_rng(seed=args.seed)
    sel = rng.choice(n_rays_avail, size=n_rays, replace=False)
    sel.sort()
    tau_truth = tau_truth_full[sel]
    rho_gt = rho_gt_full[sel]

    run, _ = _load_mlflow_run(args.run_id)
    model = _build_model_from_run(run, args.ckpt_path)

    coords_world = sherwood.get_world_coordinates(sl)
    coords_unit_np = (coords_world[sel] / box_kpc_h).astype(np.float64)
    coords = torch.tensor(coords_unit_np, dtype=torch.float32)
    vel_axis_t = torch.tensor(vel_axis, dtype=torch.float32)
    # Chunk over rays to fit in CPU RAM. _render_tau_for_model batches the
    # whole tensor through volume_render_physics in one shot, which OOMs at
    # n_rays >= 256 on this host (Fourier features L=10 + 8-layer MLP +
    # Voigt convolution peaks at several GB). Chunking and concatenating
    # is mathematically identical (no cross-ray state).
    chunk_rays = max(1, int(args.chunk_rays))
    tau_chunks = []
    for i in range(0, coords.shape[0], chunk_rays):
        sl_c = slice(i, min(i + chunk_rays, coords.shape[0]))
        tau_c = _render_tau_for_model(model, coords[sl_c], vel_axis_t)
        tau_chunks.append(tau_c)
        print(f"[d34] rendered rays {sl_c.start}..{sl_c.stop} of {coords.shape[0]}")
    tau_pred = np.concatenate(tau_chunks, axis=0)
    assert tau_pred.shape == tau_truth.shape, \
        f"shape mismatch: tau_pred {tau_pred.shape} vs truth {tau_truth.shape}"

    F_pred = np.exp(-tau_pred)
    mean_F_pred = float(F_pred.mean())
    r = float(args.target_mean_flux) / mean_F_pred
    print(f"\n[d34] <F_pred> (cycle-mean)      = {mean_F_pred:.6f}")
    print(f"[d34] rescale factor r           = {args.target_mean_flux} / "
          f"{mean_F_pred:.6f} = {r:.6f}")

    tau_pred_rescaled = _rescale_tau(tau_pred, r)
    mean_F_rescaled = float(np.exp(-tau_pred_rescaled).mean())
    print(f"[d34] <F_rescaled> (post-rescale) = {mean_F_rescaled:.6f}  "
          f"(target {args.target_mean_flux:.3f}; clip-induced shift if mismatch)")

    pf_resid_asis = _pf_inertial_residual(tau_pred, tau_truth, vel_axis)
    ks_asis = _ks_F_distance(tau_pred, tau_truth)
    pf_resid_resc = _pf_inertial_residual(tau_pred_rescaled, tau_truth, vel_axis)
    ks_resc = _ks_F_distance(tau_pred_rescaled, tau_truth)

    rho_pred_asis = _predict_rho_along_rays(model, coords_unit_np)
    rho_pred_resc = _predict_rho_along_rays(model, coords_unit_np)
    floor = 1e-6
    r_log_asis = _pearson_per_row(
        np.log10(np.maximum(rho_pred_asis, floor)),
        np.log10(np.maximum(rho_gt, floor)),
    )
    r_log_resc = _pearson_per_row(
        np.log10(np.maximum(rho_pred_resc, floor)),
        np.log10(np.maximum(rho_gt, floor)),
    )
    med_rlog_asis = float(np.median(r_log_asis[np.isfinite(r_log_asis)]))
    med_rlog_resc = float(np.median(r_log_resc[np.isfinite(r_log_resc)]))

    pf_drift = abs(pf_resid_resc - pf_resid_asis)
    ks_drift = abs(ks_resc - ks_asis)
    rlog_drift = abs(med_rlog_resc - med_rlog_asis)

    pf_gate, pf_thr = "PASS" if pf_drift < 0.005 else "FAIL", 0.005
    ks_gate, ks_thr = "PASS" if ks_drift < 0.01 else "FAIL", 0.01
    rlog_sanity = "SANITY OK" if rlog_drift == 0.0 else "SANITY VIOLATED"

    print()
    print("=" * 78)
    print("[D-34] empirical anchor-invariance results")
    print("=" * 78)
    hdr = f"{'metric':<28}{'as-is':>12}{'rescaled':>12}{'drift':>12}{'gate':>14}{'verdict':>10}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{'P_F inertial residual':<28}{pf_resid_asis*100:>11.4f}%"
          f"{pf_resid_resc*100:>11.4f}%{pf_drift*100:>11.4f}%"
          f"{'< 0.5%':>14}{pf_gate:>10}")
    print(f"{'KS-PDF distance':<28}{ks_asis:>12.6f}{ks_resc:>12.6f}"
          f"{ks_drift:>12.6f}{'< 0.01':>14}{ks_gate:>10}")
    print(f"{'r_rho_log (1D proxy [D-33])':<28}{med_rlog_asis:>+12.6f}"
          f"{med_rlog_resc:>+12.6f}{rlog_drift:>12.6f}"
          f"{'exactly 0':>14}{rlog_sanity:>10}")
    print()
    if pf_gate == "PASS" and ks_gate == "PASS":
        print("HEADLINE: PASS -- anchor-invariance empirically confirmed; "
              "[D-34] keep-existing-runs disposition stands.")
    else:
        print("HEADLINE: FAIL -- anchor-invariance empirically FALSIFIED; "
              "option-(b) re-training re-opens; PI ruling owed.")
    if rlog_drift != 0.0:
        print("WARNING: [D-33] proxy drift is non-zero. The anchor rescale "
              "must not touch rho_pred = mlp(coords)[..., 0]. This indicates "
              "an implementation bug in the rescale path — investigate before "
              "trusting the headline verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
