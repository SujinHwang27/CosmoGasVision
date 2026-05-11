"""Per-k-bin P_F dump on the [D-40] sat-aware Tier-1 checkpoint.

Empirically tests Hypothesis C from the [D-40] FAIL ruling: does the
relative-residual P_F loss admit a trivial constant-prediction solution?
A constant P_F_pred(k) = c would give a specific signature (flat curve
across the inertial band). A scale-distorted solution (P_F_pred shape ≈
P_F_truth shape, amplitude depressed) is a different mechanism.

Usage::

    PYTHONPATH=. uv run python scripts/diag_pf_per_bin.py \\
        --ckpt-path cloud_runs/sat-aware-smoke-step10k.pt \\
        --run-id 87dcf9e63564465489f770266fcec197 \\
        --output-dir experiments/nerf/artifacts/eval/sat_aware_hypc/ \\
        --n-rays-eval 1024
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_anchor_invariance_d34 import (  # noqa: E402
    _build_model_with_fallback,
)
from src.analysis.flux_power import compute_PF_1d  # noqa: E402
from src.analysis.stage2b_report import _render_tau_for_model  # noqa: E402
from src.data.loader import SherwoodLoader  # noqa: E402


def _render_tau_chunked(model, coords, vel_axis_t, chunk_rays: int = 32):
    """Chunked render to bound peak host RAM on the Voigt kernel.
    Mirrors scripts/make_pf_overlay_fig.py._render_tau_chunked; same
    chunk-rays default and per-chunk logging convention.
    """
    tau_chunks = []
    n_rays_total = coords.shape[0]
    for i in range(0, n_rays_total, chunk_rays):
        sl = slice(i, min(i + chunk_rays, n_rays_total))
        tau_c = _render_tau_for_model(model, coords[sl], vel_axis_t)
        if isinstance(tau_c, torch.Tensor):
            tau_c = tau_c.detach().cpu().numpy()
        tau_chunks.append(tau_c)
        print(f"[diag-pf]   rendered rays {sl.start}..{sl.stop} of {n_rays_total}",
              flush=True)
    return np.concatenate(tau_chunks, axis=0)

# [D-13] inertial-range gate band.
_PF_BAND = (10 ** -2.5, 10 ** -1.5)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--run-id", required=True,
                   help="MLflow run id for provenance (may not resolve locally; "
                        "loader falls back to production defaults).")
    p.add_argument("--output-dir", default="experiments/nerf/artifacts/eval/sat_aware_hypc/")
    p.add_argument("--physics-id", type=int, default=1)
    p.add_argument("--redshift", type=float, default=0.3)
    p.add_argument("--n-rays-eval", type=int, default=1024)
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--n-kbins", type=int, default=20)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Truth + render
    sherwood = SherwoodLoader("Sherwood")
    sl_full = sherwood.load_sightlines(args.physics_id, args.redshift)
    tau_truth_full = np.asarray(sl_full["tau_h1"], dtype=np.float64)
    vel_axis = np.asarray(sl_full["vel_axis"], dtype=np.float64)
    box_kpc_h = float(sl_full["header"]["box_kpc_h"])
    n_rays_avail = tau_truth_full.shape[0]
    n_rays = min(args.n_rays_eval, n_rays_avail)
    rng = np.random.default_rng(seed=args.eval_seed)
    sel = rng.choice(n_rays_avail, size=n_rays, replace=False)
    sel.sort()
    tau_truth = tau_truth_full[sel]

    model = _build_model_with_fallback(args.run_id, args.ckpt_path)
    coords_world = sherwood.get_world_coordinates(sl_full)
    coords = torch.tensor(coords_world[sel] / box_kpc_h, dtype=torch.float32)
    vel_axis_t = torch.tensor(vel_axis, dtype=torch.float32)
    # Chunked render matches the W1-A / Task C / overlay convention
    # (chunk_rays=32) so the Voigt kernel doesn't blow up host RAM at
    # n_rays=1024. The previous un-chunked call hung silently.
    tau_pred = _render_tau_chunked(model, coords, vel_axis_t, chunk_rays=32)

    # P_F on identical k bins for both.
    k_axis, P_truth = compute_PF_1d(tau_truth, vel_axis, n_kbins=args.n_kbins)
    _, P_pred = compute_PF_1d(tau_pred, vel_axis, n_kbins=args.n_kbins)

    in_band = (k_axis >= _PF_BAND[0]) & (k_axis <= _PF_BAND[1])
    rel_diff = np.where(np.isfinite(P_truth) & (P_truth > 0),
                        (P_pred - P_truth) / P_truth, np.nan)
    ratio = np.where(np.isfinite(P_truth) & (P_truth > 0),
                     P_pred / P_truth, np.nan)

    # Hypothesis C diagnostics. Constant-prediction → log-std(P_pred) ≈ 0
    # across the inertial band; structure preserved → log-std(P_pred) ≈
    # log-std(P_truth) up to scale; amplitude-suppressed → ratio < 1
    # roughly uniform.
    band_pred = P_pred[in_band]
    band_truth = P_truth[in_band]
    log_std_pred = float(np.nanstd(np.log10(band_pred[band_pred > 0])))
    log_std_truth = float(np.nanstd(np.log10(band_truth[band_truth > 0])))
    band_ratio = np.where(band_truth > 0, band_pred / band_truth, np.nan)
    band_ratio_mean = float(np.nanmean(band_ratio))
    band_ratio_std = float(np.nanstd(band_ratio))
    band_resid_mean = float(np.nanmean(np.abs(rel_diff[in_band])))
    # Pearson correlation in log-space (shape-preservation test).
    finite = (
        np.isfinite(np.log10(np.where(band_pred > 0, band_pred, np.nan)))
        & np.isfinite(np.log10(np.where(band_truth > 0, band_truth, np.nan)))
    )
    if finite.sum() >= 3:
        logp = np.log10(band_pred[finite])
        logt = np.log10(band_truth[finite])
        pearson_log = float(np.corrcoef(logp, logt)[0, 1])
    else:
        pearson_log = float("nan")

    # Per-bin table
    print()
    print("=" * 88)
    print(f"  [D-40] Hypothesis C: per-bin P_F dump on sat-aware Tier-1 checkpoint")
    print(f"  ckpt={args.ckpt_path}  physics_id={args.physics_id} z={args.redshift}")
    print(f"  n_rays={n_rays}  eval_seed={args.eval_seed}  n_kbins={args.n_kbins}")
    print("=" * 88)
    print(f"  {'bin':>3}  {'k_||':>10}  {'P_truth':>10}  {'P_pred':>10}  "
          f"{'P_pred/P_t':>10}  {'(p-t)/t':>10}  in_band")
    for i in range(args.n_kbins):
        flag = "  <-- inertial" if in_band[i] else ""
        print(f"  {i:>3}  {k_axis[i]:>10.4e}  {P_truth[i]:>10.4e}  "
              f"{P_pred[i]:>10.4e}  {ratio[i]:>10.4f}  {rel_diff[i]:>+10.4f}{flag}")

    print()
    print("=" * 88)
    print("  Hypothesis C signature tests (inertial band, k_|| in [10^-2.5, 10^-1.5] s/km)")
    print("=" * 88)
    print(f"  log10-std(P_pred)   = {log_std_pred:.4f}   "
          f"(constant-prediction → 0; truth gives {log_std_truth:.4f})")
    print(f"  log10-std(P_truth)  = {log_std_truth:.4f}")
    print(f"  pred/truth mean     = {band_ratio_mean:.4f}   "
          f"(1.0 = matched amplitude; <1 = pred-amplitude-suppressed)")
    print(f"  pred/truth std      = {band_ratio_std:.4f}   "
          f"(0 = scale-uniform suppression; large = bin-to-bin noisy)")
    print(f"  pearson(logP_pred, logP_truth) = {pearson_log:.4f}   "
          f"(1.0 = shape-preserved; 0 = no relation; <0 = anti-correlated)")
    print(f"  mean |ΔP_F/P_F| in band = {band_resid_mean:.4f}   "
          f"(headline residual; pub-t1 baseline was 0.4155)")

    # Verdict logic (heuristic, not strict).
    print()
    if log_std_pred < 0.10 * log_std_truth:
        verdict = "CONFIRMED-constant-prediction (Hypothesis C, strict form)"
    elif pearson_log > 0.7 and band_ratio_mean < 0.85:
        verdict = ("REFUTED-strict-C: shape preserved (pearson>0.7) but "
                   "amplitude suppressed; degeneracy is scale-distortion, not "
                   "constant-collapse")
    elif pearson_log < 0.3:
        verdict = ("CONFIRMED-shape-collapse (weaker form of Hypothesis C): "
                   "predicted shape has lost relation to truth")
    else:
        verdict = "AMBIGUOUS: neither constant nor cleanly shape-preserved"

    print(f"  HYPOTHESIS C VERDICT: {verdict}")

    # Persist JSON.
    payload = {
        "run_id": args.run_id,
        "ckpt_path": args.ckpt_path,
        "physics_id": args.physics_id,
        "redshift": args.redshift,
        "n_rays": int(n_rays),
        "eval_seed": int(args.eval_seed),
        "n_kbins": int(args.n_kbins),
        "pf_band_s_per_km": list(_PF_BAND),
        "k_axis": k_axis.tolist(),
        "P_truth": P_truth.tolist(),
        "P_pred": P_pred.tolist(),
        "P_pred_over_P_truth": ratio.tolist(),
        "rel_diff_per_bin": rel_diff.tolist(),
        "in_band_mask": in_band.tolist(),
        "diagnostics": {
            "log10_std_pred_in_band": log_std_pred,
            "log10_std_truth_in_band": log_std_truth,
            "band_ratio_mean": band_ratio_mean,
            "band_ratio_std": band_ratio_std,
            "pearson_log_in_band": pearson_log,
            "mean_abs_rel_diff_in_band": band_resid_mean,
        },
        "hypothesis_c_verdict": verdict,
    }
    out_json = out / f"{args.run_id}_pf_per_bin.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print()
    print(f"  per-bin numerics + diagnostics written to {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
