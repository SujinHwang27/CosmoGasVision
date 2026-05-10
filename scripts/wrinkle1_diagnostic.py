"""Wrinkle-1 disambiguation driver — host-only, CPU, no new training.

Context
-------
[D-39] T1 publication-class returned PASS_T1_pub FAIL with P_F failing in
all four physics cells. P1 pub-t1 P_F = 0.4155 is *larger* than the
cost-survey post-hoc rescaled P_F = 0.2825 (from [D-35], same 0.979
anchor). 4× more training + corrected anchor INCREASED P_F.

This driver disambiguates Wrinkle-1's three candidate mechanisms by
running four cheap evals against the existing P1 checkpoints:

  W1-A : P1 pub-t1 step_050000.pt , eval_seed=42 , trained-at-0.979 anchor.
         Reproduces the [D-39] P_F=0.4155 baseline → validates this driver's
         eval pipeline against the canonical one (scripts/eval_partial_d13.py).
  W1-B : P1 pub-t1 step_050000.pt , eval_seed=1  , trained-at-0.979 anchor.
         Same model, different held-out sightline draw → isolates
         sightline-selection sensitivity (mechanism b).
  W1-C : P1 cost-survey step_010000.pt , eval_seed=42 , post-hoc rescale
         (r = 0.979 / <F_pred>). Reproduces the [D-35] 0.2825 number →
         validates the rescale pattern at the same seed as W1-A.
  W1-D : P1 cost-survey step_010000.pt , eval_seed=1  , post-hoc rescale.
         Cross-check W1-B sensitivity on the rescaled cost-survey side.

Decision matrix (verbatim from PI dispatch):
  - If |W1-A − W1-B|/W1-A > 0.20 AND |W1-C − W1-D|/W1-C > 0.20
     → mechanism (b) sightline-selection sensitivity; T4 conditionally
       unlocks (1-cell pilot).
  - If both pair-shifts < 0.10 AND W1-C reproduces ~0.2825
     → mechanism (c) rescale-vs-trained intrinsic divergence; T4 stays
       BLOCKED; successor work is P_F-aware loss / regularizer.
  - If W1-A ≫ W1-C at the same seed AND pair-shifts small → ambiguous
       between (a) and (c); commission W1-E (step_012500.pt seed=42 trained)
       and ask for authorization before running it.

Anti-bias guardrails (per [D-37]):
  - Lead with empirical numbers in the JSON; the mechanism call lives at
    the end and follows directly from the decision matrix.
  - If neither matrix arm fires, mechanism_call = "ambiguous" — no spin.

Usage
-----
    PYTHONPATH=. uv run python scripts/wrinkle1_diagnostic.py

No CLI args needed for the standard recipe; everything is wired to the
two checkpoint paths and seeds {42, 1}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

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

_KS_F_RANGE = (0.05, 0.95)
_PF_BAND = (10 ** -2.5, 10 ** -1.5)


# ----------------------------------------------------- model construction


def _build_model_with_fallback(run_id: str, ckpt_path: str) -> IGMNeRF:
    """Try MLflow lookup for the run's hyperparameters; if it fails (local
    tracker down / run absent), fall back to the production defaults
    (hidden_dim=256, num_layers=8, L=10) — these are the locked
    publication-class values per [D-23] / [D-24] and match both the pub-t1
    bundle and the T3 cost-survey runs.

    A checkpoint-load mismatch (state_dict keys disagree with the model
    instance) raises loudly here rather than producing silent-bogus
    metrics — same posture as eval_partial_d13.py.
    """
    run, _ = _load_mlflow_run(run_id)
    if run is None:
        print(f"[w1] MLflow lookup miss for {run_id}; using production "
              f"defaults (hidden_dim=256, num_layers=8, L=10).")
        model = IGMNeRF(hidden_dim=256, num_layers=8, L=10)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("model_state", "model_state_dict"):
                if key in state:
                    state = state[key]
                    break
        model.load_state_dict(state)
        model.eval()
        print(f"[w1] Loaded weights from {ckpt_path}")
        return model
    return _build_model_from_run(run, ckpt_path)


# ----------------------------------------------------- metric helpers


def _pf_inertial_residual(tau_pred: np.ndarray, tau_truth: np.ndarray,
                          vel_axis: np.ndarray) -> float:
    """Identical computation to stage2b_report._fig_pf_compare's headline
    statistic — mean fractional |ΔP_F / P_F| over the [D-13] inertial band.
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


def _ks_F(tau_pred: np.ndarray, tau_truth: np.ndarray) -> float:
    F_pred = np.exp(-np.asarray(tau_pred)).ravel()
    F_truth = np.exp(-np.asarray(tau_truth)).ravel()
    return ks_distance(F_pred, F_truth, F_range=_KS_F_RANGE)


def _rescale_tau(tau_pred: np.ndarray, r: float,
                 eps: float = 1e-12) -> np.ndarray:
    """Anchor-target uniform rescale F → r*F, then back to τ. Mirrors
    scripts/eval_anchor_invariance_d34.py._rescale_tau exactly."""
    F = np.exp(-np.asarray(tau_pred, dtype=np.float64))
    F_rescaled = np.clip(r * F, eps, 1.0)
    return -np.log(F_rescaled)


# ----------------------------------------------------- per-case runner


def _render_for_seed(
    model: IGMNeRF,
    sherwood: SherwoodLoader,
    physics_id: int,
    redshift: float,
    n_rays_eval: int,
    eval_seed: int,
    chunk_rays: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render tau on a seed-dependent sightline draw. Returns (tau_pred,
    tau_truth, vel_axis). Mirrors eval_partial_d13.py's main path but
    parametrizes the previously-hardcoded eval_seed=42 at line 73.
    """
    sl_full = sherwood.load_sightlines(physics_id, redshift)
    tau_truth_full = np.asarray(sl_full["tau_h1"], dtype=np.float64)
    vel_axis = np.asarray(sl_full["vel_axis"], dtype=np.float64)
    box_kpc_h = float(sl_full["header"]["box_kpc_h"])
    n_rays_avail = tau_truth_full.shape[0]
    n_rays = min(n_rays_eval, n_rays_avail)
    rng = np.random.default_rng(seed=eval_seed)
    sel = rng.choice(n_rays_avail, size=n_rays, replace=False)
    sel.sort()
    tau_truth = tau_truth_full[sel]

    coords_world = sherwood.get_world_coordinates(sl_full)
    coords = torch.tensor(
        coords_world[sel] / box_kpc_h, dtype=torch.float32
    )
    vel_axis_t = torch.tensor(vel_axis, dtype=torch.float32)

    # Chunk over rays — host CPU OOMs at ~256 rays in a single
    # volume_render_physics call (Fourier features L=10 + 8-layer MLP +
    # Voigt convolution peaks at several GB). Cross-ray-state-free, so
    # concatenation is mathematically identical.
    chunks = []
    n_rays_total = coords.shape[0]
    for i in range(0, n_rays_total, chunk_rays):
        sl_c = slice(i, min(i + chunk_rays, n_rays_total))
        tau_c = _render_tau_for_model(model, coords[sl_c], vel_axis_t)
        chunks.append(tau_c)
        print(f"[w1]   rendered rays {sl_c.start}..{sl_c.stop} "
              f"of {n_rays_total}")
    tau_pred = np.concatenate(chunks, axis=0)
    return tau_pred, tau_truth, vel_axis


def _run_case(
    label: str,
    ckpt_path: Path,
    run_id: str,
    eval_seed: int,
    anchor_mode: str,         # "trained" or "rescale"
    physics_id: int,
    redshift: float,
    n_rays_eval: int,
    target_mean_flux: float,
    chunk_rays: int,
    model_cache: dict,
) -> dict:
    """Evaluate one of the four W1 cases. model_cache memoizes the loaded
    IGMNeRF instance per ckpt_path so we don't reload weights twice for
    the two seed variants on each side."""
    print(f"\n[w1] === {label} ===")
    print(f"[w1]   ckpt        : {ckpt_path}")
    print(f"[w1]   eval_seed   : {eval_seed}")
    print(f"[w1]   anchor_mode : {anchor_mode}")

    if not ckpt_path.exists():
        return {
            "checkpoint": str(ckpt_path),
            "eval_seed": eval_seed,
            "anchor": anchor_mode,
            "error": f"checkpoint not found: {ckpt_path}",
        }

    ck_str = str(ckpt_path)
    if ck_str not in model_cache:
        model_cache[ck_str] = _build_model_with_fallback(run_id, ck_str)
    model = model_cache[ck_str]

    sherwood = SherwoodLoader(str(REPO_ROOT / "Sherwood"))
    tau_pred_raw, tau_truth, vel_axis = _render_for_seed(
        model, sherwood, physics_id, redshift, n_rays_eval, eval_seed,
        chunk_rays=chunk_rays,
    )

    F_pred_raw = np.exp(-tau_pred_raw)
    mean_F_raw = float(F_pred_raw.mean())

    if anchor_mode == "trained":
        tau_pred = tau_pred_raw
        r = 1.0
        mean_F_used = mean_F_raw
    elif anchor_mode == "rescale":
        r = float(target_mean_flux) / mean_F_raw
        tau_pred = _rescale_tau(tau_pred_raw, r)
        mean_F_used = float(np.exp(-tau_pred).mean())
    else:
        raise ValueError(f"unknown anchor_mode {anchor_mode!r}")

    pf_resid = _pf_inertial_residual(tau_pred, tau_truth, vel_axis)
    ks = _ks_F(tau_pred, tau_truth)

    print(f"[w1]   <F_pred> raw     = {mean_F_raw:.6f}")
    if anchor_mode == "rescale":
        print(f"[w1]   rescale factor r = {target_mean_flux}/"
              f"{mean_F_raw:.6f} = {r:.6f}")
        print(f"[w1]   <F_pred> rescaled= {mean_F_used:.6f}")
    print(f"[w1]   P_F residual    = {pf_resid:.6f}")
    print(f"[w1]   KS distance     = {ks:.6f}")

    return {
        "checkpoint": str(ckpt_path),
        "run_id": run_id,
        "eval_seed": eval_seed,
        "anchor": anchor_mode,
        "rescale_r": r if anchor_mode == "rescale" else None,
        "mean_F_pred_raw": mean_F_raw,
        "mean_F_pred_used": mean_F_used,
        "P_F": pf_resid,
        "KS": ks,
    }


# ----------------------------------------------------- decision logic


def _safe_rel(num_a: float, num_b: float) -> Optional[float]:
    if not np.isfinite(num_a) or not np.isfinite(num_b) or num_a == 0:
        return None
    return float(abs(num_a - num_b) / abs(num_a))


def _mechanism_call(results: dict) -> dict:
    """Apply the PI decision matrix verbatim. Returns the decision block
    that goes into diagnostic.json. Honest-reporting: if neither arm
    matches, mechanism_call='ambiguous' — do not force a call."""
    pf_a = results["W1-A"].get("P_F", float("nan"))
    pf_b = results["W1-B"].get("P_F", float("nan"))
    pf_c = results["W1-C"].get("P_F", float("nan"))
    pf_d = results["W1-D"].get("P_F", float("nan"))

    s_pubt1 = _safe_rel(pf_a, pf_b)
    s_costsurvey = _safe_rel(pf_c, pf_d)
    diff_rescale_vs_trained = (
        float(pf_a - pf_c) if np.isfinite(pf_a) and np.isfinite(pf_c)
        else None
    )

    # Arm 1: mechanism (b) — both seed-pair shifts large.
    arm_b = (
        s_pubt1 is not None and s_pubt1 > 0.20
        and s_costsurvey is not None and s_costsurvey > 0.20
    )
    # Arm 2: mechanism (c) — both seed-pair shifts small AND W1-C reproduces
    # the historical [D-35] 0.2825 value within a 10% tolerance band.
    cs_reproduces_0_2825 = (
        np.isfinite(pf_c) and abs(pf_c - 0.2825) / 0.2825 < 0.10
    )
    arm_c = (
        s_pubt1 is not None and s_pubt1 < 0.10
        and s_costsurvey is not None and s_costsurvey < 0.10
        and cs_reproduces_0_2825
    )
    # Arm 3: ambiguous (a) vs (c) — W1-A >> W1-C and pair-shifts small.
    a_much_bigger_c = (
        np.isfinite(pf_a) and np.isfinite(pf_c) and pf_a > 1.3 * pf_c
    )
    arm_ac_ambiguous = (
        s_pubt1 is not None and s_pubt1 < 0.10
        and s_costsurvey is not None and s_costsurvey < 0.10
        and a_much_bigger_c
        and not cs_reproduces_0_2825
    )

    if arm_b:
        call = "b-seed-sensitivity"
    elif arm_c:
        call = "c-rescale-vs-trained-intrinsic"
    elif arm_ac_ambiguous:
        call = "ambiguous-a-or-c-commission-W1-E"
    else:
        call = "ambiguous"

    fmt_pct = lambda v: f"{v*100:.2f}%" if v is not None else "n/a"
    return {
        "seed_sensitivity_pubt1": (
            f"|W1-A - W1-B|/W1-A = {fmt_pct(s_pubt1)} "
            f"(W1-A={pf_a:.4f}, W1-B={pf_b:.4f})"
        ),
        "seed_sensitivity_costsurvey": (
            f"|W1-C - W1-D|/W1-C = {fmt_pct(s_costsurvey)} "
            f"(W1-C={pf_c:.4f}, W1-D={pf_d:.4f})"
        ),
        "rescale_vs_trained_same_seed": (
            f"P_F(W1-A) - P_F(W1-C) = "
            f"{diff_rescale_vs_trained:+.4f}" if diff_rescale_vs_trained
            is not None else "n/a"
        ) + " (at eval_seed=42)",
        "costsurvey_reproduces_d35_0_2825": cs_reproduces_0_2825,
        "decision_matrix": {
            "arm_b_seed_sensitivity": arm_b,
            "arm_c_rescale_intrinsic": arm_c,
            "arm_ac_ambiguous_commission_W1E": arm_ac_ambiguous,
        },
        "mechanism_call": call,
    }


# ----------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--p1-pubt1-ckpt",
        default=str(
            REPO_ROOT / "cloud_runs" / "pub-t1-extracted"
            / "P1-N64-S0-1778430089-7f65fe" / "checkpoints"
            / "step_050000.pt"
        ),
    )
    p.add_argument(
        "--p1-pubt1-run-id",
        default="31acdf9d900e447081e6d051f7d42c0e",
        help="MLflow run-id from [D-39] entry; used only for hyperparam "
             "lookup. Defaults will be applied if the local tracker is down.",
    )
    p.add_argument(
        "--p1-costsurvey-ckpt",
        default=str(
            REPO_ROOT / "cloud_runs" / "prong3-p1-t3"
            / "P1-N1024-S0-1778229084-c08848" / "checkpoints"
            / "step_010000.pt"
        ),
    )
    p.add_argument(
        "--p1-costsurvey-run-id",
        default="f74dbb669c9641568ab883023a84d1fa",
        help="MLflow run-id of the T3 cost-survey P1 cell; same anchor as "
             "the [D-35] table (post-hoc rescaled P_F = 0.2825 there).",
    )
    p.add_argument("--physics-id", type=int, default=1)
    p.add_argument("--redshift", type=float, default=0.3)
    p.add_argument("--n-rays-eval", type=int, default=1024)
    p.add_argument("--target-mean-flux", type=float, default=0.979,
                   help="Kirkman+ 2007 corrected anchor.")
    p.add_argument("--chunk-rays", type=int, default=32)
    p.add_argument(
        "--output",
        default=str(
            REPO_ROOT / "experiments" / "nerf" / "artifacts" / "eval"
            / "wrinkle1" / "diagnostic.json"
        ),
    )
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cases = [
        # label, ckpt, run_id, eval_seed, anchor_mode
        ("W1-A", args.p1_pubt1_ckpt,      args.p1_pubt1_run_id,      42, "trained"),
        ("W1-B", args.p1_pubt1_ckpt,      args.p1_pubt1_run_id,       1, "trained"),
        ("W1-C", args.p1_costsurvey_ckpt, args.p1_costsurvey_run_id, 42, "rescale"),
        ("W1-D", args.p1_costsurvey_ckpt, args.p1_costsurvey_run_id,  1, "rescale"),
    ]

    results: dict[str, dict] = {}
    model_cache: dict[str, IGMNeRF] = {}
    for label, ckpt, run_id, eval_seed, anchor in cases:
        results[label] = _run_case(
            label=label,
            ckpt_path=Path(ckpt),
            run_id=run_id,
            eval_seed=eval_seed,
            anchor_mode=anchor,
            physics_id=args.physics_id,
            redshift=args.redshift,
            n_rays_eval=args.n_rays_eval,
            target_mean_flux=args.target_mean_flux,
            chunk_rays=args.chunk_rays,
            model_cache=model_cache,
        )

    decision = _mechanism_call(results)
    output = {
        **results,
        "decision": decision,
        "metadata": {
            "physics_id": args.physics_id,
            "redshift": args.redshift,
            "n_rays_eval": args.n_rays_eval,
            "target_mean_flux": args.target_mean_flux,
            "p_f_band_s_per_km": list(_PF_BAND),
            "ks_F_range": list(_KS_F_RANGE),
            "driver_version": "wrinkle1_diagnostic.py v1 (2026-05-10)",
        },
    }
    out_path.write_text(json.dumps(output, indent=2))

    print()
    print("=" * 78)
    print("[W1] Wrinkle-1 disambiguation results")
    print("=" * 78)
    hdr = (f"{'label':<6}{'eval_seed':>10}{'anchor':>12}"
           f"{'mean_F_used':>14}{'P_F':>10}{'KS':>10}")
    print(hdr)
    print("-" * len(hdr))
    for label in ("W1-A", "W1-B", "W1-C", "W1-D"):
        r = results[label]
        if "error" in r:
            print(f"{label:<6}  ERROR: {r['error']}")
            continue
        print(
            f"{label:<6}{r['eval_seed']:>10}{r['anchor']:>12}"
            f"{r['mean_F_pred_used']:>14.6f}"
            f"{r['P_F']:>10.4f}{r['KS']:>10.4f}"
        )
    print()
    print(f"seed-sensitivity pubt1   : {decision['seed_sensitivity_pubt1']}")
    print(f"seed-sensitivity cost-sv : {decision['seed_sensitivity_costsurvey']}")
    print(f"rescale-vs-trained (seed=42): {decision['rescale_vs_trained_same_seed']}")
    print(f"W1-C reproduces [D-35] 0.2825 within 10%: "
          f"{decision['costsurvey_reproduces_d35_0_2825']}")
    print(f"\nMECHANISM CALL: {decision['mechanism_call']}")
    print(f"\nFull JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
