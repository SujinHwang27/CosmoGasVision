"""
[D-71] RUNG 3 — Saturation-band per-bin gradient-norm diagnostic.

PI-authorized 2026-06-02 per LEDGER Section I Part 3(b) + Q4 SQ4 defense-panel
verdict 2026-06-02 (commit 4182e63): "the cheapest single experiment that
disambiguates identifiability-ceiling from model-capacity, and it can run on
existing checkpoints without any Q4 work."

Hypothesis under test
---------------------
At tau >> 1, F = exp(-tau) -> 0, so dF/dtau -> 0 AND dL/dF -> small as both
flux values approach 0. Net: dL/dtau_pred -> 0 in the saturation regime.
This script EMPIRICALLY MEASURES |dL/dtau_pred| per saturation-band bin on
an existing Stage 2b checkpoint to disambiguate:

  - identifiability ceiling (gradient near-zero in saturated stratum)
  - model-capacity failure  (gradient non-trivial across all strata)

Loss form is the canonical [D-24] log1p-MSE on tau, masked + capped at
TAU_MAX=10. We report grad-norms w.r.t. both tau_pred (linear) and
log1p(tau_pred) (the actual training surrogate).

Pre-committed thresholds ([D-37] rule 5 symmetric-disclosure)
-------------------------------------------------------------
Let R = mean(|grad|)_saturated / mean(|grad|)_unsaturated, where
saturated  = F_truth in [0.00, 0.05] and
unsaturated = F_truth in [0.95, 1.00].

  R < 0.1            -> identifiability-ceiling supported
  R > 0.33           -> model-capacity supported
  R in [0.1, 0.33]   -> inconclusive
  any stratum N_bin < 100 -> inconclusive (underpowered)

Usage
-----
    PYTHONPATH=. python -u scripts/d71_I_grad_norm_saturation.py \
        --checkpoint cloud_runs/pub-t1-extracted/P1-N64-S0-1778430089-7f65fe/checkpoints/step_050000.pt \
        --physics 1 --redshift 0.3 --n_rays 64 \
        --data_root D:/Data/sujin/CosmoGasVision/Sherwood \
        --out_dir experiments/nerf/artifacts/d71_I_grad_norm_saturation

CPU is fine: 64 rays * 2048 bins, one forward + one backward.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch

# Allow `src.` imports when invoked as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.loader import SherwoodLoader
from src.models.nerf import IGMNeRF, volume_render_physics


# Canonical training constants (mirror pipeline.py).
TAU_MAX_DEFAULT = 10.0

# Pre-committed saturation strata on F_truth.
STRATA = [
    ("unsaturated_F95_100", 0.95, 1.00),
    ("weak_F70_95",         0.70, 0.95),
    ("moderate_F30_70",     0.30, 0.70),
    ("strong_F05_30",       0.05, 0.30),
    ("saturated_F00_05",    0.00, 0.05),
]

# Pre-committed verdict thresholds (record BEFORE looking at numbers).
THRESHOLDS = {
    "ratio_definition":
        "R = mean(|grad|)_F00_05 / mean(|grad|)_F95_100",
    "identifiability_ceiling_supported": "R < 0.1",
    "model_capacity_supported":          "R > 0.33",
    "inconclusive_band":                 "0.1 <= R <= 0.33",
    "underpowered_floor":                "any stratum N_bins < 100",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to step_*.pt Stage 2b checkpoint.")
    p.add_argument("--physics", type=int, default=1)
    p.add_argument("--redshift", type=float, default=0.3)
    p.add_argument("--n_rays", type=int, default=64,
                   help="Subset of sightlines to use (matches pub-t1 train).")
    p.add_argument("--data_root", type=str,
                   default=r"D:/Data/sujin/CosmoGasVision/Sherwood")
    p.add_argument("--tau_max", type=float, default=TAU_MAX_DEFAULT)
    p.add_argument("--out_dir", type=str,
                   default="experiments/nerf/artifacts/d71_I_grad_norm_saturation")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def stratify(F_truth: np.ndarray, mask: np.ndarray, vals: np.ndarray):
    """Return per-stratum stats dict keyed by stratum name.

    F_truth, mask, vals all shape (n_rays, n_bins). vals = |dL/dtau| samples.
    `mask` is the DLA-include mask (True = include).
    """
    out = {}
    F_flat = F_truth.flatten()
    m_flat = mask.flatten()
    v_flat = vals.flatten()
    for name, lo, hi in STRATA:
        if name == "unsaturated_F95_100":
            sel = (F_flat >= lo) & (F_flat <= hi) & m_flat
        else:
            sel = (F_flat >= lo) & (F_flat < hi) & m_flat
        v_sel = v_flat[sel]
        n = int(sel.sum())
        if n == 0:
            out[name] = {
                "N_bins": 0, "mean": None, "median": None, "std": None,
                "F_lo": lo, "F_hi": hi,
            }
        else:
            out[name] = {
                "N_bins": n,
                "mean": float(np.mean(v_sel)),
                "median": float(np.median(v_sel)),
                "std": float(np.std(v_sel)),
                "F_lo": lo,
                "F_hi": hi,
            }
    return out


def verdict_from_strata(strata_lin, strata_log):
    """Apply pre-committed thresholds. Returns dict with verdict + rationale."""
    sat_lin = strata_lin["saturated_F00_05"]
    unsat_lin = strata_lin["unsaturated_F95_100"]
    sat_log = strata_log["saturated_F00_05"]
    unsat_log = strata_log["unsaturated_F95_100"]

    # Underpowered check first (any stratum N_bins < 100).
    underpowered = []
    for stratum_dict, tag in [(strata_lin, "linear"), (strata_log, "log1p")]:
        for name, s in stratum_dict.items():
            if s["N_bins"] < 100:
                underpowered.append(f"{tag}:{name}:N={s['N_bins']}")
    if underpowered:
        return {
            "label": "inconclusive_underpowered",
            "R_linear": None, "R_log1p": None,
            "rationale": f"At least one stratum has N_bins < 100: "
                         f"{', '.join(underpowered)}",
        }

    def _ratio(sat, unsat):
        if sat["mean"] is None or unsat["mean"] is None:
            return None
        if unsat["mean"] == 0.0:
            return float("inf")
        return sat["mean"] / unsat["mean"]

    R_lin = _ratio(sat_lin, unsat_lin)
    R_log = _ratio(sat_log, unsat_log)

    # Verdict ladder uses the LINEAR form (matches PI's dL/dtau_pred framing
    # in the dispatch brief). Log1p form reported for completeness.
    if R_lin is None:
        label = "inconclusive_missing_strata"
        rationale = "Could not compute R_linear (empty stratum)."
    elif R_lin < 0.1:
        label = "identifiability_ceiling_supported"
        rationale = (f"R_linear = {R_lin:.4g} < 0.1 -> saturated-stratum "
                     f"gradient norm is >10x smaller than unsaturated.")
    elif R_lin > 0.33:
        label = "model_capacity_supported"
        rationale = (f"R_linear = {R_lin:.4g} > 0.33 -> gradient signal "
                     f"comparable across strata; model failing to extract.")
    else:
        label = "inconclusive_intermediate"
        rationale = (f"R_linear = {R_lin:.4g} in [0.1, 0.33] -> neither "
                     f"hypothesis decisively supported.")

    return {
        "label": label,
        "R_linear": R_lin,
        "R_log1p": R_log,
        "rationale": rationale,
    }


def render_histogram(strata_lin, strata_log, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [s[0] for s in STRATA]
    short = ["F95-100", "F70-95", "F30-70", "F05-30", "F00-05"]
    mean_lin = [strata_lin[n]["mean"] if strata_lin[n]["mean"] is not None
                else 0.0 for n in names]
    std_lin = [strata_lin[n]["std"] if strata_lin[n]["std"] is not None
               else 0.0 for n in names]
    n_lin = [strata_lin[n]["N_bins"] for n in names]

    mean_log = [strata_log[n]["mean"] if strata_log[n]["mean"] is not None
                else 0.0 for n in names]
    std_log = [strata_log[n]["std"] if strata_log[n]["std"] is not None
               else 0.0 for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(short))

    axes[0].bar(x, mean_lin, yerr=std_lin, capsize=4, color="steelblue",
                alpha=0.85)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short, rotation=20)
    axes[0].set_ylabel("mean |dL/dtau_pred|  (log scale)")
    axes[0].set_title("Linear: gradient w.r.t. tau_pred")
    for i, n in enumerate(n_lin):
        axes[0].text(i, mean_lin[i] if mean_lin[i] > 0 else 1e-12,
                     f"N={n}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, mean_log, yerr=std_log, capsize=4, color="indianred",
                alpha=0.85)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short, rotation=20)
    axes[1].set_ylabel("mean |dL/dlog1p(tau_pred)|  (log scale)")
    axes[1].set_title("Log1p: gradient w.r.t. log1p(tau_pred) [training surrogate]")

    fig.suptitle("[D-71] Saturation-band per-bin gradient norm")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # -------------------------------------------------------------- checkpoint
    print(f"[d71-I] loading checkpoint: {args.checkpoint}", flush=True)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state" not in state:
        raise RuntimeError(
            f"Checkpoint at {args.checkpoint} missing 'model_state' key; got "
            f"{list(state.keys()) if isinstance(state, dict) else type(state)}"
        )

    # Single-physics pub-t1 -> defaults (no g, no physics embedding).
    model = IGMNeRF(hidden_dim=256, num_layers=8, L=10,
                    use_velocity_gradient_conditioning=False,
                    use_physics_embedding=False,
                    body_arch="current").to(device)
    model.load_state_dict(state["model_state"])
    model.eval()  # eval: no dropout/BN in this model, but be explicit.

    log_tau_amp = state["log_tau_amp"].to(device).detach()
    tau_amp = torch.exp(log_tau_amp)
    print(f"[d71-I] step={state.get('step')} log_tau_amp={float(log_tau_amp):.4f} "
          f"-> tau_amp={float(tau_amp):.4f}", flush=True)

    # -------------------------------------------------------------- data
    print(f"[d71-I] loading sightlines: physics={args.physics} z={args.redshift} "
          f"n_rays={args.n_rays}", flush=True)
    loader = SherwoodLoader(args.data_root)
    sl = loader.load_sightlines(args.physics, args.redshift)
    coords_raw = loader.get_world_coordinates(sl)
    box_max = sl["header"]["box_kpc_h"]

    n_rays = args.n_rays
    coords = torch.tensor(coords_raw[:n_rays], dtype=torch.float32,
                          device=device) / box_max
    vel_axis = torch.tensor(sl["vel_axis"], dtype=torch.float32, device=device)
    tau_gt = torch.tensor(sl["tau_h1"][:n_rays], dtype=torch.float32,
                          device=device)
    mask = torch.tensor(sl["mask_no_dla"][:n_rays], dtype=torch.bool,
                        device=device)
    n_bins = coords.shape[1]
    print(f"[d71-I] data: rays={n_rays} bins={n_bins} "
          f"DLA-excluded bins={int((~mask).sum())}", flush=True)

    # -------------------------------------------------------------- forward
    # Build tau_pred with requires_grad so we can autograd w.r.t. it directly.
    # Strategy: render tau_pred (carries autograd through the MLP), then
    # detach + re-attach as a leaf to isolate the gradient on tau_pred itself.
    # That gives us |dL/dtau_pred| per bin without going all the way to the
    # MLP weights — which is exactly the PI's identifiability question.
    print(f"[d71-I] rendering tau_pred ...", flush=True)
    with torch.no_grad():
        tau_pred_raw = volume_render_physics(
            model, coords, vel_axis=vel_axis, tau_amp=tau_amp,
            g=None, physics_id=None,
        )  # (n_rays, n_bins)

    # Re-attach tau_pred as a leaf with autograd, then compute the canonical
    # log1p-MSE loss and backprop. This isolates dL/dtau_pred.
    tau_pred = tau_pred_raw.detach().clone().requires_grad_(True)

    # Match pipeline.py [D-24] loss form exactly:
    #   diff = log1p(cap(tau_pred)) - log1p(cap(tau_gt))
    #   L_bin = diff**2 ; reduce = masked-mean
    # For per-bin grad we backprop the per-bin sum (mask-weighted) so that
    # autograd assigns the correct per-bin partial. Since the reduction is a
    # SUM here (not mean), grad scales linearly with N — that's fine because
    # the verdict is RATIO-based and stratum-comparable.
    tau_pred_eff = tau_pred.clamp_max(args.tau_max)
    tau_gt_eff = tau_gt.clamp_max(args.tau_max)
    diff = torch.log1p(tau_pred_eff) - torch.log1p(tau_gt_eff)
    diff_sq = diff * diff
    # Mask: include only non-DLA bins (training default).
    L_total = (diff_sq * mask.float()).sum()
    L_total.backward()

    grad_lin = tau_pred.grad.detach().clone()  # dL/dtau_pred per bin

    # -------------------------------------------------------------- log1p form
    # Repeat with leaf = log1p(tau_pred) to capture dL/d(log1p(tau_pred)).
    log1p_tau = torch.log1p(tau_pred_raw.detach().clamp_max(args.tau_max))
    log1p_tau_leaf = log1p_tau.clone().requires_grad_(True)
    diff_log = log1p_tau_leaf - torch.log1p(tau_gt_eff)
    L_log = ((diff_log * diff_log) * mask.float()).sum()
    L_log.backward()
    grad_log = log1p_tau_leaf.grad.detach().clone()

    # -------------------------------------------------------------- stratify
    F_truth = torch.exp(-tau_gt_eff)
    F_pred = torch.exp(-tau_pred_raw.detach().clamp_max(args.tau_max))

    F_truth_np = F_truth.cpu().numpy()
    F_pred_np = F_pred.cpu().numpy()
    mask_np = mask.cpu().numpy()
    grad_lin_abs = grad_lin.abs().cpu().numpy()
    grad_log_abs = grad_log.abs().cpu().numpy()
    F_err_abs = np.abs(F_pred_np - F_truth_np)

    strata_lin = stratify(F_truth_np, mask_np, grad_lin_abs)
    strata_log = stratify(F_truth_np, mask_np, grad_log_abs)
    strata_F_err = stratify(F_truth_np, mask_np, F_err_abs)

    # Attach mean |F_pred - F_truth| context to each stratum.
    for name in strata_lin:
        strata_lin[name]["mean_abs_F_err"] = strata_F_err[name]["mean"]
        strata_log[name]["mean_abs_F_err"] = strata_F_err[name]["mean"]

    verdict = verdict_from_strata(strata_lin, strata_log)

    # -------------------------------------------------------------- capsule
    capsule = {
        "diagnostic": "[D-71] RUNG 3 saturation-band per-bin grad-norm",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pre_committed_thresholds": THRESHOLDS,
        "checkpoint": {
            "path": os.path.abspath(args.checkpoint),
            "step": int(state.get("step", -1)),
            "log_tau_amp": float(log_tau_amp),
            "tau_amp": float(tau_amp),
            "mlflow_run_id": state.get("mlflow_run_id"),
        },
        "config": {
            "physics": args.physics,
            "redshift": args.redshift,
            "n_rays": n_rays,
            "n_bins": int(n_bins),
            "tau_max": args.tau_max,
            "seed": args.seed,
            "device": args.device,
            "loss_form": "log1p_mse_capped_masked  (canonical [D-24])",
            "reduction_for_backward": "sum (per-bin grad isolation)",
            "stratification_axis": "F_truth = exp(-clamp(tau_gt, tau_max))",
        },
        "observation_first": {
            "n_bins_total": int(n_rays * n_bins),
            "n_bins_after_dla_mask": int(mask_np.sum()),
            "tau_pred_stats": {
                "min": float(tau_pred_raw.min()),
                "max": float(tau_pred_raw.max()),
                "mean": float(tau_pred_raw.mean()),
                "median": float(tau_pred_raw.median()),
            },
            "tau_gt_stats": {
                "min": float(tau_gt.min()),
                "max": float(tau_gt.max()),
                "mean": float(tau_gt.mean()),
                "median": float(tau_gt.median()),
            },
            "F_pred_stats": {
                "min": float(F_pred.min()),
                "max": float(F_pred.max()),
                "mean": float(F_pred.mean()),
            },
            "F_truth_stats": {
                "min": float(F_truth.min()),
                "max": float(F_truth.max()),
                "mean": float(F_truth.mean()),
            },
        },
        "grad_norm_per_stratum_linear": strata_lin,
        "grad_norm_per_stratum_log1p": strata_log,
        "verdict": verdict,
    }

    capsule_path = os.path.join(args.out_dir, "capsule.json")
    with open(capsule_path, "w") as f:
        json.dump(capsule, f, indent=2)
    print(f"[d71-I] capsule -> {capsule_path}", flush=True)

    png_path = os.path.join(args.out_dir, "histogram.png")
    render_histogram(strata_lin, strata_log, png_path)
    print(f"[d71-I] histogram -> {png_path}", flush=True)

    # -------------------------------------------------------------- console
    print("\n=== [D-71] RUNG 3 RESULT (observation first) ===")
    print(f"{'stratum':<22} {'N':>8} {'mean|dL/dtau|':>16} "
          f"{'mean|dL/dlog1p|':>18} {'mean|F-Fhat|':>14}")
    for name, _, _ in STRATA:
        s_lin = strata_lin[name]
        s_log = strata_log[name]
        n = s_lin["N_bins"]
        m_lin = s_lin["mean"]
        m_log = s_log["mean"]
        m_F = s_lin["mean_abs_F_err"]
        print(f"{name:<22} {n:>8d} "
              f"{(f'{m_lin:.4e}' if m_lin is not None else 'NA'):>16} "
              f"{(f'{m_log:.4e}' if m_log is not None else 'NA'):>18} "
              f"{(f'{m_F:.4e}' if m_F is not None else 'NA'):>14}")
    print(f"\nR_linear (sat/unsat)  = {verdict['R_linear']}")
    print(f"R_log1p  (sat/unsat)  = {verdict['R_log1p']}")
    print(f"VERDICT: {verdict['label']}")
    print(f"  rationale: {verdict['rationale']}")


if __name__ == "__main__":
    main()
