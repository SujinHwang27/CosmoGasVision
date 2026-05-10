"""τ-binned residual decomposition — positive identification of mechanism (c).

Context
-------
[D-39] Wrinkle-1 disambiguation addendum (2026-05-10, PI ruling) downgraded
the four-case W1-A/B/C/D outcome to AMBIGUOUS (seed sensitivity
13.76%–16.21% lay between the (b)-arm >20% and (c)-arm <10% thresholds).
The PI's positive-identification recipe owed is a τ-binned residual
decomposition: if mechanism (c) ("rescale-vs-trained intrinsic divergence
in nonlinear F(τ)=exp(−τ) saturation") is correct, the P_F gap should
*concentrate* in the saturation regime (τ_truth ~ 1–10, F ~ 0.05–0.3). If
the residual is uniform across τ_truth, (c) is *not* the right story and
the LEDGER claim must back off.

Honest-reporting [D-37]: this driver reports per-bin numerics first;
mechanism call follows mechanically from the bin where the F-residual
mass concentrates. Both outcomes are scientifically useful.

Usage
-----
    PYTHONPATH=. uv run python scripts/tau_binned_residual.py

Outputs (under experiments/nerf/artifacts/eval/tau_binned/):
    p1_residual.json   — per-bin stats + aggregate P_F + mechanism call
    p1_residual.png    — diagnostic figure (signed F-residual + count)

Out of scope: P2/P3/P4 (cross-physics extension after P1 ruling), paper
modifications (orchestrator handles).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.flux_power import compute_PF_1d  # noqa: E402
from src.analysis.stage2b_report import (  # noqa: E402
    _build_model_from_run,
    _load_mlflow_run,
    _render_tau_for_model,
)
from src.data.loader import SherwoodLoader  # noqa: E402
from src.models.nerf import IGMNeRF  # noqa: E402

_PF_BAND = (10 ** -2.5, 10 ** -1.5)
_SATURATION_F_RANGE = (0.05, 0.30)  # the "where the gap should live" zone


# Cross-physics extension (PI dispatch 2026-05-10) — same pattern as
# scripts/eval_anchor_invariance_d34.py PUB_T1_CELLS. Each cell binds a
# physics_id to its juno-trained step_050000 checkpoint and source MLflow
# run-id. Source run-IDs are recorded for provenance; the fallback loader
# handles missing local MLflow entries.
PUB_T1_CELLS = {
    "P1": {
        "label": "P1",
        "physics_id": 1,
        "redshift": 0.3,
        "ckpt_path": str(
            REPO_ROOT / "cloud_runs" / "pub-t1-extracted"
            / "P1-N64-S0-1778430089-7f65fe" / "checkpoints" / "step_050000.pt"
        ),
        "run_id": "31acdf9d900e447081e6d051f7d42c0e",
        "expected_P_F_from_d39_W1A": 0.4155,
    },
    "P2": {
        "label": "P2",
        "physics_id": 2,
        "redshift": 0.3,
        "ckpt_path": str(
            REPO_ROOT / "cloud_runs" / "pub-t1-extracted"
            / "P2-N64-S0-1778430089-0f7fc8" / "checkpoints" / "step_050000.pt"
        ),
        "run_id": "f7fafa2320164a9cb7c9c29fad74474d",
        "expected_P_F_from_d39_W1A": None,
    },
    "P3": {
        "label": "P3",
        "physics_id": 3,
        "redshift": 0.3,
        "ckpt_path": str(
            REPO_ROOT / "cloud_runs" / "pub-t1-extracted"
            / "P3-N64-S0-1778430089-b9dad4" / "checkpoints" / "step_050000.pt"
        ),
        "run_id": "62aeb93aacd44cb0aeca5b51f802a352",
        "expected_P_F_from_d39_W1A": None,
    },
    "P4": {
        "label": "P4",
        "physics_id": 4,
        "redshift": 0.3,
        "ckpt_path": str(
            REPO_ROOT / "cloud_runs" / "pub-t1-extracted"
            / "P4-N64-S0-1778430089-b18fc5" / "checkpoints" / "step_050000.pt"
        ),
        "run_id": "fc3817b3b3114cae8b134800aedf20e1",
        "expected_P_F_from_d39_W1A": None,
    },
}


# ---------------------------------------------------- model construction
# Mirrors scripts/wrinkle1_diagnostic.py._build_model_with_fallback exactly.


def _build_model_with_fallback(run_id: str, ckpt_path: str) -> IGMNeRF:
    run, _ = _load_mlflow_run(run_id)
    if run is None:
        print(f"[tbr] MLflow lookup miss for {run_id}; using production "
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
        print(f"[tbr] Loaded weights from {ckpt_path}")
        return model
    return _build_model_from_run(run, ckpt_path)


# ---------------------------------------------------- sightline rendering


def _render_for_seed(
    model: IGMNeRF,
    sherwood: SherwoodLoader,
    physics_id: int,
    redshift: float,
    n_rays_eval: int,
    eval_seed: int,
    chunk_rays: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirrors scripts/wrinkle1_diagnostic.py._render_for_seed (same seed
    convention as eval_partial_d13.py / W1-A). Returns (tau_pred,
    tau_truth, vel_axis) all in numpy."""
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

    chunks = []
    n_rays_total = coords.shape[0]
    for i in range(0, n_rays_total, chunk_rays):
        sl_c = slice(i, min(i + chunk_rays, n_rays_total))
        tau_c = _render_tau_for_model(model, coords[sl_c], vel_axis_t)
        chunks.append(tau_c)
        print(f"[tbr]   rendered rays {sl_c.start}..{sl_c.stop} of "
              f"{n_rays_total}")
    tau_pred = np.concatenate(chunks, axis=0)
    return tau_pred, tau_truth, vel_axis


# ---------------------------------------------------- aggregate P_F


def _pf_inertial_residual(tau_pred: np.ndarray, tau_truth: np.ndarray,
                          vel_axis: np.ndarray) -> float:
    """Headline P_F residual on the [D-13] band — same calculation as
    [D-39] gate / W1-A. Used here only to assert reproducibility of the
    P1 pub-t1 0.4155 number from the W1-A row of the wrinkle1 JSON."""
    k_p, P_p = compute_PF_1d(tau_pred, vel_axis)
    k_t, P_t = compute_PF_1d(tau_truth, vel_axis)
    band = (
        (k_t >= _PF_BAND[0]) & (k_t <= _PF_BAND[1])
        & np.isfinite(P_t) & np.isfinite(P_p)
    )
    if not band.any():
        return float("nan")
    return float(np.nanmean(np.abs(P_p[band] - P_t[band]) / P_t[band]))


# ---------------------------------------------------- per-bin decomposition


def _bin_residuals(
    tau_pred: np.ndarray,
    tau_truth: np.ndarray,
    bin_edges: np.ndarray,
) -> list[dict]:
    """For each τ_truth bin, compute count + signed residuals + per-bin
    Pearson r. All bins are evaluated on the *flattened* (N_rays * N_vel)
    pixel grid; no aggregation along velocity (one (sightline, v) pair per
    sample).

    Notes
    -----
    - F = exp(−τ) on raw (pre-cap) τ; we do NOT apply the [D-24] τ_max=10
      cap here because the cap is a *loss-domain* operation. The decomposition
      analyzes the rendered τ field as-is.
    - For τ_truth == 0 (saturated-mask voxels with zero optical depth), we
      put them in the leftmost bin (no special-case dropout).
    """
    tau_pred_f = np.asarray(tau_pred, dtype=np.float64).ravel()
    tau_truth_f = np.asarray(tau_truth, dtype=np.float64).ravel()
    F_pred = np.exp(-tau_pred_f)
    F_truth = np.exp(-tau_truth_f)

    # np.digitize uses right-exclusive bins by default (bin i covers
    # [edges[i-1], edges[i])). bin index 0 => below first edge,
    # bin index len(edges) => above last edge.
    # We map τ_truth=0 to bin 1 (first real bin) by clipping a tiny floor.
    tau_truth_clipped = np.maximum(tau_truth_f, bin_edges[0] * 1.0001)
    idx = np.digitize(tau_truth_clipped, bin_edges, right=False)
    # idx in [1, len(edges)-1] are the real bins.

    bins_out = []
    n_bins = len(bin_edges) - 1
    for b in range(1, n_bins + 1):
        mask = idx == b
        n = int(mask.sum())
        bin_lo = float(bin_edges[b - 1])
        bin_hi = float(bin_edges[b])
        if n < 2:
            bins_out.append({
                "bin_index": b,
                "tau_truth_lo": bin_lo,
                "tau_truth_hi": bin_hi,
                "F_truth_lo": float(np.exp(-bin_hi)),
                "F_truth_hi": float(np.exp(-bin_lo)),
                "count": n,
                "mean_tau_residual": None,
                "median_tau_residual": None,
                "mean_F_residual": None,
                "median_F_residual": None,
                "pearson_r_tau": None,
                "in_saturation_band": False,
                "frac_of_total_pixels": 0.0,
            })
            continue
        tau_res = tau_pred_f[mask] - tau_truth_f[mask]
        F_res = F_pred[mask] - F_truth[mask]
        # Pearson r — guard zero-variance (e.g., all-saturated bin).
        tp = tau_pred_f[mask]
        tt = tau_truth_f[mask]
        if tp.std() == 0 or tt.std() == 0:
            r = None
        else:
            r = float(np.corrcoef(tp, tt)[0, 1])
        # Is this bin centered in the saturation band F∈[0.05, 0.30]?
        F_bin_center = float(np.exp(-0.5 * (bin_lo + bin_hi)))
        in_sat = (
            _SATURATION_F_RANGE[0] <= F_bin_center <= _SATURATION_F_RANGE[1]
        )
        bins_out.append({
            "bin_index": b,
            "tau_truth_lo": bin_lo,
            "tau_truth_hi": bin_hi,
            "F_truth_lo": float(np.exp(-bin_hi)),
            "F_truth_hi": float(np.exp(-bin_lo)),
            "count": n,
            "mean_tau_residual": float(tau_res.mean()),
            "median_tau_residual": float(np.median(tau_res)),
            "mean_F_residual": float(F_res.mean()),
            "median_F_residual": float(np.median(F_res)),
            "pearson_r_tau": r,
            "in_saturation_band": in_sat,
            "frac_of_total_pixels": float(n) / float(tau_truth_f.size),
        })
    return bins_out


# ---------------------------------------------------- mechanism call


def _mechanism_call(bins_out: list[dict]) -> dict:
    """Apply a residual-concentration test to the per-bin signed F-residual.

    Test: compute the share of total |F-residual mass| that falls in
    saturation-band bins (F_center ∈ [0.05, 0.30]) versus the share of
    pixels those bins represent. If the residual-mass share is more than
    ~1.5× the pixel-share, the residual concentrates in the saturation
    regime — positive ID of (c). If the two shares are comparable
    (within ~1.2×), the residual is roughly uniform across τ_truth and
    the (c) claim should be backed off.

    Honest-reporting [D-37]: thresholds are stated up front; the call is
    mechanical from the per-bin numbers.
    """
    # Aggregate residual mass = sum_b |mean_F_residual_b| * count_b
    # (proxy for L1 contribution; sign is reported per-bin separately).
    total_mass = 0.0
    sat_mass = 0.0
    total_pix = 0
    sat_pix = 0
    for b in bins_out:
        if b["mean_F_residual"] is None:
            continue
        m = abs(b["mean_F_residual"]) * b["count"]
        total_mass += m
        total_pix += b["count"]
        if b["in_saturation_band"]:
            sat_mass += m
            sat_pix += b["count"]
    if total_mass == 0 or total_pix == 0:
        return {
            "saturation_residual_mass_share": None,
            "saturation_pixel_share": None,
            "concentration_ratio": None,
            "mechanism_call": "no-data",
        }
    sat_mass_share = sat_mass / total_mass
    sat_pix_share = sat_pix / total_pix
    if sat_pix_share == 0:
        ratio = None
    else:
        ratio = sat_mass_share / sat_pix_share

    if ratio is None:
        call = "no-saturation-bins-in-grid"
    elif ratio >= 1.5:
        call = "c-positive-id-residual-concentrates-in-saturation"
    elif ratio >= 1.2:
        call = "c-weakened-id-mild-saturation-concentration"
    elif ratio >= 0.83:  # 1/1.2; the symmetric "roughly uniform" band
        call = "against-c-residual-roughly-uniform-across-tau"
    else:
        call = "against-c-residual-concentrates-outside-saturation"

    return {
        "saturation_band_F": list(_SATURATION_F_RANGE),
        "saturation_residual_mass_share": float(sat_mass_share),
        "saturation_pixel_share": float(sat_pix_share),
        "concentration_ratio": float(ratio) if ratio is not None else None,
        "concentration_thresholds": {
            "positive_id": ">=1.5",
            "weakened_id": ">=1.2",
            "uniform_band": "0.83-1.2",
            "anti": "<0.83",
        },
        "mechanism_call": call,
    }


# ---------------------------------------------------- figure


def _plot(bins_out: list[dict], pf_aggregate: float, out_path: Path,
          cell_label: str = "P1") -> None:
    centers = []
    counts = []
    mean_F_res = []
    in_sat = []
    for b in bins_out:
        if b["mean_F_residual"] is None:
            continue
        c = 0.5 * (b["tau_truth_lo"] + b["tau_truth_hi"])
        centers.append(c)
        counts.append(b["count"])
        mean_F_res.append(b["mean_F_residual"])
        in_sat.append(b["in_saturation_band"])
    centers = np.asarray(centers)
    counts = np.asarray(counts)
    mean_F_res = np.asarray(mean_F_res)
    in_sat = np.asarray(in_sat, dtype=bool)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
    )

    # Top: signed F-residual per τ_truth bin.
    colors = np.where(in_sat, "#d62728", "#1f77b4")
    ax1.bar(centers, mean_F_res, width=np.diff(np.log10(centers).tolist()
                                               + [np.log10(centers[-1]) + 0.5])
                                               * 0 + 0,  # widths set below
            color=colors)
    # Re-draw with per-bin widths spanning [lo, hi] in log-x.
    ax1.clear()
    for b in bins_out:
        if b["mean_F_residual"] is None:
            continue
        col = "#d62728" if b["in_saturation_band"] else "#1f77b4"
        ax1.bar(
            0.5 * (b["tau_truth_lo"] + b["tau_truth_hi"]),
            b["mean_F_residual"],
            width=(b["tau_truth_hi"] - b["tau_truth_lo"]),
            color=col, edgecolor="black", linewidth=0.4,
        )
    ax1.axhline(0, color="black", lw=0.6)
    # Saturation band F∈[0.05,0.30] ↔ τ∈[-ln0.30, -ln0.05] ≈ [1.20, 3.00].
    tau_sat_lo = -np.log(_SATURATION_F_RANGE[1])
    tau_sat_hi = -np.log(_SATURATION_F_RANGE[0])
    ax1.axvspan(tau_sat_lo, tau_sat_hi, color="orange", alpha=0.15,
                label=f"saturation band F∈[{_SATURATION_F_RANGE[0]},"
                      f"{_SATURATION_F_RANGE[1]}]")
    ax1.set_xscale("log")
    ax1.set_ylabel("mean (F_pred − F_truth) per bin")
    ax1.set_title(
        f"{cell_label} pub-t1 step_050000 — τ-binned residual decomposition  "
        f"(aggregate P_F={pf_aggregate:.4f}, gate <0.10)"
    )
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.3)

    # Bottom: pixel count per bin (log y).
    for b in bins_out:
        if b["count"] == 0:
            continue
        col = "#d62728" if b["in_saturation_band"] else "#1f77b4"
        ax2.bar(
            0.5 * (b["tau_truth_lo"] + b["tau_truth_hi"]),
            b["count"],
            width=(b["tau_truth_hi"] - b["tau_truth_lo"]),
            color=col, edgecolor="black", linewidth=0.4,
        )
    ax2.axvspan(tau_sat_lo, tau_sat_hi, color="orange", alpha=0.15)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("τ_truth bin center  (log scale)")
    ax2.set_ylabel("pixel count per bin")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[tbr] figure -> {out_path}")


# ---------------------------------------------------- four-panel composite


def _plot_composite(per_cell: dict, out_path: Path) -> None:
    """Four-panel composite: one (signed F-residual + pixel count) twin
    plot per cell, shared y-scale across cells. Mirrors the per-cell
    figure layout but stacks P1..P4 in a 2x2 grid for cross-physics
    comparison.
    """
    cells_order = [k for k in ("P1", "P2", "P3", "P4") if k in per_cell]
    n = len(cells_order)
    if n == 0:
        print("[tbr] no cells to composite")
        return

    # Find global F-residual extent for shared y-scale.
    all_F = []
    for k in cells_order:
        for b in per_cell[k]["bins"]:
            if b["mean_F_residual"] is not None:
                all_F.append(b["mean_F_residual"])
    if not all_F:
        return
    F_lim = max(abs(min(all_F)), abs(max(all_F))) * 1.05

    tau_sat_lo = -np.log(_SATURATION_F_RANGE[1])
    tau_sat_hi = -np.log(_SATURATION_F_RANGE[0])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, label in zip(axes, cells_order):
        cell = per_cell[label]
        bins_out = cell["bins"]
        pf_agg = cell["aggregate_P_F_residual"]
        decision = cell["decision"]
        ratio = decision.get("concentration_ratio")
        call = decision.get("mechanism_call", "?")
        for b in bins_out:
            if b["mean_F_residual"] is None:
                continue
            col = "#d62728" if b["in_saturation_band"] else "#1f77b4"
            ax.bar(
                0.5 * (b["tau_truth_lo"] + b["tau_truth_hi"]),
                b["mean_F_residual"],
                width=(b["tau_truth_hi"] - b["tau_truth_lo"]),
                color=col, edgecolor="black", linewidth=0.4,
            )
        ax.axhline(0, color="black", lw=0.6)
        ax.axvspan(tau_sat_lo, tau_sat_hi, color="orange", alpha=0.15)
        ax.set_xscale("log")
        ax.set_ylim(-F_lim, F_lim)
        ratio_str = f"{ratio:.2f}" if ratio is not None else "n/a"
        ax.set_title(
            f"{label}  P_F={pf_agg:.3f}  R={ratio_str}\n{call}",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
        ax.set_xlabel("τ_truth bin center (log)")
        ax.set_ylabel("mean (F_pred − F_truth)")

    fig.suptitle(
        "pub-t1 P1-P4 — τ-binned residual decomposition (cross-physics)  "
        f"[saturation band F∈[{_SATURATION_F_RANGE[0]},"
        f"{_SATURATION_F_RANGE[1]}] in orange]",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[tbr] composite figure -> {out_path}")


# ---------------------------------------------------- per-cell evaluator


def _evaluate_cell(
    cell: dict,
    *,
    n_rays_eval: int,
    eval_seed: int,
    chunk_rays: int,
    tau_bin_min: float,
    tau_bin_max: float,
    n_bins: int,
    sherwood: SherwoodLoader,
) -> dict:
    """Run the full τ-binned residual decomposition for one cell. Returns
    a JSON-ready dict identical in schema to the P1 baseline.
    """
    label = cell["label"]
    physics_id = int(cell["physics_id"])
    redshift = float(cell["redshift"])
    ckpt_path = cell["ckpt_path"]
    run_id = cell["run_id"]
    expected_pf = cell.get("expected_P_F_from_d39_W1A")

    print(f"\n[tbr] === {label} (physics={physics_id}, z={redshift}) ===")
    print(f"[tbr]   ckpt    : {ckpt_path}")
    print(f"[tbr]   run_id  : {run_id}")
    print(f"[tbr]   seed    : {eval_seed}")
    print(f"[tbr]   rays    : {n_rays_eval}")
    print(f"[tbr]   τ bins  : {n_bins} log-spaced over "
          f"[{tau_bin_min}, {tau_bin_max}]")

    model = _build_model_with_fallback(run_id, ckpt_path)

    tau_pred, tau_truth, vel_axis = _render_for_seed(
        model, sherwood, physics_id, redshift,
        n_rays_eval, eval_seed,
        chunk_rays=chunk_rays,
    )

    pf_aggregate = _pf_inertial_residual(tau_pred, tau_truth, vel_axis)
    if expected_pf is not None:
        print(f"[tbr] aggregate P_F residual = {pf_aggregate:.4f}  "
              f"(expected ~{expected_pf} vs [D-39] W1-A; "
              f"reproducibility check)")
    else:
        print(f"[tbr] aggregate P_F residual = {pf_aggregate:.4f}")

    bin_edges = np.logspace(
        np.log10(tau_bin_min), np.log10(tau_bin_max),
        n_bins + 1,
    )
    bins_out = _bin_residuals(tau_pred, tau_truth, bin_edges)
    decision = _mechanism_call(bins_out)

    output = {
        "physics_id": physics_id,
        "redshift": redshift,
        "n_rays_eval": n_rays_eval,
        "eval_seed": eval_seed,
        "checkpoint": str(ckpt_path),
        "run_id": run_id,
        "aggregate_P_F_residual": pf_aggregate,
        "expected_P_F_from_d39_W1A": expected_pf,
        "tau_bin_edges": bin_edges.tolist(),
        "bins": bins_out,
        "decision": decision,
        "metadata": {
            "p_f_band_s_per_km": list(_PF_BAND),
            "saturation_F_range": list(_SATURATION_F_RANGE),
            "driver_version": "tau_binned_residual.py v2 (2026-05-10, "
                              "cross-physics P1-P4)",
            "cell_label": label,
        },
    }
    return output


def _print_summary_table(label: str, output: dict) -> None:
    bins_out = output["bins"]
    decision = output["decision"]
    print()
    print("=" * 96)
    print(f"[TBR] τ-binned residual decomposition — {label} pub-t1 "
          f"step_050000 (seed={output['eval_seed']}, "
          f"{output['n_rays_eval']} rays)")
    print("=" * 96)
    hdr = (f"{'bin':>3}  {'tau_lo':>9}  {'tau_hi':>9}  {'F_lo':>7}  "
           f"{'F_hi':>7}  {'N':>9}  {'<dtau>':>9}  {'<dF>':>9}  "
           f"{'r_pear':>7}  sat?")
    print(hdr)
    print("-" * len(hdr))
    for b in bins_out:
        sat_marker = "*" if b["in_saturation_band"] else " "
        if b["mean_F_residual"] is None:
            print(f"{b['bin_index']:>3}  {b['tau_truth_lo']:>9.3e}  "
                  f"{b['tau_truth_hi']:>9.3e}  {b['F_truth_lo']:>7.3f}  "
                  f"{b['F_truth_hi']:>7.3f}  {b['count']:>9}  "
                  f"{'-':>9}  {'-':>9}  {'-':>7}   {sat_marker}")
            continue
        r_str = (f"{b['pearson_r_tau']:>7.3f}"
                 if b["pearson_r_tau"] is not None else f"{'n/a':>7}")
        print(f"{b['bin_index']:>3}  {b['tau_truth_lo']:>9.3e}  "
              f"{b['tau_truth_hi']:>9.3e}  {b['F_truth_lo']:>7.3f}  "
              f"{b['F_truth_hi']:>7.3f}  {b['count']:>9}  "
              f"{b['mean_tau_residual']:>9.3e}  "
              f"{b['mean_F_residual']:>9.3e}  "
              f"{r_str}   {sat_marker}")
    print()
    print(f"saturation-band F in [{_SATURATION_F_RANGE[0]},"
          f"{_SATURATION_F_RANGE[1]}] residual-mass share = "
          f"{decision.get('saturation_residual_mass_share')}")
    print(f"saturation-band pixel share              = "
          f"{decision.get('saturation_pixel_share')}")
    print(f"concentration ratio (mass/pixel)         = "
          f"{decision.get('concentration_ratio')}")
    print(f"\nMECHANISM CALL [{label}]: {decision['mechanism_call']}")


# ---------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--cell", choices=("P1", "P2", "P3", "P4", "all"),
        default="P1",
        help="Cross-physics cell selector (PUB_T1_CELLS dict). 'all' runs "
             "P1..P4 sequentially and writes a 4-panel composite. Default "
             "'P1' preserves the original P1-only invocation.",
    )
    p.add_argument("--n-rays-eval", type=int, default=1024)
    p.add_argument("--eval-seed", type=int, default=42,
                   help="[D-39] gate convention; do not change.")
    p.add_argument("--chunk-rays", type=int, default=32)
    p.add_argument(
        "--tau-bin-min", type=float, default=1e-3,
        help="Left edge of the lowest τ_truth bin (linear regime).",
    )
    p.add_argument(
        "--tau-bin-max", type=float, default=1e2,
        help="Right edge of the highest τ_truth bin (DLA tail; the "
             "[D-24] cap is τ_max=10 but we leave headroom to see "
             "whether any rendered τ exceeds the cap).",
    )
    p.add_argument(
        "--n-bins", type=int, default=10,
        help="Number of log-spaced τ_truth bins.",
    )
    p.add_argument(
        "--out-dir",
        default=str(
            REPO_ROOT / "experiments" / "nerf" / "artifacts" / "eval"
            / "tau_binned"
        ),
        help="Directory for per-cell JSON + PNG outputs.",
    )
    # Legacy overrides (kept so the old P1 invocation still works verbatim).
    p.add_argument("--output", default=None,
                   help="(Legacy) override JSON path for single-cell run.")
    p.add_argument("--figure", default=None,
                   help="(Legacy) override PNG path for single-cell run.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sherwood = SherwoodLoader(str(REPO_ROOT / "Sherwood"))

    if args.cell == "all":
        labels = ["P1", "P2", "P3", "P4"]
    else:
        labels = [args.cell]

    per_cell: dict[str, dict] = {}
    for label in labels:
        cell = PUB_T1_CELLS[label]
        output = _evaluate_cell(
            cell,
            n_rays_eval=int(args.n_rays_eval),
            eval_seed=int(args.eval_seed),
            chunk_rays=int(args.chunk_rays),
            tau_bin_min=float(args.tau_bin_min),
            tau_bin_max=float(args.tau_bin_max),
            n_bins=int(args.n_bins),
            sherwood=sherwood,
        )
        per_cell[label] = output

        # Per-cell paths. Legacy overrides apply only to single-cell mode.
        if args.cell != "all" and args.output is not None:
            out_path = Path(args.output)
        else:
            out_path = out_dir / f"{label.lower()}_residual.json"
        if args.cell != "all" and args.figure is not None:
            fig_path = Path(args.figure)
        else:
            fig_path = out_dir / f"{label.lower()}_residual.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig_path.parent.mkdir(parents=True, exist_ok=True)

        out_path.write_text(json.dumps(output, indent=2))
        _print_summary_table(label, output)
        _plot(output["bins"], output["aggregate_P_F_residual"],
              fig_path, cell_label=label)
        print(f"\n[{label}] Full JSON: {out_path}")
        print(f"[{label}] Figure   : {fig_path}")

    if args.cell == "all" and len(per_cell) >= 2:
        composite_path = out_dir / "all_residual.png"
        _plot_composite(per_cell, composite_path)

        # Print cross-cell R table.
        print()
        print("=" * 72)
        print("Cross-physics summary: pub-t1 P1-P4 concentration ratio R")
        print("=" * 72)
        print(f"{'Cell':<6}{'P_F_agg':>10}{'R (sat mass/pix)':>20}"
              f"  mechanism call")
        print("-" * 72)
        for label in ("P1", "P2", "P3", "P4"):
            if label not in per_cell:
                continue
            d = per_cell[label]["decision"]
            r = d.get("concentration_ratio")
            r_str = f"{r:.3f}" if r is not None else "n/a"
            print(f"{label:<6}"
                  f"{per_cell[label]['aggregate_P_F_residual']:>10.4f}"
                  f"{r_str:>20}  {d['mechanism_call']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
