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


def _plot(bins_out: list[dict], pf_aggregate: float, out_path: Path) -> None:
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
        f"P1 pub-t1 step_050000 — τ-binned residual decomposition  "
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


# ---------------------------------------------------- main


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
    )
    p.add_argument("--physics-id", type=int, default=1)
    p.add_argument("--redshift", type=float, default=0.3)
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
        "--output",
        default=str(
            REPO_ROOT / "experiments" / "nerf" / "artifacts" / "eval"
            / "tau_binned" / "p1_residual.json"
        ),
    )
    p.add_argument(
        "--figure",
        default=str(
            REPO_ROOT / "experiments" / "nerf" / "artifacts" / "eval"
            / "tau_binned" / "p1_residual.png"
        ),
    )
    args = p.parse_args()

    out_path = Path(args.output)
    fig_path = Path(args.figure)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[tbr] P1 pub-t1 step_050000 — τ-binned residual decomposition")
    print(f"[tbr]   ckpt    : {args.p1_pubt1_ckpt}")
    print(f"[tbr]   seed    : {args.eval_seed}")
    print(f"[tbr]   rays    : {args.n_rays_eval}")
    print(f"[tbr]   τ bins  : {args.n_bins} log-spaced over "
          f"[{args.tau_bin_min}, {args.tau_bin_max}]")

    model = _build_model_with_fallback(args.p1_pubt1_run_id,
                                       args.p1_pubt1_ckpt)
    sherwood = SherwoodLoader(str(REPO_ROOT / "Sherwood"))

    tau_pred, tau_truth, vel_axis = _render_for_seed(
        model, sherwood, args.physics_id, args.redshift,
        args.n_rays_eval, args.eval_seed,
        chunk_rays=args.chunk_rays,
    )

    pf_aggregate = _pf_inertial_residual(tau_pred, tau_truth, vel_axis)
    print(f"[tbr] aggregate P_F residual = {pf_aggregate:.4f}  "
          f"(expected ~0.4155 vs [D-39] W1-A; reproducibility check)")

    bin_edges = np.logspace(
        np.log10(args.tau_bin_min), np.log10(args.tau_bin_max),
        args.n_bins + 1,
    )
    bins_out = _bin_residuals(tau_pred, tau_truth, bin_edges)

    decision = _mechanism_call(bins_out)

    output = {
        "physics_id": args.physics_id,
        "redshift": args.redshift,
        "n_rays_eval": args.n_rays_eval,
        "eval_seed": args.eval_seed,
        "checkpoint": str(args.p1_pubt1_ckpt),
        "run_id": args.p1_pubt1_run_id,
        "aggregate_P_F_residual": pf_aggregate,
        "expected_P_F_from_d39_W1A": 0.4155,
        "tau_bin_edges": bin_edges.tolist(),
        "bins": bins_out,
        "decision": decision,
        "metadata": {
            "p_f_band_s_per_km": list(_PF_BAND),
            "saturation_F_range": list(_SATURATION_F_RANGE),
            "driver_version": "tau_binned_residual.py v1 (2026-05-10)",
        },
    }
    out_path.write_text(json.dumps(output, indent=2))

    # Console summary table.
    print()
    print("=" * 96)
    print("[TBR] τ-binned residual decomposition — P1 pub-t1 step_050000 "
          "(seed=42, 1024 rays)")
    print("=" * 96)
    hdr = (f"{'bin':>3}  {'tau_lo':>9}  {'tau_hi':>9}  {'F_lo':>7}  "
           f"{'F_hi':>7}  {'N':>9}  {'<Δτ>':>9}  {'<ΔF>':>9}  "
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
    print(f"saturation-band F∈[{_SATURATION_F_RANGE[0]},"
          f"{_SATURATION_F_RANGE[1]}] residual-mass share = "
          f"{decision.get('saturation_residual_mass_share')}")
    print(f"saturation-band pixel share              = "
          f"{decision.get('saturation_pixel_share')}")
    print(f"concentration ratio (mass/pixel)         = "
          f"{decision.get('concentration_ratio')}")
    print(f"\nMECHANISM CALL: {decision['mechanism_call']}")

    _plot(bins_out, pf_aggregate, fig_path)
    print(f"\nFull JSON: {out_path}")
    print(f"Figure   : {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
