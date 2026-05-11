"""§3.5 P_F overlay falsification figure — driver.

Produces the two-panel P_F(k_||) overlay called for by the
\\todo{} placeholder in paper_cvpr/sec/3_experiments.tex:97. The figure
is the visual counterpart to the τ-binned residual (all_residual.png):
the τ-binned plot shows where the residual lives, this one shows the
shape-vs-scale calibration failure on its own axis.

Curves
------
(a) Sherwood ground-truth P_F(k_||) at (physics_id=1, z=0.3), 1024
    sightlines drawn with eval_seed=42 (project convention).
(b) Cost-survey checkpoint (T3, run-id f74dbb669c9641568ab883023a84d1fa,
    step_010000.pt) after the [D-35] uniform rescale F → r*F,
    r = 0.979 / ⟨F_pred⟩. Same rays as (a). Should reproduce ~0.2825
    inertial-band residual.
(c) Pub-t1 trained-at-target checkpoint (run-id
    31acdf9d900e447081e6d051f7d42c0e, step_050000.pt). Same rays, no
    rescale (trained against 0.979 directly). Should reproduce 0.4155
    inertial-band residual exactly (same eval as W1-A / Task C).

Output: paper_cvpr/figures/pf_overlay_falsification.png plus a
companion .txt file with the falsification-verdict caption.

Reuses primitives from:
  - scripts/eval_anchor_invariance_d34.py (model build with fallback,
    chunked tau render, anchor rescale)
  - src/analysis/p_flux.py (canonical [D-13] P_F estimator,
    post-[D-35] δ_F = F/⟨F⟩ - 1 convention)

Usage
-----
    PYTHONPATH=. uv run python scripts/make_pf_overlay_fig.py

No CLI knobs — the (a)/(b)/(c) cells are wired to the source data
the §3.5 \\todo{} block names verbatim. Two optional flags are exposed
for the orchestrator: --skip-validate (don't error on >10% drift from
the published 0.2825 / 0.4155 numbers; figure still saves) and
--out-dir (override the default paper_cvpr/figures destination).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse the exact eval primitives the [D-35] / [D-39] / Wrinkle-1
# numbers were produced with — no metric-side reimplementation.
from scripts.eval_anchor_invariance_d34 import (  # noqa: E402
    _build_model_with_fallback,
    _rescale_tau,
)
from src.analysis.flux_power import compute_PF_1d  # noqa: E402
from src.analysis.stage2b_report import _render_tau_for_model  # noqa: E402
from src.data.loader import SherwoodLoader  # noqa: E402

# Defer matplotlib import so --help is fast and so that headless runs
# can swap the backend before the first figure is created.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# [D-13] inertial-range gate band, identical to compute_PF_1d defaults.
_PF_BAND = (10 ** -2.5, 10 ** -1.5)
_GATE = 0.10  # |ΔP_F/P_F| pass/fail line per [D-13]
_PUBLISHED_COSTSURVEY = 0.2825   # [D-35] preview, target reproduction
_PUBLISHED_PUBT1 = 0.4155        # [D-39] W1-A / Task C, exact target


def _render_tau_chunked(model, coords_unit_np, vel_axis, chunk_rays=32):
    """Mirror of scripts/eval_anchor_invariance_d34._eval_one_cell's
    render loop; broken out so we can call it twice (cost-survey,
    pub-t1) against the same selected ray indices.
    """
    coords = torch.tensor(coords_unit_np, dtype=torch.float32)
    vel_axis_t = torch.tensor(vel_axis, dtype=torch.float32)
    tau_chunks = []
    n_rays_total = coords.shape[0]
    for i in range(0, n_rays_total, chunk_rays):
        sl = slice(i, min(i + chunk_rays, n_rays_total))
        tau_c = _render_tau_for_model(model, coords[sl], vel_axis_t)
        tau_chunks.append(tau_c)
        print(f"[pf-overlay]   rendered rays {sl.start}..{sl.stop} of {n_rays_total}")
    return np.concatenate(tau_chunks, axis=0)


def _pf_inertial_residual(P_pred, P_truth, k_axis):
    """Mean fractional |ΔP_F / P_F| over the [D-13] inertial band.
    Identical formula to scripts/eval_anchor_invariance_d34
    ._pf_inertial_residual and scripts/wrinkle1_diagnostic. Inputs are
    already-binned P_F arrays from compute_PF_1d (so we can also draw
    the curves with the same bins)."""
    band = (
        (k_axis >= _PF_BAND[0]) & (k_axis <= _PF_BAND[1])
        & np.isfinite(P_truth) & np.isfinite(P_pred)
    )
    if not band.any():
        return float("nan"), band
    return float(np.nanmean(np.abs(P_pred[band] - P_truth[band]) / P_truth[band])), band


def _figure(
    k_axis,
    P_truth,
    P_costsurvey,
    P_pubt1,
    resid_costsurvey,
    resid_pubt1,
    band_mask,
    out_path,
):
    """Two-panel log-log overlay + fractional-residual sub-panel."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.0, 6.4),
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.08},
        sharex=True,
    )

    # Top: P_F curves.
    ax_top.loglog(
        k_axis, P_truth,
        color="black", lw=2.0, label="(a) Sherwood truth", zorder=3,
    )
    ax_top.loglog(
        k_axis, P_costsurvey,
        color="#1f77b4", lw=1.6, marker="o", ms=4,
        label=r"(b) cost-survey rescaled ($r{=}0.979/\langle F\rangle$)",
        zorder=2,
    )
    ax_top.loglog(
        k_axis, P_pubt1,
        color="#d62728", lw=1.6, marker="s", ms=4,
        label=r"(c) pub-t1 trained-at-0.979",
        zorder=2,
    )

    # Shade [D-13] inertial band on top.
    ax_top.axvspan(_PF_BAND[0], _PF_BAND[1], color="grey",
                   alpha=0.12, zorder=0,
                   label="[D-13] inertial range")
    ax_top.set_ylabel(r"$P_F(k_\parallel)$  [s/km]")
    ax_top.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax_top.set_title(
        r"$P_F(k_\parallel)$ overlay, $P_1$, $z{=}0.3$, $n_\mathrm{rays}{=}1024$",
        fontsize=11,
    )
    ax_top.grid(True, which="both", ls=":", alpha=0.5)

    # Bottom: fractional residuals + 0.10 gate line.
    rel_cs = np.abs(P_costsurvey - P_truth) / np.where(
        np.isfinite(P_truth) & (P_truth > 0), P_truth, np.nan
    )
    rel_pt = np.abs(P_pubt1 - P_truth) / np.where(
        np.isfinite(P_truth) & (P_truth > 0), P_truth, np.nan
    )
    ax_bot.semilogx(
        k_axis, rel_cs,
        color="#1f77b4", lw=1.4, marker="o", ms=3.5,
        label=fr"(b) residual {resid_costsurvey*100:.2f}%",
    )
    ax_bot.semilogx(
        k_axis, rel_pt,
        color="#d62728", lw=1.4, marker="s", ms=3.5,
        label=fr"(c) residual {resid_pubt1*100:.2f}%",
    )
    ax_bot.axhline(_GATE, color="black", ls=":", lw=1.2,
                   label=r"$|\Delta P_F/P_F|=0.10$ gate")
    ax_bot.axvspan(_PF_BAND[0], _PF_BAND[1], color="grey",
                   alpha=0.12, zorder=0)
    ax_bot.set_xlabel(r"$k_\parallel$  [s/km]")
    ax_bot.set_ylabel(r"$|\Delta P_F/P_F|$")
    ax_bot.set_yscale("log")
    ax_bot.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_bot.grid(True, which="both", ls=":", alpha=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[pf-overlay] figure saved: {out_path}")


def _caption_text(resid_costsurvey, resid_pubt1):
    """Falsification-verdict caption per [D-37] honest-reporting rule.
    Latex-author will paste this into the figure caption when they
    replace the §3.5 \\todo{} block. Leads with the empirical numbers
    before the verdict framing."""
    delta = resid_pubt1 - resid_costsurvey
    return (
        rf"$P_F(k_\parallel)$ overlay at $P_1$, $z{{=}}0.3$, "
        rf"$n_\mathrm{{rays}}{{=}}1024$. Curves: (a) Sherwood ground "
        rf"truth, (b) cost-survey checkpoint after uniform rescale to "
        rf"$\langle F\rangle{{=}}0.979$ ([D-35] preview, $|\Delta P_F/P_F|"
        rf"={resid_costsurvey*100:.2f}\%$), (c) \texttt{{pub-t1}} "
        rf"corrected-anchor full-schedule reconstruction "
        rf"($|\Delta P_F/P_F|={resid_pubt1*100:.2f}\%$). Shaded band: "
        rf"[D-13] inertial range $k_\parallel \in [10^{{-2.5}}, 10^{{-1.5}}]$ "
        rf"s/km; dotted line: $|\Delta P_F/P_F|{{=}}0.10$ gate. "
        rf"$4\times$ more training and the corrected anchor increased "
        rf"the $P_F$ residual by $\Delta={delta*100:+.2f}$ pp; the "
        rf"shape-vs-scale rescale framework is falsified."
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--cost-survey-ckpt",
        default=str(
            REPO_ROOT / "cloud_runs" / "prong3-p1-t3"
            / "P1-N1024-S0-1778229084-c08848" / "checkpoints"
            / "step_010000.pt"
        ),
    )
    p.add_argument(
        "--cost-survey-run-id",
        default="f74dbb669c9641568ab883023a84d1fa",
    )
    p.add_argument(
        "--pubt1-ckpt",
        default=str(
            REPO_ROOT / "cloud_runs" / "pub-t1-extracted"
            / "P1-N64-S0-1778430089-7f65fe" / "checkpoints"
            / "step_050000.pt"
        ),
    )
    p.add_argument(
        "--pubt1-run-id",
        default="31acdf9d900e447081e6d051f7d42c0e",
    )
    p.add_argument("--physics-id", type=int, default=1)
    p.add_argument("--redshift", type=float, default=0.3)
    p.add_argument("--n-rays-eval", type=int, default=1024)
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--target-mean-flux", type=float, default=0.979)
    p.add_argument("--chunk-rays", type=int, default=32)
    p.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "paper_cvpr" / "figures"),
    )
    p.add_argument(
        "--skip-validate", action="store_true",
        help=("Don't error if either residual drifts >10%% from the "
              "published [D-35] / [D-39] numbers; figure still saves."),
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    fig_path = out_dir / "pf_overlay_falsification.png"
    cap_path = out_dir / "pf_overlay_falsification.caption.txt"

    cost_survey_ckpt = Path(args.cost_survey_ckpt)
    pubt1_ckpt = Path(args.pubt1_ckpt)
    if not cost_survey_ckpt.exists():
        sys.exit(f"FATAL: cost-survey checkpoint not found: {cost_survey_ckpt}")
    if not pubt1_ckpt.exists():
        sys.exit(f"FATAL: pub-t1 checkpoint not found: {pubt1_ckpt}")

    print(f"[pf-overlay] cost-survey ckpt : {cost_survey_ckpt}")
    print(f"[pf-overlay]   run-id         : {args.cost_survey_run_id}")
    print(f"[pf-overlay] pub-t1 ckpt      : {pubt1_ckpt}")
    print(f"[pf-overlay]   run-id         : {args.pubt1_run_id}")
    print(f"[pf-overlay] eval cell         : P{args.physics_id} z={args.redshift}")
    print(f"[pf-overlay] n_rays / seed     : {args.n_rays_eval} / {args.eval_seed}")
    print(f"[pf-overlay] anchor (rescale)  : {args.target_mean_flux}")

    # --- Load Sherwood truth + select the eval ray draw exactly once ---
    sherwood = SherwoodLoader(str(REPO_ROOT / "Sherwood"))
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

    coords_world = sherwood.get_world_coordinates(sl_full)
    coords_unit_np = (coords_world[sel] / box_kpc_h).astype(np.float64)
    print(f"[pf-overlay] selected {n_rays} rays from {n_rays_avail} available")

    # --- (b) cost-survey: render + uniform rescale to anchor 0.979 ---
    print(f"\n[pf-overlay] === (b) cost-survey checkpoint ===")
    model_cs = _build_model_with_fallback(
        args.cost_survey_run_id, str(cost_survey_ckpt)
    )
    tau_cs_raw = _render_tau_chunked(
        model_cs, coords_unit_np, vel_axis, chunk_rays=args.chunk_rays
    )
    F_cs_raw = np.exp(-tau_cs_raw)
    mean_F_cs = float(F_cs_raw.mean())
    r_cs = float(args.target_mean_flux) / mean_F_cs
    tau_cs_rescaled = _rescale_tau(tau_cs_raw, r_cs)
    mean_F_cs_resc = float(np.exp(-tau_cs_rescaled).mean())
    print(f"[pf-overlay]   <F_pred> raw      = {mean_F_cs:.6f}")
    print(f"[pf-overlay]   rescale r         = {args.target_mean_flux}/"
          f"{mean_F_cs:.6f} = {r_cs:.6f}")
    print(f"[pf-overlay]   <F_pred> rescaled = {mean_F_cs_resc:.6f}")

    # --- (c) pub-t1: render as-is (trained at target) ---
    print(f"\n[pf-overlay] === (c) pub-t1 checkpoint ===")
    model_pt = _build_model_with_fallback(
        args.pubt1_run_id, str(pubt1_ckpt)
    )
    tau_pt = _render_tau_chunked(
        model_pt, coords_unit_np, vel_axis, chunk_rays=args.chunk_rays
    )
    F_pt = np.exp(-tau_pt)
    mean_F_pt = float(F_pt.mean())
    print(f"[pf-overlay]   <F_pred>          = {mean_F_pt:.6f}")

    # --- Compute P_F on identical k bins for all three curves ---
    print("\n[pf-overlay] computing P_F(k_||) on identical 20-bin grid")
    k_axis, P_truth = compute_PF_1d(tau_truth, vel_axis)
    _, P_costsurvey = compute_PF_1d(tau_cs_rescaled, vel_axis)
    _, P_pubt1 = compute_PF_1d(tau_pt, vel_axis)

    resid_cs, band = _pf_inertial_residual(P_costsurvey, P_truth, k_axis)
    resid_pt, _ = _pf_inertial_residual(P_pubt1, P_truth, k_axis)

    # --- Empirical numbers FIRST (honest-reporting per [D-37]) ---
    print()
    print("=" * 72)
    print("Empirical observation (lead, before any framing):")
    print("=" * 72)
    print(f"  (b) cost-survey rescaled at P1, eval_seed=42:")
    print(f"      |ΔP_F/P_F| inertial band = {resid_cs:.6f} "
          f"({resid_cs*100:.4f}%)")
    print(f"  (c) pub-t1 trained-at-0.979 at P1, eval_seed=42:")
    print(f"      |ΔP_F/P_F| inertial band = {resid_pt:.6f} "
          f"({resid_pt*100:.4f}%)")
    print(f"  delta (c - b)               = {resid_pt - resid_cs:+.6f} "
          f"({(resid_pt - resid_cs)*100:+.4f} pp)")

    # --- Validation against published numbers ---
    drift_cs = abs(resid_cs - _PUBLISHED_COSTSURVEY) / _PUBLISHED_COSTSURVEY
    drift_pt = abs(resid_pt - _PUBLISHED_PUBT1) / _PUBLISHED_PUBT1
    print()
    print("Validation vs. published numbers:")
    print(f"  cost-survey rescaled : target {_PUBLISHED_COSTSURVEY:.4f}, "
          f"got {resid_cs:.4f}, drift {drift_cs*100:.2f}% "
          f"(tolerance 10%)")
    print(f"  pub-t1 trained       : target {_PUBLISHED_PUBT1:.4f}, "
          f"got {resid_pt:.4f}, drift {drift_pt*100:.2f}% "
          f"(tolerance 1%)")

    fail = []
    if drift_cs > 0.10:
        fail.append(f"cost-survey drift {drift_cs*100:.2f}% > 10%")
    if drift_pt > 0.01:
        fail.append(f"pub-t1 drift {drift_pt*100:.2f}% > 1%")

    if fail and not args.skip_validate:
        print()
        print("VALIDATION FAILED:")
        for f in fail:
            print(f"  - {f}")
        print("Pass --skip-validate to save the figure anyway and surface "
              "for review.")
        return 1
    elif fail:
        print()
        print("VALIDATION DRIFT BEYOND TOLERANCE (saving anyway "
              "per --skip-validate):")
        for f in fail:
            print(f"  - {f}")

    # --- Draw the figure ---
    _figure(
        k_axis=k_axis,
        P_truth=P_truth,
        P_costsurvey=P_costsurvey,
        P_pubt1=P_pubt1,
        resid_costsurvey=resid_cs,
        resid_pubt1=resid_pt,
        band_mask=band,
        out_path=fig_path,
    )

    # --- Write companion caption text ---
    caption = _caption_text(resid_cs, resid_pt)
    cap_path.parent.mkdir(parents=True, exist_ok=True)
    cap_path.write_text(caption + "\n", encoding="utf-8")
    print(f"[pf-overlay] caption saved   : {cap_path}")
    print()
    print("Caption (for latex-author to paste at the §3.5 \\todo{}):")
    print("-" * 72)
    print(caption)
    print("-" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
