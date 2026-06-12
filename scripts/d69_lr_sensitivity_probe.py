"""[D-69 Deliverable A] lr-sensitivity probe for the Stage-1a pretrain path.

Three CPU cells at lr_max in {1e-4, 5e-4, 1e-3} to disambiguate amendment-6's
R-b-pre1 fire on the production-matched lr_max=5e-4 preflight. Per the
pre-committed verdict matrix:

    PASS         : r_pre_at_M1 <= 0.5
    FAIL_MOVING  : r_pre_at_M1 > 0.5 AND var_ratio strictly increasing
                   from step 100 to fire_step (or step 1000 if NO_FIRE)
    FAIL_FLAT    : var_ratio fluctuates within +/- 10% across trajectory

Overall verdict:
    (a) confirmed : >= 1 cell PASS                 -> ADVANCE_JUNO_WITH_LR={best}
    (a)+(b) mixed : 0 PASS but >=1 FAIL_MOVING
                    with var_ratio improving       -> EXTEND_CPU_PROBE_LONGER
    (c) confirmed : ALL FAIL_FLAT                  -> FREEZE_ESCALATE_D62_LADDER_REFRAME

The harness imports helpers from experiments.nerf.pipeline and replicates the
train_pretrain step loop inline so trajectory probes can be inserted at the
fixed step grid {100, 250, 500, 750, 1000} without touching pipeline.py.

Honest-reporting (CLAUDE.md / [D-37] rule (a)): the driver records what the
probe observes and emits the pre-committed verdict mechanically. It does not
recommend routing beyond the matrix; PI routes from the summary.

---------------------------------------------------------------------------
[D-73] §E A1 head-probe extension (2026-06-10, amendments AM-1..AM-5 binding):

* ``--head {softplus,linear-log}`` plumbed to IGMNeRF(density_head=...).
  Under linear-log, out[..., 0] IS log10(rho/<rho> + 1e-3); the training
  loss is computed DIRECTLY on the raw output vs log10(rho_truth + 1e-3)
  (AM-5: no round-trip through clamp(10**out) — the clamp kills gradient
  below out = -3). The 10**-conversion is probe-side only.
* AM-1: default artifact dir is experiments/nerf/artifacts/d73_a1_head_probe;
  every cell JSON carries a "head" field; the [D-69] artifact dir
  experiments/nerf/artifacts/d69_lr_probe is READ-ONLY (hard-guarded).
  The [D-69] verdict matrix above is VOID / non-binding for the [D-73]
  gate — it is still computed per cell for continuity but the summary is
  tagged "d69_matrix": "VOID-for-D73"; the D73 verdict is computed solely
  from the AM-2 rule below.
* AM-2: every recorded probe step (PLUS a step-0 point) records, on the
  same fixed probe sample, ratio = Var(rho_theta)/Var(rho_truth_fixed)
  (linear rho/<rho> space) AND pearson_r = corr(rho_theta, rho_truth) over
  the same voxels. [D-73] verdict rule (pre-registered): ESCAPE iff in >=1
  lr cell there exists a recorded step <= 1000 where median-across-3-seeds
  ratio > 0.1 AND median pearson_r >= 0.2; ratio-pass/corr-fail at every
  such step -> ESCAPE-UNSTABLE, routed as COLLAPSE; otherwise COLLAPSE.
* AM-3: Softplus control cells (seeds {1,2} x 3 lr) run into the same d73
  artifact dir; seed-0 Softplus = existing [D-69] artifacts, cite-only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# Pull the pipeline helpers (single source of truth for the pretrain math).
from experiments.nerf.pipeline import (  # noqa: E402
    PRETRAIN_LOG_EPS,
    _m1_verdict,
    _pretrain_density_head,
    _pretrain_load_rho_field,
    _pretrain_loss,
    _pretrain_sample_voxels,
    build_lr_lambda,
    set_global_seed,
)
from src.data.loader import SherwoodLoader  # noqa: E402
from src.models.nerf import IGMNeRF  # noqa: E402


# Pre-committed grid + budgets (per dispatch spec).
PROBE_STEPS = [100, 250, 500, 750, 1000]
DEFAULT_CELLS = [1e-4, 5e-4, 1e-3]
# AM-1: [D-69] dir is READ-ONLY; the [D-73] dir is the write default.
D69_ARTIFACT_DIR = Path("experiments/nerf/artifacts/d69_lr_probe")
ARTIFACT_DIR = Path("experiments/nerf/artifacts/d73_a1_head_probe")

# [D-73] §E + amendment-1 AM-2 verdict rule (pre-registered; verbatim in
# d73_summary.json). Thresholds are [D-70] M0 / R-b-pre1-frame (linear-space).
D73_RATIO_THRESHOLD = 0.1
D73_PEARSON_THRESHOLD = 0.2
D73_RULE_TEXT = (
    "ESCAPE iff in >=1 lr cell there exists a recorded step <= 1000 where "
    "median-across-3-seeds ratio > 0.1 AND median pearson_r >= 0.2 at the "
    "same step; ratio-pass/corr-fail at every such step -> ESCAPE-UNSTABLE, "
    "routed as COLLAPSE per [D-37] rule (a); otherwise COLLAPSE. "
    "(median-of-3 == 2-of-3.)"
)


def _head_loss(rho_theta_head, rho_truth, head: str):
    """Head-aware L_pre.

    softplus  : rho_theta_head is linear-space rho/<rho>; the canonical
                _pretrain_loss applies log10(. + eps) to both sides.
    linear-log: rho_theta_head IS log10(rho/<rho> + eps) raw (AM-5) — the
                loss is computed DIRECTLY on it vs log10(truth + eps); no
                round-trip through clamp(10**out) (clamp kills gradient
                below out = -3).
    """
    if head == "softplus":
        return _pretrain_loss(rho_theta_head, rho_truth)
    diff = rho_theta_head - torch.log10(rho_truth + PRETRAIN_LOG_EPS)
    return (diff * diff).mean()


def _theta_linear(rho_theta_head, head: str):
    """Linear-space rho_theta for the variance/correlation probes ONLY."""
    if head == "softplus":
        return rho_theta_head
    return IGMNeRF.density_log_to_linear(rho_theta_head)


def _format_lr_tag(lr: float) -> str:
    """File-system-safe lr tag (e.g. 1e-4 -> '1e-04')."""
    return f"{lr:.0e}".replace("+", "")


def _var_ratio_probe(model, rho_field_torch, coords_fixed, var_truth_fixed,
                     device, truth_fixed=None,
                     head: str = "softplus") -> dict[str, float]:
    """Forward the model on a FIXED probe coordinate set and return
    (var_rho_theta, var_rho_truth_fixed, ratio[, pearson_r]).

    DEVIATION FROM SPEC (recorded per [D-37] rule (a)): the dispatch brief
    said record ``{step, var_rho_theta, var_rho_truth, ratio}`` at each
    trajectory step. The naive reading samples fresh crops at each step,
    but the truth-side sample then becomes a confound — its variance
    swings 2x-4x cell-to-cell from random crop draws on a 64^3 field,
    which is much larger than any plausible model-side trajectory signal.
    Fix: probe on a fixed coord+truth sample drawn ONCE per cell. The
    model side (rho_theta) carries the entirety of the per-step movement;
    ratio = var_theta / var_truth_fixed isolates model-side behavior.

    Mirrors the R-b-pre1 backstop comparison in train_pretrain (pipeline.py
    lines ~893-919) on the rho_theta side. The var_truth denominator comes
    from the SAME single fixed coord+truth draw — one call, same voxel set
    for theta and truth (AM-5 docstring fix: an earlier version of this
    docstring wrongly claimed a separate freshly-sampled probe call; the
    implementation has always used the one fixed sample, which is correct).

    AM-2: when ``truth_fixed`` is provided, also returns ``pearson_r`` =
    Pearson correlation(rho_theta_linear, rho_truth) over the same voxels.
    Both variance and correlation are computed in linear rho/<rho> space;
    under the linear-log head the raw log output is converted via
    IGMNeRF.density_log_to_linear (probe-side only, never in the loss).
    """
    with torch.no_grad():
        theta_eval = _pretrain_density_head(model, coords_fixed)
        theta_lin = _theta_linear(theta_eval, head)
        var_theta = float(theta_lin.var(unbiased=True).item())
    out = {
        "var_rho_theta": var_theta,
        "var_rho_truth": var_truth_fixed,
        "ratio": var_theta / max(var_truth_fixed, 1e-30),
    }
    if truth_fixed is not None:
        t = theta_lin.detach().cpu().numpy().astype(np.float64)
        u = truth_fixed.detach().cpu().numpy().astype(np.float64)
        if np.std(t) < 1e-30 or np.std(u) < 1e-30:
            out["pearson_r"] = 0.0  # degenerate (constant) prediction
        else:
            out["pearson_r"] = float(np.corrcoef(t, u)[0, 1])
    return out


def _l_pre_probe(model, rho_field_torch, microbatch, n_crops, crop_size,
                 device, rng_seed: int, head: str = "softplus") -> float:
    """Compute L_pre on a fresh microbatch with no_grad — diagnostic only."""
    rng = torch.Generator(device=device).manual_seed(rng_seed)
    with torch.no_grad():
        coords, truth = _pretrain_sample_voxels(
            rho_field_torch,
            microbatch=microbatch,
            n_crops=n_crops,
            crop_size=crop_size,
            generator=rng, device=device,
        )
        theta = _pretrain_density_head(model, coords)
        return float(_head_loss(theta, truth, head).item())


def run_cell(lr_max: float, args, log_dir: Path) -> dict[str, Any]:
    """Run a single lr-sensitivity cell. Returns the cell summary dict.

    The training loop is a reduced replica of train_pretrain in pipeline.py:
    same model constructor, optimizer, schedule, sampler, and loss. We omit
    MLflow + checkpointing (we're a probe, not a production run) and add
    explicit per-step-grid var_ratio + L_pre trajectory recording.
    """
    set_global_seed(args.seed)
    device = torch.device("cpu")  # spec: CPU only
    print(f"[probe] === lr_max={lr_max:.0e} seed={args.seed} ===", flush=True)

    # Per-cell stdout log file (open before MLflow-free training loop runs).
    head_tag = "" if args.head == "softplus" else "_linlog"
    cell_log_path = (log_dir /
                     f"cell_lr{_format_lr_tag(lr_max)}_seed{args.seed}{head_tag}.log")
    log_fh = open(cell_log_path, "w", encoding="utf-8")
    t_cell_start = time.time()

    def _emit(msg: str) -> None:
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    try:
        # --- Model + optimizer + schedule (production-matched). ---
        model = IGMNeRF(hidden_dim=256, num_layers=8, L=10,
                        density_head=args.head).to(device)
        _emit(f"[probe] density_head={args.head}")
        optimizer = optim.AdamW(
            model.parameters(), lr=lr_max,
            betas=(0.9, 0.999), weight_decay=1e-6,
        )
        lr_lambda = build_lr_lambda(
            args.warmup_steps, args.max_steps, lr_max, args.lr_min,
        )
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        # --- Load rho-field via the production extract_rho_crops path. ---
        loader = SherwoodLoader(args.data_root)
        rho_field_torch = _pretrain_load_rho_field(
            loader, physics_id=args.physics, redshift=0.3,
            n_grid=args.pretrain_n_grid, device=device,
        )
        _emit(
            f"[probe] rho_field shape={tuple(rho_field_torch.shape)} "
            f"min={float(rho_field_torch.min()):.3e} "
            f"max={float(rho_field_torch.max()):.3e}"
        )

        pre_rng = torch.Generator(device=device).manual_seed(args.seed + 6900)

        # --- Fixed probe sample: one coord+truth draw, reused at every
        # trajectory step (per DEVIATION note in _var_ratio_probe docstring).
        probe_rng_fixed = torch.Generator(device=device).manual_seed(
            args.seed + 7777
        )
        with torch.no_grad():
            probe_coords_fixed, probe_truth_fixed = _pretrain_sample_voxels(
                rho_field_torch,
                microbatch=args.pretrain_microbatch,
                n_crops=args.pretrain_crops_per_step,
                crop_size=args.pretrain_crop_size,
                generator=probe_rng_fixed, device=device,
            )
            probe_var_truth_fixed = float(
                probe_truth_fixed.var(unbiased=True).item()
            )
        _emit(
            f"[probe] fixed probe sample: n_vox={probe_coords_fixed.shape[0]} "
            f"var_truth_fixed={probe_var_truth_fixed:.4e}"
        )

        # --- Step-0 baseline L_pre (matches train_pretrain). ---
        with torch.no_grad():
            coords0, rho_truth0 = _pretrain_sample_voxels(
                rho_field_torch,
                microbatch=args.pretrain_microbatch,
                n_crops=args.pretrain_crops_per_step,
                crop_size=args.pretrain_crop_size,
                generator=pre_rng, device=device,
            )
            rho_theta0 = _pretrain_density_head(model, coords0)
            L_pre_step0 = float(
                _head_loss(rho_theta0, rho_truth0, args.head).item()
            )
        _emit(f"[probe] L_pre @ step 0 = {L_pre_step0:.6e}")

        var_ratio_traj: list[dict[str, Any]] = []

        # AM-2: step-0 probe point (before any optimizer step) so a
        # decaying-init-variance trajectory is visible in the record.
        vr0 = _var_ratio_probe(
            model, rho_field_torch,
            coords_fixed=probe_coords_fixed,
            var_truth_fixed=probe_var_truth_fixed,
            device=device,
            truth_fixed=probe_truth_fixed,
            head=args.head,
        )
        var_ratio_traj.append({"step": 0, **vr0})
        _emit(
            f"[probe] step 0: ratio={vr0['ratio']:.4e} "
            f"pearson_r={vr0.get('pearson_r', float('nan')):.4f}"
        )
        l_pre_traj: list[dict[str, Any]] = []
        r_b_pre1_fire_step: int | None = None
        r_pre_at_M1: float | None = None
        L_pre_M1: float | None = None
        last_step_reached = 0

        # --- Training loop ---
        for step in range(1, args.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            coords_mb, rho_truth_mb = _pretrain_sample_voxels(
                rho_field_torch,
                microbatch=args.pretrain_microbatch,
                n_crops=args.pretrain_crops_per_step,
                crop_size=args.pretrain_crop_size,
                generator=pre_rng, device=device,
            )
            rho_theta_mb = _pretrain_density_head(model, coords_mb)

            # Skip the production R-b-pre3 nan/inf ring buffer — the probe
            # is short enough (1k steps x 3 cells) that nan/inf is observable
            # via the per-step prints. We DO want to detect R-b-pre1 inline
            # at the M1 step.
            loss = _head_loss(rho_theta_mb, rho_truth_mb, args.head)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            last_step_reached = step

            # Trajectory probe at the fixed step grid. Uses the FIXED
            # coord+truth sample drawn pre-loop (see DEVIATION note in
            # _var_ratio_probe) so per-step ratio movement is purely
            # model-side.
            if step in PROBE_STEPS:
                vr = _var_ratio_probe(
                    model, rho_field_torch,
                    coords_fixed=probe_coords_fixed,
                    var_truth_fixed=probe_var_truth_fixed,
                    device=device,
                    truth_fixed=probe_truth_fixed,
                    head=args.head,
                )
                lp = _l_pre_probe(
                    model, rho_field_torch,
                    microbatch=args.pretrain_microbatch,
                    n_crops=args.pretrain_crops_per_step,
                    crop_size=args.pretrain_crop_size,
                    device=device,
                    rng_seed=args.seed + 8000 + step,
                    head=args.head,
                )
                var_ratio_traj.append({"step": step, **vr})
                l_pre_traj.append({"step": step, "l_pre": lp})
                _emit(
                    f"[probe] step {step}: L_pre={float(loss.item()):.4e} "
                    f"var_theta={vr['var_rho_theta']:.4e} "
                    f"var_truth={vr['var_rho_truth']:.4e} "
                    f"ratio={vr['ratio']:.4e} "
                    f"pearson_r={vr.get('pearson_r', float('nan')):.4f} "
                    f"L_pre_probe={lp:.4e}"
                )

            # M1 hook — log R_pre and check R-b-pre1 (constant-density basin)
            # using the same probe machinery the production loop uses.
            if step == args.warmup_steps:
                L_pre_M1 = float(loss.item())
                r_pre_at_M1 = L_pre_M1 / max(L_pre_step0, 1e-30)
                m1v = _m1_verdict(r_pre_at_M1)
                _emit(
                    f"[probe M1] step {step}: L_pre={L_pre_M1:.4e} / "
                    f"L_pre_0={L_pre_step0:.4e} -> R_pre={r_pre_at_M1:.4f} "
                    f"[{m1v}]"
                )

                # R-b-pre1 backstop probe — fires if var_theta < 0.1 * var_truth.
                # Uses a fresh sample drawn from m3_rng-style seed to mirror
                # the production backstop exactly (NOT the fixed probe).
                m1_rng = torch.Generator(device=device).manual_seed(args.seed + 6903)
                m1_mb = min(args.pretrain_microbatch, args.pretrain_crop_size ** 3)
                m1_nc = min(args.pretrain_crops_per_step,
                            args.pretrain_m3_n_crops)
                with torch.no_grad():
                    coords_m1, truth_m1 = _pretrain_sample_voxels(
                        rho_field_torch, microbatch=m1_mb, n_crops=m1_nc,
                        crop_size=args.pretrain_crop_size,
                        generator=m1_rng, device=device,
                    )
                    theta_m1 = _theta_linear(
                        _pretrain_density_head(model, coords_m1), args.head
                    )
                    vr_m1 = {
                        "var_rho_theta": float(theta_m1.var(unbiased=True).item()),
                        "var_rho_truth": float(truth_m1.var(unbiased=True).item()),
                    }
                if vr_m1["var_rho_theta"] < 0.1 * vr_m1["var_rho_truth"]:
                    r_b_pre1_fire_step = step
                    _emit(
                        f"[probe R-b-pre1] FIRE @ step {step}: "
                        f"var_theta={vr_m1['var_rho_theta']:.4e} < "
                        f"0.1 * var_truth={vr_m1['var_rho_truth']:.4e}"
                    )
                    # NOTE: probe DOES NOT abort on R-b-pre1 — we want the
                    # full trajectory to the verdict grid so the matrix can
                    # distinguish FAIL_MOVING vs FAIL_FLAT. Production
                    # train_pretrain aborts here; this is a deviation per
                    # [D-37] rule (a), surfaced in the report-back.

        # --- Per-cell verdict per pre-committed matrix. ---
        # DEVIATION (per [D-37] rule (a)): the dispatch matrix said
        # "record up to fire step if R-b-pre1 fires earlier". With our
        # probe-grid {100, 250, 500, 750, 1000} and warmup_steps=200,
        # R-b-pre1 fires at step 200 — BEFORE any post-fire probe step.
        # Bounding the trajectory to step <= 200 leaves a 1-point series
        # and the FAIL_FLAT / FAIL_MOVING checks become undefined.
        #
        # We use the FULL trajectory (the probe deliberately does NOT abort
        # on R-b-pre1, so we have all 5 points), because the matrix's
        # underlying question is "does the model move OUT of the basin
        # given more steps?" — and the 250/500/750/1000 points answer that
        # question directly. This widens FAIL_MOVING's reach.
        # d69-matrix continuity: exclude the AM-2 step-0 point so the legacy
        # flat/monotone checks keep their original {100..1000} semantics.
        ratios = [r["ratio"] for r in var_ratio_traj if r["step"] > 0]
        if len(ratios) >= 2:
            mean_r = sum(ratios) / len(ratios)
            if mean_r > 0:
                spread = (max(ratios) - min(ratios)) / mean_r
            else:
                spread = float("nan")
            is_flat = math.isfinite(spread) and spread <= 0.10
            is_monotone_increasing = all(
                ratios[i + 1] > ratios[i] for i in range(len(ratios) - 1)
            )
            is_monotone_decreasing = all(
                ratios[i + 1] < ratios[i] for i in range(len(ratios) - 1)
            )
        else:
            spread = float("nan")
            is_flat = False
            is_monotone_increasing = False
            is_monotone_decreasing = False

        if r_pre_at_M1 is not None and r_pre_at_M1 <= 0.5:
            verdict = "PASS"
        elif (r_pre_at_M1 is not None and r_pre_at_M1 > 0.5
              and is_monotone_increasing):
            verdict = "FAIL_MOVING"
        elif is_flat:
            verdict = "FAIL_FLAT"
        elif is_monotone_decreasing:
            # Model is sinking deeper into the constant-collapse basin
            # (var_theta -> 0 monotonically). This is the OPPOSITE of
            # FAIL_MOVING (which requires monotone-increasing toward 1).
            # Per [D-37] rule (a), surface as a distinct FAIL_SINKING tag
            # rather than collapsing into FAIL_FLAT — the substance is
            # different and PI must see it. Treated as FAIL_FLAT for
            # overall-matrix routing (also a "no improvement" outcome).
            verdict = "FAIL_SINKING"
        else:
            verdict = "UNKNOWN_NON_MONOTONIC"

        elapsed = time.time() - t_cell_start
        _emit(f"[probe] cell verdict: {verdict}  (elapsed {elapsed:.1f}s)")

        cell_summary: dict[str, Any] = {
            "lr_max": lr_max,
            "seed": args.seed,
            "head": args.head,
            "n_steps_run": last_step_reached,
            "r_b_pre1_fire_step": r_b_pre1_fire_step,
            "r_pre_at_M1": r_pre_at_M1,
            "L_pre_step0": L_pre_step0,
            "L_pre_at_M1": L_pre_M1,
            "var_ratio_trajectory": var_ratio_traj,
            "var_ratio_spread_over_mean": spread,
            "var_ratio_monotone_increasing": is_monotone_increasing,
            "l_pre_trajectory": l_pre_traj,
            "verdict": verdict,
            "elapsed_sec": elapsed,
            "cell_log_path": str(cell_log_path),
        }
        return cell_summary

    except Exception as e:  # noqa: BLE001
        _emit(f"[probe] EXCEPTION: {e}\n{traceback.format_exc()}")
        return {
            "lr_max": lr_max,
            "seed": args.seed,
            "head": args.head,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "verdict": "ERROR",
            "cell_log_path": str(cell_log_path),
        }
    finally:
        log_fh.close()


def emit_summary(cell_results: list[dict[str, Any]],
                 out_path: Path) -> dict[str, Any]:
    """Compute overall verdict per the pre-committed matrix and write
    summary.json. Returns the summary dict for stdout convenience."""
    verdicts = [c.get("verdict") for c in cell_results]
    pass_cells = [c for c in cell_results if c.get("verdict") == "PASS"]
    moving_cells = [c for c in cell_results
                    if c.get("verdict") == "FAIL_MOVING"]
    # FAIL_FLAT and FAIL_SINKING both represent no-improvement outcomes
    # (matrix branch (c)). Distinguished in per-cell verdicts; collapsed
    # for the overall routing decision.
    flat_or_sinking_cells = [c for c in cell_results
                             if c.get("verdict") in ("FAIL_FLAT",
                                                      "FAIL_SINKING")]

    if pass_cells:
        # Pick the lr with the lowest r_pre_at_M1 among PASS cells.
        best = min(pass_cells,
                   key=lambda c: c.get("r_pre_at_M1", float("inf")))
        overall_verdict = f"ADVANCE_JUNO_WITH_LR={best['lr_max']:.0e}"
        route = "(a)_confirmed"
    elif moving_cells:
        # Check if any FAIL_MOVING cell has monotone-improving var-ratio
        # (var_theta moving toward var_truth, i.e. ratio approaching 1).
        improving = []
        for c in moving_cells:
            traj = c.get("var_ratio_trajectory", [])
            ratios = [r["ratio"] for r in traj]
            if len(ratios) >= 2 and ratios[-1] < 1.0 and ratios[0] < ratios[-1]:
                improving.append(c)
        if improving:
            overall_verdict = "EXTEND_CPU_PROBE_LONGER"
            route = "(a)+(b)_mixed"
        else:
            overall_verdict = "FREEZE_ESCALATE_D62_LADDER_REFRAME"
            route = "fallback_no_clear_improvement"
    elif (len(flat_or_sinking_cells) == len(cell_results)
          and cell_results):
        overall_verdict = "FREEZE_ESCALATE_D62_LADDER_REFRAME"
        route = "(c)_confirmed"
    else:
        # Mixed flat / unknown / error — emit conservatively.
        overall_verdict = "FREEZE_ESCALATE_D62_LADDER_REFRAME"
        route = "fallback_mixed_outcome"

    summary = {
        "d69_matrix": "VOID-for-D73",  # AM-1: [D-73] verdict comes solely
        # from the AM-2 rule over var_ratio_trajectory (see d73_summary.json)
        "cells": [
            {
                "lr_max": c.get("lr_max"),
                "head": c.get("head"),
                "verdict": c.get("verdict"),
                "r_pre_at_M1": c.get("r_pre_at_M1"),
                "r_b_pre1_fire_step": c.get("r_b_pre1_fire_step"),
                "var_ratio_spread_over_mean": c.get("var_ratio_spread_over_mean"),
                "var_ratio_monotone_increasing": c.get("var_ratio_monotone_increasing"),
                "n_steps_run": c.get("n_steps_run"),
                "elapsed_sec": c.get("elapsed_sec"),
            }
            for c in cell_results
        ],
        "per_cell_verdicts": verdicts,
        "overall_verdict": overall_verdict,
        "verdict_matrix_route": route,
        "pre_committed_matrix": {
            "PASS_cell_rule": "r_pre_at_M1 <= 0.5",
            "FAIL_MOVING_cell_rule": ("r_pre_at_M1 > 0.5 AND var_ratio "
                                       "strictly increasing 100 -> fire/end"),
            "FAIL_FLAT_cell_rule": "var_ratio (max-min)/mean <= 0.10",
            "overall_a_confirmed": ">=1 cell PASS",
            "overall_a_plus_b": ("0 cells PASS, >=1 FAIL_MOVING with "
                                  "var-ratio improving"),
            "overall_c_confirmed": "ALL cells FAIL_FLAT",
        },
    }
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def emit_d73_verdict(artifact_dir: Path) -> dict[str, Any]:
    """[D-73] §E + AM-2 verdict aggregator (pre-registered rule, applied
    mechanically). Reads every linear-log cell JSON in artifact_dir, groups
    by lr cell, computes per-recorded-step median-across-seeds of ratio and
    pearson_r, and applies D73_RULE_TEXT. Writes d73_summary.json.
    """
    cell_files = sorted(
        f for f in artifact_dir.glob("cell_*.json")
        if not f.name.startswith("._")  # ExFAT AppleDouble junk
    )
    cells = [json.loads(f.read_text()) for f in cell_files]
    ll_cells = [c for c in cells if c.get("head") == "linear-log"
                and "error" not in c]
    by_lr: dict[float, list[dict]] = {}
    for c in ll_cells:
        by_lr.setdefault(c["lr_max"], []).append(c)

    escape_hits = []          # steps where ratio AND pearson both pass
    unstable_hits = []        # ratio passes, pearson fails
    per_cell_medians: dict[str, list[dict[str, float]]] = {}
    for lr_max, group in sorted(by_lr.items()):
        steps = sorted({pt["step"] for c in group
                        for pt in c["var_ratio_trajectory"]})
        med_traj = []
        for s in steps:
            ratios = [pt["ratio"] for c in group
                      for pt in c["var_ratio_trajectory"] if pt["step"] == s]
            rs = [pt.get("pearson_r") for c in group
                  for pt in c["var_ratio_trajectory"]
                  if pt["step"] == s and pt.get("pearson_r") is not None]
            med_ratio = float(np.median(ratios)) if ratios else float("nan")
            med_r = float(np.median(rs)) if rs else float("nan")
            med_traj.append({"step": s, "median_ratio": med_ratio,
                             "median_pearson_r": med_r,
                             "n_seeds": len(ratios)})
            if med_ratio > D73_RATIO_THRESHOLD:
                hit = {"lr_max": lr_max, "step": s,
                       "median_ratio": med_ratio, "median_pearson_r": med_r}
                if med_r >= D73_PEARSON_THRESHOLD:
                    escape_hits.append(hit)
                else:
                    unstable_hits.append(hit)
        per_cell_medians[f"lr={lr_max:.0e}"] = med_traj

    if escape_hits:
        verdict = "ESCAPE"
        routed = "ESCAPE"
    elif unstable_hits:
        verdict = "ESCAPE-UNSTABLE"
        routed = "COLLAPSE"  # per AM-2 / [D-37] rule (a)
    else:
        verdict = "COLLAPSE"
        routed = "COLLAPSE"

    softplus_cells = [c for c in cells if c.get("head") == "softplus"
                      and "error" not in c]
    d73 = {
        "gate": "[D-73] §E A1 linear log-rho head probe",
        "rule_verbatim": D73_RULE_TEXT,
        "ratio_threshold": D73_RATIO_THRESHOLD,
        "pearson_threshold": D73_PEARSON_THRESHOLD,
        "verdict": verdict,
        "routed_as": routed,
        "escape_hits": escape_hits,
        "escape_unstable_hits": unstable_hits,
        "median_trajectories_by_lr_cell": per_cell_medians,
        "n_linear_log_cells": len(ll_cells),
        "n_softplus_control_cells_in_dir": len(softplus_cells),
        "softplus_seed0_controls": "experiments/nerf/artifacts/d69_lr_probe "
                                   "(read-only, cite-only per AM-3)",
        "scope": "(n_grid=64, (gamma) direct rho-MSE, P1 z=0.3) per R8/R9",
    }
    out_path = artifact_dir / "d73_summary.json"
    out_path.write_text(json.dumps(d73, indent=2))
    print(f"[d73] verdict={verdict} (routed_as={routed})", flush=True)
    print(json.dumps(d73, indent=2), flush=True)
    print(f"[d73] wrote {out_path}", flush=True)
    return d73


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--physics", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--data_root", type=str, default="Sherwood")
    p.add_argument("--pretrain_n_grid", type=int, default=64)
    p.add_argument("--pretrain_crop_size", type=int, default=32)
    p.add_argument("--pretrain_microbatch", type=int, default=1024)
    p.add_argument("--pretrain_crops_per_step", type=int, default=4)
    p.add_argument("--pretrain_m3_n_crops", type=int, default=16)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--lr_min", type=float, default=5e-6)
    p.add_argument("--cells", type=str, default=None,
                   help="Comma-separated lr_max values; default 1e-4,5e-4,1e-3")
    p.add_argument("--artifact_dir", type=str, default=str(ARTIFACT_DIR))
    p.add_argument("--head", type=str, default="softplus",
                   choices=["softplus", "linear-log"],
                   help="[D-73] §E density head selector")
    p.add_argument("--emit_d73_verdict", action="store_true",
                   help="Skip training; aggregate cell JSONs in artifact_dir "
                        "and emit d73_summary.json per the AM-2 rule")
    args = p.parse_args(argv)

    log_dir = Path(args.artifact_dir)

    # AM-1 hard guard: the [D-69] artifact dir is READ-ONLY.
    if log_dir.resolve() == D69_ARTIFACT_DIR.resolve():
        print("[probe] REFUSED: experiments/nerf/artifacts/d69_lr_probe is "
              "READ-ONLY per [D-73] amendment-1 AM-1. Use the default "
              "d73_a1_head_probe dir or pass another --artifact_dir.",
              flush=True)
        return 2

    if args.emit_d73_verdict:
        emit_d73_verdict(log_dir)
        return 0

    if args.cells:
        cells = [float(x.strip()) for x in args.cells.split(",") if x.strip()]
    else:
        cells = list(DEFAULT_CELLS)

    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[probe] cells={cells} artifact_dir={log_dir}", flush=True)
    t_total = time.time()

    cell_results: list[dict[str, Any]] = []
    for lr_max in cells:
        cell_summary = run_cell(lr_max, args, log_dir)
        head_tag = "" if args.head == "softplus" else "_linlog"
        cell_path = (log_dir /
                     f"cell_lr{_format_lr_tag(lr_max)}_seed{args.seed}{head_tag}.json")
        cell_path.write_text(json.dumps(cell_summary, indent=2))
        print(f"[probe] wrote {cell_path}", flush=True)
        cell_results.append(cell_summary)

    summary_path = log_dir / "summary.json"
    summary = emit_summary(cell_results, summary_path)
    elapsed = time.time() - t_total
    print(f"[probe] === SUMMARY (total elapsed {elapsed:.1f}s) ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[probe] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
