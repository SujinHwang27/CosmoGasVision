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
ARTIFACT_DIR = Path("experiments/nerf/artifacts/d69_lr_probe")


def _format_lr_tag(lr: float) -> str:
    """File-system-safe lr tag (e.g. 1e-4 -> '1e-04')."""
    return f"{lr:.0e}".replace("+", "")


def _var_ratio_probe(model, rho_field_torch, coords_fixed, var_truth_fixed,
                     device) -> dict[str, float]:
    """Forward the model on a FIXED probe coordinate set and return
    (var_rho_theta, var_rho_truth_fixed, ratio).

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
    lines ~893-919) on the rho_theta side; the var_truth denominator is
    drawn from a SEPARATE freshly-sampled M1-style probe call.
    """
    with torch.no_grad():
        theta_eval = _pretrain_density_head(model, coords_fixed)
        var_theta = float(theta_eval.var(unbiased=True).item())
    return {
        "var_rho_theta": var_theta,
        "var_rho_truth": var_truth_fixed,
        "ratio": var_theta / max(var_truth_fixed, 1e-30),
    }


def _l_pre_probe(model, rho_field_torch, microbatch, n_crops, crop_size,
                 device, rng_seed: int) -> float:
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
        return float(_pretrain_loss(theta, truth).item())


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
    cell_log_path = log_dir / f"cell_lr{_format_lr_tag(lr_max)}_seed{args.seed}.log"
    log_fh = open(cell_log_path, "w", encoding="utf-8")
    t_cell_start = time.time()

    def _emit(msg: str) -> None:
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    try:
        # --- Model + optimizer + schedule (production-matched). ---
        model = IGMNeRF(hidden_dim=256, num_layers=8, L=10).to(device)
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
            L_pre_step0 = float(_pretrain_loss(rho_theta0, rho_truth0).item())
        _emit(f"[probe] L_pre @ step 0 = {L_pre_step0:.6e}")

        var_ratio_traj: list[dict[str, Any]] = []
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
            loss = _pretrain_loss(rho_theta_mb, rho_truth_mb)
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
                )
                lp = _l_pre_probe(
                    model, rho_field_torch,
                    microbatch=args.pretrain_microbatch,
                    n_crops=args.pretrain_crops_per_step,
                    crop_size=args.pretrain_crop_size,
                    device=device,
                    rng_seed=args.seed + 8000 + step,
                )
                var_ratio_traj.append({"step": step, **vr})
                l_pre_traj.append({"step": step, "l_pre": lp})
                _emit(
                    f"[probe] step {step}: L_pre={float(loss.item()):.4e} "
                    f"var_theta={vr['var_rho_theta']:.4e} "
                    f"var_truth={vr['var_rho_truth']:.4e} "
                    f"ratio={vr['ratio']:.4e} L_pre_probe={lp:.4e}"
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
                    theta_m1 = _pretrain_density_head(model, coords_m1)
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
        ratios = [r["ratio"] for r in var_ratio_traj]
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
        "cells": [
            {
                "lr_max": c.get("lr_max"),
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
    args = p.parse_args(argv)

    if args.cells:
        cells = [float(x.strip()) for x in args.cells.split(",") if x.strip()]
    else:
        cells = list(DEFAULT_CELLS)

    log_dir = Path(args.artifact_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[probe] cells={cells} artifact_dir={log_dir}", flush=True)
    t_total = time.time()

    cell_results: list[dict[str, Any]] = []
    for lr_max in cells:
        cell_summary = run_cell(lr_max, args, log_dir)
        cell_path = log_dir / f"cell_lr{_format_lr_tag(lr_max)}_seed{args.seed}.json"
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
