"""
D70 Wilcoxon gate harness — MLflow-direct read.

Per D70 Rev 5.1 §0.9 F2 path-β: this harness reads Stage 1a (1b) per-seed
metrics DIRECTLY from MLflow via `mlflow.search_runs` + `MlflowClient.
get_metric_history`, rather than from a placeholder JSON written by a
hand-rolled post-Juno aggregation step. This reduces hand-off surface and
defers failure modes (tag-key drift, filter syntax, store-URI handling)
to the harness layer itself.

Spec
----
Gate-1 criterion for Stage 1a (1b) skip-rich-mlp:
  - Two-stage Bonferroni adaptive design per S3:
      Stage 1 (n=10, α=0.05 one-sided IMPROVE):
        p ≤ 0.025      → PASS
        0.025 < p ≤ 0.05 → MARGINAL (re-bake to n=20, evaluate stage-2)
        p > 0.05       → FAIL
      Stage 2 (n=20, α=0.025 one-sided Bonferroni):
        p ≤ 0.025 → PASS, else FAIL

Bin-D sub-clause (ii) per §2.2 amended applies the same Wilcoxon harness
to per-bin log-MSE (`s3_log_mse_bin_D`). Sub-clause (i) (constant-floor 10×)
was RETIRED per S2; only (ii) is gating.

Per-seed metric extraction
--------------------------
Stage 1a pipeline.py only logs `m3_var_pred_log`/`m3_var_truth_log` at
step = max_steps (M3 fires ONLY at max_steps; see pipeline.py L948-980).
There is NO step-0 baseline for these metrics; the only step-0 metric
logged is `L_pre` (line 819). Same for `s3_log_mse_bin_*` (logged only
at M2 step, line 996-997).

** DESIGN-QUESTION-SURFACED **
The PI brief specifies the test as per-seed Wilcoxon on
  Var_ratio(500) − Var_ratio(0).
But Var_ratio(0) is NOT logged. This harness interprets the per-seed
delta as:
  Δ_seed = R_real(step=500) − 1.0
where R_real = m3_var_pred_log / m3_var_truth_log is the canonical M3
metric (pipeline.py L561). The null hypothesis "no improvement" maps to
R_real = 1.0 (predicted variance matches truth-side variance up to
sampling); IMPROVE direction is Δ > 0 (predicted variance recovers
truth-side variance from below — the canonical M3 K3 calibration band
target). This is honest framing: the gate tests whether the n=10 sweep
recovers truth-side variance, not whether step-500 beats a missing
step-0 baseline.

Tag-emission contract (verified against pipeline.py + submit_juno_stage1a_1b.sh)
------------------------------------------------------------------------------
Tags emitted by pipeline.py train_pretrain (L777-785):
  model_type=nerf, stage="1a-density-pretrain", physics_id, redshift,
  pretrain_target, design_doc, loss_variant
Tags emitted by post-run tagger (sbatch L139-145):
  body_arch="skip-rich-mlp", compute="juno",
  juno_batch="stage1a-1b-skiprich", seed="<int>",
  stage_substep="1a-(1b)", design_doc_ref, framing_amendment_A

Canonical filter:
  tags.stage = '1a-density-pretrain' AND
  tags.body_arch = 'skip-rich-mlp' AND
  tags.juno_batch = 'stage1a-1b-skiprich'

MLflow store handling (file:// vs http://)
------------------------------------------
Stage 1a Juno sbatch sets `MLFLOW_TRACKING_URI=file://${RUN_DIR}/mlflow`
PER SEED (one file-store per run; n=10 separate stores). Host-side
aggregation requires either (a) replaying each store into the local
http:// tracker, or (b) iterating multiple --tracking-uri values.

This harness supports both:
  --tracking-uri file:///path/to/store  (single store, multiple runs)
  --tracking-uri http://127.0.0.1:5000  (default, post-replay)
For the n=10 file://-per-run layout, the post-Juno step is expected to
either run host-side `sagemaker_stage2b_import_mlflow.py`-style replay
or call this harness once per store and aggregate externally. The
--multi-store flag enables the latter path.

R20 fail-loud-not-skip-log
--------------------------
--dry-run runs the search against the configured tracking URI and asserts
that ≥ N_STAGE1 matching runs are present. Empty result set → loud
AssertionError. NEVER returns PASS-by-default when no runs match.

Out-of-scope (DO NOT execute under this dispatch)
-------------------------------------------------
Authoring only. The harness will be executed AFTER the Juno sweep returns
and (optionally) after a file://-store import step.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import statistics
from scipy.stats import wilcoxon

# Late-imported so unit tests can mock mlflow without requiring the real lib
# at import time. The CLI path imports unconditionally; tests inject mocks.
try:  # pragma: no cover - thin import shim
    import mlflow
    from mlflow.tracking import MlflowClient
    _MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore
    _MLFLOW_AVAILABLE = False


ALPHA_STAGE1 = 0.05
ALPHA_PASS_STRICT = 0.025
ALPHA_STAGE2 = 0.025

# PI B2 binding (LEDGER §3 [D-70] 2026-05-26 absorption block, ε pre-registration
# block): physical-escape threshold on |Δ_seed|, anchored to Boera+2019 Fig. 4
# ~5% systematic floor on observational P_F at the k-bands of interest.
# DISTINCT from ALPHA_STAGE1 (statistical significance) — this is a domain-
# physics floor, not an α-level.
# R26 banking provenance: F1-β P2 smoke surfaced R_real semantic mismatch;
# F4 redefined Δ_seed = R_real_linear(500) − R_real_linear(0); this constant
# binds the MDE-block guard below. Wilcoxon signed-rank n=10 α=0.05 one-sided
# ARE 0.955 ⇒ MDE ≈ 0.9·σ_seed; BLOCK condition: 0.9·σ_seed > ε.
EPSILON_PHYSICAL_ESCAPE = 0.05
MDE_ARE_COEFF = 0.9  # Wilcoxon signed-rank n=10 ARE for normal-shift alternative

N_STAGE1 = 10
N_STAGE2 = 20

# Canonical metric + tag keys (verified in-session against pipeline.py /
# submit_juno_stage1a_1b.sh — see module docstring).
#
# PI F1-β + F4 (2026-05-25): the gate observable is `m3_r_real_linear`
# (linear-space Var(ρ_θ)/Var(ρ_truth)), NOT the log-space R_real used in
# pre-F1-β builds. Boera+2019 5% observational floor lives in linear-
# space P_F units; gating on log-space R_real conflated frames.
# `m3_r_real_log` (== legacy m3_r_real) retained as diagnostic for
# pre-flight C drift continuity.
METRIC_R_REAL_LINEAR = "m3_r_real_linear"
METRIC_R_REAL_LOG = "m3_r_real_log"
METRIC_VAR_PRED = "m3_var_pred_log"
METRIC_VAR_TRUTH = "m3_var_truth_log"
METRIC_BIN_D = "s3_log_mse_bin_D"
METRIC_BIN_B = "s3_log_mse_bin_B"

TAG_STAGE = "stage"
TAG_STAGE_VALUE = "1a-density-pretrain"
TAG_ARCH = "body_arch"
TAG_ARCH_VALUE = "skip-rich-mlp"
TAG_JUNO_BATCH = "juno_batch"
TAG_JUNO_BATCH_VALUE = "stage1a-1b-skiprich"
TAG_SEED = "seed"
TAG_ABORT_REASON = "abort_reason"
TAG_ABORT_REASON_NONE = "none"

# PI F4 (2026-05-25) Δ_seed baseline source: linear-space variance ratio
# expected from a frozen-init MLP (no training). Per pre-flight B
# empirical observation, the linear-space ratio sits near 2.4e-6 for the
# canonical n_grid=64 / crop_size=8 / m3_n_crops=100 grid. Used as a
# fallback baseline ONLY when the per-run step-0 metric is missing
# (legacy runs predating the F1-β step-0 M3 emission). Live runs use
# the per-run step-0 metric, not this constant.
FROZEN_INIT_R_REAL_LINEAR_PRIOR = 2.4e-6

DEFAULT_EXPERIMENT_NAME = "CosmoGasVision/NeRF"
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Bin-D coverage gate per S5 / Rev 5.1 §2.2 amended.
BIN_D_MIN_SAMPLES_PER_CROP = 5
BIN_D_MAX_EXCLUDED_FRACTION = 0.20


# ---------------------------------------------------------------------------
# MLflow read layer
# ---------------------------------------------------------------------------

def _build_client(tracking_uri: str):
    """Construct an MlflowClient bound to ``tracking_uri``.

    Kept thin + monkeypatchable for unit tests.
    """
    if not _MLFLOW_AVAILABLE:  # pragma: no cover
        raise RuntimeError(
            "mlflow not importable; install with `uv add mlflow` or run on a "
            "machine with the project venv activated."
        )
    return MlflowClient(tracking_uri=tracking_uri)


def _stage1a_filter_string(arch_value: str = TAG_ARCH_VALUE,
                           juno_batch_value: str = TAG_JUNO_BATCH_VALUE) -> str:
    """Canonical Stage-1a filter string.

    NB: MLflow's `search_runs` filter language requires backticks around
    tag keys containing periods; plain `tags.X = 'Y'` is fine for our keys.
    """
    return (
        f"tags.{TAG_STAGE} = '{TAG_STAGE_VALUE}' AND "
        f"tags.{TAG_ARCH} = '{arch_value}' AND "
        f"tags.{TAG_JUNO_BATCH} = '{juno_batch_value}'"
    )


def _search_stage1a_runs(client, experiment_name: str,
                         filter_string: str) -> List:
    """Return the list of stage-1a runs matching the canonical filter,
    further filtered to abort_reason ∈ {"none", MISSING}.

    PI F2-α (2026-05-25): the `abort_reason` tag is post-N3 (2026-05-26)
    machinery; historical runs from before the N3 patch will not carry
    the tag at all. The harness MUST treat a MISSING `abort_reason` tag
    as semantically equivalent to ``abort_reason = "none"`` (no abort
    occurred — back-compat for legacy runs), while explicit
    `abort_reason = "<flag_name>"` values still gate the run out.

    This filter is applied post-search because MLflow's filter language
    does not natively express "tag absent OR equal to X".
    """
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(
            f"MLflow experiment '{experiment_name}' not found at the "
            f"configured tracking URI; refusing to compute Wilcoxon verdict."
        )
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_string,
        max_results=2 * N_STAGE2,  # ample headroom; gate later
    )
    # F2-α: keep abort_reason ∈ {"none", MISSING, "", None}
    kept = []
    for r in runs:
        ar = r.data.tags.get(TAG_ABORT_REASON)
        if ar is None or ar == "" or ar == TAG_ABORT_REASON_NONE:
            kept.append(r)
        else:
            print(
                f"[d70_wilcoxon] INFO: run {r.info.run_id} excluded "
                f"(abort_reason='{ar}').",
                file=sys.stderr,
            )
    return kept


def _last_metric_at(client, run_id: str, key: str,
                    step: Optional[int] = None) -> Optional[float]:
    """Return the metric value at ``step`` (or the last logged step if
    ``step`` is None). Uses get_metric_history for time-series access.
    Returns None when the key is absent from the run."""
    hist = client.get_metric_history(run_id, key)
    if not hist:
        return None
    if step is None:
        # Pick the highest step recorded for this key.
        last = max(hist, key=lambda m: m.step)
        return float(last.value)
    matches = [m for m in hist if m.step == step]
    if not matches:
        return None
    return float(matches[-1].value)


def _extract_per_seed_metrics(client, runs,
                              max_steps: int) -> Dict[int, Dict[str, float]]:
    """Build per-seed payload from the canonical metrics.

    PI F1-β + F4 (2026-05-25): the gate observable is the linear-space
    R_real, and Δ_seed = R_real_linear(max_steps) − R_real_linear(0).
    The step-0 value is read from the per-run MLflow log (pipeline.py
    emits m3_r_real_linear at step 0 post-F1-β). For legacy runs missing
    the step-0 metric, fall back to FROZEN_INIT_R_REAL_LINEAR_PRIOR
    (Δ_seed_baseline_source = "frozen_init_prior") and surface the
    substitution in the per-seed payload for honest-reporting audit.

    Returns a dict keyed by integer seed:
      {seed: {
        "run_id": str,
        "var_pred": float | None,        # m3_var_pred_log @ max_steps
        "var_truth": float | None,
        "r_real": float | None,          # legacy log-space ratio (back-compat)
        "r_real_linear": float | None,   # m3_r_real_linear @ max_steps
        "r_real_linear_step0": float | None,
        "delta_baseline_source": "per_run_step0" | "frozen_init_prior" | None,
        "delta": float | None,           # F4: r_real_linear − baseline
        "bin_d": float | None,
        "bin_b": float | None,
      }, ...}
    Runs missing the `seed` tag are skipped with a stderr warning.
    """
    out: Dict[int, Dict[str, float]] = {}
    for r in runs:
        tags = r.data.tags
        seed_raw = tags.get(TAG_SEED)
        if seed_raw is None:
            print(
                f"[d70_wilcoxon] WARN: run {r.info.run_id} missing 'seed' "
                f"tag; skipping.",
                file=sys.stderr,
            )
            continue
        try:
            seed = int(seed_raw)
        except ValueError:
            print(
                f"[d70_wilcoxon] WARN: run {r.info.run_id} has non-int seed "
                f"tag '{seed_raw}'; skipping.",
                file=sys.stderr,
            )
            continue

        vp = _last_metric_at(client, r.info.run_id, METRIC_VAR_PRED,
                             step=max_steps)
        vt = _last_metric_at(client, r.info.run_id, METRIC_VAR_TRUTH,
                             step=max_steps)
        bd = _last_metric_at(client, r.info.run_id, METRIC_BIN_D,
                             step=max_steps)
        bb = _last_metric_at(client, r.info.run_id, METRIC_BIN_B,
                             step=max_steps)
        r_real_lin_end = _last_metric_at(
            client, r.info.run_id, METRIC_R_REAL_LINEAR, step=max_steps,
        )
        r_real_lin_0 = _last_metric_at(
            client, r.info.run_id, METRIC_R_REAL_LINEAR, step=0,
        )

        # Back-compat: legacy log-space r_real for diagnostic only.
        r_real_log = None
        if vp is not None and vt is not None and vt > 0:
            r_real_log = vp / vt

        # F4 Δ_seed: linear-space, improvement-over-frozen-init baseline.
        delta = None
        baseline_source: Optional[str] = None
        if r_real_lin_end is not None and math.isfinite(r_real_lin_end):
            if (r_real_lin_0 is not None
                    and math.isfinite(r_real_lin_0)):
                delta = r_real_lin_end - r_real_lin_0
                baseline_source = "per_run_step0"
            else:
                delta = (
                    r_real_lin_end - FROZEN_INIT_R_REAL_LINEAR_PRIOR
                )
                baseline_source = "frozen_init_prior"
                print(
                    f"[d70_wilcoxon] WARN: run {r.info.run_id} missing "
                    f"step-0 m3_r_real_linear; fell back to "
                    f"FROZEN_INIT_R_REAL_LINEAR_PRIOR="
                    f"{FROZEN_INIT_R_REAL_LINEAR_PRIOR:.3e}.",
                    file=sys.stderr,
                )

        if seed in out:
            print(
                f"[d70_wilcoxon] WARN: duplicate seed {seed} (runs "
                f"{out[seed]['run_id']} + {r.info.run_id}); keeping latest.",
                file=sys.stderr,
            )
        out[seed] = {
            "run_id": r.info.run_id,
            "var_pred": vp,
            "var_truth": vt,
            "r_real": r_real_log,
            "r_real_linear": r_real_lin_end,
            "r_real_linear_step0": r_real_lin_0,
            "delta_baseline_source": baseline_source,
            "delta": delta,
            "bin_d": bd,
            "bin_b": bb,
        }
    return out


# ---------------------------------------------------------------------------
# Wilcoxon engine
# ---------------------------------------------------------------------------

def _wilcoxon_one_sided(deltas: List[float]) -> float:
    """One-sided Wilcoxon signed-rank against zero, IMPROVE direction.

    Returns the one-sided p-value (alternative='greater').
    """
    arr = np.array([d for d in deltas if d is not None and math.isfinite(d)])
    if len(arr) < 2:
        raise ValueError(
            f"Need ≥ 2 valid samples for Wilcoxon; got {len(arr)}."
        )
    res = wilcoxon(arr, alternative="greater", zero_method="wilcox")
    return float(res.pvalue)


def _compute_sigma_seed(delta_seed_values: List[Optional[float]]) -> Tuple[float, str, int]:
    """Compute empirical sigma_seed across per-seed Δ_seed values for MDE-block guard.

    Per PI B2 binding (LEDGER §3 [D-70] 2026-05-26):
      - Use full n=10 sample when available.
      - Fall back to first-3-seeds only if seeds 4-10 are missing.
      - Return (sigma_seed, source_label, n_used) where source_label ∈
        {"full_n10", "first_3", "insufficient"}.

    Uses sample standard deviation (ddof=1), matching scipy.stats and
    statistics.stdev. statistics is preferred over numpy here so the
    estimator is self-contained for unit-test mocking.
    """
    valid_values = [
        v for v in delta_seed_values
        if v is not None and not (math.isnan(v) or math.isinf(v))
    ]
    if len(valid_values) >= 10:
        return (float(statistics.stdev(valid_values[:10])), "full_n10", 10)
    if len(valid_values) >= 3:
        return (float(statistics.stdev(valid_values[:3])), "first_3", 3)
    return (float("nan"), "insufficient", len(valid_values))


def _gate(p_stage1: float, p_stage2: Optional[float],
          n1: int, n2: Optional[int]) -> str:
    """Two-stage Bonferroni adaptive gate."""
    if p_stage2 is None:
        if n1 < N_STAGE1:
            return "INSUFFICIENT-N"
        if p_stage1 <= ALPHA_PASS_STRICT:
            return "PASS"
        if p_stage1 <= ALPHA_STAGE1:
            return "MARGINAL"
        return "FAIL"
    # Stage 2 evaluated.
    if n2 is None or n2 < N_STAGE2:
        return "INSUFFICIENT-N-STAGE2"
    return "PASS" if p_stage2 <= ALPHA_STAGE2 else "FAIL"


# ---------------------------------------------------------------------------
# Bin-D sub-clause (ii)
# ---------------------------------------------------------------------------

def _bin_d_wilcoxon(per_seed: Dict[int, Dict[str, float]]) -> Dict[str, object]:
    """Run Wilcoxon on per-seed Bin-D log-MSE.

    Per Rev 5.1 §2.2 amended sub-clause (ii): IMPROVE direction means Bin-D
    log-MSE DECREASES (negative delta). We test `−bin_d` increasing.
    The PI memo describes "Wilcoxon-decreasing"; we encode this as the
    one-sided ``alternative='less'``-equivalent via sign flip.

    Coverage gate per S5: count seeds with non-None Bin-D values. If
    fraction of seeds excluded > BIN_D_MAX_EXCLUDED_FRACTION ⇒ flag
    INSUFFICIENT-COVERAGE.
    """
    total = len(per_seed)
    valid = [(s, v["bin_d"]) for s, v in per_seed.items()
             if v.get("bin_d") is not None and math.isfinite(v["bin_d"])]
    excluded_frac = (total - len(valid)) / max(total, 1)
    if excluded_frac > BIN_D_MAX_EXCLUDED_FRACTION:
        return {
            "verdict": "INSUFFICIENT-COVERAGE",
            "excluded_fraction": excluded_frac,
            "n_valid": len(valid),
            "p_value": None,
        }
    if len(valid) < 2:
        return {
            "verdict": "INSUFFICIENT-N",
            "excluded_fraction": excluded_frac,
            "n_valid": len(valid),
            "p_value": None,
        }
    # IMPROVE = decrease in log-MSE relative to Bin-B reference.
    # Per S2 retirement of the constant-floor sub-clause (i), the test is
    # framed as per-seed Bin-D log-MSE decreasing toward Bin-B-comparable
    # error levels — we test the delta (Bin-D − Bin-B) against zero with
    # alternative='less'.
    deltas = []
    for _s, _ in valid:
        v = per_seed[_s]
        if v.get("bin_b") is None or not math.isfinite(v["bin_b"]):
            continue
        deltas.append(v["bin_d"] - v["bin_b"])
    if len(deltas) < 2:
        return {
            "verdict": "INSUFFICIENT-N-BIN-B",
            "excluded_fraction": excluded_frac,
            "n_valid": len(deltas),
            "p_value": None,
        }
    arr = np.array(deltas)
    res = wilcoxon(arr, alternative="less", zero_method="wilcox")
    p = float(res.pvalue)
    verdict = "PASS" if p <= ALPHA_STAGE1 else "FAIL"
    return {
        "verdict": verdict,
        "excluded_fraction": excluded_frac,
        "n_valid": len(deltas),
        "p_value": p,
    }


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------

def run_gate(tracking_uri: str,
             experiment_name: str,
             max_steps: int,
             arch_value: str = TAG_ARCH_VALUE,
             juno_batch_value: str = TAG_JUNO_BATCH_VALUE,
             stage2: bool = False,
             dry_run: bool = False,
             client=None,
             enforce_mde_block: bool = False,
             mde_block_exit: bool = False) -> Dict[str, object]:
    """End-to-end harness. ``client`` is injectable for unit tests."""
    if client is None:
        if _MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(tracking_uri)
        client = _build_client(tracking_uri)

    filter_string = _stage1a_filter_string(arch_value, juno_batch_value)
    runs = _search_stage1a_runs(client, experiment_name, filter_string)

    # R20 twin-gate: empty result set ⇒ loud AssertionError, NEVER
    # silently return PASS-by-default.
    if dry_run:
        assert len(runs) >= N_STAGE1, (
            f"Stage 1a sweep returned {len(runs)} matching runs; refusing to "
            f"compute Wilcoxon verdict (expected ≥ {N_STAGE1})."
        )

    if len(runs) == 0:
        raise AssertionError(
            "Stage 1a sweep returned 0 matching runs; refusing to compute "
            "Wilcoxon verdict."
        )

    per_seed = _extract_per_seed_metrics(client, runs, max_steps)
    n_obs = len(per_seed)

    valid_deltas = [v["delta"] for v in per_seed.values()
                    if v["delta"] is not None and math.isfinite(v["delta"])]

    # PI B2 binding ruling (LEDGER §3 [D-70] 2026-05-26 absorption block):
    # MDE-block guard — refuse to compute Wilcoxon when the gate is dead-on-
    # arrival underpowered (MDE > ε_physical_escape). Wilcoxon signed-rank
    # n=10 α=0.05 one-sided ARE 0.955 ⇒ MDE ≈ 0.9·σ_seed; if that exceeds
    # ε=0.05, the gate cannot detect a physically-meaningful improvement
    # even when one exists, so the verdict-readout itself is suppressed.
    #
    # `enforce_mde_block=False` is the test-only opt-out used by regression
    # tests that pre-date this guard (σ=0.1 synthetic deltas would trip the
    # block). Production CLI keeps the default True.
    sigma_seed, sigma_source, n_sigma = _compute_sigma_seed(valid_deltas)
    mde_estimate = MDE_ARE_COEFF * sigma_seed
    mde_block_payload = {
        "enforced": bool(enforce_mde_block),
        "epsilon_physical_escape": EPSILON_PHYSICAL_ESCAPE,
        "mde_are_coeff": MDE_ARE_COEFF,
        "sigma_seed": (None if math.isnan(sigma_seed) else float(sigma_seed)),
        "sigma_source": sigma_source,
        "sigma_n_used": n_sigma,
        "mde_estimate": (None if math.isnan(mde_estimate) else float(mde_estimate)),
        "verdict_blocked": False,
        "block_reason": None,
    }

    if enforce_mde_block:
        if sigma_source == "insufficient":
            msg = (
                f"BLOCK: gate underpowered — insufficient valid seeds "
                f"(n_valid={n_sigma} < 3); halt + re-spec required"
            )
            print(msg, flush=True)
            mde_block_payload["verdict_blocked"] = True
            mde_block_payload["block_reason"] = "insufficient_seeds"
            blocked_out = {
                "tracking_uri": tracking_uri,
                "experiment_name": experiment_name,
                "filter_string": filter_string,
                "n_runs_matched": len(runs),
                "n_seeds_extracted": n_obs,
                "n_valid_deltas": len(valid_deltas),
                "verdict": "BLOCKED-MDE-UNDERPOWERED",
                "mde_block": mde_block_payload,
                "p_stage1": None,
                "p_stage2": None,
                "bin_d": None,
                "design_doc": (
                    "D70 Rev 5.1 §2.2 amended + PI B2 [D-70] 2026-05-26 "
                    "absorption block (MDE-block guard)"
                ),
            }
            # Always return the structured payload; CLI main() handles the
            # non-zero exit + JSON emission. mde_block_exit is retained as a
            # parameter for API compatibility but no longer triggers sys.exit
            # here — keeping exits centralised in main() avoids two stdout
            # writers stomping each other.
            return blocked_out

        if mde_estimate > EPSILON_PHYSICAL_ESCAPE:
            msg = (
                f"BLOCK: gate underpowered — MDE={mde_estimate:.4f} > "
                f"ε={EPSILON_PHYSICAL_ESCAPE:.4f} (σ_seed={sigma_seed:.4f} "
                f"from {sigma_source}); halt + re-spec required"
            )
            print(msg, flush=True)
            mde_block_payload["verdict_blocked"] = True
            mde_block_payload["block_reason"] = "mde_underpowered"
            blocked_out = {
                "tracking_uri": tracking_uri,
                "experiment_name": experiment_name,
                "filter_string": filter_string,
                "n_runs_matched": len(runs),
                "n_seeds_extracted": n_obs,
                "n_valid_deltas": len(valid_deltas),
                "verdict": "BLOCKED-MDE-UNDERPOWERED",
                "mde_block": mde_block_payload,
                "p_stage1": None,
                "p_stage2": None,
                "bin_d": None,
                "design_doc": (
                    "D70 Rev 5.1 §2.2 amended + PI B2 [D-70] 2026-05-26 "
                    "absorption block (MDE-block guard)"
                ),
            }
            if mde_block_exit:
                print(json.dumps(blocked_out, indent=2, default=str), flush=True)
                sys.exit(1)
            return blocked_out

    # Choose the n for stage classification: if stage2 flag set, expect n_20.
    n_target = N_STAGE2 if stage2 else N_STAGE1
    n_used = min(n_obs, n_target)

    p_stage1: Optional[float] = None
    p_stage2: Optional[float] = None
    try:
        p_stage1 = _wilcoxon_one_sided(valid_deltas[:N_STAGE1])
    except ValueError as e:
        print(f"[d70_wilcoxon] Stage 1 Wilcoxon failed: {e}", file=sys.stderr)

    if stage2 and len(valid_deltas) >= N_STAGE2:
        try:
            p_stage2 = _wilcoxon_one_sided(valid_deltas[:N_STAGE2])
        except ValueError as e:
            print(
                f"[d70_wilcoxon] Stage 2 Wilcoxon failed: {e}",
                file=sys.stderr,
            )

    n1 = min(len(valid_deltas), N_STAGE1)
    n2 = len(valid_deltas) if stage2 else None
    if p_stage1 is None:
        verdict = "INSUFFICIENT-N"
    else:
        verdict = _gate(p_stage1, p_stage2, n1, n2)

    bin_d_result = _bin_d_wilcoxon(per_seed)

    out = {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "filter_string": filter_string,
        "n_runs_matched": len(runs),
        "n_seeds_extracted": n_obs,
        "n_valid_deltas": len(valid_deltas),
        "per_seed": {
            str(s): {k: v for k, v in payload.items() if k != "run_id"}
                    | {"run_id": payload["run_id"]}
            for s, payload in per_seed.items()
        },
        "p_stage1": p_stage1,
        "p_stage2": p_stage2,
        "verdict": verdict,
        "bin_d": bin_d_result,
        "mde_block": mde_block_payload,
        "design_doc": "D70 Rev 5.1 §2.2 amended + PI F1-β/F4 (2026-05-25)",
        "framing_note": (
            "Per-seed delta = R_real_linear(max_steps) − R_real_linear(0) "
            "per PI F4. Step-0 R_real_linear is per-run when the pipeline "
            "emits it (post-F1-β builds); legacy runs fall back to "
            f"FROZEN_INIT_R_REAL_LINEAR_PRIOR="
            f"{FROZEN_INIT_R_REAL_LINEAR_PRIOR:.3e} with "
            "delta_baseline_source='frozen_init_prior' annotated per-seed. "
            "Wilcoxon H1: median(Δ_seed) > 0 (improvement; "
            "alternative='greater'). Linear-space ratio chosen per Boera+"
            "2019 5% observational floor framing."
        ),
    }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI,
                   help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI "
                        "or http://127.0.0.1:5000).")
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT_NAME,
                   help="MLflow experiment name "
                        f"(default: {DEFAULT_EXPERIMENT_NAME}).")
    p.add_argument("--max-steps", type=int, default=500,
                   help="Step at which M3 metrics were logged (default 500).")
    p.add_argument("--arch", default=TAG_ARCH_VALUE,
                   help=f"body_arch tag value (default '{TAG_ARCH_VALUE}').")
    p.add_argument("--juno-batch", default=TAG_JUNO_BATCH_VALUE,
                   help=f"juno_batch tag value "
                        f"(default '{TAG_JUNO_BATCH_VALUE}').")
    p.add_argument("--stage2", action="store_true",
                   help="Evaluate Stage-2 Bonferroni n=20 path.")
    p.add_argument("--dry-run", action="store_true",
                   help="R20 twin-gate: assert ≥ N_STAGE1 matching runs "
                        "exist at the configured URI; do not write output.")
    p.add_argument("--output-json", default=None,
                   help="Path to write the full verdict payload.")
    p.add_argument("--no-mde-block", action="store_true",
                   help="Disable the PI B2 MDE-block guard (LEDGER §3 [D-70] "
                        "2026-05-26). Default: guard ENFORCED. Use only with "
                        "explicit PI authorization — disabling allows the gate "
                        "to rule even when MDE > ε_physical_escape, which "
                        "violates the absorption-block binding.")
    args = p.parse_args(argv)

    result = run_gate(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
        max_steps=args.max_steps,
        arch_value=args.arch,
        juno_batch_value=args.juno_batch,
        stage2=args.stage2,
        dry_run=args.dry_run,
        enforce_mde_block=(not args.no_mde_block),
        mde_block_exit=(not args.no_mde_block),
    )

    print(json.dumps(result, indent=2, default=str))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2,
                                                     default=str))

    verdict = result["verdict"]
    if verdict == "BLOCKED-MDE-UNDERPOWERED":
        return 1
    return 0 if verdict == "PASS" else (1 if verdict == "MARGINAL" else 2)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
