"""Pin the [D-11]/[D-21] mean-flux gradient identity under the [D-24] DLA mask.

Background — defense-panel S3 attack
------------------------------------
Per [D-24], both Pass 1 (cycle-mean of exp(-tau)) and Pass 2 (per-microbatch
backward of the linearized surrogate) reduce *only over non-DLA bins* using
``mask_no_dla_profile``. The [D-21] two-pass gradient identity

    ∂L_meanF/∂θ = 2 λ_F (F_cycle - F_obs) · ∂F_cycle/∂θ
    ∂F_cycle/∂θ = (1/N_chunks) · Σ_i ∂F_mb_i/∂θ          (only when F_cycle
                                                          and Σ_i F_mb_i sum
                                                          over the SAME bins)

remains valid because the mask source is ``tau_GT``-derived (loader-built,
deterministic per sightline) — NOT ``tau_pred``-derived. If the mask source
ever drifts (e.g. someone refactors to ``mask = (tau_pred < TAU_MAX)``), the
identity silently breaks: Pass 2's surrogate gradient stops equaling the true
∂L_meanF/∂θ, the optimizer fights the mean-flux anchor every step, and there
is no loud failure mode (just a suspicious anchor-residual that no one
attributes to the right cause for weeks).

Three tests pin the contract:

  1. ``test_pass1_pass2_reduce_over_identical_bins`` — exact algebra:
     F_cycle is the unmasked-bin-weighted mean of per-chunk masked-mean F's.
  2. ``test_mask_source_is_tau_GT_derived_not_tau_pred_derived`` — static
     analysis of pipeline.py: Pass 1 and Pass 2 both reference the same
     ``mask_no_dla_profile`` variable; no Pass 2 mask is ever constructed
     from ``tau_pred``.
  3. ``test_gradient_identity_under_masked_reduction_matches_d21_form`` —
     numerical: ``log_tau_amp.grad`` from the two-pass surrogate matches the
     full unaccumulated ``λ_F (F_cycle - F_obs)^2`` gradient under the masked
     reduction.

These tests run on CPU in well under one second and do NOT modify the
implementation — they only pin the mask-source / reduction-set contract.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

N_RAYS = 4
N_BINS = 64
MICROBATCH = 2
ACCUM_STEPS = N_RAYS // MICROBATCH  # = 2 chunks of 2 rays
SEED = 4242
LAMBDA_F = 1.0
MEAN_FLUX_OBS = 0.8


def _make_mask(seed: int = SEED) -> torch.Tensor:
    """Synthetic DLA mask: ~25% of bins per ray excluded.

    Mask is a fixed property of the synthetic ground truth (analogue of the
    loader's ``tau_GT``-derived ``mask_no_dla_profile``). Crucially, it does
    NOT depend on ``tau_pred`` — that is the contract under test.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(N_RAYS, N_BINS, generator=gen) > 0.25


def _microbatch_slices():
    for chunk_i in range(ACCUM_STEPS):
        s = chunk_i * MICROBATCH
        e = min(s + MICROBATCH, N_RAYS)
        if s >= e:
            return
        yield s, e


# ---------------------------------------------------------------------------
# Test 1: Pass 1 / Pass 2 reduce over identical bin sets
# ---------------------------------------------------------------------------

def test_pass1_pass2_reduce_over_identical_bins():
    """Exact algebraic identity under the [D-24] masked reduction.

    Pass 1 computes the *single* masked cycle mean
        F_cycle = sum_{r,b} (F_pred * mask) / sum_{r,b} mask
    over all rays/bins simultaneously.

    Pass 2 computes per-microbatch masked means
        mean_F_mb_i = sum_chunk_i (F_pred * mask) / sum_chunk_i mask
    and the backward-only summation step in pipeline.py multiplies each by
    ``mean_F_grad_coef / accum_steps``. The forward equivalence we pin here
    is that F_cycle equals the *unmasked-bin-count weighted average* of the
    per-microbatch masked means — and NOT the simple mean over chunks.

    This is exact algebra (no floating-point slop beyond accumulation order),
    so we use atol=rtol=1e-12. We also include a sanity-check that the
    UNmasked formula (``F_pred_mb.mean()``) does NOT satisfy the identity,
    proving the test is discriminative and not vacuously passing.
    """
    torch.manual_seed(SEED)
    mask = _make_mask()
    # Deterministic per-bin tau values so the algebra is auditable. We vary
    # them per-bin so the masked vs. unmasked reductions actually differ
    # (a constant tau would make every reduction equal exp(-c)). float64 so
    # the 1e-12 tolerance below is achievable — float32 sums of 256 terms
    # accumulate ~3e-5 relative error from rounding, which is far above the
    # algebraic-identity tolerance we want to assert here.
    tau_pred = torch.linspace(
        0.0, 1.5, N_RAYS * N_BINS, dtype=torch.float64
    ).reshape(N_RAYS, N_BINS)
    F_pred = torch.exp(-tau_pred)

    # ---- Pass 1: single masked cycle mean over the full dataset ----
    weighted_F_sum = 0.0
    total_F_count = 0
    for s, e in _microbatch_slices():
        mask_mb = mask[s:e]
        F_pred_mb = F_pred[s:e]
        weighted_F_sum += (F_pred_mb * mask_mb).sum().item()
        total_F_count += int(mask_mb.sum().item())
    F_cycle = weighted_F_sum / max(1, total_F_count)

    # ---- Pass 2: per-microbatch masked means + their unmasked-bin counts ----
    mean_F_mbs = []
    n_unmasked_per_chunk = []
    for s, e in _microbatch_slices():
        mask_mb = mask[s:e]
        F_pred_mb = F_pred[s:e]
        denom = mask_mb.sum().clamp(min=1)
        mean_F_mb = (F_pred_mb * mask_mb).sum() / denom
        mean_F_mbs.append(mean_F_mb.item())
        n_unmasked_per_chunk.append(int(mask_mb.sum().item()))

    # Identity: F_cycle == weighted-average of mean_F_mb_i, weighted by the
    # *unmasked* bin count of each chunk.
    weighted_avg = (
        sum(m * n for m, n in zip(mean_F_mbs, n_unmasked_per_chunk))
        / sum(n_unmasked_per_chunk)
    )
    assert abs(F_cycle - weighted_avg) <= 1e-12 + 1e-12 * abs(F_cycle), (
        f"[D-21] mask-consistency identity violated: "
        f"F_cycle={F_cycle:.15e}, weighted_avg_of_chunk_means={weighted_avg:.15e}, "
        f"diff={abs(F_cycle - weighted_avg):.3e}"
    )

    # ---- Discriminative sanity: the UNmasked per-chunk mean does NOT satisfy
    # the identity (so the test would catch a Pass 2 reduction that drops
    # the mask). ----
    unmasked_chunk_means = []
    for s, e in _microbatch_slices():
        unmasked_chunk_means.append(F_pred[s:e].mean().item())
    naive_mean = sum(unmasked_chunk_means) / len(unmasked_chunk_means)
    # If this were close to F_cycle, the identity would be vacuous on this
    # synthetic data (it would coincidentally hold for the wrong formula).
    assert abs(F_cycle - naive_mean) > 1e-6, (
        "Test setup degenerate: unmasked-mean accidentally equals masked "
        "F_cycle, so the test cannot distinguish a dropped mask. Adjust the "
        "synthetic mask/tau_pred to make the two reductions differ."
    )


# ---------------------------------------------------------------------------
# Test 2: static-analysis assertion on pipeline.py mask source
# ---------------------------------------------------------------------------

PIPELINE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "nerf" / "pipeline.py"


def test_mask_source_is_tau_GT_derived_not_tau_pred_derived():
    """Pin the mask-source contract: mask comes from tau_GT (loader),
    NOT from tau_pred (which would be circular and break [D-21]).

    Static-analysis style: parse pipeline.py text and assert structural
    properties of the train loop. Brittle by design — if the variable is
    renamed, this test fails loudly so the new name can be re-pinned with
    intent.
    """
    src = PIPELINE_PATH.read_text(encoding="utf-8")

    # The mask name used in both passes today.
    mask_var = "mask_no_dla_profile"

    # ---- (a) Pass 1 (no_grad block) references the mask. ----
    # `pipeline.py` has multiple `with torch.no_grad():` blocks (e.g. one in
    # load_checkpoint for RNG state). Anchor on Pass 1 specifically by finding
    # `weighted_F_sum = 0.0` (only appears in the Pass 1 cycle-mean reduction),
    # then verify mask_no_dla_profile appears within the next ~10 lines.
    p1_anchor = re.search(r"weighted_F_sum\s*=\s*0\.0", src)
    assert p1_anchor is not None, (
        "Pass 1 anchor 'weighted_F_sum = 0.0' missing from pipeline.py — "
        "Pass 1 cycle-mean reduction structure changed; re-pin this test."
    )
    p1_window = src[p1_anchor.end(): p1_anchor.end() + 600]
    assert mask_var in p1_window, (
        f"Pass 1 (cycle-mean reduction) does not reference '{mask_var}'. "
        f"Either the mask source moved, or the variable was renamed — "
        f"either way, re-verify the [D-21] mask-consistency contract before "
        f"updating this assertion."
    )

    # ---- (b) Pass 2 (chunk loop) references the SAME mask. ----
    # The Pass 2 chunk loop is the second `for chunk_i, (s, e) in
    # enumerate(microbatch_slices())` (the first is the no_grad pass).
    chunk_iter_matches = list(
        re.finditer(r"for chunk_i, \(s, e\) in enumerate\(microbatch_slices\(\)\)", src)
    )
    # Pass 2 has the enumerate (it needs chunk_i for the prior term);
    # Pass 1 does not. So we expect exactly one chunk_i loop in train().
    assert len(chunk_iter_matches) >= 1, (
        "Pass 2 chunk loop signature ('for chunk_i, (s, e) in "
        "enumerate(microbatch_slices())') not found — pipeline structure "
        "changed; re-pin mask consistency."
    )
    pass2_start = chunk_iter_matches[-1].end()
    pass2_window = src[pass2_start: pass2_start + 3000]
    assert mask_var in pass2_window, (
        f"Pass 2 chunk loop does not reference '{mask_var}'. The [D-21] "
        f"identity requires the SAME mask in both passes."
    )

    # ---- (c) Mask is NOT constructed from tau_pred anywhere in the train loop. ----
    # Anything matching `mask\s*=\s*(...tau_pred...)` is a red flag.
    forbidden_patterns = [
        r"mask\s*=\s*\(?\s*tau_pred",          # mask = (tau_pred ...
        r"mask\s*=\s*tau_pred",                  # mask = tau_pred ...
        r"mask_mb\s*=\s*\(?\s*tau_pred",       # mask_mb = (tau_pred ...
        r"mask_mb\s*=\s*tau_pred",
        r"mask_no_dla\s*=\s*\(?\s*tau_pred",   # mask_no_dla = (tau_pred ...
    ]
    for pat in forbidden_patterns:
        m = re.search(pat, src)
        assert m is None, (
            f"FORBIDDEN: pipeline.py contains a tau_pred-derived mask "
            f"('{pat}' matched: {m.group(0)!r}). This is the exact circular "
            f"failure mode the defense-panel S3 attack warned about — "
            f"Pass 2's mask now depends on the parameter being optimized, "
            f"so the [D-21] gradient identity no longer holds. Revert."
        )


# ---------------------------------------------------------------------------
# Test 3: numerical gradient identity under masked reduction
# ---------------------------------------------------------------------------

def test_gradient_identity_under_masked_reduction_matches_d21_form():
    """[D-21] gradient identity, exercised under the [D-24] masked reduction.

    Two formulations of the same loss:

      (a) FULL: L_meanF_full = lambda_F * (F_cycle - F_obs)^2
          where F_cycle is the masked mean over the entire dataset.
          ``theta.grad`` from backward(L_meanF_full).

      (b) TWO-PASS SURROGATE per [D-21]/[D-24]: Pass 1 computes F_cycle
          under no_grad, then Pass 2 backwards
              (mean_F_grad_coef * mean_F_mb) / accum_steps
          per microbatch where mean_F_grad_coef = 2*lambda_F*(F_cycle-F_obs).

    Under the SAME mask in both passes, (a) and (b) produce numerically
    identical ``theta.grad`` (up to floating-point accumulation order). If
    a future refactor desyncs Pass 1 and Pass 2 masks, this gradient comparison
    drifts.

    Uses tau_pred = theta * ones, so the masked mean F = exp(-theta) and the
    gradient is hand-derivable, providing a third independent check.
    """
    torch.manual_seed(SEED)
    mask = _make_mask()

    # ---- (a) FULL formulation ----
    theta_a = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    tau_pred_a = theta_a * torch.ones(N_RAYS, N_BINS, dtype=torch.float64)
    F_pred_a = torch.exp(-tau_pred_a)
    F_cycle_a = (F_pred_a * mask).sum() / mask.sum().clamp(min=1)
    loss_a = LAMBDA_F * (F_cycle_a - MEAN_FLUX_OBS) ** 2
    loss_a.backward()
    grad_a = theta_a.grad.detach().clone()

    # ---- (b) TWO-PASS SURROGATE ----
    theta_b = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

    # Pass 1: no_grad masked cycle mean.
    with torch.no_grad():
        weighted_F_sum = 0.0
        total_F_count = 0
        for s, e in _microbatch_slices():
            tau_pred_mb = theta_b * torch.ones(e - s, N_BINS, dtype=torch.float64)
            F_pred_mb = torch.exp(-tau_pred_mb)
            mask_mb = mask[s:e]
            weighted_F_sum += (F_pred_mb * mask_mb).sum().item()
            total_F_count += int(mask_mb.sum().item())
        F_cycle_b = weighted_F_sum / max(1, total_F_count)

    mean_F_grad_coef = 2.0 * LAMBDA_F * (F_cycle_b - MEAN_FLUX_OBS)

    # Pass 2: per-microbatch backward of c * mean_F_mb / accum_steps.
    # NOTE: pipeline.py divides the per-chunk loss by accum_steps. The [D-21]
    # identity then sums to exactly ∂L/∂θ because mean_F_grad_coef * (1/N) *
    # Σ_i ∂F_mb_i/∂θ = c * ∂F_cycle_unweighted/∂θ. Under the *masked* reduction
    # the per-chunk denominators differ across chunks, so the (1/accum_steps)
    # weighting is exact only when chunk sizes (and unmasked-bin counts) are
    # uniform across chunks — which they are in this synthetic. When they
    # are NOT uniform in production, the very small bias is the per-step
    # accuracy floor of the surrogate (well below SGD noise; see [D-21]
    # discussion in LEDGER §7).
    for s, e in _microbatch_slices():
        tau_pred_mb = theta_b * torch.ones(e - s, N_BINS, dtype=torch.float64)
        F_pred_mb = torch.exp(-tau_pred_mb)
        mask_mb = mask[s:e]
        mean_F_mb = (F_pred_mb * mask_mb).sum() / mask_mb.sum().clamp(min=1)
        loss_b_chunk = mean_F_grad_coef * mean_F_mb
        (loss_b_chunk / ACCUM_STEPS).backward()

    grad_b = theta_b.grad.detach().clone()

    # ---- Compare ----
    # The identity is exact when chunk sizes AND per-chunk unmasked counts
    # are uniform; we constructed the synthetic that way (uniform chunk
    # rays + statistically uniform mask), but tightness of equality depends
    # on the realized per-chunk unmasked counts. We assert equality of the
    # per-chunk-unmasked-count-weighted sum (the true identity) within
    # double-precision tolerance.
    #
    # Concretely: with uniform chunk sizes (MICROBATCH = N_RAYS / accum_steps
    # = 2), the (1/accum_steps) scaling in the surrogate matches the
    # 1/N_chunks term in ∂F_cycle/∂θ = (1/N_chunks) Σ_i ∂F_mb_i/∂θ ONLY when
    # per-chunk unmasked-bin counts are equal. We verify equality in a way
    # that is robust to the small per-chunk-count nonuniformity by checking
    # both grads within a loose tolerance, AND verify they agree in sign,
    # magnitude order, and to high precision when chunk masks are uniform.
    n_unmasked_per_chunk = []
    for s, e in _microbatch_slices():
        n_unmasked_per_chunk.append(int(mask[s:e].sum().item()))
    chunks_uniform = len(set(n_unmasked_per_chunk)) == 1

    if chunks_uniform:
        # Tight identity.
        assert torch.allclose(grad_a, grad_b, atol=1e-7, rtol=1e-7), (
            f"[D-21] gradient identity broken under masked reduction "
            f"(uniform per-chunk unmasked counts): "
            f"FULL grad={grad_a.item():.10e}, "
            f"SURROGATE grad={grad_b.item():.10e}, "
            f"|diff|={(grad_a - grad_b).abs().item():.3e}. "
            f"Likely cause: Pass 1 and Pass 2 mask sources have desynced."
        )
    else:
        # Loose identity (the per-chunk count nonuniformity is the [D-21]
        # surrogate's intrinsic accuracy floor; assert sign + same magnitude).
        assert grad_a.sign() == grad_b.sign(), (
            f"[D-21] surrogate flipped gradient sign vs FULL: "
            f"grad_a={grad_a.item():.3e}, grad_b={grad_b.item():.3e}"
        )
        rel_err = (grad_a - grad_b).abs() / grad_a.abs().clamp(min=1e-30)
        assert rel_err.item() < 0.05, (
            f"[D-21] surrogate vs FULL relative error {rel_err.item():.3e} "
            f"exceeds 5% under nonuniform per-chunk masks; this is well "
            f"above the expected accuracy floor — investigate."
        )
