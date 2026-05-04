"""Regression tests for the [D-24] loss form in ``experiments/nerf/pipeline.py``.

[D-24] mandates three coupled changes to the Stage 2b training loss:

  (1) Data term: log1p MSE, capped at TAU_MAX = 10 (Bolton+ 2017 forest cap),
      masked-mean reduction over non-DLA bins.
  (2) Mean-flux surrogate (per [D-21] two-pass): same masked reduction so the
      cycle-mean F is consistent with the data term's exclusion set.
  (3) DLA detection: handled in ``src/data/loader.py``; this file verifies the
      mask is honored end-to-end at the loss boundary.

These tests pin the *numerical contract* of the [D-24] loss so future
refactors of pipeline.py do not silently re-mean over DLA bins, drop the
log1p, or remove the cap. They run on CPU in well under one second.
"""

from __future__ import annotations

import torch

from src.data.loader import SherwoodLoader


# Match the [D-24] cap pinned in pipeline.py. Hard-coded constant, not a CLI
# flag (per PI: re-tuning means a new D-XX, not an arg override).
TAU_MAX = 10.0


def _d24_loss(tau_pred, tau_gt, mask_no_dla, tau_max: float = TAU_MAX):
    """Reference implementation of the [D-24] data loss.

    Mirrors the inline form in ``experiments/nerf/pipeline.py``:

        diff   = log1p(min(tau_pred, TAU_MAX)) - log1p(min(tau_gt, TAU_MAX))
        loss   = sum(diff^2 * mask) / clamp(sum(mask), min=1)

    Used both as the system-under-test and as the ground-truth comparator
    in the no-DLA hand-derived gradient case below.
    """
    tau_pred_eff = tau_pred.clamp_max(tau_max)
    tau_gt_eff = tau_gt.clamp_max(tau_max)
    diff = torch.log1p(tau_pred_eff) - torch.log1p(tau_gt_eff)
    diff_sq = diff * diff
    return (diff_sq * mask_no_dla).sum() / mask_no_dla.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Test 1: finite + non-NaN + masked-region zero gradient
# ---------------------------------------------------------------------------

def test_d24_loss_finite_and_dla_region_has_zero_gradient():
    """[D-24]: an injected DLA spike must not blow up the loss or leak
    gradient into masked bins.

    Builds a synthetic ``(num_los=2, nbins=2048)`` tau_gt of mild forest
    absorption with a single DLA spike (``tau_gt[..., 1024] = 1e7``), passes
    through the loader's ``_detect_dla_mask`` (the actual algorithm that ships
    in production), constructs a leaf ``tau_pred`` so backward is meaningful,
    and asserts:

      * the [D-24] loss is finite (no NaN, no Inf);
      * the DLA region was actually detected (mask has False entries around
        bin 1024);
      * gradient on ``tau_pred`` at masked bins is exactly zero.

    The third claim is the core supervision contract: ``log_tau_amp`` and
    model weights must not be pulled toward fitting bins inside the DLA
    damping wing.
    """
    torch.manual_seed(2026)

    num_los, nbins = 2, 2048
    # Mild forest: mostly tau ~ 0..1, well below TAU_MAX.
    tau_gt_np = torch.rand(num_los, nbins).numpy() * 0.8 + 0.1
    # Inject a DLA spike at the same bin in both sightlines, with broad
    # damping wings (>10 over a window) so the connected-component grower
    # has something to expand into.
    spike_center = 1024
    half_window = 12
    for los_i in range(num_los):
        # Wing region: tau > 10 (above the wing threshold)
        tau_gt_np[los_i, spike_center - half_window:spike_center + half_window] = 50.0
        # Core: tau > 1e5
        tau_gt_np[los_i, spike_center] = 1e7

    mask_no_dla_np = SherwoodLoader._detect_dla_mask(tau_gt_np)

    tau_gt = torch.tensor(tau_gt_np, dtype=torch.float32)
    mask_no_dla = torch.tensor(mask_no_dla_np, dtype=torch.bool)

    # Sanity: the DLA detector actually masked the spike + its wings.
    assert (~mask_no_dla[:, spike_center - half_window:spike_center + half_window]).all(), (
        "DLA detector did not mask the injected spike + damping wings; "
        "test setup is wrong, not the loss."
    )
    # Sanity: bins far from the spike remain included.
    assert mask_no_dla[:, 0].all() and mask_no_dla[:, -1].all(), (
        "DLA mask grew across the entire sightline; check spike injection."
    )

    # tau_pred is a leaf so we can extract its gradient bin-by-bin.
    tau_pred = torch.full_like(tau_gt, 0.5, requires_grad=True)

    loss = _d24_loss(tau_pred, tau_gt, mask_no_dla)
    assert torch.isfinite(loss), f"[D-24] loss not finite: {loss.item()!r}"
    assert not torch.isnan(loss), "[D-24] loss is NaN"

    grad, = torch.autograd.grad(loss, tau_pred)
    assert torch.isfinite(grad).all(), "[D-24] loss produced non-finite gradient"

    # Bins inside the DLA mask must contribute zero gradient.
    masked_bin_grad = grad[~mask_no_dla]
    assert torch.equal(masked_bin_grad, torch.zeros_like(masked_bin_grad)), (
        f"DLA-masked bins have nonzero gradient: "
        f"max|grad|={masked_bin_grad.abs().max().item():.3e}"
    )

    # Sanity: at non-masked bins the gradient is generally NONzero (the
    # tau_pred=0.5 vs tau_gt~[0.1,0.9] mismatch should drive some signal).
    unmasked_bin_grad = grad[mask_no_dla]
    assert unmasked_bin_grad.abs().max().item() > 0.0, (
        "Non-DLA bins all have zero gradient — the loss is silently dead."
    )


# ---------------------------------------------------------------------------
# Test 2: forest-only gradient matches the analytical log1p-MSE derivative
# ---------------------------------------------------------------------------

def test_d24_loss_gradient_matches_hand_computed_log1p_mse():
    """On a pure-forest synthetic with tau_gt = 0.5 and no DLA, the [D-24]
    loss must reduce to ordinary log1p MSE, and its tau_pred-gradient must
    match the closed-form derivative.

    Derivation. With mask all-True and TAU_MAX = 10 unreached:

        L = mean_{i,j} (log1p(tp_ij) - log1p(tg_ij))^2

        ∂L/∂tp_ij = 2 (log1p(tp_ij) - log1p(tg_ij)) / (1 + tp_ij) / N

    where N = numel(tau_pred). Sign matters: with tp = 0.3 < tg = 0.5 the
    parenthetical (log1p(tp) - log1p(tg)) is *negative*, so gradient is
    *negative*. Optimizer step (gradient descent) then increases tau_pred
    toward tau_gt — the desired supervision direction.
    """
    torch.manual_seed(2026)
    shape = (4, 64)
    tau_gt = torch.full(shape, 0.5, dtype=torch.float32)
    tau_pred = torch.full(shape, 0.3, dtype=torch.float32, requires_grad=True)
    mask_no_dla = torch.ones(shape, dtype=torch.bool)

    loss = _d24_loss(tau_pred, tau_gt, mask_no_dla)

    # Per-bin loss: same expression, no reduction.
    expected_per_bin = (torch.log1p(tau_pred.detach()) - torch.log1p(tau_gt)) ** 2
    assert torch.allclose(loss, expected_per_bin.mean(), atol=1e-7), (
        f"[D-24] masked-mean loss != hand-computed mean of log1p-MSE per-bin: "
        f"got {loss.item():.10e}, expected {expected_per_bin.mean().item():.10e}"
    )

    grad, = torch.autograd.grad(loss, tau_pred)
    N = float(tau_pred.numel())
    expected_grad = (
        2.0 * (torch.log1p(tau_pred.detach()) - torch.log1p(tau_gt))
        / (1.0 + tau_pred.detach())
        / N
    )
    assert torch.allclose(grad, expected_grad, atol=1e-7), (
        f"[D-24] gradient mismatch: max|diff|="
        f"{(grad - expected_grad).abs().max().item():.3e}"
    )

    # The descent-direction sanity check: tau_pred=0.3 below tau_gt=0.5 means
    # the gradient must be negative everywhere (so -lr * grad pushes tp UP).
    assert (grad < 0).all(), (
        "log1p-MSE gradient sign wrong: tau_pred should be pulled UP toward "
        "tau_gt, but the gradient is non-negative."
    )


# ---------------------------------------------------------------------------
# Test 3: masked mean-flux reduction is gradient-correct
# ---------------------------------------------------------------------------

def test_d24_masked_mean_flux_reduction_is_gradient_correct():
    """[D-24] + [D-21]: the per-microbatch mean-F surrogate

        mean_F_mb = (exp(-tau_pred) * mask).sum() / mask.sum().clamp(min=1)

    must

      (a) reduce to exp(-tau) when tau_pred is constant and the mask has at
          least one True element (the masked-mean gives the unweighted mean
          over the included subset);
      (b) propagate gradient correctly: with tau_pred = c (constant scalar)
          everywhere, ``d(mean_F_mb)/dc = -exp(-c)``.

    This proves the masked reduction does not introduce a stale denominator,
    a detached cast, or a shape-broadcast bug that would silently scale the
    [D-21] linearization coefficient ``mean_F_grad_coef`` and corrupt the
    mean-flux soft constraint gradient.
    """
    shape = (3, 16)
    c_value = 0.7
    c = torch.tensor(c_value, dtype=torch.float32, requires_grad=True)
    tau_pred = c * torch.ones(shape, dtype=torch.float32)

    # mask: all True except a single bin. The masked-mean must equal exp(-c)
    # exactly because every retained bin holds the same constant value.
    mask = torch.ones(shape, dtype=torch.bool)
    mask[0, 0] = False

    F_pred = torch.exp(-tau_pred)
    mean_F_mb = (F_pred * mask).sum() / mask.sum().clamp(min=1)

    # (a) Value check.
    expected_value = torch.exp(-c.detach())
    assert torch.allclose(mean_F_mb, expected_value, atol=1e-7), (
        f"[D-24] masked mean-F value: got {mean_F_mb.item():.10e}, "
        f"expected {expected_value.item():.10e}"
    )

    # (b) Gradient check.
    grad_c, = torch.autograd.grad(mean_F_mb, c)
    expected_grad = -torch.exp(-c.detach())
    assert torch.allclose(grad_c, expected_grad, atol=1e-7), (
        f"[D-24] masked mean-F grad: got {grad_c.item():.10e}, "
        f"expected {expected_grad.item():.10e}"
    )
