"""Regression test for the [D-14] gradient-accumulation contract.

Background
----------
Stage 2b microbatched accumulation (per [D-14]) splits ``n_rays`` into
``accum_steps`` chunks, calls ``volume_render_physics`` per chunk, and
backwards ``loss_mb / accum_steps`` per chunk to keep peak activation memory
bounded. The B-2 cloud smoke (P1, n_rays=16384, microbatch=1024 -> accum=16)
crashed at ``experiments/nerf/pipeline.py:395`` with::

    RuntimeError: Trying to backward through the graph a second time ...

The cause was that ``tau_amp = torch.exp(log_tau_amp)`` was computed once per
training step, *outside* the chunk loop, so the ``torch.exp`` autograd node
linking ``tau_amp`` to the leaf parameter ``log_tau_amp`` was shared across
chunks. After chunk 0's ``.backward()`` freed that node, chunk 1's backward
crashed.

Fix
---
``tau_amp`` is recomputed inside the Pass-2 chunk loop body so each chunk
owns its own autograd subgraph rooted at ``log_tau_amp``.

Contract enforced by this test
------------------------------
For a fixed model + data + ``log_tau_amp`` value, the per-step accumulated
gradient ``log_tau_amp.grad`` MUST be numerically identical (within
``rtol=1e-5``) across ``accum_steps in {1, 4, 16}``. Likewise for
``model.parameters()``. This is the mathematical equivalence claim that
microbatched accumulation depends on. If a future refactor reintroduces a
graph-shared tensor, this test trips.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.models.nerf import IGMNeRF, volume_render_physics


# ---------------------------------------------------------------------------
# Tiny synthetic setup
# ---------------------------------------------------------------------------

N_RAYS = 16
N_BINS = 32
SEED = 1234
MEAN_FLUX_OBS = 0.877
LAMBDA_F = 1.0


def _build_fixture(device: torch.device):
    """Return a fresh (model, log_tau_amp, coords, vel_axis, tau_gt, mask) tuple.

    Identical across calls because we re-seed before each construction.

    The synthetic data has no DLAs by construction, so ``mask`` is all-True
    (every bin included). This preserves the [D-14] gradient-invariance proof
    — a constant mask is a per-bin scalar that factors out of the per-chunk
    reduction identity — while exercising the [D-24] code path.
    """
    torch.manual_seed(SEED)
    # Tiny model -- exercise the same code path (Fourier encoding, skip
    # connection, physical-bound output heads) without burning CI time.
    model = IGMNeRF(hidden_dim=16, num_layers=4, L=2).to(device)
    log_tau_amp = torch.nn.Parameter(torch.tensor(0.0, device=device))

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    coords = torch.rand(N_RAYS, N_BINS, 3, generator=gen).to(device)
    vel_axis = torch.linspace(0.0, 6000.0, N_BINS, device=device)
    tau_gt = torch.rand(N_RAYS, N_BINS, generator=gen).to(device)
    mask_no_dla = torch.ones(N_RAYS, N_BINS, dtype=torch.bool, device=device)
    return model, log_tau_amp, coords, vel_axis, tau_gt, mask_no_dla


def _step_grads(accum_steps: int, device: torch.device):
    """Run one optimizer.zero_grad -> two-pass forward/backward cycle.

    Mirrors ``experiments/nerf/pipeline.py`` Stage 2b training step exactly:
      Pass 1: cycle-mean of exp(-tau) under ``no_grad``.
      Pass 2: per-microbatch backward of (data_mse + linearized meanF) /
              accum_steps, with ``tau_amp`` recomputed PER CHUNK.

    Returns the post-backward (pre-step) ``.grad`` of ``log_tau_amp`` and a
    flat clone of every ``model.parameters()`` ``.grad``.

    Mirrors the [D-24] loss form (log1p MSE capped at TAU_MAX, masked) plus
    the [D-24] masked mean-F reduction. Since the synthetic mask is all-True
    the masked-mean reduces to the ordinary mean — but the code path that
    exercises ``(diff_sq * mask).sum() / mask.sum()`` is what we're regression
    testing, so the test must drive it.
    """
    model, log_tau_amp, coords, vel_axis, tau_gt, mask_no_dla = _build_fixture(device)

    assert N_RAYS % accum_steps == 0, "Test rigged so chunks tile exactly."
    microbatch = N_RAYS // accum_steps
    TAU_MAX = 10.0  # [D-24] Bolton+ 2017 forest cap

    def slices():
        for i in range(accum_steps):
            s = i * microbatch
            e = min(s + microbatch, N_RAYS)
            if s >= e:
                return
            yield s, e

    # zero grads before the cycle
    if log_tau_amp.grad is not None:
        log_tau_amp.grad = None
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    # Pass 1 (no grad): cycle-mean F, masked per [D-24].
    with torch.no_grad():
        tau_amp_p1 = torch.exp(log_tau_amp)
        weighted_F_sum = 0.0
        total_F_count = 0
        for s, e in slices():
            tau_pred = volume_render_physics(
                model, coords[s:e], vel_axis=vel_axis, tau_amp=tau_amp_p1,
            )
            mask_mb = mask_no_dla[s:e]
            F_pred = torch.exp(-tau_pred)
            weighted_F_sum += (F_pred * mask_mb).sum().item()
            total_F_count += int(mask_mb.sum().item())
        mean_F_pred_val = weighted_F_sum / max(1, total_F_count)

    mean_F_grad_coef = 2.0 * LAMBDA_F * (mean_F_pred_val - MEAN_FLUX_OBS)

    # Pass 2: per-chunk backward. tau_amp MUST be recomputed inside the loop.
    for s, e in slices():
        tau_amp_chunk = torch.exp(log_tau_amp)
        tau_pred = volume_render_physics(
            model, coords[s:e], vel_axis=vel_axis, tau_amp=tau_amp_chunk,
        )
        # [D-24] log1p MSE, capped at TAU_MAX, masked.
        tau_pred_eff = tau_pred.clamp_max(TAU_MAX)
        tau_gt_eff = tau_gt[s:e].clamp_max(TAU_MAX)
        mask_mb = mask_no_dla[s:e]
        diff = torch.log1p(tau_pred_eff) - torch.log1p(tau_gt_eff)
        diff_sq = diff * diff
        loss_data = (diff_sq * mask_mb).sum() / mask_mb.sum().clamp(min=1)
        # [D-24] masked mean-F surrogate.
        F_pred = torch.exp(-tau_pred)
        mean_F_mb = (F_pred * mask_mb).sum() / mask_mb.sum().clamp(min=1)
        loss_mb = loss_data + mean_F_grad_coef * mean_F_mb
        (loss_mb / accum_steps).backward()

    log_tau_amp_grad = log_tau_amp.grad.detach().clone()
    model_grads = {
        name: p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    return log_tau_amp_grad, model_grads


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def grads_by_accum(device):
    """Compute grads at accum_steps in {1, 4, 16}; cache for both tests."""
    out = {}
    for k in (1, 4, 16):
        out[k] = _step_grads(k, device)
    return out


@pytest.mark.parametrize("accum_steps", [1, 4, 16])
def test_log_tau_amp_grad_invariant_under_accumulation(
    accum_steps, grads_by_accum,
):
    """[D-14] contract: log_tau_amp.grad is invariant to accum_steps.

    If this fails, a graph-shared tensor has been reintroduced into the
    Pass-2 chunk loop (or the per-chunk loss scaling has drifted away from
    1/accum_steps). See pipeline.py:344-409 and the docstring above.
    """
    ref_grad, _ = grads_by_accum[1]
    test_grad, _ = grads_by_accum[accum_steps]

    assert torch.isfinite(test_grad).all(), (
        f"log_tau_amp.grad has non-finite entries at accum_steps={accum_steps}"
    )
    assert torch.allclose(test_grad, ref_grad, rtol=1e-5, atol=1e-7), (
        f"log_tau_amp.grad mismatch at accum_steps={accum_steps}: "
        f"got {test_grad.item():.10e}, expected {ref_grad.item():.10e}, "
        f"abs_diff={(test_grad - ref_grad).abs().item():.3e}"
    )


@pytest.mark.parametrize("accum_steps", [4, 16])
def test_model_param_grads_invariant_under_accumulation(
    accum_steps, grads_by_accum,
):
    """Sister contract: every model parameter's accumulated grad is invariant.

    Catches the symmetric bug where some intermediate (e.g. a positional
    encoding tensor) gets cached outside the chunk loop and shared across
    backwards.
    """
    _, ref_grads = grads_by_accum[1]
    _, test_grads = grads_by_accum[accum_steps]

    assert set(ref_grads) == set(test_grads), (
        "Parameter set drifted between accum_steps runs."
    )
    for name in ref_grads:
        a, b = ref_grads[name], test_grads[name]
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-7), (
            f"grad mismatch on parameter '{name}' at accum_steps={accum_steps}: "
            f"max|diff|={ (a - b).abs().max().item():.3e}, "
            f"max|ref|={a.abs().max().item():.3e}"
        )


def test_pass2_does_not_share_graph_across_chunks():
    """Direct reproduction of the original RuntimeError.

    Builds the SAME setup but with ``tau_amp`` computed ONCE outside the loop
    (the pre-fix code path). Asserts that the second chunk's backward raises
    the 'backward through the graph a second time' RuntimeError. If this
    test stops raising, PyTorch has changed its retain-graph semantics and
    the regression-detection logic in this file needs revisiting.
    """
    device = torch.device("cpu")
    model, log_tau_amp, coords, vel_axis, tau_gt, _mask = _build_fixture(device)
    mse_loss = torch.nn.MSELoss()

    accum_steps = 4
    microbatch = N_RAYS // accum_steps

    # The buggy pattern: single tau_amp, graph shared across chunks.
    tau_amp = torch.exp(log_tau_amp)

    raised = False
    try:
        for i in range(accum_steps):
            s = i * microbatch
            e = s + microbatch
            tau_pred = volume_render_physics(
                model, coords[s:e], vel_axis=vel_axis, tau_amp=tau_amp,
            )
            loss = mse_loss(tau_pred, tau_gt[s:e])
            (loss / accum_steps).backward()
    except RuntimeError as err:
        msg = str(err).lower()
        assert "backward through the graph a second time" in msg or (
            "saved" in msg and "freed" in msg
        ), f"Unexpected RuntimeError text: {err!r}"
        raised = True

    assert raised, (
        "Expected the buggy single-tau_amp pattern to raise on chunk 2, but "
        "it didn't. Either PyTorch's autograd retain semantics changed, or "
        "the test fixture no longer exercises the shared-graph pattern."
    )
