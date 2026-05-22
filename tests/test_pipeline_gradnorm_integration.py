"""[D2 2026-05-22] Integration test for bug #2 fix in
``experiments/nerf/pipeline.py::_assemble_step_losses_and_gradnorm``.

Bug #2 (gate-pilot iteration-2, SageMaker job 201669): the helper's
``.item() -> np.mean -> float -> torch.tensor(float)`` cascade graph-broke the
per-chunk task losses on their way to ``GradNormWrapper.compute_gradnorm_loss``.
Under ``simplified=False`` (full Chen+ 2018 path) the wrapper's
``torch.autograd.grad(w_i * L_i, shared_params)`` then traversed a graph-dead
scalar and ``G_tau == G_pf == 0`` -> ``r_tau == r_pf == 1.0`` -> w_tau, w_pf
pinned at init forever. Pilot tripped the step-100 degeneracy assertion with
``w_tau=w_pf=1.000000``.

D2 fix:
1. New ``data_loss_chunks_live`` / ``l1_loss_chunks_live`` accumulators feed
   ``torch.stack(chunks_live).mean()`` into the helper (graph-live).
2. Empty-branch guard: when either live-chunks list is empty, the helper
   SKIPS the GradNorm step (returns ``l1_gn_metrics=None`` +
   ``gradnorm_guard_skipped=True``); no synthetic zero is forged.

This integration test is the load-bearing complement to the unit-tier
``tests/test_gradnorm_weights_diverge.py``. It exercises the SAME helper the
pipeline calls (no parallel reimplementation) on a tiny IGM-NeRF + synthetic
data and asserts (i) ``loss_tau_scalar.grad_fn is not None``,
(ii) ``loss_pf_scalar.grad_fn is not None``, and (iii) after >=2 helper calls
+ optimizer step both ``w_tau`` AND ``w_pf`` have moved from init (the
half-graph-break failure mode PI flagged).

Target: <5s on CPU. Marked ``integration`` (not ``slow``).

Reference: PI D2 spec 2026-05-22; D1 commit ``f9bfb20``.
"""

from __future__ import annotations

import argparse
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.nerf.pipeline import _assemble_step_losses_and_gradnorm  # noqa: E402
from src.models.nerf import IGMNeRF  # noqa: E402
from src.training.p_flux_loss import GradNormWrapper, pf_log_mse_loss  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _smoke_args() -> argparse.Namespace:
    """Minimal argparse.Namespace mirroring smoke defaults for the helper."""
    return argparse.Namespace(
        enable_l1_pf_loss=True,
        tau_max=10.0,
    )


def _tiny_model(seed: int = 0) -> IGMNeRF:
    """64-hidden, 2-layer IGM-NeRF (~3k params) per PI D2 spec."""
    torch.manual_seed(seed)
    # num_layers must be >= 4 by IGMNeRF construction (layers1 has 4 entries
    # by design; layers2 = num_layers - 4). We pick num_layers=4 (smallest
    # allowed: 4 pre-skip layers, 0 post-skip layers). L=2 keeps the encoding
    # cheap.
    return IGMNeRF(hidden_dim=64, num_layers=4, L=2)


def _synthetic_step_chunks(
    model: IGMNeRF,
    *,
    n_rays: int = 8,
    n_bins: int = 64,
    n_chunks: int = 3,
    seed: int = 1,
):
    """Build per-chunk live-graph task-loss tensors via a tiny forward pass.

    Returns
    -------
    (data_loss_chunks, l1_loss_chunks, data_loss_chunks_live, l1_loss_chunks_live)
    """
    torch.manual_seed(seed)
    data_loss_chunks = []
    l1_loss_chunks = []
    data_loss_chunks_live = []
    l1_loss_chunks_live = []

    chunk_size = max(1, n_rays // n_chunks)
    # vel_axis used by pf_log_mse_loss: monotonic, dv = 10 km/s.
    vel_axis = torch.arange(n_bins, dtype=torch.float32) * 10.0
    # F_truth target: a smooth ~1 - exp(-tau) toy field so the log-MSE is
    # finite + has a real gradient.
    F_truth = torch.clamp(
        0.5 + 0.3 * torch.sin(2.0 * torch.pi * vel_axis / (n_bins * 10.0)),
        min=0.05, max=0.95,
    ).unsqueeze(0).expand(chunk_size, n_bins).contiguous()

    for c in range(n_chunks):
        # Tiny coord batch in unit cube.
        coords = torch.rand(chunk_size, n_bins, 3)
        out = model(coords)
        # IGMNeRF returns a stacked (..., 4) tensor over (rho, T, X_HI, v_pec).
        # Build a tau surrogate from rho * X_HI so the graph reaches the model
        # parameters.
        rho = out[..., 0]
        x_hi = out[..., 2]
        tau_pred = (rho * x_hi).clamp_max(10.0)
        # Synthetic tau target: 0.5 (forces nonzero loss).
        tau_target = torch.full_like(tau_pred, 0.5)
        # Data loss: MSE in tau-space (matches pipeline's loss_data_mb).
        loss_data_mb = ((tau_pred - tau_target) ** 2).mean()
        # P_F log-MSE loss (matches pipeline's loss_pf_mb).
        tau_capped = tau_pred.clamp_max(10.0)
        F_pred = torch.exp(-tau_capped)
        loss_pf_mb = pf_log_mse_loss(F_pred, F_truth, vel_axis)

        # Mirror pipeline conventions: detached for diagnostic + live for
        # GradNorm.
        data_loss_chunks.append(loss_data_mb.detach())
        l1_loss_chunks.append(loss_pf_mb.detach())
        data_loss_chunks_live.append(loss_data_mb)
        l1_loss_chunks_live.append(loss_pf_mb)

    return (
        data_loss_chunks, l1_loss_chunks,
        data_loss_chunks_live, l1_loss_chunks_live,
    )


def _make_gn():
    """GradNormWrapper + dedicated optimizer (mirrors pipeline.py wiring).

    ``simplified=False`` exercises the full Chen+ 2018 second-order path —
    the path that revealed bug #2. Without the D2 fix the helper raises
    inside ``torch.autograd.grad`` (graph-dead scalar) or returns a no-op
    that pins w_tau == w_pf == 1.0.
    """
    gn = GradNormWrapper(initial_w=(1.0, 1.0), alpha=0.12, simplified=False)
    gn_opt = torch.optim.Adam([gn.w_tau, gn.w_pf], lr=0.05)
    return gn, gn_opt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("n_chunks", [1, 3])
def test_helper_emits_graph_live_scalars_and_runs(n_chunks: int) -> None:
    """(i)+(ii): loss_tau_scalar and loss_pf_scalar must have ``grad_fn`` set.

    We instrument the wrapper to capture the scalars the helper feeds in,
    then assert ``grad_fn is not None`` and ``requires_grad`` on both.

    Covers the degenerate ``n_chunks=1`` path and the multi-chunk
    accumulation path that exercises ``torch.stack(...).mean()`` over real
    per-chunk live-graph tensors.
    """
    model = _tiny_model()
    args = _smoke_args()
    gn, gn_opt = _make_gn()

    (data_chunks, l1_chunks,
     data_chunks_live, l1_chunks_live) = _synthetic_step_chunks(
        model, n_chunks=n_chunks,
    )

    # Monkey-patch compute_gradnorm_loss to capture the scalar arguments.
    captured = {}
    orig_compute = gn.compute_gradnorm_loss

    def _spy(loss_tau, loss_pf, shared_params):
        captured["loss_tau_scalar"] = loss_tau
        captured["loss_pf_scalar"] = loss_pf
        return orig_compute(loss_tau, loss_pf, shared_params=shared_params)

    gn.compute_gradnorm_loss = _spy  # type: ignore[assignment]

    out = _assemble_step_losses_and_gradnorm(
        model, gn, gn_opt, args, torch.device("cpu"),
        data_loss_chunks=data_chunks,
        l1_loss_chunks=l1_chunks,
        step=1,
        data_loss_chunks_live=data_chunks_live,
        l1_loss_chunks_live=l1_chunks_live,
    )

    # The helper must NOT short-circuit when live chunks are populated.
    assert out.get("gradnorm_guard_skipped", False) is False
    assert out["l1_gn_metrics"] is not None

    lt = captured["loss_tau_scalar"]
    lp = captured["loss_pf_scalar"]

    # (i) loss_tau_scalar graph-live
    assert lt.grad_fn is not None, (
        f"loss_tau_scalar.grad_fn is None — D2 fix is dead, the helper "
        f"reverted to .item()/np.mean/float/torch.tensor cascade. "
        f"(n_chunks={n_chunks})"
    )
    assert lt.requires_grad, (
        f"loss_tau_scalar.requires_grad is False (n_chunks={n_chunks})"
    )
    # (ii) loss_pf_scalar graph-live
    assert lp.grad_fn is not None, (
        f"loss_pf_scalar.grad_fn is None — half-graph-break failure mode. "
        f"(n_chunks={n_chunks})"
    )
    assert lp.requires_grad, (
        f"loss_pf_scalar.requires_grad is False (n_chunks={n_chunks})"
    )


@pytest.mark.integration
def test_helper_empty_l1_chunks_guard_skipped() -> None:
    """Empty ``l1_loss_chunks_live`` -> GradNorm step is SKIPPED, not zero-forged.

    PI D2 ruling: ``0.0 * sum(p.sum() ...)`` is semantically dishonest.
    Helper must return ``l1_gn_metrics=None`` and ``gradnorm_guard_skipped=True``.
    Weights stay at init (no update happened).
    """
    model = _tiny_model()
    args = _smoke_args()
    gn, gn_opt = _make_gn()

    (data_chunks, _l1_chunks,
     data_chunks_live, _l1_chunks_live) = _synthetic_step_chunks(
        model, n_chunks=2,
    )

    w_tau_before = float(gn.w_tau.detach().item())
    w_pf_before = float(gn.w_pf.detach().item())

    out = _assemble_step_losses_and_gradnorm(
        model, gn, gn_opt, args, torch.device("cpu"),
        data_loss_chunks=data_chunks,
        l1_loss_chunks=[],  # detached list also empty for parity
        step=1,
        data_loss_chunks_live=data_chunks_live,
        l1_loss_chunks_live=[],  # <-- the guarded branch
    )

    assert out.get("gradnorm_guard_skipped") is True, (
        "helper failed to set gradnorm_guard_skipped=True on empty "
        "l1_loss_chunks_live; PI D2 explicit ruling violated."
    )
    assert out["l1_gn_metrics"] is None, (
        "helper returned non-None metrics on guarded step; metrics would "
        "lie to MLflow about a step where the task had no observable data."
    )
    # Weights MUST be unchanged (no GradNorm step happened).
    assert float(gn.w_tau.detach().item()) == pytest.approx(w_tau_before)
    assert float(gn.w_pf.detach().item()) == pytest.approx(w_pf_before)


@pytest.mark.integration
def test_both_weights_diverge_after_two_helper_calls() -> None:
    """(iii): after >=2 helper calls + opt steps, BOTH w_tau AND w_pf must move.

    This catches the half-graph-break failure mode (one task graph-live, the
    other graph-dead -> only one weight moves -> Chen+ 2018 sum invariant
    drags the dead-graph weight in lockstep but BOTH move; with both broken,
    NEITHER moves -> the gate-pilot bug-#2 signature). The strict
    ``> 1e-6`` threshold per PI spec is conservative — the unit-tier
    ``tests/test_gradnorm_weights_diverge.py`` asserts ``> 0.05`` after 50
    steps; here we only need detectable motion to prove the graph carries.
    """
    model = _tiny_model()
    args = _smoke_args()
    gn, gn_opt = _make_gn()

    n_steps = 4
    for step_i in range(1, n_steps + 1):
        # Rebuild chunks per step (the per-chunk losses have to be FRESH —
        # the previous .backward()-free path here means graphs stay live
        # across step boundaries within one chunk-batch, but each helper
        # call needs its own forward pass to be honest about the model
        # state evolving).
        (data_chunks, l1_chunks,
         data_chunks_live, l1_chunks_live) = _synthetic_step_chunks(
            model, n_chunks=3, seed=step_i,
        )

        out = _assemble_step_losses_and_gradnorm(
            model, gn, gn_opt, args, torch.device("cpu"),
            data_loss_chunks=data_chunks,
            l1_loss_chunks=l1_chunks,
            step=step_i,
            data_loss_chunks_live=data_chunks_live,
            l1_loss_chunks_live=l1_chunks_live,
        )
        # Helper internally calls gn_opt.step() + renormalize_weights().
        assert out.get("gradnorm_guard_skipped", False) is False, (
            f"unexpected guard skip at step {step_i}"
        )

    w_tau_final = float(gn.w_tau.detach().item())
    w_pf_final = float(gn.w_pf.detach().item())

    assert abs(w_tau_final - 1.0) > 1e-6, (
        f"w_tau did not move from init after {n_steps} steps: "
        f"w_tau={w_tau_final:.8f}. D2 fix is dead or half-broken — "
        f"the graph between per-chunk losses and the wrapper scalar "
        f"is severed. (gate-pilot bug-#2 signature.)"
    )
    assert abs(w_pf_final - 1.0) > 1e-6, (
        f"w_pf did not move from init after {n_steps} steps: "
        f"w_pf={w_pf_final:.8f}. D2 fix is dead or half-broken on the "
        f"P_F branch — pf_log_mse_loss output is not reaching GradNorm "
        f"graph-live."
    )
    # Sanity: the sum invariant from renormalize_weights still holds.
    assert abs((w_tau_final + w_pf_final) - 2.0) < 1e-5, (
        f"Sum invariant violated: w_tau + w_pf = "
        f"{w_tau_final + w_pf_final:.8f} (expected 2.0). Wrapper "
        f"renormalize_weights misbehaved under live-graph input."
    )
