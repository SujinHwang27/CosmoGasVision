"""Sprint-L1 commit-B behavioral test: GradNorm task weights must actually
diverge when wired correctly.

This is the structural fix for the test-coverage gap that allowed the
``shared_params=[l1_gn.w_tau]`` placeholder bug to ship to the gate-pilot
(R20-candidate). The pre-existing tests asserted the wrapper API was correct
but never asserted that ``w_tau`` and ``w_pf`` move under a forward-/-backward
update loop driven by deliberately imbalanced losses.

Setup:
- 2-param ``nn.Linear(1, 1)`` dummy model (so ``shared_params`` is non-empty
  and the full Chen+ 2018 second-order path can compute meaningful gradient
  norms).
- ``loss_tau = 100 * (out - y_a).pow(2).mean()`` vs.
  ``loss_pf = (out - y_b).pow(2).mean()`` — 100x loss-ratio divergence by
  construction, which forces GradNorm to shift the weights toward the smaller
  task (Chen+ 2018 §3 "relative inverse training rate" balancing).
- ``GradNormWrapper(initial_w=(1.0, 1.0), alpha=0.12, simplified=False)``
  with a dedicated ``Adam([wrapper.w_tau, wrapper.w_pf], lr=0.025)``.
- 50 steps of: forward -> ``compute_gradnorm_loss(...)`` -> ``backward()``
  -> ``step()`` -> ``renormalize_weights()``.

Assertions at step 50:
- ``abs(w_tau - 1.0) > 0.05`` AND ``abs(w_pf - 1.0) > 0.05`` (weights moved).
- ``abs((w_tau + w_pf) - 2.0) < 1e-5`` (sum invariant T = 2 for 2 tasks).

Both ``simplified=False`` (the path that exercises the Commit-B call-site fix)
and ``simplified=True`` (the gate-pilot default per [D-60] job 201587) are
covered to prevent regression on either branch.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.training.p_flux_loss import GradNormWrapper  # noqa: E402


@pytest.mark.parametrize("simplified", [False, True])
def test_gradnorm_weights_diverge_under_imbalanced_losses(simplified: bool) -> None:
    """Weights must move > 0.05 from init while preserving sum == T."""
    torch.manual_seed(0)

    # 2-param dummy model so shared_params is non-empty for simplified=False.
    model = nn.Linear(1, 1)

    # Two disjoint regression targets -> two task losses.
    x = torch.tensor([[1.0]])
    y_a = torch.tensor([[10.0]])  # tau target
    y_b = torch.tensor([[-10.0]])  # pf target

    wrapper = GradNormWrapper(
        initial_w=(1.0, 1.0), alpha=0.12, simplified=simplified,
    )
    # Dedicated optimizer for the task weights ONLY (mirrors pipeline.py:l1_gn_opt).
    gn_opt = torch.optim.Adam([wrapper.w_tau, wrapper.w_pf], lr=0.025)

    for _step in range(50):
        out = model(x)
        loss_tau = 100.0 * (out - y_a).pow(2).mean()
        loss_pf = (out - y_b).pow(2).mean()

        # Pin L0 on first step (idempotent).
        wrapper.initialize_L0(loss_tau, loss_pf)

        gn_loss = wrapper.compute_gradnorm_loss(
            loss_tau, loss_pf,
            shared_params=list(model.parameters()),
        )
        gn_opt.zero_grad()
        gn_loss.backward()
        gn_opt.step()
        wrapper.renormalize_weights()

    w_tau_final = float(wrapper.w_tau.detach().item())
    w_pf_final = float(wrapper.w_pf.detach().item())

    # Load-bearing claim: weights moved meaningfully under imbalanced losses.
    assert abs(w_tau_final - 1.0) > 0.05, (
        f"w_tau did not diverge from init: w_tau={w_tau_final:.6f} "
        f"(simplified={simplified}). GradNorm balancer is dead."
    )
    assert abs(w_pf_final - 1.0) > 0.05, (
        f"w_pf did not diverge from init: w_pf={w_pf_final:.6f} "
        f"(simplified={simplified}). GradNorm balancer is dead."
    )
    # Sum-invariant under renormalize_weights (T = 2 for 2 tasks).
    assert abs((w_tau_final + w_pf_final) - 2.0) < 1e-5, (
        f"Sum invariant violated: w_tau + w_pf = {w_tau_final + w_pf_final:.8f} "
        f"(expected 2.0, simplified={simplified})."
    )
