"""[D-69] Stage 1a — estimator-equivalence test for the pretrain loss path.

Verifies the in-loop autograd ``_pretrain_loss`` (log10 + per-voxel MSE, with
the 1e-3 stabilizer) matches a hand-coded NumPy reference on a fixed-seed
minibatch to 1e-5 relative tolerance.

Per [D-60] precedent (sprint-L1 gate-4 K2 deliverable): every new training-
loss path that lands in ``pipeline.py`` requires an estimator-equivalence
test before any Juno commitment. See
``tests/test_torch_pf_estimator_equivalence.py`` for the [D-60] template; this
file is the [D-69] analog for the L_pre log-MSE path.

Spec source: ``experiments/nerf/design/D69_stage1_pretraining_scoping.md``
Revision 5 §2 single-equation form:

    L_pre = mean((log10(rho_theta + 1e-3) - log10(rho_truth + 1e-3)) ** 2)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

# Make ``experiments/nerf/pipeline.py`` importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
NERF_EXP = REPO_ROOT / "experiments" / "nerf"
if str(NERF_EXP) not in sys.path:
    sys.path.insert(0, str(NERF_EXP))

from pipeline import _pretrain_loss, PRETRAIN_LOG_EPS  # noqa: E402


SEED = 20260524
N_BATCHES = 10
B = 1024
REL_TOL = 1.0e-5


def _numpy_reference_loss(rho_theta_np: np.ndarray,
                          rho_truth_np: np.ndarray,
                          eps: float = PRETRAIN_LOG_EPS) -> float:
    """Hand-coded NumPy reference for L_pre. Float64 throughout."""
    a = np.log10(rho_theta_np.astype(np.float64) + eps)
    b = np.log10(rho_truth_np.astype(np.float64) + eps)
    diff = a - b
    return float((diff * diff).mean())


def test_pretrain_loss_matches_numpy_reference():
    """10 fixed-seed batches, all within 1e-5 relative tolerance."""
    rng = np.random.default_rng(SEED)
    max_rel = 0.0
    for batch_idx in range(N_BATCHES):
        # rho_truth from a Sherwood-realistic distribution: lognormal
        # overdensity in [1e-3, ~1e3] with a heavy filament tail.
        # rho_theta from a perturbed copy so the loss is nonzero (we test
        # numerical equivalence of the estimator path, not loss==0 degeneracy).
        rho_truth_np = rng.lognormal(mean=0.0, sigma=1.5, size=(B,))
        # Inject zeros at 25% rate so the +eps stabilizer is exercised.
        zero_mask = rng.uniform(size=(B,)) < 0.25
        rho_truth_np = np.where(zero_mask, 0.0, rho_truth_np).astype(np.float32)
        rho_theta_np = (rho_truth_np
                        * rng.uniform(0.5, 2.0, size=(B,))).astype(np.float32)
        # Sanity: both are >= 0 (Softplus post-activation invariant).
        assert (rho_truth_np >= 0).all() and (rho_theta_np >= 0).all()

        # Torch in-loop path — autograd live.
        rho_theta_t = torch.from_numpy(rho_theta_np).clone().requires_grad_(True)
        rho_truth_t = torch.from_numpy(rho_truth_np)
        L_torch = _pretrain_loss(rho_theta_t, rho_truth_t)
        L_torch_val = float(L_torch.item())

        # NumPy reference.
        L_np_val = _numpy_reference_loss(rho_theta_np, rho_truth_np)

        rel = abs(L_torch_val - L_np_val) / max(abs(L_np_val), 1e-30)
        max_rel = max(max_rel, rel)
        assert rel <= REL_TOL, (
            f"[batch {batch_idx}] _pretrain_loss != NumPy ref: "
            f"torch={L_torch_val:.10e}, numpy={L_np_val:.10e}, "
            f"rel={rel:.3e}, tol={REL_TOL:.0e}"
        )
    print(f"\n[D-69 equivalence] {N_BATCHES} batches of B={B}: "
          f"max_rel_dev={max_rel:.3e} (tol={REL_TOL:.0e})")


def test_pretrain_loss_autograd_through_rho_theta():
    """Gradient flow sanity — L_pre.backward() produces finite, non-zero grads
    in rho_theta. Mirrors test_torch_pf_estimator_equivalence's autograd probe.
    """
    rng = np.random.default_rng(SEED + 1)
    rho_truth_np = rng.lognormal(0.0, 1.5, size=(B,)).astype(np.float32)
    rho_theta_np = (rho_truth_np * rng.uniform(0.5, 2.0, size=(B,))).astype(np.float32)
    rho_theta_t = torch.from_numpy(rho_theta_np).clone().requires_grad_(True)
    rho_truth_t = torch.from_numpy(rho_truth_np)
    loss = _pretrain_loss(rho_theta_t, rho_truth_t)
    loss.backward()
    assert rho_theta_t.grad is not None, "no grad on rho_theta — graph severed"
    assert torch.isfinite(rho_theta_t.grad).all(), (
        "non-finite gradient in rho_theta.grad"
    )
    assert float(rho_theta_t.grad.abs().max().item()) > 0.0, (
        "rho_theta.grad is identically zero — _pretrain_loss severed the graph"
    )


def test_pretrain_loss_with_zero_inputs_finite():
    """log10(0 + 1e-3) = -3 is finite; the loss must not produce NaN/Inf when
    both inputs are zero."""
    rho_theta_t = torch.zeros(B, requires_grad=True)
    rho_truth_t = torch.zeros(B)
    loss = _pretrain_loss(rho_theta_t, rho_truth_t)
    val = float(loss.item())
    assert math.isfinite(val), f"L_pre(zeros, zeros) = {val}, expected 0.0"
    assert abs(val) < 1e-12, f"L_pre(zeros, zeros) should be 0; got {val}"
