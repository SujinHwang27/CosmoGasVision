"""Regression tests for the Tepper-García Voigt kernel.

Covers the 2026-05-04 small-|x| Taylor branch added to disarm the
H(a, 0) ~ 0 cancellation pathology while leaving the production regime
(|x| >~ 0.1 on the Sherwood velocity grid) unchanged within float32 noise.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.models.nerf import tepper_garcia_voigt


# Typical Lyman-alpha damping parameter at b = 12.85 km/s (T = 10^4 K).
_A_LYA = 4.7e-4 / 12.85


def test_line_center_returns_one_minus_2a_over_sqrtpi():
    """At x = 0 the analytic limit is H = 1 - 2a/sqrt(pi)."""
    a = torch.tensor(_A_LYA, dtype=torch.float64)
    x = torch.tensor(0.0, dtype=torch.float64)
    expected = 1.0 - 2.0 * _A_LYA / math.sqrt(math.pi)

    H = tepper_garcia_voigt(a, x).item()

    assert H == pytest.approx(expected, rel=1e-12, abs=1e-12), (
        f"H(a, 0) = {H}, expected {expected}"
    )


def test_taylor_branch_matches_main_branch_at_handoff():
    """Both branches must agree at the |x|^2 = 1e-4 boundary.

    Evaluating *both* formulas at the *same* x near the threshold tests
    branch continuity. The leading-order disagreement is O(a*x^2),
    i.e. ~4e-9 at x^2 = 1e-4 with a ~ 4e-5.
    """
    a = torch.tensor(_A_LYA, dtype=torch.float64)
    x2 = torch.tensor(1.0e-4, dtype=torch.float64)
    x = torch.sqrt(x2)

    # Evaluate the main branch directly (no torch.where dispatch)
    exp_x2 = torch.exp(-x2)
    exp_2x2 = torch.exp(-2.0 * x2)
    poly = 4 * x2 ** 2 + 7 * x2 + 4 + 1.5 / x2
    bracket = exp_2x2 * poly - 1.5 / x2 - 1.0
    H_main = (exp_x2 - (a / (math.sqrt(math.pi) * x2)) * bracket).item()

    # Taylor branch
    H_small = (torch.exp(-x2) - 2.0 * a / math.sqrt(math.pi)).item()

    assert abs(H_main - H_small) < 1e-7, (
        f"Branch discontinuity at x^2 = 1e-4: H_main = {H_main}, "
        f"H_small = {H_small}, diff = {H_main - H_small}"
    )


def test_production_regime_unchanged_within_float32_noise():
    """For |x| in [0.05, 5.0] (the production regime on the Sherwood grid), the
    Taylor branch is never taken; values are the analytic main-branch evaluation
    and must be reproducible to float32 precision."""
    a = torch.tensor(_A_LYA, dtype=torch.float64)
    x = torch.linspace(0.05, 5.0, 100, dtype=torch.float64)

    H = tepper_garcia_voigt(a, x)

    # Main-branch closed form (should be exactly what tepper_garcia_voigt computes
    # since |x|^2 >= 0.0025 > 1e-4 throughout)
    P = x * x
    R = torch.exp(-P)
    Q = 1.5 / P
    bracket = R * R * (4 * P * P + 7 * P + 4 + Q) - Q - 1.0
    H_ref = R - (a / (math.sqrt(math.pi) * P)) * bracket
    H_ref = torch.clamp(H_ref, min=0.0)

    diff = (H - H_ref).abs().max().item()
    assert diff < 1e-12, f"Main-branch values drifted: max abs diff = {diff}"


def test_gradient_finite_at_line_center():
    """The Taylor branch must produce finite gradients at x = 0 (no NaN/Inf
    poisoning from the dead main branch)."""
    a = torch.tensor(_A_LYA, dtype=torch.float64, requires_grad=True)
    x = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    H = tepper_garcia_voigt(a, x)
    H.backward()

    assert torch.isfinite(a.grad).all(), f"a.grad = {a.grad}"
    assert torch.isfinite(x.grad).all(), f"x.grad = {x.grad}"
    # dH/da at x=0: -2/sqrt(pi)
    assert a.grad.item() == pytest.approx(-2.0 / math.sqrt(math.pi), rel=1e-12)


def test_no_clamp_activation_for_typical_lya_inputs():
    """The clamp(min=0) backstop should never activate for realistic Lya inputs."""
    a = torch.tensor(_A_LYA, dtype=torch.float64)
    x = torch.linspace(-10.0, 10.0, 1001, dtype=torch.float64)

    H = tepper_garcia_voigt(a, x)

    # Recompute without final clamp to confirm pre-clamp values are already >= 0
    x2 = x * x
    small = x2 < 1e-4
    x2_safe = torch.where(small, torch.ones_like(x2), x2)
    exp_x2 = torch.exp(-x2_safe)
    exp_2x2 = torch.exp(-2.0 * x2_safe)
    poly = 4 * x2_safe ** 2 + 7 * x2_safe + 4 + 1.5 / x2_safe
    bracket = exp_2x2 * poly - 1.5 / x2_safe - 1.0
    H_main = exp_x2 - (a / (math.sqrt(math.pi) * x2_safe)) * bracket
    H_small = torch.exp(-x2) - 2.0 * a / math.sqrt(math.pi)
    H_pre_clamp = torch.where(small, H_small, H_main)

    assert (H_pre_clamp >= 0).all(), (
        "clamp activated for typical Lya range — investigate"
    )
    assert torch.allclose(H, H_pre_clamp)


def test_symmetry_in_x():
    """H(a, x) is even in x."""
    a = torch.tensor(_A_LYA, dtype=torch.float64)
    x = torch.linspace(0.01, 5.0, 50, dtype=torch.float64)

    H_pos = tepper_garcia_voigt(a, x)
    H_neg = tepper_garcia_voigt(a, -x)

    assert torch.allclose(H_pos, H_neg, atol=1e-14)
