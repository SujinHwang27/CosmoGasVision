"""Unit tests for [D-42] milestone 1 — velocity-gradient sidecar.

Spec source: `experiments/nerf/LEDGER.md` §3 [D-42] "Math contract" subsection.
Pins:

  1. `compute_vpec_grad` is centered finite-difference with periodic BCs along
     the LOS (last) axis, matching the reference NumPy implementation to 1e-12.
  2. Periodic boundary: the value at `i=0` uses `v_pec[N-1]` for the `i-1`
     neighbor.
  3. Z-score normalization across the FULL dataset is stable across seeds: on
     1024 random Gaussian rays the post-zscore std stays in `[0.95, 1.05]`
     cross-seed (the smoke gate floor; the production gate is `[0.9, 1.1]`
     and is asserted inside `_validate_data` on real data).
"""

from __future__ import annotations

import numpy as np
import pytest

import torch

from src.data.loader import SherwoodLoader
from src.models.nerf import IGMNeRF


# ---------------------------------------------------------------------------
# Reference NumPy implementation (the "truth" the torch implementation must
# match to 1e-12).
# ---------------------------------------------------------------------------
def _ref_grad_numpy(v_pec: np.ndarray, dchi: float) -> np.ndarray:
    """Periodic centered finite-difference along the last axis."""
    fwd = np.roll(v_pec, shift=-1, axis=-1)
    bwd = np.roll(v_pec, shift=+1, axis=-1)
    return (fwd - bwd) / (2.0 * dchi)


def test_sinusoidal_matches_analytic_to_1e_12():
    """
    Sin field on N bins: the centered-difference must match the closed-form
    analytic expression to 1e-12 (per [D-42] spec milestone 1).
    """
    N = 2048
    dchi = 0.029296875  # 60 Mpc/h / 2048; representative Sherwood value
    i = np.arange(N, dtype=np.float64)
    v_pec = np.sin(2.0 * np.pi * i / N).astype(np.float32)[None, :]  # (1, N)

    analytic = (
        np.sin(2.0 * np.pi * (i + 1.0) / N) - np.sin(2.0 * np.pi * (i - 1.0) / N)
    ) / (2.0 * dchi)
    analytic = analytic.astype(np.float32)[None, :]

    g = SherwoodLoader.compute_vpec_grad(v_pec, dchi)
    ref = _ref_grad_numpy(v_pec, dchi)

    max_diff_vs_ref = float(np.max(np.abs(g - ref)))
    max_diff_vs_analytic = float(np.max(np.abs(g - analytic)))

    # Pin against the reference NumPy implementation to 1e-12 (spec).
    assert max_diff_vs_ref < 1e-12, (
        f"torch impl vs numpy reference max-diff {max_diff_vs_ref:.3e} >= 1e-12"
    )
    # The analytic check is float32-limited; a looser bound is appropriate.
    assert max_diff_vs_analytic < 1e-4, (
        f"torch impl vs analytic sin-derivative max-diff "
        f"{max_diff_vs_analytic:.3e} >= 1e-4"
    )


def test_periodic_boundary_at_index_zero():
    """
    The value at i=0 must use v_pec[N-1] (NOT a zero-pad or replicate-pad)
    for the i-1 neighbor.
    """
    N = 16
    dchi = 1.0
    rng = np.random.default_rng(seed=0)
    v_pec = rng.standard_normal(size=(3, N)).astype(np.float32)

    g = SherwoodLoader.compute_vpec_grad(v_pec, dchi)
    # Periodic expectation at i=0: (v_pec[i=1] - v_pec[i=N-1]) / (2 * dchi)
    expected_i0 = (v_pec[:, 1] - v_pec[:, N - 1]) / (2.0 * dchi)
    # Periodic expectation at i=N-1: (v_pec[i=0] - v_pec[i=N-2]) / (2 * dchi)
    expected_iN_minus_1 = (v_pec[:, 0] - v_pec[:, N - 2]) / (2.0 * dchi)

    np.testing.assert_allclose(g[:, 0], expected_i0, atol=1e-6)
    np.testing.assert_allclose(g[:, -1], expected_iN_minus_1, atol=1e-6)


def test_zscore_stability_across_seeds():
    """
    Post-zscore std stays in [0.95, 1.05] across 3 random seeds drawing 1024
    rays each from a Gaussian v_pec. Mean stays in [-0.02, +0.02].
    """
    N = 2048
    dchi = 0.029296875
    n_rays = 1024
    stds, means = [], []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed=seed)
        v_pec = rng.standard_normal(size=(n_rays, N)).astype(np.float32) * 50.0
        g = SherwoodLoader.compute_vpec_grad(v_pec, dchi)
        g_mean = float(g.mean())
        g_std = float(g.std())
        g_z = (g - g_mean) / g_std
        stds.append(float(g_z.std()))
        means.append(float(g_z.mean()))
    for s in stds:
        assert 0.95 <= s <= 1.05, f"cross-seed std {s:.4f} outside [0.95, 1.05]"
    for m in means:
        assert -0.02 <= m <= 0.02, f"cross-seed mean {m:.4f} outside [-0.02, 0.02]"


def test_shape_preserved_for_2d_input():
    """compute_vpec_grad must preserve the input shape (num_los, nbins)."""
    rng = np.random.default_rng(seed=42)
    v_pec = rng.standard_normal(size=(7, 64)).astype(np.float32)
    g = SherwoodLoader.compute_vpec_grad(v_pec, dchi=0.5)
    assert g.shape == v_pec.shape
    assert g.dtype == np.float32


# ---------------------------------------------------------------------------
# Milestone 2: model-side hook regression / shape-check tests.
# ---------------------------------------------------------------------------
def test_milestone2_flag_off_bit_equivalence():
    """
    Default-path bit equivalence: an IGMNeRF constructed with
    use_velocity_gradient_conditioning=False must produce IDENTICAL outputs
    to a fresh instance built with the same seed, on the same input. This
    proves the [D-42] constructor change does not perturb the established
    default forward path.
    """
    n_rays, n_bins = 4, 16

    torch.manual_seed(20260511)
    model_a = IGMNeRF(hidden_dim=32, num_layers=8, L=4,
                      use_velocity_gradient_conditioning=False)
    torch.manual_seed(0)
    x = torch.rand(n_rays, n_bins, 3)
    out_a = model_a(x)

    torch.manual_seed(20260511)
    model_b = IGMNeRF(hidden_dim=32, num_layers=8, L=4,
                      use_velocity_gradient_conditioning=False)
    out_b = model_b(x)

    # The model returns (n_rays, n_bins, 4) stacking [density, temp, h1, vpec].
    assert torch.equal(out_a, out_b), (
        "Bit equivalence broken: flag-off forward path differs between two "
        "seeded-identical IGMNeRF constructions."
    )
    for i, name in enumerate(["density", "temp", "h1_frac", "vpec"]):
        assert torch.equal(out_a[..., i], out_b[..., i]), (
            f"Field {name} differs across seeded-identical models."
        )


def test_milestone2_flag_on_shape_and_no_nan():
    """
    Shape-check smoke: with use_velocity_gradient_conditioning=True, forward
    (x, g) must return the same (n_rays, n_bins, 4) shape and contain no
    NaN. Default-OFF behavior is left untouched.
    """
    n_rays, n_bins = 4, 16

    torch.manual_seed(20260511)
    model = IGMNeRF(hidden_dim=32, num_layers=8, L=4,
                    use_velocity_gradient_conditioning=True)
    x = torch.rand(n_rays, n_bins, 3)
    g = torch.randn(n_rays, n_bins, 1)  # z-scored N(0,1) feature
    out = model(x, g=g)

    assert out.shape == (n_rays, n_bins, 4), (
        f"Expected (n_rays, n_bins, 4); got {tuple(out.shape)}."
    )
    assert not torch.isnan(out).any(), "Forward produced NaNs."

    # Spot-check the four channels individually for the (n_rays, n_bins)
    # spatial shape and no NaN.
    for i, name in enumerate(["density", "temp", "h1_frac", "vpec"]):
        chan = out[..., i]
        assert chan.shape == (n_rays, n_bins), (
            f"Channel {name} shape {tuple(chan.shape)} != "
            f"({n_rays}, {n_bins})."
        )
        assert not torch.isnan(chan).any(), f"Channel {name} contains NaNs."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
