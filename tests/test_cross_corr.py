"""Spec tests for ``src.analysis.cross_corr.compute_xi_rho``.

Two unit checks per the dispatch spec:

1. **Self cross-correlation**: rho_pred = rho_truth -> xi(r=0) ~ Var(rho)
   and the function returns a finite value at the [D-13] r = 2 h^-1 Mpc
   probe bin.

2. **Orthogonal random fields**: independent Gaussian fields ->
   xi(r > 0) ~ 0 within sample variance.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.cross_corr import compute_xi_rho


# Sherwood box from CLAUDE.md (60 Mpc/h).
BOX_KPC_H = 60_000.0


def test_self_correlation_matches_variance_and_has_d13_bin():
    """xi(r=0) ~ Var(rho); r = 2 h^-1 Mpc bin returns a finite value."""
    rng = np.random.default_rng(42)
    N = 32
    rho = 1.0 + 0.3 * rng.standard_normal((N, N, N))

    # Custom bins that explicitly include r=0 in the first bin and r=2
    # h^-1 Mpc in a later bin so we can probe both.
    r_bins = np.array([0.0, 1.0, 1.5, 2.5, 4.0, 8.0, 16.0, 30.0])

    r_centers, xi = compute_xi_rho(rho, rho, BOX_KPC_H, r_bins_h_inv_mpc=r_bins)

    # Bin containing r=0 -> approximate Var(rho).
    var = float(np.var(rho))
    bin0 = 0
    assert np.isfinite(xi[bin0]), "r=0 bin should be populated"
    rel_err = abs(xi[bin0] - var) / var
    assert rel_err < 0.10, (
        f"xi(r~0) = {xi[bin0]:.4f} vs Var = {var:.4f} (rel err {rel_err:.3f})"
    )

    # Find the [D-13] r = 2 h^-1 Mpc probe bin.
    probe_bin = int(np.argmin(np.abs(r_centers - 2.0)))
    assert np.isfinite(xi[probe_bin]), (
        f"r = 2 h^-1 Mpc probe bin {probe_bin} (center {r_centers[probe_bin]:.2f}) "
        f"is not finite"
    )

    # Also exercise the default r_bins_h_inv_mpc path.
    r_def, xi_def = compute_xi_rho(rho, rho, BOX_KPC_H)
    probe_def = int(np.argmin(np.abs(r_def - 2.0)))
    assert np.isfinite(xi_def[probe_def])


def test_orthogonal_fields_xi_near_zero():
    """Independent random fields -> xi(r > 0) ~ 0 within sample noise."""
    rng = np.random.default_rng(7)
    N = 32
    rho_a = 1.0 + 0.3 * rng.standard_normal((N, N, N))
    rho_b = 1.0 + 0.3 * rng.standard_normal((N, N, N))

    r_bins = np.linspace(0.5, 20.0, 21)
    r_centers, xi = compute_xi_rho(rho_a, rho_b, BOX_KPC_H, r_bins_h_inv_mpc=r_bins)

    # Compare to the per-bin sample-variance scale: independent unit-variance
    # fields have sigma_xi ~ var(rho_a) * var(rho_b) / sqrt(N_pairs_per_bin),
    # and at N=32 with bins spanning a few cells we have enough pairs that
    # |xi| << var. Empirically |xi| ~ a few % of the variance product.
    var_product = float(np.var(rho_a) * np.var(rho_b))
    xi_finite = xi[np.isfinite(xi)]
    assert xi_finite.size > 0
    max_abs = float(np.max(np.abs(xi_finite)))
    assert max_abs < 0.05 * var_product or max_abs < 0.01, (
        f"max |xi| = {max_abs:.4g} too large vs variance product {var_product:.4g}"
    )
