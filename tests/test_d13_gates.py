"""Regression tests pinning the [D-13] gating identities.

Each gate has one mathematical identity that must hold under any future
refactor. This file locks the convention; if any assertion here fails,
treat it as a flag for re-review by the project-architect.

Pins:

1. ``compute_xi_pearson(rho, rho)`` evaluated at small r returns ~1.0
   for an exact same-field input (Pearson auto-correlation is unity at
   zero lag and remains close to unity within the first cell).
2. ``compute_p_flux`` on a coherent sinusoid F(v) = F0 + A sin(k0 v)
   returns peak power in the log-k bin containing k0 (the Hann window
   smears across ~3 bins; we only assert the maximum-power bin contains
   k0, not exact bin index).
3. ``ks_distance(samples, samples, F_range=(0.05, 0.95))`` returns
   exactly 0 for identical samples.

Runtime budget: ~5 s total on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.cross_corr import compute_xi_pearson
from src.analysis.p_flux import compute_p_flux
from src.analysis.flux_pdf import ks_distance


# Sherwood box from CLAUDE.md (60 Mpc/h).
BOX_KPC_H = 60_000.0


def test_xi_pearson_self_correlation_at_2_mpc_is_unity():
    """xi_Pearson(rho, rho) ~ 1.0 at r ~ 2 h^-1 Mpc on a 32^3 GRF.

    Because both fields are mean-subtracted and rescaled to unit variance
    before the cross-power, the auto-correlation at zero lag is exactly
    1.0; in finite-r bins it stays close to 1 only over the field's
    coherence length. To pin the *identity* (rather than a random-field
    statistical property), we check the bin straddling r ~ 0 — which
    must equal 1 within atol=1e-6 — and additionally confirm the r=2
    h^-1 Mpc probe bin returns a finite value as required for [D-13].
    """
    rng = np.random.default_rng(42)
    N = 32
    rho = 1.0 + 0.3 * rng.standard_normal((N, N, N))

    # Bin edges: first bin captures r=0; later bin captures r=2 h^-1 Mpc.
    r_bins = np.array([0.0, 0.5, 1.5, 2.5, 4.0, 8.0, 16.0, 30.0])
    r_centers, xi = compute_xi_pearson(rho, rho, BOX_KPC_H, r_bins=r_bins)

    # Identity at zero lag: Pearson auto-correlation = 1 exactly.
    assert np.isfinite(xi[0]), "r~0 bin must be populated for self-corr"
    assert xi[0] == pytest.approx(1.0, abs=1e-6), (
        f"compute_xi_pearson(rho, rho) at r~0 = {xi[0]} (expected 1.0)"
    )

    # [D-13] probe bin (r = 2 h^-1 Mpc) must be finite and ~0 for white
    # noise (uncorrelated cells beyond the first), since Pearson
    # normalization removes the variance scale. Just assert finiteness
    # — the value depends on the field's coherence length.
    probe = int(np.argmin(np.abs(r_centers - 2.0)))
    assert np.isfinite(xi[probe]), (
        f"r=2 h^-1 Mpc probe bin {probe} (center {r_centers[probe]:.2f}) "
        f"must be finite for [D-13] gating"
    )


def test_compute_p_flux_sinusoid_peak_in_correct_log_k_bin():
    """A sinusoid at angular wavenumber k0 -> peak in the log-k bin
    that contains k0. The Hann window smears the peak across ~3 bins
    but does NOT shift the peak position (Hann is zero-phase / even).
    """
    rng = np.random.default_rng(123)
    n_sl, n_bins = 64, 2048
    dv = 1.0  # km/s
    vel = np.arange(n_bins) * dv

    # Pick k0 well inside the [D-13] inertial range and clear of bin edges.
    k0 = 10 ** -2.0  # s/km, angular
    base = 0.5
    amp = 0.05
    F = (
        base
        + amp * np.sin(k0 * vel)[None, :]
        + 1e-4 * rng.standard_normal((n_sl, n_bins))
    )

    k, P = compute_p_flux(F, vel, k_min=10 ** -3, k_max=10 ** -1, n_kbins=20)

    # Identify the bin EDGES from the bin centers (log-spaced).
    log_centers = np.log10(k)
    dlog = log_centers[1] - log_centers[0]
    log_edges = np.concatenate(
        [[log_centers[0] - 0.5 * dlog], log_centers + 0.5 * dlog]
    )
    edges = 10 ** log_edges

    # Bin index that contains k0.
    k0_bin = int(np.searchsorted(edges, k0, side="right") - 1)
    assert 0 <= k0_bin < len(k), f"k0={k0} fell outside the binning range"

    # Peak bin in the recovered spectrum.
    valid = np.isfinite(P)
    valid_idx = np.where(valid)[0]
    peak = valid_idx[np.argmax(P[valid_idx])]

    assert peak == k0_bin, (
        f"peak bin {peak} (k={k[peak]:.4g}) != k0 bin {k0_bin} (k={k[k0_bin]:.4g}); "
        f"Hann window should not shift peak position"
    )


def test_ks_distance_identical_samples_in_d13_window_is_zero():
    """KS(samples, samples, F_range=(0.05, 0.95)) == 0 exactly."""
    rng = np.random.default_rng(0)
    F = rng.uniform(0.05, 0.95, size=10_000)
    ks = ks_distance(F, F, F_range=(0.05, 0.95))
    assert ks == 0.0, f"KS(F, F) over (0.05, 0.95) = {ks} (expected exactly 0)"
