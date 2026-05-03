"""Spec tests for ``src.analysis.p_flux.compute_p_flux``.

Two unit checks per the dispatch spec:

1. **Flat spectrum**: feed white-noise sightlines (Gaussian fluctuations
   on a flat baseline) and assert P_F(k) is approximately k-independent
   over the [D-13] inertial range.

2. **Sinusoid**: feed a coherent cosine at a known angular wavenumber
   ``k0`` and assert the recovered spectrum spikes at that bin (i.e.
   the peak bin's k-center is the closest log-bin to ``k0``).

Both fixtures are sized for ~5 s wall-clock on a Windows CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.p_flux import compute_p_flux


def test_white_noise_is_flat():
    """White-noise flux fluctuations -> approximately flat P_F(k)."""
    rng = np.random.default_rng(0)
    # Larger n_bins gives finer k resolution; more sightlines averages out
    # chi-squared sampling noise per FFT bin. 512 x 4096 keeps the test
    # well under 5 s on a CPU.
    n_sl, n_bins = 512, 4096
    dv = 1.0  # km/s -> Nyquist k = pi ~ 3.14 s/km
    vel = np.arange(n_bins) * dv

    # Mean F = 0.7, sigma = 0.05 -> safely in (0, 1) and uncorrelated.
    F = 0.7 + 0.05 * rng.standard_normal((n_sl, n_bins))

    k, P = compute_p_flux(F, vel, k_min=10 ** -2.5, k_max=10 ** -1.5, n_kbins=8)

    valid = np.isfinite(P)
    assert valid.sum() >= 5, f"too few populated bins: {valid.sum()}"
    P_v = P[valid]
    # White noise: PSD should be roughly constant. Per-bin chi-squared(2)
    # sampling noise is ~1/sqrt(n_sl * bin_width) ~ 5-10%; allow factor 2.5.
    ratio = P_v.max() / P_v.min()
    assert ratio < 2.5, f"white-noise PSD too non-flat (max/min={ratio:.2f})"


def test_sinusoid_peaks_at_k0():
    """A coherent cosine at k0 -> peak bin centered near k0."""
    rng = np.random.default_rng(1)
    n_sl, n_bins = 64, 2048
    dv = 1.0  # km/s
    vel = np.arange(n_bins) * dv

    # Choose k0 inside the [D-13] inertial range, well clear of bin edges.
    k0 = 10 ** -2.0  # s/km
    # Add a small DC offset so F stays positive after the cosine swing.
    base = 0.5
    amp = 0.05
    cos = amp * np.cos(k0 * vel)
    # Tiny noise floor to avoid degenerate FFT bins.
    F = base + cos[None, :] + 1e-4 * rng.standard_normal((n_sl, n_bins))

    k, P = compute_p_flux(F, vel, k_min=10 ** -3, k_max=10 ** -1, n_kbins=20)

    valid_idx = np.where(np.isfinite(P))[0]
    assert valid_idx.size > 0
    # Pick the populated bin nearest k0 in log space, and the peak bin.
    nearest = valid_idx[np.argmin(np.abs(np.log10(k[valid_idx]) - np.log10(k0)))]
    peak = valid_idx[np.argmax(P[valid_idx])]
    # The peak must coincide with the bin that contains k0 (or a neighbor).
    assert abs(peak - nearest) <= 1, (
        f"sinusoid spike at k={k[peak]:.4g} but expected near k0={k0:.4g} "
        f"(nearest bin k={k[nearest]:.4g})"
    )
    # And it must dominate: peak power >> typical noise-floor power.
    others = np.delete(P[valid_idx], np.where(valid_idx == peak)[0])
    assert P[peak] > 10.0 * np.median(others), (
        f"sinusoid peak {P[peak]:.3g} not dominant over noise floor "
        f"(median other={np.median(others):.3g})"
    )
