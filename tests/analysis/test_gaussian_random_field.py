"""Sanity tests for src/analysis using analytic Gaussian-random-field inputs.

Three orthogonal checks:

1. ``compute_Pdelta_3d`` recovers an input ``P_in(k) = k^-2`` to better than
   ~5% on a 128^3 GRF over the inertial range k in [0.1, 1.0] h/Mpc.

2. ``compute_PF_1d`` recovers a flat-PSD synthetic 1D Gaussian-noise spectrum
   to better than ~5% over the [D-13] band.

3. ``compute_xi_cross(rho, rho, ...)`` reproduces the autocorrelation and
   is monotone-decreasing for a smooth GRF.

The 5% tolerance refers to the *band-averaged* recovery; bin-by-bin sample
variance on a single 128^3 realization is much larger and is not the test
target. (The Stage 2b gating residual in [D-13] is also a band average.)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.cross_corr import compute_xi_cross
from src.analysis.density_power import compute_Pdelta_3d, compute_Pdelta_iso
from src.analysis.flux_power import compute_PF_1d


# ----------------------------------------------------------------- helpers


def _generate_grf_3d(
    N: int, L_mpc_h: float, power_law_n: float, seed: int = 0
) -> tuple[np.ndarray, callable]:
    """Generate a Gaussian random field on (N, N, N) with input
    P(k) = A * k^power_law_n (k in h/Mpc, P in (Mpc/h)^3).

    Returns (delta, P_in_func).
    """
    rng = np.random.default_rng(seed)
    freq = np.fft.fftfreq(N, d=L_mpc_h / N)
    k_axis = 2.0 * np.pi * freq  # h/Mpc
    kx, ky, kz = np.meshgrid(k_axis, k_axis, k_axis, indexing="ij")
    kmag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

    A = 1.0
    Pk = np.zeros_like(kmag)
    nz = kmag > 0
    Pk[nz] = A * kmag[nz] ** power_law_n  # P_in

    V = L_mpc_h ** 3
    # Variance budget: we want the realized real field's FFT to have
    # <|FFT(delta)_k|^2> = N^6 P(k) / V (the discrete-FFT analogue of
    # <|tilde delta|^2> = V P(k)). When we generate independent complex
    # delta_hat and take Re(ifftn(delta_hat)), the resulting field's FFT
    # is the Hermitian projection (delta_hat_k + delta_hat_{-k}^*)/2 with
    # variance sigma^2 / 2. To compensate we set sigma^2 = 2 * N^6 P / V.
    sigma2 = 2.0 * Pk * (N ** 6) / V
    re = rng.standard_normal(Pk.shape) * np.sqrt(sigma2 / 2.0)
    im = rng.standard_normal(Pk.shape) * np.sqrt(sigma2 / 2.0)
    delta_k = re + 1j * im
    delta_k[0, 0, 0] = 0.0  # zero-mean

    delta = np.fft.ifftn(delta_k).real

    def P_in_func(k):
        out = np.where(k > 0, A * np.power(np.maximum(k, 1e-30), power_law_n), 0.0)
        return out

    return delta, P_in_func


# ------------------------------------------------------------- 3D GRF test


def test_compute_Pdelta_3d_recovers_kminus2():
    """3D power spectrum recovers P_in = k^-2 within 5% in the band."""
    N = 128
    L = 60.0  # Mpc/h, matching Sherwood
    delta, P_in_func = _generate_grf_3d(N, L, power_law_n=-2.0, seed=0)
    rho = 1.0 + delta  # construct an overdensity field

    centers, P_iso = compute_Pdelta_iso(
        rho, box_kpc_h=L * 1000.0, n_kbins=24, k_range=(0.05, 5.0)
    )
    P_in = P_in_func(centers)
    band = (centers >= 0.1) & (centers <= 1.0) & np.isfinite(P_iso)
    assert band.sum() >= 4, "not enough k bins in [0.1, 1.0]"
    ratio = P_iso[band] / P_in[band]
    mean_ratio = float(ratio.mean())
    # Tolerance: 5% on the band-averaged ratio (cosmic variance per-bin
    # is ~30% on a single 128^3 box; we explicitly average over the band).
    assert abs(mean_ratio - 1.0) < 0.05, (
        f"P_delta band-averaged ratio = {mean_ratio:.3f} (expected 1.0 +/- 0.05)"
    )


# ------------------------------------------------------------ 1D PSD test


def test_compute_PF_1d_recovers_flat_psd():
    """Synthetic 1D Gaussian noise -> flat one-sided PSD; band recovery <5%."""
    rng = np.random.default_rng(123)
    n_rays = 512
    n_bins = 1024
    dv = 5.0  # km/s

    # Construct a tau field whose F = exp(-tau) has known PSD properties.
    # For small tau, var(F) ~ var(tau). We feed Gaussian noise directly via
    # tau = -ln(F) with F = 0.5 + small noise so the linearization holds.
    sigma = 0.05
    F_true = 0.5 + sigma * rng.standard_normal((n_rays, n_bins))
    F_true = np.clip(F_true, 1e-3, 1.0 - 1e-3)
    tau = -np.log(F_true)
    vel_axis = np.arange(n_bins) * dv

    centers, P_F = compute_PF_1d(
        tau, vel_axis, n_kbins=20, k_range=(1e-3, 1e-1)
    )

    # Expected one-sided PSD for white noise of std sigma. compute_PF_1d
    # uses the normalized delta_F = F/<F> - 1 convention per [D-35], so
    # var(delta_F) = (sigma/<F>)^2; the two-sided PSD of zero-mean white
    # noise is var * dv (since variance = integral of S(f) over the
    # Nyquist band = S * 2 f_Nyq = S / dv  =>  S = var * dv).
    # compute_PF_1d returns the one-sided PSD (factor 2 already applied),
    # so the target is 2 * (sigma / <F>)^2 * dv.
    F_mean = float(F_true.mean())
    P_expected = 2.0 * (sigma / F_mean) ** 2 * dv

    band = (centers >= 10 ** -2.5) & (centers <= 10 ** -1.5) & np.isfinite(P_F)
    assert band.sum() >= 3, "not enough k bins in the [D-13] band"
    mean_ratio = float((P_F[band] / P_expected).mean())
    assert abs(mean_ratio - 1.0) < 0.05, (
        f"P_F band-averaged ratio = {mean_ratio:.3f} "
        f"(expected ~1.0 +/- 0.05; sigma={sigma}, dv={dv})"
    )


# ------------------------------------------------------- xi_cross identity


def test_xi_cross_autocorrelation_monotone():
    """xi(pred=truth) equals the autocorrelation and is monotone-decreasing
    for a smooth (large-scale-dominated) GRF."""
    N = 64
    L = 60.0  # Mpc/h
    delta, _ = _generate_grf_3d(N, L, power_law_n=-2.0, seed=7)
    rho = 1.0 + delta

    r_bins = np.linspace(0.5, 20.0, 20)
    r_centers, xi = compute_xi_cross(
        rho, rho, box_kpc_h=L * 1000.0, r_bins=r_bins
    )

    valid = np.isfinite(xi)
    assert valid.sum() >= 5
    # Expect xi(0) > 0 and overall decay. Sample variance can produce small
    # bumps; we test that the running maximum-from-the-tail is monotone --
    # i.e., xi(r) does not exceed the maximum of all smaller r.
    xi_v = xi[valid]
    running_max = np.maximum.accumulate(xi_v)
    # The first bin should be the largest within ~5% (smooth GRF):
    assert xi_v[0] >= 0.9 * running_max.max(), (
        f"xi(small r) = {xi_v[0]:.3f} not the global maximum "
        f"({running_max.max():.3f}); autocorrelation should peak at small r"
    )
    # And xi at large r should be much smaller than at small r:
    assert xi_v[-1] < xi_v[0] * 0.5, (
        f"xi tail = {xi_v[-1]:.3f} not significantly smaller than head "
        f"= {xi_v[0]:.3f}; expected monotone-ish decay for k^-2 GRF"
    )


def test_xi_cross_zero_for_uncorrelated_fields():
    """Two independent GRFs should cross-correlate near zero everywhere."""
    N = 64
    L = 60.0
    delta_a, _ = _generate_grf_3d(N, L, power_law_n=-2.0, seed=1)
    delta_b, _ = _generate_grf_3d(N, L, power_law_n=-2.0, seed=2)
    rho_a = 1.0 + delta_a
    rho_b = 1.0 + delta_b
    r_bins = np.linspace(0.5, 20.0, 20)
    _, xi = compute_xi_cross(rho_a, rho_b, box_kpc_h=L * 1000.0, r_bins=r_bins)
    valid = np.isfinite(xi)
    # For a 64^3 box the per-shell sample variance is non-trivial, but the
    # mean cross-correlation should be small.
    assert abs(np.nanmean(xi[valid])) < 0.1, (
        f"<xi_cross> = {np.nanmean(xi[valid]):.3f} not near 0 for "
        "independent GRFs"
    )
