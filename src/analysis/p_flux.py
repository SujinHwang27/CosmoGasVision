"""1D Lyman-alpha flux power spectrum P_F(k_||) — spec [D-13] gating module.

This is the canonical Lyα-forest community statistic (Walther+ 2018,
Boera+ 2019). The Stage 2b gating window per [D-13] is the inertial
range k_|| in [10^-2.5, 10^-1.5] s/km.

Conventions
-----------
- Input is the transmitted flux F = exp(-tau), shape (n_sightlines, n_bins).
- Velocity-axis FFT: k_|| has units of s/km (angular wavenumber 2*pi*f).
- Per-sightline mean subtraction before FFT (kills DC contamination).
- Hann-windowed periodogram with $dv/\\sum w^2$ normalization, matching
  Walther+ 2018 / Boera+ 2019 pipeline convention. The window suppresses
  spectral leakage from the periodic-FFT discontinuity at the sightline
  endpoints; the $\\sum w^2$ denominator compensates for the power lost
  to apodization so the resulting one-sided PSD has units of s/km.
- Output binning: log-spaced k bins between ``k_min`` and ``k_max``.

This module is intentionally NumPy-only (no SciPy required) and vectorized
over sightlines so that ``n_sightlines = 16384`` runs in seconds.
"""

from __future__ import annotations

import numpy as np


def compute_p_flux(
    F: np.ndarray,
    vel_axis_kms: np.ndarray,
    k_min: float = 10 ** -3,
    k_max: float = 10 ** -1,
    n_kbins: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 1D flux power spectrum averaged over sightlines.

    Parameters
    ----------
    F : (n_sightlines, n_bins) ndarray
        Transmitted flux F = exp(-tau) on a uniform velocity grid.
    vel_axis_kms : (n_bins,) ndarray
        Monotone, uniformly spaced velocity grid in km/s.
    k_min, k_max : float
        Log-bin edges for the output spectrum, in s/km. Defaults bracket
        the [D-13] inertial range [10^-2.5, 10^-1.5] s/km comfortably.
    n_kbins : int
        Number of log-spaced k bins between ``k_min`` and ``k_max``.

    Returns
    -------
    k_parallel_s_per_km : (n_kbins,) ndarray
        Geometric-mean centers of the log-k bins, in s/km.
    P_F_k_parallel : (n_kbins,) ndarray
        Sightline-averaged P_F(k_||) in s/km. Empty bins -> NaN.
    """
    F = np.asarray(F)
    vel_axis_kms = np.asarray(vel_axis_kms)
    if F.ndim != 2:
        raise ValueError(f"F must be 2D (n_sightlines, n_bins); got {F.shape}")
    n_sl, n_bins = F.shape
    if vel_axis_kms.shape != (n_bins,):
        raise ValueError(
            f"vel_axis_kms shape {vel_axis_kms.shape} != ({n_bins},)"
        )

    dv = float(vel_axis_kms[1] - vel_axis_kms[0])
    if dv <= 0:
        raise ValueError("vel_axis_kms must be strictly increasing")
    if not np.allclose(np.diff(vel_axis_kms), dv, rtol=1e-3):
        raise ValueError("vel_axis_kms must be uniformly spaced for FFT-based PSD")

    # normalized delta_F = F/<F> - 1; anchor-invariant under uniform F -> r*F
    # ([D-35] fix; the mean-subtracted form delta_F = F - <F> picks up an
    # overall r-scaling and breaks the [D-13] invariance gate). The Lyα
    # mean transmitted flux is bounded below by ~0.5 at z=0.3, so
    # divide-by-zero is structurally impossible — the two-line form is
    # kept for debuggability of F_mean if a future loader violates that.
    F_mean = F.mean(axis=1, keepdims=True)
    delta_F = F / F_mean - 1.0

    # Hann window apodization to suppress periodic-window leakage.
    # Walther+ 2018 / Boera+ 2019 convention. Normalization below uses
    # sum(window**2) (leakage compensation), not n_bins.
    window = np.hanning(n_bins)
    delta_F = delta_F * window[None, :]

    # Real FFT along the velocity (frequency) axis. Vectorized over rays.
    F_k = np.fft.rfft(delta_F, axis=1)

    # PSD normalization for a one-sided periodogram in s/km.
    # |F_k|^2 * dv / sum(window**2) gives the two-sided PSD; multiply
    # positive frequencies (excluding DC and Nyquist) by 2 for the
    # one-sided form.
    psd = (np.abs(F_k) ** 2) * (dv / np.sum(window ** 2))
    if n_bins % 2 == 0:
        psd[:, 1:-1] *= 2.0
    else:
        psd[:, 1:] *= 2.0

    # Average over sightlines.
    psd_mean = psd.mean(axis=0)

    f = np.fft.rfftfreq(n_bins, d=dv)
    # Angular wavenumber k_|| = 2*pi*f, units s/km. Matches Walther+ 2018
    # Fig. 5 / Boera+ 2019 convention. Ordinary-frequency f = k/(2*pi)
    # would shift inertial-range labels by log10(2*pi) = 0.798.
    k_axis = 2.0 * np.pi * f

    # Log-spaced binning over the requested k range.
    log_edges = np.linspace(np.log10(k_min), np.log10(k_max), n_kbins + 1)
    edges = 10 ** log_edges
    centers = 10 ** (0.5 * (log_edges[:-1] + log_edges[1:]))

    P_binned = np.full(n_kbins, np.nan, dtype=np.float64)
    valid = k_axis > 0
    k_pos = k_axis[valid]
    psd_pos = psd_mean[valid]

    # Vectorized binning via np.digitize + bincount.
    idx = np.digitize(k_pos, edges) - 1
    in_range = (idx >= 0) & (idx < n_kbins)
    idx_ir = idx[in_range]
    psd_ir = psd_pos[in_range]
    if idx_ir.size > 0:
        sums = np.bincount(idx_ir, weights=psd_ir, minlength=n_kbins)
        cnts = np.bincount(idx_ir, minlength=n_kbins)
        nz = cnts > 0
        P_binned[nz] = sums[nz] / cnts[nz]

    return centers, P_binned
