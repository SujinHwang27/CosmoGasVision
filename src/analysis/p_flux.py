"""1D Lyman-alpha flux power spectrum P_F(k_||) — spec [D-13] gating module.

This is the canonical Lyα-forest community statistic (Walther+ 2018,
Boera+ 2019). The Stage 2b gating window per [D-13] is the inertial
range k_|| in [10^-2.5, 10^-1.5] s/km.

Conventions
-----------
- Input is the transmitted flux F = exp(-tau), shape (n_sightlines, n_bins).
- Velocity-axis FFT: k_|| has units of s/km (angular wavenumber 2*pi*f).
- Per-sightline mean subtraction before FFT (kills DC contamination).
- Window-power normalization yields a one-sided PSD in s/km.
- No window is applied (raw periodogram); apodization is the caller's
  responsibility — the spec asks for the standard FFT-and-square pipeline.
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

    # Per-sightline mean subtraction; standard for the Lyα flux power.
    delta_F = F - F.mean(axis=1, keepdims=True)

    # Real FFT along the velocity (frequency) axis. Vectorized over rays.
    F_k = np.fft.rfft(delta_F, axis=1)

    # PSD normalization for a one-sided periodogram in s/km.
    # |F_k|^2 * dv / n_bins gives the two-sided PSD; multiply positive
    # frequencies (excluding DC and Nyquist) by 2 for the one-sided form.
    psd = (np.abs(F_k) ** 2) * (dv / n_bins)
    if n_bins % 2 == 0:
        psd[:, 1:-1] *= 2.0
    else:
        psd[:, 1:] *= 2.0

    # Average over sightlines.
    psd_mean = psd.mean(axis=0)

    # Convert FFT frequency (cycles/(km/s)) -> angular wavenumber k (s/km).
    f = np.fft.rfftfreq(n_bins, d=dv)
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
