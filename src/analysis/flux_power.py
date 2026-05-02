"""1D Lyman-alpha flux power spectrum P_F(k_||) along the velocity axis.

This is the canonical Lyα-forest community statistic (Walther+ 2018,
Boera+ 2019). The Stage 2b success criterion [D-13] gates on
|delta P_F / P_F| < 10% averaged over k_|| in [10^-2.5, 10^-1.5] s/km.

Conventions:
- Velocity-axis FFT, so k_|| has units of s/km.
- Per-ray mean subtraction before windowing (removes the DC mode).
- Hann window via np.hanning(n_bins); window-power normalization
  divides by sum(window**2) and the bin spacing dv to yield s/km units.
- Real-FFT only, factor of 2 applied on positive-frequency bins
  except DC and Nyquist; matches the standard 1-sided PSD convention.
"""

from __future__ import annotations

import numpy as np


def compute_PF_1d(
    tau_arr: np.ndarray,
    vel_axis: np.ndarray,
    n_kbins: int = 20,
    k_range: tuple[float, float] = (1e-3, 1e-1),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 1D flux power spectrum averaged over rays.

    Parameters
    ----------
    tau_arr : (n_rays, n_bins) ndarray
        Per-ray HI Lyα optical depth profiles on a uniform velocity grid.
    vel_axis : (n_bins,) ndarray
        Monotone velocity grid in km/s. Used for the bin spacing dv.
    n_kbins : int
        Number of log-spaced k bins between ``k_range``.
    k_range : (k_min, k_max)
        Inclusive log-k bin edges in s/km. Default brackets the
        [D-13] inertial range [10^-2.5, 10^-1.5] s/km comfortably.

    Returns
    -------
    k_par : (n_kbins,) ndarray
        Geometric-mean bin centers in s/km.
    P_F : (n_kbins,) ndarray
        Ray-averaged power in s/km. Empty bins are set to NaN.
    """
    if tau_arr.ndim != 2:
        raise ValueError(f"tau_arr must be 2D (n_rays, n_bins); got {tau_arr.shape}")
    n_rays, n_bins = tau_arr.shape
    if vel_axis.shape != (n_bins,):
        raise ValueError(f"vel_axis shape {vel_axis.shape} != ({n_bins},)")

    # Flux from optical depth
    F = np.exp(-tau_arr)
    # Per-ray mean subtraction (kills DC; real Lyα analyses subtract <F>)
    F = F - F.mean(axis=1, keepdims=True)

    # Apodize to suppress periodic-window leakage
    window = np.hanning(n_bins)
    Fw = F * window[None, :]

    # Velocity-grid spacing in km/s
    dv = float(vel_axis[1] - vel_axis[0])
    if not np.allclose(np.diff(vel_axis), dv, rtol=1e-3):
        raise ValueError("vel_axis must be uniformly spaced for FFT-based PSD")

    # Real FFT along the velocity axis
    F_k = np.fft.rfft(Fw, axis=1)
    # PSD in s/km: |F_k|^2 * dv / sum(window**2) for two-sided -> need x2 factor
    # for one-sided (positive-freq only). Apply factor 2 except at DC/Nyquist.
    norm = dv / np.sum(window ** 2)
    psd = (np.abs(F_k) ** 2) * norm
    if n_bins % 2 == 0:
        # rfft has DC and Nyquist as one-sided; double the rest.
        psd[:, 1:-1] *= 2.0
    else:
        psd[:, 1:] *= 2.0

    # Ray-averaged PSD
    psd_mean = psd.mean(axis=0)

    # k-axis in s/km from rfftfreq (cycles/length); convert to angular if needed.
    # Convention here: P_F(k) with k = 2*pi*f follows the cosmological convention.
    f = np.fft.rfftfreq(n_bins, d=dv)        # cycles per km/s
    k_axis = 2.0 * np.pi * f                  # s/km (angular)

    # Log-space binning over k_range
    k_min, k_max = k_range
    log_edges = np.linspace(np.log10(k_min), np.log10(k_max), n_kbins + 1)
    edges = 10 ** log_edges
    centers = 10 ** (0.5 * (log_edges[:-1] + log_edges[1:]))

    P_binned = np.full(n_kbins, np.nan, dtype=np.float64)
    # Skip k=0 (DC) which has no log10 representation
    valid = k_axis > 0
    k_pos = k_axis[valid]
    psd_pos = psd_mean[valid]
    for i in range(n_kbins):
        mask = (k_pos >= edges[i]) & (k_pos < edges[i + 1])
        if mask.any():
            P_binned[i] = psd_pos[mask].mean()

    return centers, P_binned
