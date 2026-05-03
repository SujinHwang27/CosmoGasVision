"""1D Lyman-alpha flux power spectrum P_F(k_||) along the velocity axis.

Legacy tau-input wrapper around the canonical
:func:`src.analysis.p_flux.compute_p_flux` gating function. Preserved
because ``stage2b_report.py`` calls ``compute_PF_1d(tau, vel_axis)``
directly. The two routines now share an identical Hann-windowed
periodogram pipeline (Walther+ 2018 / Boera+ 2019) — see ``p_flux.py``
for the convention details.
"""

from __future__ import annotations

import numpy as np

from .p_flux import compute_p_flux


def compute_PF_1d(
    tau_arr: np.ndarray,
    vel_axis: np.ndarray,
    n_kbins: int = 20,
    k_range: tuple[float, float] = (1e-3, 1e-1),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 1D flux power spectrum averaged over rays (tau input).

    Thin wrapper that converts optical depth -> transmitted flux and
    delegates to :func:`src.analysis.p_flux.compute_p_flux`. Both share
    the Hann-windowed periodogram with $dv/\\sum w^2$ normalization and
    angular wavenumber $k = 2\\pi f$.

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
    F = np.exp(-tau_arr)
    return compute_p_flux(
        F, vel_axis, k_min=k_range[0], k_max=k_range[1], n_kbins=n_kbins
    )
