"""Real-space density cross-correlation xi(r).

The Stark+ 2015 sparse-tomography reference metric: compute the cross-power
between predicted and ground-truth overdensity fields in Fourier space,
inverse-transform to real space, and bin into spherical shells of |r|.

The headline number for [D-13] is xi(r=2 h^-1 Mpc).
"""

from __future__ import annotations

import numpy as np


def compute_xi_cross(
    rho_pred: np.ndarray,
    rho_truth: np.ndarray,
    box_kpc_h: float,
    r_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Density-density cross-correlation function.

    Parameters
    ----------
    rho_pred, rho_truth : (N, N, N) ndarray
        Overdensity fields rho/<rho>. Must share the shape and box.
    box_kpc_h : float
        Comoving box length in kpc/h.
    r_bins : (n_bins+1,) ndarray
        Bin EDGES for |r| in h^-1 Mpc.

    Returns
    -------
    r_centers : (n_bins,) ndarray
        Geometric centers of the r bins (h^-1 Mpc).
    xi : (n_bins,) ndarray
        Mean cross-correlation in each shell. Empty bins -> NaN.
    """
    if rho_pred.shape != rho_truth.shape:
        raise ValueError(
            f"shape mismatch: pred {rho_pred.shape} vs truth {rho_truth.shape}"
        )
    if rho_pred.ndim != 3 or len(set(rho_pred.shape)) != 1:
        raise ValueError("inputs must be cubic 3D")
    N = rho_pred.shape[0]
    L = box_kpc_h / 1000.0  # Mpc/h

    # NORMALIZE both fields to zero-mean, unit-variance so xi is bounded
    # in [-1, 1] and the [D-13] threshold of 0.6 is well-defined.
    delta_pred = rho_pred - rho_pred.mean()
    delta_truth = rho_truth - rho_truth.mean()
    sp = delta_pred.std()
    st = delta_truth.std()
    if sp == 0 or st == 0:
        raise ValueError("zero-variance field, cross-correlation undefined")
    delta_pred = delta_pred / sp
    delta_truth = delta_truth / st

    # Cross-power (real part) -> inverse FFT yields the real-space xi(r).
    Fp = np.fft.fftn(delta_pred)
    Ft = np.fft.fftn(delta_truth)
    Px = (Fp * np.conj(Ft)).real
    # Volume normalization: <delta_p delta_t>(r) = (1/N^3) IFFT[F_p F_t*]
    xi_3d = np.fft.ifftn(Px).real / (N ** 3)

    # Build the |r| grid (in h^-1 Mpc) accounting for periodic wrap-around.
    cell = L / N
    coord = np.arange(N) * cell
    coord = np.where(coord > L / 2.0, coord - L, coord)  # signed offset
    rx, ry, rz = np.meshgrid(coord, coord, coord, indexing="ij")
    rmag = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

    # Spherical-shell binning
    n_bins = len(r_bins) - 1
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    flat_r = rmag.ravel()
    flat_xi = xi_3d.ravel()
    idx = np.digitize(flat_r, r_bins) - 1
    in_range = (idx >= 0) & (idx < n_bins)
    idx = idx[in_range]
    flat_xi = flat_xi[in_range]

    sum_xi = np.bincount(idx, weights=flat_xi, minlength=n_bins)
    cnt = np.bincount(idx, minlength=n_bins)
    xi = np.full(n_bins, np.nan, dtype=np.float64)
    nz = cnt > 0
    xi[nz] = sum_xi[nz] / cnt[nz]

    return r_centers, xi
