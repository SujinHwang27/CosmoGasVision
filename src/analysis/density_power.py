"""3D density auto-power spectrum P_delta(k_||, k_perp).

Probes whether the reconstructed field has the right anisotropic
structure: tomography should recover transverse modes even where the
sightlines are sparse, and the (k_||, k_perp) decomposition is the
diagnostic that exposes this. See LEDGER §5.

Conventions:
- Input ``rho_grid`` is rho/<rho> (already mean-divided).
- delta = rho - 1.
- ``k_par`` is the "z" axis by convention (last index); ``k_perp`` is
  the magnitude of the (k_x, k_y) pair. This matches the typical
  Ly-alpha-forest tomography geometry where the redshift direction is
  treated as the line-of-sight.
- k axis converted from cycles/(kpc/h) to h/Mpc via 2*pi * 1000 (i.e.,
  multiply np.fft.fftfreq by 2*pi to get angular wavenumber, then
  multiply by 1000 to convert kpc^-1 -> Mpc^-1; the factor is folded
  in below).
"""

from __future__ import annotations

import numpy as np


def compute_Pdelta_3d(
    rho_grid: np.ndarray,
    box_kpc_h: float,
    n_kpar: int = 16,
    n_kperp: int = 16,
    k_range: tuple[float, float] = (0.05, 5.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cylindrically binned 3D power spectrum.

    Parameters
    ----------
    rho_grid : (N, N, N) ndarray
        Overdensity rho/<rho>. Must have mean ~1; we subtract 1 to form
        delta and FFT.
    box_kpc_h : float
        Comoving box length in kpc/h. The Sherwood 60-Mpc/h box is 60000.
    n_kpar, n_kperp : int
        Number of log bins along each axis.
    k_range : (k_min, k_max) in h/Mpc
        Common log-k range used for both axes.

    Returns
    -------
    k_par_centers : (n_kpar,)
    k_perp_centers : (n_kperp,)
    P : (n_kpar, n_kperp)
        Volume-averaged power in (h/Mpc)^-3 (i.e., (Mpc/h)^3).
        Empty bins -> NaN.
    """
    if rho_grid.ndim != 3 or len(set(rho_grid.shape)) != 1:
        raise ValueError(f"rho_grid must be cubic 3D; got shape {rho_grid.shape}")
    N = rho_grid.shape[0]

    delta = rho_grid - 1.0
    delta_k = np.fft.fftn(delta)

    # Volume normalization: P(k) = |delta_k|^2 * (V / N^6) where V = L^3,
    # N^3 cells, FFT convention is the unnormalized forward transform.
    # Equivalent: |delta_k * (L/N)^3|^2 / V.
    L_mpc_h = box_kpc_h / 1000.0  # Mpc/h
    V = L_mpc_h ** 3
    cell_vol = V / (N ** 3)
    pk_3d = (np.abs(delta_k) ** 2) * (cell_vol ** 2) / V  # (Mpc/h)^3

    # k-axes in h/Mpc (angular wavenumber convention)
    # np.fft.fftfreq returns cycles/length when d is in length units.
    # Multiplying by 2*pi gives angular k.
    freq = np.fft.fftfreq(N, d=L_mpc_h / N)  # cycles per Mpc/h
    k_axis = 2.0 * np.pi * freq               # h/Mpc

    kx, ky, kz = np.meshgrid(k_axis, k_axis, k_axis, indexing="ij")
    k_par = np.abs(kz)
    k_perp = np.sqrt(kx ** 2 + ky ** 2)

    # Log bins, common range
    k_min, k_max = k_range
    par_edges = 10 ** np.linspace(np.log10(k_min), np.log10(k_max), n_kpar + 1)
    perp_edges = 10 ** np.linspace(np.log10(k_min), np.log10(k_max), n_kperp + 1)
    par_centers = np.sqrt(par_edges[:-1] * par_edges[1:])
    perp_centers = np.sqrt(perp_edges[:-1] * perp_edges[1:])

    # Drop DC and any cell outside the bin range
    flat_pk = pk_3d.ravel()
    flat_kpar = k_par.ravel()
    flat_kperp = k_perp.ravel()
    keep = (flat_kpar > 0) | (flat_kperp > 0)
    flat_pk = flat_pk[keep]
    flat_kpar = flat_kpar[keep]
    flat_kperp = flat_kperp[keep]

    # 2D digitize
    par_idx = np.digitize(flat_kpar, par_edges) - 1
    perp_idx = np.digitize(flat_kperp, perp_edges) - 1
    in_range = (
        (par_idx >= 0) & (par_idx < n_kpar) &
        (perp_idx >= 0) & (perp_idx < n_kperp)
    )
    par_idx = par_idx[in_range]
    perp_idx = perp_idx[in_range]
    flat_pk = flat_pk[in_range]

    P = np.full((n_kpar, n_kperp), np.nan, dtype=np.float64)
    # Sum and count via flat 2D index
    flat_bin = par_idx * n_kperp + perp_idx
    sum_pk = np.bincount(flat_bin, weights=flat_pk, minlength=n_kpar * n_kperp)
    cnt = np.bincount(flat_bin, minlength=n_kpar * n_kperp)
    nz = cnt > 0
    P_flat = np.full(n_kpar * n_kperp, np.nan, dtype=np.float64)
    P_flat[nz] = sum_pk[nz] / cnt[nz]
    P = P_flat.reshape(n_kpar, n_kperp)

    return par_centers, perp_centers, P


def compute_Pdelta_iso(
    rho_grid: np.ndarray,
    box_kpc_h: float,
    n_kbins: int = 24,
    k_range: tuple[float, float] = (0.05, 5.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic 1D-binned 3D power, useful for the GRF sanity test."""
    if rho_grid.ndim != 3:
        raise ValueError("rho_grid must be 3D")
    N = rho_grid.shape[0]
    delta = rho_grid - rho_grid.mean()
    delta_k = np.fft.fftn(delta)
    L = box_kpc_h / 1000.0
    V = L ** 3
    cell_vol = V / (N ** 3)
    pk_3d = (np.abs(delta_k) ** 2) * (cell_vol ** 2) / V

    freq = np.fft.fftfreq(N, d=L / N)
    k_axis = 2.0 * np.pi * freq
    kx, ky, kz = np.meshgrid(k_axis, k_axis, k_axis, indexing="ij")
    kmag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

    edges = 10 ** np.linspace(np.log10(k_range[0]), np.log10(k_range[1]), n_kbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    flat_k = kmag.ravel()
    flat_p = pk_3d.ravel()
    keep = flat_k > 0
    flat_k = flat_k[keep]
    flat_p = flat_p[keep]

    idx = np.digitize(flat_k, edges) - 1
    in_range = (idx >= 0) & (idx < n_kbins)
    idx = idx[in_range]
    flat_p = flat_p[in_range]

    sum_p = np.bincount(idx, weights=flat_p, minlength=n_kbins)
    cnt = np.bincount(idx, minlength=n_kbins)
    P = np.full(n_kbins, np.nan)
    nz = cnt > 0
    P[nz] = sum_p[nz] / cnt[nz]
    return centers, P
