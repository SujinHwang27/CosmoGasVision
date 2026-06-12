"""Stark+2015 / Lee+2018 (CLAMATO)-style Wiener-filter density reconstruction.

A linear (non-neural) classical baseline for IGM tomography. Maps observed
Lyman-alpha flux contrast delta_F sampled along sightlines to a 3D density
estimate on a regular grid, using an assumed signal + noise covariance.

Method (Stark+ 2015 Eq. 12-13; Lee+ 2014/2018 CLAMATO; Caucci+ 2008
Wiener-filter tomography)
-------------------------------------------------------------------------
The reconstructed field on a set of map voxels ``m`` from data pixels ``d`` is

    rho_rec(m) = C_mm @ (C_dd + N)^-1 @ d                          (Wiener, MMSE)

where, in the *covariance-model* CLAMATO formulation actually used in survey
practice (Stark+2015 §3.1; the matrices are NOT built from a measured P(k)
but from a prescribed two-point correlation kernel):

    C_dd(i,j) = sigma_signal^2 * K(r_perp_ij, r_para_ij)   pixel-pixel signal cov
    C_mm(m,i) = sigma_signal^2 * K(r_perp_mi, r_para_mi)   voxel-pixel  cross cov
    K(r_perp, r_para) = exp(-r_perp^2 / (2 L_perp^2))
                        * exp(-r_para^2 / (2 L_para^2))      Gaussian kernel
    N = sigma_noise^2 * I                                    diagonal pixel noise

The Gaussian-correlation kernel with separate transverse / line-of-sight
correlation lengths (``L_perp``, ``L_para``) is the standard CLAMATO choice
(Stark+2015 use L_perp = L_para ≈ a few Mpc/h). The data ``d`` is the flux
overdensity ``delta_F = F/<F> - 1`` mapped (Gunn-Peterson sign convention) to
a density tracer: high density -> high absorption -> low F -> negative
delta_F, so the natural density tracer is ``-delta_F``. We fit a single global
linear gain ``b`` (a "bias", absorbing tau_amp/FGPA slope) by least squares
against the truth on a HELD-OUT calibration so the cross-correlation metric is
gain-invariant; the reported xi/Pearson metrics are themselves gain-invariant
(Pearson is scale-free), so the gain only affects the power-ratio diagnostic.

Idealization (stated honestly per [D-37])
-----------------------------------------
- **Noiseless mock**: sigma_noise is a small regularizer (default 1e-3 of the
  signal variance), NOT a survey-realistic continuum-fitting + photon noise
  model. This is the idealized-covariance / best-case classical reconstruction.
- **Gaussian-kernel prior**, NOT a measured nonlinear matter P(k); the
  correlation lengths are prescribed, not fit to the simulation.
- **Real-space pixel positions**: we place each sightline pixel at its
  comoving (x,y,z) using the loader world coordinates; redshift-space
  distortions in the data are NOT deconvolved (no RSD treatment), matching the
  "map the observed redshift-space flux directly" CLAMATO convention.
- The reconstruction lives on the same regular grid as the ground-truth rho
  cube so the [D-13] 3D xi_{rho_hat,rho} estimator can score it directly.

References
----------
- Stark et al. 2015, MNRAS 453, 311 (Wiener IGM tomography forecast).
- Lee et al. 2014 ApJL 788 L49; Lee et al. 2018 ApJS 237 31 (CLAMATO).
- Caucci et al. 2008 MNRAS 386, 211 (Wiener-filter Lya tomography).
- experiments/nerf/LEDGER.md [D-13] (gate metrics), [D-36] (xi bar provenance).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


@dataclass
class WienerConfig:
    L_perp_mpc_h: float = 2.0      # transverse correlation length (Mpc/h)
    L_para_mpc_h: float = 2.0      # line-of-sight correlation length (Mpc/h)
    noise_rel: float = 0.05        # sigma_noise^2 / sigma_signal^2 (Wiener noise floor)
    pixel_stride: int = 64         # subsample pixels along each ray (cost control)
    cg_tol: float = 1e-5           # CG relative tolerance for the (C_dd+N) solve
    cg_maxiter: int = 4000


def _gaussian_kernel(dperp2, dpara2, L_perp, L_para):
    return np.exp(-0.5 * dperp2 / (L_perp ** 2)) * np.exp(-0.5 * dpara2 / (L_para ** 2))


def wiener_reconstruct(
    pixel_xyz_mpc_h: np.ndarray,   # (Npix, 3) comoving pixel positions, Mpc/h
    pixel_data: np.ndarray,        # (Npix,) density tracer = -(delta_F)
    voxel_xyz_mpc_h: np.ndarray,   # (Nvox, 3) map voxel centers, Mpc/h
    box_mpc_h: float,
    cfg: WienerConfig,
) -> np.ndarray:
    """Return the Wiener density-tracer estimate at each voxel (Nvox,).

    Periodic minimum-image convention is used for all separations (the
    Sherwood box is periodic). Signal variance is normalized to 1 (the
    Pearson/xi metrics are gain-invariant); the returned field is the
    dimensionless Wiener tracer, to be compared to (rho/<rho> - 1) up to a
    global linear gain.
    """
    Npix = pixel_xyz_mpc_h.shape[0]
    half = box_mpc_h / 2.0
    px = pixel_xyz_mpc_h
    # Isotropic kernel (L_perp == L_para is the common CLAMATO choice); when
    # they differ we cannot cleanly split perp/para for a mixed-axis sightline
    # set, so we use the isotropic 3D Gaussian with the mean L and treat all
    # separations equally. Documented idealization.
    L = 0.5 * (cfg.L_perp_mpc_h + cfg.L_para_mpc_h)
    inv2L2 = 0.5 / (L ** 2)

    # Solve (C_dd + N) w = d ; reconstruction = C_md @ w. C_dd+N is SPD and
    # well-conditioned for a finite noise floor (cond ~ lambda_max/noise_rel),
    # so a direct dense factorization is robust and fast for Npix up to ~12k.
    # Above that we fall back to matrix-free CG to bound RAM.
    DIRECT_MAX = 12000
    if Npix <= DIRECT_MAX:
        dx = px[:, None, 0] - px[None, :, 0]
        dy = px[:, None, 1] - px[None, :, 1]
        dz = px[:, None, 2] - px[None, :, 2]
        for d in (dx, dy, dz):
            d[d > half] -= box_mpc_h
            d[d < -half] += box_mpc_h
        A = np.exp(-(dx ** 2 + dy ** 2 + dz ** 2) * inv2L2)
        A[np.diag_indices(Npix)] += cfg.noise_rel
        w = np.linalg.solve(A, pixel_data)
    else:
        def _apply_Cdd(vec):
            out = np.empty(Npix, dtype=np.float64)
            blk = 2048
            for s in range(0, Npix, blk):
                e = min(s + blk, Npix)
                ddx = px[s:e, None, 0] - px[None, :, 0]
                ddy = px[s:e, None, 1] - px[None, :, 1]
                ddz = px[s:e, None, 2] - px[None, :, 2]
                for d in (ddx, ddy, ddz):
                    d[d > half] -= box_mpc_h
                    d[d < -half] += box_mpc_h
                K = np.exp(-(ddx ** 2 + ddy ** 2 + ddz ** 2) * inv2L2)
                out[s:e] = K @ vec
            out += cfg.noise_rel * vec
            return out
        op = LinearOperator((Npix, Npix), matvec=_apply_Cdd, dtype=np.float64)
        w, info = cg(op, pixel_data, rtol=cfg.cg_tol, maxiter=cfg.cg_maxiter)
        if info != 0:
            print(f"[wiener] WARNING: CG did not converge cleanly (info={info}).")

    # --- C_md: voxel-pixel cross covariance, applied row-block-wise to bound RAM ---
    Nvox = voxel_xyz_mpc_h.shape[0]
    rec = np.empty(Nvox, dtype=np.float64)
    block = 4096
    for s in range(0, Nvox, block):
        e = min(s + block, Nvox)
        vx = voxel_xyz_mpc_h[s:e]
        mdx = vx[:, None, 0] - pixel_xyz_mpc_h[None, :, 0]
        mdy = vx[:, None, 1] - pixel_xyz_mpc_h[None, :, 1]
        mdz = vx[:, None, 2] - pixel_xyz_mpc_h[None, :, 2]
        for d in (mdx, mdy, mdz):
            d[d > half] -= box_mpc_h
            d[d < -half] += box_mpc_h
        mr2 = mdx ** 2 + mdy ** 2 + mdz ** 2
        C_md = np.exp(-0.5 * mr2 / (L ** 2))
        # np.errstate guards a spurious numpy-2.0 BLAS matmul over/underflow
        # warning; C_md in [0,1] and w finite, so the product is finite.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            rec[s:e] = C_md @ w
    if not np.all(np.isfinite(rec)):
        raise FloatingPointError(
            "wiener_reconstruct produced non-finite voxels; check conditioning."
        )
    return rec
