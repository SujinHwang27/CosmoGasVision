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
    cg_maxiter: int = 20000
    # [D-73] A4': sparse-kernel cutoff. The Gaussian kernel decays as
    # exp(-r^2/2L^2); beyond r_cut = sparse_n_sigma * L the entries are
    # < exp(-sparse_n_sigma^2/2) and dropped. This turns the O(Npix^2)-per-
    # matvec dense CG into an O(Npix * neighbours) sparse CG, the only tractable
    # path at >=70 px/ray (Npix ~ 76k) on CPU.
    #
    # SPD-SAFETY (PROBE-7): a Gaussian truncated mid-body is NOT positive-
    # definite -> CG (which requires SPD) fails to converge and xi collapses.
    # The truncation is SPD-safe only when the kernel value AT the cutoff is
    # <= the noise-floor diagonal (which regularizes the dropped tail). With
    # noise_rel n, the safe radius is r_cut >= L*sqrt(2*ln(1/n)); for n=1e-3
    # that is 3.72*L, so n_sigma=4 (kernel(4L)=3.4e-4 < 1e-3) is SPD-safe.
    # An ABSOLUTE r_cut cap below this radius is FORBIDDEN -- it truncates the
    # kernel body, breaks SPD, and destroys convergence (observed L>=4 with a
    # 12 Mpc/h cap -> info=4000, xi -> ~0). So no absolute cap is applied; cost
    # is bounded instead by the L-sweep range (large L is genuinely expensive).
    sparse_kernel: bool = False
    sparse_n_sigma: float = 5.0
    # Jacobi (diagonal) preconditioner for the sparse CG. The (C_dd+N) diagonal
    # is 1+noise_rel (constant), so the Jacobi preconditioner is a near-identity
    # scaling; included for robustness against the wider-kernel (larger-L)
    # conditioning growth.
    use_jacobi_precond: bool = True


def _gaussian_kernel(dperp2, dpara2, L_perp, L_para):
    return np.exp(-0.5 * dperp2 / (L_perp ** 2)) * np.exp(-0.5 * dpara2 / (L_para ** 2))


def _build_sparse_Cdd(px, box_mpc_h, L, noise_rel, n_sigma):
    """Sparse (C_dd + N) as a float32 CSR matrix using a KD-tree neighbour cutoff.

    Periodic minimum-image is handled by cKDTree(boxsize=...). Returns the
    (Npix, Npix) symmetric CSR Gaussian-kernel signal covariance truncated at
    r_cut = n_sigma * L, plus the noise_rel diagonal. SPD-safe cutoff (see
    WienerConfig). scipy's C-level sparse_distance_matrix builds the neighbour
    list; the COO data is recast to the kernel and dropped to a float32 CSR to
    halve RAM (the COO row/col int arrays are the transient peak).
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix, identity

    Npix = px.shape[0]
    inv2L2 = 0.5 / (L ** 2)
    r_cut = n_sigma * L
    pxw = np.mod(px, box_mpc_h)
    tree = cKDTree(pxw, boxsize=box_mpc_h)
    coo = tree.sparse_distance_matrix(tree, r_cut, output_type="coo_matrix")
    data = np.exp(-(coo.data ** 2) * inv2L2)
    A = csr_matrix((data, (coo.row, coo.col)), shape=(Npix, Npix))
    del coo, data
    A = A + (noise_rel * identity(Npix, format="csr", dtype=np.float64))
    return A


def _build_sparse_Cmd(vox, px, box_mpc_h, L, n_sigma):
    """Sparse voxel-pixel cross covariance C_md as float32 CSR, KD-tree cutoff."""
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix

    Nvox = vox.shape[0]
    Npix = px.shape[0]
    inv2L2 = 0.5 / (L ** 2)
    r_cut = n_sigma * L
    voxw = np.mod(vox, box_mpc_h)
    pxw = np.mod(px, box_mpc_h)
    tree_px = cKDTree(pxw, boxsize=box_mpc_h)
    tree_vox = cKDTree(voxw, boxsize=box_mpc_h)
    coo = tree_vox.sparse_distance_matrix(tree_px, r_cut, output_type="coo_matrix")
    data = np.exp(-(coo.data ** 2) * inv2L2)
    return csr_matrix((data, (coo.row, coo.col)), shape=(Nvox, Npix))


def wiener_reconstruct(
    pixel_xyz_mpc_h: np.ndarray,   # (Npix, 3) comoving pixel positions, Mpc/h
    pixel_data: np.ndarray,        # (Npix,) density tracer = -(delta_F)
    voxel_xyz_mpc_h: np.ndarray,   # (Nvox, 3) map voxel centers, Mpc/h
    box_mpc_h: float,
    cfg: WienerConfig,
    return_info: bool = False,
):
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
    # CG convergence flag (PROBE-7): 0 = converged for the direct path (no CG
    # invoked); set by scipy.sparse.linalg.cg on the matrix-free path. A
    # non-zero info means a non-converged solve, which biases xi LOW.
    cg_info = 0
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
    if cfg.sparse_kernel:
        # [D-73] A4' sparse-CG path: KD-tree-truncated Gaussian kernel solved
        # with scipy CG. Tractable at >=70 px/ray (Npix ~ 76k) on CPU; matvec
        # cost is O(Npix * neighbours), not O(Npix^2).
        A = _build_sparse_Cdd(px, box_mpc_h, L, cfg.noise_rel, cfg.sparse_n_sigma)
        M = None
        if cfg.use_jacobi_precond:
            from scipy.sparse import diags
            diag = A.diagonal()
            diag = np.where(diag != 0, diag, 1.0)
            M = diags(1.0 / diag, format="csr")
        w, info = cg(A, pixel_data, rtol=cfg.cg_tol, maxiter=cfg.cg_maxiter, M=M)
        cg_info = int(info)
        if info != 0:
            print(f"[wiener] WARNING: sparse-CG did not converge (info={info}).")
        C_md = _build_sparse_Cmd(
            voxel_xyz_mpc_h, px, box_mpc_h, L, cfg.sparse_n_sigma,
        )
        rec = np.asarray(C_md @ w).ravel()
        if not np.all(np.isfinite(rec)):
            raise FloatingPointError(
                "wiener_reconstruct (sparse) produced non-finite voxels."
            )
        if return_info:
            return rec, cg_info
        return rec
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
        cg_info = int(info)
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
    if return_info:
        return rec, cg_info
    return rec
