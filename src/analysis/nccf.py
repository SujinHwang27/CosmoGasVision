"""[D-75] v3 corrected-metric estimator library (spec of record:
experiments/nerf/design/d75_corrected_metric_rescore_spec_v3.md, commit 029f5b8).

Naming ruling (spec v3 §3 [K3]): the shell statistic
    NCCF(r) = C_xy(r) / sqrt(C_xx(r) * C_yy(r))
is the **normalized cross-correlation function** (NCCF), a configuration-space
coherence/stochasticity ratio (Dekel & Lahav 1999 lineage). It is NOT a lagged
Pearson coefficient and is not called one anywhere: Cauchy-Schwarz does not
bound the shell-averaged ratio (|NCCF| > 1 reachable), and every mean-subtracted
autocorrelation in a periodic box has a forced zero-crossing. The zero-lag
Pearson of smoothed fields, r_s(sigma), IS a true Pearson coefficient and keeps
that name.

Estimator contract (v3 §3):
- FFT-based, periodic box, mean-subtracted fields, float64 end-to-end.
- Ratio-of-shell-means: shell-mean C_xy, C_xx, C_yy are computed FIRST, the
  ratio second (pre-registered against mean-of-ratios).
- Validity domain: only shells with min(C_xx(r), C_yy(r)) > eps_d * C_xx(0),
  eps_d = 0.01 pinned; masked NaN otherwise.
- Degenerate-variance policy: std(x) < 1e-12 => statistic UNDEFINED (returned
  as None / NaN with an 'undefined' flag, never silently 0).

Legacy `src/analysis/cross_corr.py:compute_xi_pearson` is record-preserved and
NOT modified; this module lives alongside it.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

EPS_D_PINNED = 0.01          # v3 §3 validity-domain threshold (fraction of C_xx(0))
DEGENERATE_STD = 1e-12       # v3 §3 degenerate-variance policy


# --------------------------------------------------------------------------- #
# core FFT machinery
# --------------------------------------------------------------------------- #

def _check_cubes(x: np.ndarray, y: np.ndarray) -> int:
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.ndim != 3 or len(set(x.shape)) != 1:
        raise ValueError(f"inputs must be cubic 3D, got {x.shape}")
    return x.shape[0]


def periodic_r_grid(n: int, box_mpc_h: float) -> np.ndarray:
    """|r| lattice (h^-1 Mpc) with periodic (signed minimum-image) offsets."""
    cell = box_mpc_h / n
    coord = np.arange(n) * cell
    coord = np.where(coord > box_mpc_h / 2.0, coord - box_mpc_h, coord)
    rx, ry, rz = np.meshgrid(coord, coord, coord, indexing="ij")
    return np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)


def cross_corr_cube(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Real-space cross-covariance cube C_xy(r_vec) of mean-subtracted fields.

    C_xy(r_vec) = (1/N^3) sum_u dx(u) dy(u + r_vec), periodic; float64.
    """
    n = _check_cubes(x, y)
    dx = np.asarray(x, dtype=np.float64)
    dy = np.asarray(y, dtype=np.float64)
    dx = dx - dx.mean()
    dy = dy - dy.mean()
    Fx = np.fft.rfftn(dx)
    Fy = np.fft.rfftn(dy)
    # IFFT of conj(Fx)*Fy gives C(r) = <dx(u) dy(u+r)>; real by construction
    # for real inputs up to roundoff.
    c = np.fft.irfftn(np.conj(Fx) * Fy, s=dx.shape) / float(n) ** 3
    return c


def shell_bin(cube3d: np.ndarray, rmag: np.ndarray,
              r_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Shell-mean of a real-space lag cube over |r| bins. Returns (means, counts)."""
    n_bins = len(r_edges) - 1
    idx = np.digitize(rmag.ravel(), r_edges) - 1
    ok = (idx >= 0) & (idx < n_bins)
    idxo = idx[ok]
    vals = cube3d.ravel()[ok]
    s = np.bincount(idxo, weights=vals, minlength=n_bins)
    c = np.bincount(idxo, minlength=n_bins)
    means = np.full(n_bins, np.nan)
    nz = c > 0
    means[nz] = s[nz] / c[nz]
    return means, c


def default_r_edges() -> np.ndarray:
    """v3 §3 pinned NCCF bins: log-spaced, first edge 2*dx = 0.625 h^-1 Mpc
    (192^3 in a 60 Mpc/h box), upper edge 15 h^-1 Mpc."""
    return np.geomspace(0.625, 15.0, 16)


def nccf(x: np.ndarray, y: np.ndarray, box_mpc_h: float,
         r_edges: Optional[np.ndarray] = None,
         eps_d: float = EPS_D_PINNED) -> Dict:
    """Normalized cross-correlation function (ratio-of-shell-means, v3 §3).

    Returns dict with r_centers, nccf (NaN outside validity domain), shell
    means of C_xy/C_xx/C_yy, mode counts, validity mask, zero-lag values,
    zero-crossing radii of the shell-mean autocovariances, and the zero-lag
    Pearson (true Pearson at r=0).
    """
    n = _check_cubes(x, y)
    if r_edges is None:
        r_edges = default_r_edges()
    sx = float(np.asarray(x, dtype=np.float64).std())
    sy = float(np.asarray(y, dtype=np.float64).std())
    if sx < DEGENERATE_STD or sy < DEGENERATE_STD:
        return {"undefined": True,
                "reason": f"degenerate variance (std_x={sx:.3e}, std_y={sy:.3e})"}

    cxy = cross_corr_cube(x, y)
    cxx = cross_corr_cube(x, x)
    cyy = cross_corr_cube(y, y)
    rmag = periodic_r_grid(n, box_mpc_h)

    m_xy, cnt = shell_bin(cxy, rmag, r_edges)
    m_xx, _ = shell_bin(cxx, rmag, r_edges)
    m_yy, _ = shell_bin(cyy, rmag, r_edges)

    cxx0 = float(cxx[0, 0, 0])
    cyy0 = float(cyy[0, 0, 0])
    cxy0 = float(cxy[0, 0, 0])
    valid = (np.minimum(m_xx, m_yy) > eps_d * cxx0) & (cnt > 0)
    prof = np.full(len(m_xy), np.nan)
    prof[valid] = m_xy[valid] / np.sqrt(m_xx[valid] * m_yy[valid])

    return {
        "undefined": False,
        "r_edges": r_edges,
        "r_centers": np.sqrt(r_edges[:-1] * r_edges[1:]),   # geometric centers
        "nccf": prof,
        "shell_C_xy": m_xy, "shell_C_xx": m_xx, "shell_C_yy": m_yy,
        "mode_counts": cnt,
        "valid": valid,
        "eps_d": eps_d,
        "C_xx0": cxx0, "C_yy0": cyy0, "C_xy0": cxy0,
        "pearson_zero_lag": cxy0 / np.sqrt(cxx0 * cyy0),
        "r_zc_xx": shell_zero_crossing(cxx, rmag, box_mpc_h),
        "r_zc_yy": shell_zero_crossing(cyy, rmag, box_mpc_h),
    }


def shell_zero_crossing(corr_cube: np.ndarray, rmag: np.ndarray,
                        box_mpc_h: float, dr: float = 0.3125) -> float:
    """First radius (h^-1 Mpc) where the fine-binned shell-mean autocovariance
    goes <= 0. Fine linear bins of width dr (default one cell at 192^3)."""
    edges = np.arange(0.0, box_mpc_h / 2.0 + dr, dr)
    means, cnt = shell_bin(corr_cube, rmag, edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ok = cnt > 0
    m = means[ok]
    c = centers[ok]
    neg = np.where(m <= 0)[0]
    if len(neg) == 0:
        return float("nan")
    i = int(neg[0])
    if i == 0:
        return float(c[0])
    # linear interpolation between the last positive and first non-positive bin
    r0, r1 = c[i - 1], c[i]
    v0, v1 = m[i - 1], m[i]
    return float(r0 + (r1 - r0) * (v0 / (v0 - v1)))


# --------------------------------------------------------------------------- #
# smoothing + zero-lag statistics
# --------------------------------------------------------------------------- #

def k_grids(n: int, box_mpc_h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical k components (h/Mpc, angular, 2*pi*f) for rfftn layout."""
    kf = 2.0 * np.pi / box_mpc_h
    kx = np.fft.fftfreq(n, d=1.0 / n) * kf
    kz = np.fft.rfftfreq(n, d=1.0 / n) * kf
    return kx, kx.copy(), kz


def gaussian_smooth_periodic(x: np.ndarray, box_mpc_h: float,
                             sigma_mpc_h: float) -> np.ndarray:
    """Periodic FFT Gaussian smoothing, kernel exp(-k^2 sigma^2 / 2), float64."""
    xf = np.asarray(x, dtype=np.float64)
    n = xf.shape[0]
    kx, ky, kz = k_grids(n, box_mpc_h)
    F = np.fft.rfftn(xf)
    k2 = (kx[:, None, None] ** 2 + ky[None, :, None] ** 2
          + kz[None, None, :] ** 2)
    F *= np.exp(-0.5 * k2 * sigma_mpc_h ** 2)
    return np.fft.irfftn(F, s=xf.shape)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    da = a - a.mean()
    db = b - b.mean()
    sa = da.std()
    sb = db.std()
    if sa < DEGENERATE_STD or sb < DEGENERATE_STD:
        return float("nan")
    return float((da * db).mean() / (sa * sb))


def _rank(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks (float64), O(n log n)."""
    from scipy.stats import rankdata
    return rankdata(a, method="average")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(_rank(np.ravel(a)), _rank(np.ravel(b)))


# --------------------------------------------------------------------------- #
# Fourier-space coherence r(k)
# --------------------------------------------------------------------------- #

def default_k_edges(box_mpc_h: float = 60.0, k_max: float = 3.0,
                    n_bins: int = 16) -> np.ndarray:
    kf = 2.0 * np.pi / box_mpc_h
    return np.geomspace(kf * 0.999, k_max, n_bins + 1)


def rk_coherence(x: np.ndarray, y: np.ndarray, box_mpc_h: float,
                 k_edges: Optional[np.ndarray] = None) -> Dict:
    """r(k) = P_xy / sqrt(P_xx P_yy) per k shell (descriptive, v3 §3)."""
    n = _check_cubes(x, y)
    if k_edges is None:
        k_edges = default_k_edges(box_mpc_h)
    dx = np.asarray(x, dtype=np.float64)
    dy = np.asarray(y, dtype=np.float64)
    dx = dx - dx.mean()
    dy = dy - dy.mean()
    Fx = np.fft.rfftn(dx)
    Fy = np.fft.rfftn(dy)
    kx, ky, kz = k_grids(n, box_mpc_h)
    kmag = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2
                   + kz[None, None, :] ** 2)
    # rfftn double-count weights: interior kz planes represent +/- kz modes.
    w = np.full(Fx.shape, 2.0)
    w[..., 0] = 1.0
    if n % 2 == 0:
        w[..., -1] = 1.0
    pxy = (np.conj(Fx) * Fy).real * w
    pxx = (np.abs(Fx) ** 2) * w
    pyy = (np.abs(Fy) ** 2) * w

    idx = np.digitize(kmag.ravel(), k_edges) - 1
    n_bins = len(k_edges) - 1
    ok = (idx >= 0) & (idx < n_bins)
    idxo = idx[ok]
    sxy = np.bincount(idxo, weights=pxy.ravel()[ok], minlength=n_bins)
    sxx = np.bincount(idxo, weights=pxx.ravel()[ok], minlength=n_bins)
    syy = np.bincount(idxo, weights=pyy.ravel()[ok], minlength=n_bins)
    cnt = np.bincount(idxo, minlength=n_bins)
    rk = np.full(n_bins, np.nan)
    nz = (cnt > 0) & (sxx > 0) & (syy > 0)
    rk[nz] = sxy[nz] / np.sqrt(sxx[nz] * syy[nz])
    return {
        "k_edges": k_edges,
        "k_centers": np.sqrt(k_edges[:-1] * k_edges[1:]),
        "r_k": rk,
        "mode_counts": cnt,
    }


def rk_first_crossing(k_centers: np.ndarray, rk: np.ndarray,
                      level: float = 0.5) -> float:
    """First k (interpolated) where r(k) drops below `level`. NaN if never
    (within the band) or if already below at the first valid bin -> returns
    that first bin center."""
    ok = np.isfinite(rk)
    kc = k_centers[ok]
    rr = rk[ok]
    if len(rr) == 0:
        return float("nan")
    below = np.where(rr < level)[0]
    if len(below) == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(kc[0])
    k0, k1 = kc[i - 1], kc[i]
    v0, v1 = rr[i - 1], rr[i]
    return float(k0 + (k1 - k0) * ((v0 - level) / (v0 - v1)))


# --------------------------------------------------------------------------- #
# synthetic fields for the acceptance suite + controls (v3 §3, §6)
# --------------------------------------------------------------------------- #

def amplitude_matched_grf(amplitude: np.ndarray, seed: int,
                          shape: Tuple[int, int, int]) -> np.ndarray:
    """Random-phase real field with EXACTLY the given rfftn |amplitude|.

    Phases from the rfftn of a white-noise realization (guarantees the
    Hermitian symmetry a real field requires). DC mode is zeroed.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(shape)
    W = np.fft.rfftn(w)
    mag = np.abs(W)
    mag[mag == 0] = 1.0
    F = amplitude * (W / mag)
    F[0, 0, 0] = 0.0
    return np.fft.irfftn(F, s=shape)


def field_amplitude(x: np.ndarray) -> np.ndarray:
    """|rfftn| amplitude of the mean-subtracted field (measured P(k) carrier)."""
    dx = np.asarray(x, dtype=np.float64)
    return np.abs(np.fft.rfftn(dx - dx.mean()))


def phase_randomized(x: np.ndarray, seed: int) -> np.ndarray:
    """Control (d): identical |FFT| amplitudes, randomized phases; the exact
    null for 'right two-point statistics, wrong structure'. Mean restored."""
    out = amplitude_matched_grf(field_amplitude(x), seed, x.shape)
    return out + float(np.asarray(x, dtype=np.float64).mean())


def lowpass_sharp(x: np.ndarray, box_mpc_h: float, k_c: float) -> np.ndarray:
    """Control (e): sharp k-space cutoff retaining k <= k_c (h/Mpc)."""
    xf = np.asarray(x, dtype=np.float64)
    n = xf.shape[0]
    kx, ky, kz = k_grids(n, box_mpc_h)
    kmag = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2
                   + kz[None, None, :] ** 2)
    F = np.fft.rfftn(xf)
    F[kmag > k_c] = 0.0
    return np.fft.irfftn(F, s=xf.shape)


# --------------------------------------------------------------------------- #
# inference machinery (v3 §7): octants, Fisher-z paired t, block bootstrap
# --------------------------------------------------------------------------- #

def octant_slices(n: int):
    h = n // 2
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                yield (slice(i * h, (i + 1) * h),
                       slice(j * h, (j + 1) * h),
                       slice(k * h, (k + 1) * h))


def per_octant_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson computed WITHIN each of the 8 octants (v3 §7 B-ii cond 2)."""
    n = a.shape[0]
    return np.array([pearson(a[s], b[s]) for s in octant_slices(n)])


def fisher_z(r) -> np.ndarray:
    r = np.clip(np.asarray(r, dtype=np.float64), -1 + 1e-12, 1 - 1e-12)
    return np.arctanh(r)


def paired_fisher_t(r_a_oct: np.ndarray, r_b_oct: np.ndarray) -> Dict:
    """Paired t over 8 octants on Fisher-z transformed per-octant r_s.
    Threshold of record t >= t_7(0.975) = 2.365 (v3 §7 B-ii cond 2)."""
    d = fisher_z(r_a_oct) - fisher_z(r_b_oct)
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    t = m / (sd / np.sqrt(len(d))) if sd > 0 else float("inf") * np.sign(m or 1.0)
    return {"mean_dz": m, "sd_dz": sd, "t": float(t), "df": len(d) - 1,
            "t_crit_0975": 2.365, "pass": bool(t >= 2.365)}


def _block_sums(a: np.ndarray, b: np.ndarray, n_blocks_side: int):
    """Per-block sufficient statistics for fast block-bootstrap Pearson."""
    n = a.shape[0]
    bs = n // n_blocks_side
    a4 = a.reshape(n_blocks_side, bs, n_blocks_side, bs, n_blocks_side, bs)
    b4 = b.reshape(n_blocks_side, bs, n_blocks_side, bs, n_blocks_side, bs)
    ax = (1, 3, 5)
    return {
        "n_cell": float(bs ** 3),
        "sa": a4.sum(axis=ax).ravel(), "sb": b4.sum(axis=ax).ravel(),
        "saa": (a4 ** 2).sum(axis=ax).ravel(),
        "sbb": (b4 ** 2).sum(axis=ax).ravel(),
        "sab": (a4 * b4).sum(axis=ax).ravel(),
    }


def _pearson_from_sums(st, idx) -> float:
    ncell = st["n_cell"] * len(idx)
    sa = st["sa"][idx].sum(); sb = st["sb"][idx].sum()
    saa = st["saa"][idx].sum(); sbb = st["sbb"][idx].sum()
    sab = st["sab"][idx].sum()
    va = saa / ncell - (sa / ncell) ** 2
    vb = sbb / ncell - (sb / ncell) ** 2
    if va <= 0 or vb <= 0:
        return float("nan")
    return float((sab / ncell - sa * sb / ncell ** 2) / np.sqrt(va * vb))


def block_bootstrap_delta_rs(truth: np.ndarray, obj_a: np.ndarray,
                             obj_b: np.ndarray, n_blocks_side: int = 8,
                             n_boot: int = 1000, seed: int = 20260723) -> Dict:
    """Paired block bootstrap of Delta r_s = r(a,truth) - r(b,truth)
    (v3 §7 B-ii cond 3): resample sub-cube blocks with replacement, same
    resample for both objects; 95% CI must exclude 0."""
    st_a = _block_sums(truth, obj_a, n_blocks_side)
    st_b = _block_sums(truth, obj_b, n_blocks_side)
    nb = n_blocks_side ** 3
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, nb, nb)
        deltas[i] = (_pearson_from_sums(st_a, idx)
                     - _pearson_from_sums(st_b, idx))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"n_boot": n_boot, "n_blocks": nb,
            "block_cells_side": truth.shape[0] // n_blocks_side,
            "delta_mean": float(np.nanmean(deltas)),
            "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0)}
