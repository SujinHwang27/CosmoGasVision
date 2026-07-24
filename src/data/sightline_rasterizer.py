"""[U-04] Stage-1 R5 (A5): sightline -> voxel-crop rasterizer.

Spec of record: ``experiments/unet-inversion/design/u04_stage1_ratification.md``
SS2(b), commit 58ac831.

Input variable: **flux decrement delta_F = 1 - F**, NOT tau. Flux convention
is REUSED from the pipeline of record, ``F = exp(-tau)``
(src/analysis/p_flux.py:9,40; src/analysis/flux_power.py:52); tau is the
redshift-space half of the tauH1 file (``tau_h1``, the [D-24] observable).

Scale pin (SS2(b): fixed pre-registered GLOBAL constant; per-crop or
data-dependent normalization FORBIDDEN):

    DELTA_F_SCALE = 12.5

Measurement of record (scripts/u04_s1_delta_f_scale.py, recorded in
``experiments/unet-inversion/artifacts/stage1/r5_delta_f_scale.json``):
pooled std(delta_F) = 0.08202 over the first 1024 sightlines x all 4 physics
at z=0.3 (per-physics 0.0771/0.0911/0.0852/0.0731), so 1/std = 12.19 ->
round number 12.5. The spec's "~x50 expected" guess is superseded by the
measurement (disclosed deviation; the spec delegates the final value to this
pin).

Bin -> voxel (SS2(b), all reused conventions):
* along-axis voxel = ``floor(pos_axis / pitch) % n_grid`` (d75:495), carried
  by ``SightlineGeometry.bin_voxel_idx``;
* ~10.7 bins/voxel -> per-ray per-voxel value = MEAN of assigned bins;
* multiple rays through one voxel -> MEAN across rays; mask stays 1;
* ch1 = DELTA_F_SCALE * delta_F on ray-path voxels, 0 elsewhere;
  ch2 = binary ray mask. All three ray axes handled.
"""

from __future__ import annotations

import numpy as np

from src.data.unet_crop_sampler import SightlineGeometry

# Pinned global input scale (see module docstring + r5_delta_f_scale.json).
DELTA_F_SCALE = 12.5


def flux_decrement(tau: np.ndarray) -> np.ndarray:
    """delta_F = 1 - exp(-tau), float64. Reuses the pipeline flux convention
    (F = exp(-tau); src/analysis/p_flux.py, src/analysis/flux_power.py)."""
    return 1.0 - np.exp(-np.asarray(tau, dtype=np.float64))


def rasterize_crop(
    delta_f: np.ndarray,
    geom: SightlineGeometry,
    ray_indices: np.ndarray,
    corner: np.ndarray,
    crop_size: int = 64,
    scale: float = DELTA_F_SCALE,
) -> np.ndarray:
    """Rasterize a ray subset into a (2, cs, cs, cs) float32 input crop.

    Parameters
    ----------
    delta_f : (n_rays, nbins) float — flux decrement per spectral bin
        (any float dtype; accumulation is float64).
    geom : SightlineGeometry — axis/voxel geometry (same ray indexing).
    ray_indices : (m,) int — rays to deposit (from ``RayCropSampler``).
    corner : (3,) int — absolute voxel corner of the crop.
    crop_size, scale : see module docstring.

    Returns
    -------
    (2, cs, cs, cs) float32: ch0 = scale * mean delta_F (0 off-path),
    ch1 = binary ray mask.
    """
    delta_f = np.asarray(delta_f)
    ray_indices = np.asarray(ray_indices, dtype=np.int64)
    corner = np.asarray(corner, dtype=np.int64)
    cs = int(crop_size)
    n = geom.n_grid
    if delta_f.ndim != 2 or delta_f.shape[0] != geom.n_rays:
        raise ValueError(
            f"delta_f shape {delta_f.shape} incompatible with "
            f"{geom.n_rays} rays"
        )
    if delta_f.shape[1] != geom.bin_voxel_idx.shape[0]:
        raise ValueError("delta_f nbins != geometry bin_voxel_idx length")

    sum_cube = np.zeros((cs, cs, cs), dtype=np.float64)
    cnt_cube = np.zeros((cs, cs, cs), dtype=np.float64)

    # Axis-block layout: accumulate (transverse-flat, along) then transpose
    # into cube order. dims below are (other0, other1, along).
    _perm = {0: (2, 0, 1), 1: (0, 2, 1), 2: (0, 1, 2)}

    for a in range(3):
        rows = ray_indices[geom.axis[ray_indices] == a]
        if rows.size == 0:
            continue
        # ---- bin -> local along-axis voxel (shared pos_axis => shared map)
        loc_along = (geom.bin_voxel_idx - corner[a]) % n
        keep = np.nonzero(loc_along < cs)[0]
        order = np.argsort(loc_along[keep], kind="stable")
        cols = keep[order]
        lv = loc_along[cols]
        starts = np.searchsorted(lv, np.arange(cs))
        counts = np.diff(np.append(starts, lv.size))
        if counts.min() <= 0:
            raise ValueError(
                "spectral grid too coarse: some crop voxel gets zero bins "
                f"(nbins={lv.size + (loc_along.size - cols.size)}, "
                f"n_grid={n})"
            )
        # ---- per-ray per-voxel means (SS2(b): mean of assigned bins)
        vals = delta_f[rows][:, cols].astype(np.float64, copy=False)
        sums = np.add.reduceat(vals, starts, axis=1)          # (m, cs)
        means = sums / counts[None, :]
        # ---- deposit full 64-voxel lines at the transverse position
        o0, o1 = [ax for ax in range(3) if ax != a]
        l0 = (geom.voxel3[rows, o0] - corner[o0]) % n
        l1 = (geom.voxel3[rows, o1] - corner[o1]) % n
        if (l0 >= cs).any() or (l1 >= cs).any():
            raise ValueError(
                "ray outside crop passed to rasterize_crop; sampler must "
                "pre-filter with intersecting_rays"
            )
        flat = l0 * cs + l1
        acc_s = np.zeros((cs * cs, cs), dtype=np.float64)
        acc_n = np.zeros(cs * cs, dtype=np.float64)
        np.add.at(acc_s, flat, means)
        np.add.at(acc_n, flat, 1.0)
        perm = _perm[a]
        sum_cube += np.transpose(acc_s.reshape(cs, cs, cs), perm)
        cnt_cube += np.transpose(
            np.broadcast_to(acc_n.reshape(cs, cs, 1), (cs, cs, cs)), perm
        )

    mask = cnt_cube > 0
    ch0 = np.zeros((cs, cs, cs), dtype=np.float64)
    np.divide(sum_cube, cnt_cube, out=ch0, where=mask)
    ch0 *= float(scale)
    out = np.empty((2, cs, cs, cs), dtype=np.float32)
    out[0] = ch0.astype(np.float32)
    out[1] = mask.astype(np.float32)
    return out
