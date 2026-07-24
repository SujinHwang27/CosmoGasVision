"""[U-04] Stage-1 R4 (A4): random crop+ray sampler for the U-Net pair factory.

Spec of record: ``experiments/unet-inversion/design/u04_stage1_ratification.md``
SS2(a), commit 58ac831 (ratified). Duties:

* random 64^3 crop positions in the [D-49] train region (containment enforced
  by ``TruthCropProvider.crop_at`` — strict straddle rejection reused
  unchanged);
* per-example fresh random ray subset, count log-uniform in [64, 1024],
  drawn WITHOUT replacement from the rays that INTERSECT the crop;
* 90-degree transverse rotations/flips (transverse = the plane perpendicular
  to the [D-49] split axis; the split axis is never flipped);
* zero-ray crops rejected, rejection rate logged (counters on the sampler).

Geometry conventions (all reused, not re-derived):

* Sightlines are axis-aligned; ``iaxis`` in {1,2,3} = ray runs along x/y/z
  (``SherwoodLoader.get_world_coordinates``, src/data/loader.py:1338-1355).
* bin -> voxel along the ray axis: ``floor(pos_axis / pitch) % n_grid`` —
  the canonical [D-75] mapping (scripts/d75_corrected_metric_rescore.py:495).
* Transverse ray coordinates -> voxel: same ``floor(coord / pitch) % n_grid``.
* Crop transverse axes wrap periodically; the split axis does not (matches
  ``TruthCropProvider.crop_at``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from src.data.loader import distance_to_train_region
from src.data.truth_crop_provider import TruthCropProvider

# Production ray-count range (spec SS2(a)): log-uniform in [64, 1024].
N_RAYS_RANGE_DEFAULT: Tuple[int, int] = (64, 1024)


@dataclass(frozen=True)
class SightlineGeometry:
    """Voxelized geometry of an axis-aligned sightline set on an n_grid lattice.

    Attributes
    ----------
    axis : (n_rays,) int64 — ray direction, 0/1/2 (= loader iaxis - 1).
    voxel3 : (n_rays, 3) int64 — transverse voxel coords; the entry at the
        ray's own axis is a -1 sentinel (never read; overwritten in tests).
    bin_voxel_idx : (nbins,) int64 — voxel index along the ray axis for each
        spectral bin, ``floor(pos_axis / pitch) % n_grid`` (d75:495). Shared
        by all rays (one common ``pos_axis``).
    n_grid : int — lattice size (192 for Stage 1).
    """

    axis: np.ndarray
    voxel3: np.ndarray
    bin_voxel_idx: np.ndarray
    n_grid: int

    @property
    def n_rays(self) -> int:
        return int(self.axis.shape[0])


def geometry_from_sightlines(sl: Dict, n_grid: int = 192) -> SightlineGeometry:
    """Build ``SightlineGeometry`` from a ``SherwoodLoader.load_sightlines``
    dict (keys: header, iaxis, xaxis, yaxis, zaxis, pos_axis)."""
    box_kpc_h = float(sl["header"]["box_kpc_h"])
    pitch = box_kpc_h / float(n_grid)
    axis = np.asarray(sl["iaxis"], dtype=np.int64) - 1
    if axis.size and not np.isin(axis, (0, 1, 2)).all():
        raise ValueError("iaxis values outside {1,2,3}")
    coords = np.stack(
        [np.asarray(sl["xaxis"], dtype=np.float64),
         np.asarray(sl["yaxis"], dtype=np.float64),
         np.asarray(sl["zaxis"], dtype=np.float64)],
        axis=1,
    )
    voxel3 = np.floor(coords / pitch).astype(np.int64) % n_grid
    voxel3[np.arange(axis.size), axis] = -1  # own-axis sentinel
    bin_voxel_idx = (
        np.floor(np.asarray(sl["pos_axis"], dtype=np.float64) / pitch)
        .astype(np.int64) % n_grid
    )
    return SightlineGeometry(axis=axis, voxel3=voxel3,
                             bin_voxel_idx=bin_voxel_idx, n_grid=n_grid)


def intersecting_rays(
    geom: SightlineGeometry, corner: np.ndarray, crop_size: int
) -> np.ndarray:
    """Indices of rays whose voxel line passes through the crop at ``corner``.

    A ray along axis ``a`` spans the full box on ``a`` (pos_axis covers all
    voxels), so it intersects iff BOTH fixed transverse voxel coords land in
    the crop extent (periodic on transverse axes; the split axis interval is
    non-wrapping because provider corners satisfy corner+crop <= region end).
    """
    corner = np.asarray(corner, dtype=np.int64)
    loc = (geom.voxel3 - corner[None, :]) % geom.n_grid
    inside = loc < crop_size
    inside[np.arange(geom.n_rays), geom.axis] = True  # own axis: always spans
    return np.nonzero(inside.all(axis=1))[0]


@dataclass(frozen=True)
class SampleSpec:
    """One sampled training example (geometry only; rasterization is R5)."""

    corner: np.ndarray          # (3,) int64 absolute voxel corner
    ray_indices: np.ndarray     # (n_take,) int64 into the ray bundle
    n_rays_requested: int       # log-uniform draw in [64, 1024]
    n_rays_available: int       # intersecting-ray pool size for this crop
    rot_k: int                  # 0..3, 90-deg rotations in transverse plane
    flip: bool                  # flip of first transverse axis
    distance_to_train: float    # [D-49] distance at crop center (normalized)


class RayCropSampler:
    """Random (crop, ray-subset, augmentation) spec sampler over one physics.

    Stateless w.r.t. randomness: every draw consumes a caller-supplied
    ``np.random.Generator`` so datasets can derive per-index streams
    (deterministic under seed regardless of access order). Only the
    zero-ray-rejection counters are mutable state.
    """

    def __init__(
        self,
        provider: TruthCropProvider,
        geometry: SightlineGeometry,
        n_rays_range: Tuple[int, int] = N_RAYS_RANGE_DEFAULT,
        max_rejections: int = 10_000,
    ) -> None:
        if geometry.n_grid != provider.n_grid:
            raise ValueError(
                f"geometry n_grid {geometry.n_grid} != provider n_grid "
                f"{provider.n_grid}"
            )
        lo, hi = int(n_rays_range[0]), int(n_rays_range[1])
        if not (1 <= lo <= hi):
            raise ValueError(f"bad n_rays_range {n_rays_range!r}")
        self.provider = provider
        self.geometry = geometry
        self.n_rays_range = (lo, hi)
        self.max_rejections = int(max_rejections)
        # zero-ray rejection bookkeeping (spec: rejection rate logged)
        self.n_corner_draws = 0
        self.n_zero_ray_rejections = 0

    # ------------------------------------------------------------------ api

    @property
    def zero_ray_rejection_rate(self) -> float:
        if self.n_corner_draws == 0:
            return 0.0
        return self.n_zero_ray_rejections / self.n_corner_draws

    def draw_corner(self, rng: np.random.Generator) -> np.ndarray:
        """Uniform corner with split-axis containment by rejection (mirrors
        ``TruthCropProvider.sample``; transverse axes periodic)."""
        p = self.provider
        for _ in range(self.max_rejections):
            c = rng.integers(low=0, high=p.n_grid, size=3, dtype=np.int64)
            if p.corner_min <= int(c[p.scheme.axis]) <= p.corner_max_inclusive:
                return c
        raise RuntimeError(
            f"corner draw exceeded max_rejections={self.max_rejections}"
        )

    def sample_spec(
        self,
        rng: np.random.Generator,
        n_rays: Optional[int] = None,
    ) -> SampleSpec:
        """Draw one example spec. ``n_rays`` fixes the requested ray count
        (eval patterns / S4 stats); default is the log-uniform training draw.
        Zero-ray crops are rejected and redrawn (rate logged)."""
        p = self.provider
        for _ in range(self.max_rejections):
            corner = self.draw_corner(rng)
            self.n_corner_draws += 1
            pool = intersecting_rays(self.geometry, corner, p.crop_size)
            if pool.size > 0:
                break
            self.n_zero_ray_rejections += 1
        else:
            raise RuntimeError(
                f"zero-ray rejection exceeded max_rejections="
                f"{self.max_rejections} (n_rays in bundle: "
                f"{self.geometry.n_rays})"
            )
        if n_rays is None:
            lo, hi = self.n_rays_range
            n_req = int(np.round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
            n_req = int(np.clip(n_req, lo, hi))
        else:
            n_req = int(n_rays)
        n_take = min(n_req, pool.size)
        sel = rng.choice(pool.size, size=n_take, replace=False)
        ray_indices = np.sort(pool[sel])
        rot_k = int(rng.integers(0, 4))
        flip = bool(rng.integers(0, 2))
        center = (
            corner.astype(np.float64) + (p.crop_size - 1) / 2.0
        ) / float(p.n_grid)
        dist = float(
            distance_to_train_region(center[None, :], p.scheme)[0]
        )
        return SampleSpec(
            corner=corner,
            ray_indices=ray_indices,
            n_rays_requested=n_req,
            n_rays_available=int(pool.size),
            rot_k=rot_k,
            flip=flip,
            distance_to_train=dist,
        )
