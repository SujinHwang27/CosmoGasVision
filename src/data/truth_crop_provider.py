"""[U-04] Stage-1 R3 (A3): truth-crop provider over the [D-49] split at n=192.

Serves 64^3 crops of a 192^3 x-transformed truth cube, restricted to one
[D-49] region. Spec of record:
``experiments/unet-inversion/design/u04_stage1_ratification.md`` §2(a)/(c),
commit 58ac831.

Design constraints honored here:

* **x-transform is exactly the [D-75] scoring variable** — replicated from
  ``scripts/d75_corrected_metric_rescore.py:196-201``:
  ``x = log10(max(rho/<rho>, 1e-3))``, float64, floor applied BEFORE the log.
  The unit-test contract is <= 1e-12 agreement (achieved bit-exactly since
  the expression is identical).
* **[D-49] split logic is reused, not forked**: region geometry comes from
  ``src.data.loader.region_voxel_interval`` (the same helper
  ``SherwoodLoader.extract_rho_crops_split`` delegates to) together with
  ``HeldoutSplitScheme`` / ``distance_to_train_region``. Rejection sampling
  mirrors ``extract_rho_crops_split`` (one RNG advanced through accepts and
  rejects; split axis non-wrapping by acceptance; transverse axes periodic).
* **n_grid=192 vs 768**: ``HeldoutSplitScheme`` is fraction-based (train
  ``[0, 0.7)``, val ``[0.7, 0.85)``, test ``[0.85, 1.0)`` in normalized box
  coords) and therefore lattice-independent; ``region_voxel_interval``
  truncates via ``int()`` at each lattice. At n=192: train=[0,134),
  val=[134,163), test=[163,192); at n=768: train=[0,537), val=[537,652),
  test=[652,768). Consequences at 192 pitch with crop 64: train corners on
  axis 0 lie in [0, 70] (71 distinct offsets, matching the ~70 of the spec);
  val (width 29), test (width 29) and heldout (width 58) CANNOT contain a
  64^3 crop — the constructor raises, by design. Test-region evaluation is
  sliding-window full-box inference masked to test voxels (spec §2(a)), not
  crop-based.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from src.data.loader import (
    DEFAULT_SCHEME,
    HeldoutSplitScheme,
    Region,
    distance_to_train_region,
    region_voxel_interval,
)

# [D-75] v3 §3 unified hard floor on rho/<rho> before log10
# (scripts/d75_corrected_metric_rescore.py:48).
X_FLOOR = 1.0e-3


def x_transform(rho: np.ndarray) -> Tuple[np.ndarray, float]:
    """[D-75] v3 §3 scoring variable: x = log10(max(rho/<rho>, 1e-3)), float64.

    Verbatim replica of ``scripts/d75_corrected_metric_rescore.py:196-201``.
    Returns ``(x, clamped_fraction)``.
    """
    r = np.asarray(rho, dtype=np.float64)
    clamped = float((r < X_FLOOR).mean())
    return np.log10(np.maximum(r, X_FLOOR)), clamped


class TruthCropProvider:
    """Region-restricted 64^3 crop provider over a 192^3 truth cube.

    Parameters
    ----------
    cube : np.ndarray (n, n, n) or str/Path to a ``.npy``
        RAW rho/<rho> cube (e.g. ``truth_real_192.npy`` or the R2
        ``truth_real_192_p{2,3,4}.npy``). The x-transform is applied here —
        do NOT pre-transform.
    region : {"train", "val", "test", "heldout"}
        [D-49] partition the crops must be wholly contained in. Strict
        straddle rejection: a crop whose split-axis voxel range crosses a
        region boundary is rejected, never relabelled.
    crop_size : int, default 64
        Cube side length in voxels (spec §2(a): 64 at the 192 lattice).
    scheme : HeldoutSplitScheme, default DEFAULT_SCHEME (70/15/15, axis 0)
    seed : int, default 42
        Seeds a ``np.random.default_rng``; same seed => byte-identical
        sample sequence.
    """

    def __init__(
        self,
        cube: Union[np.ndarray, str, Path],
        region: Region = "train",
        crop_size: int = 64,
        scheme: HeldoutSplitScheme = DEFAULT_SCHEME,
        seed: int = 42,
    ) -> None:
        if isinstance(cube, (str, Path)):
            cube = np.load(cube)
        cube = np.asarray(cube)
        if cube.ndim != 3 or len(set(cube.shape)) != 1:
            raise ValueError(f"cube must be (n, n, n); got shape {cube.shape}")
        self.n_grid = int(cube.shape[0])
        if not isinstance(crop_size, int) or crop_size <= 0:
            raise ValueError(f"crop_size must be a positive int; got {crop_size!r}")
        if crop_size > self.n_grid:
            raise ValueError(
                f"crop_size {crop_size} exceeds n_grid {self.n_grid}"
            )
        # NaN guard (loader-track validation contract): the truth producers
        # are NaN-free by construction; a NaN here means a corrupt artifact.
        if not np.isfinite(cube).all():
            raise ValueError("cube contains non-finite values; refusing")
        if float(cube.min()) < 0.0:
            raise ValueError("cube contains negative rho/<rho>; refusing")

        self.region: Region = region
        self.crop_size = crop_size
        self.scheme = scheme
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        # x-transformed cube, float64, exact scoring variable.
        self.x_cube, self.clamped_fraction = x_transform(cube)

        # Region interval + corner acceptance interval on the split axis
        # (shared [D-49] helper; validates region/scheme).
        self.region_start, self.region_end = region_voxel_interval(
            region, self.n_grid, scheme
        )
        self.corner_min = self.region_start
        self.corner_max_inclusive = self.region_end - crop_size
        if self.corner_max_inclusive < self.corner_min:
            raise ValueError(
                f"crop_size {crop_size} too large for region {region!r}: "
                f"region width {self.region_end - self.region_start} voxels "
                f"at n_grid={self.n_grid}, scheme={scheme!r}. (At n=192 with "
                f"crop 64 only 'train' admits crops; test-region eval is "
                f"sliding-window, spec §2(a).)"
            )

    # ------------------------------------------------------------------ api

    def n_axis_offsets(self) -> int:
        """Number of distinct split-axis corner offsets (diversity metric;
        71 for train at n=192 / crop 64)."""
        return self.corner_max_inclusive - self.corner_min + 1

    def crop_at(self, corner: np.ndarray) -> np.ndarray:
        """Deterministic crop with CIC-corner ``corner`` (3 ints, voxel
        units). Split axis must be region-contained; transverse axes wrap
        periodically. Returns float64 (crop_size,)*3 view-copy of x."""
        corner = np.asarray(corner, dtype=np.int64)
        if corner.shape != (3,):
            raise ValueError(f"corner must be shape (3,); got {corner.shape}")
        split_c = int(corner[self.scheme.axis])
        if not (self.corner_min <= split_c <= self.corner_max_inclusive):
            raise ValueError(
                f"corner {split_c} on axis {self.scheme.axis} straddles or "
                f"exits region {self.region!r} "
                f"[{self.region_start}, {self.region_end}) with crop "
                f"{self.crop_size} — strict rejection, never relabel"
            )
        offset = np.arange(self.crop_size, dtype=np.int64)
        idx = [(int(corner[a]) + offset) % self.n_grid for a in range(3)]
        return self.x_cube[np.ix_(*idx)]

    def sample(
        self,
        n_crops: int,
        max_rejections: int = 100_000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw ``n_crops`` region-contained crops by rejection sampling.

        Mirrors ``extract_rho_crops_split``: corners uniform on
        ``[0, n_grid)^3``; accepted iff the split-axis corner lies in
        ``[corner_min, corner_max_inclusive]``; the single RNG advances
        through both accepts and rejects (determinism under seed).

        Returns
        -------
        crops : np.ndarray (n_crops, crop_size, crop_size, crop_size) float64
        corners : np.ndarray (n_crops, 3) int64 — absolute voxel corners
        distances : np.ndarray (n_crops,) float32 —
            ``distance_to_train_region`` at each crop CENTER (normalized).
        """
        if not isinstance(n_crops, int) or n_crops <= 0:
            raise ValueError(f"n_crops must be a positive int; got {n_crops!r}")
        accepted = np.empty((n_crops, 3), dtype=np.int64)
        n_accept = 0
        n_rej = 0
        while n_accept < n_crops:
            c = self._rng.integers(low=0, high=self.n_grid, size=3, dtype=np.int64)
            split_c = int(c[self.scheme.axis])
            if self.corner_min <= split_c <= self.corner_max_inclusive:
                accepted[n_accept] = c
                n_accept += 1
            else:
                n_rej += 1
                if n_rej > max_rejections:
                    raise RuntimeError(
                        f"rejection sampling exceeded max_rejections="
                        f"{max_rejections}; region={self.region!r}, "
                        f"crop_size={self.crop_size}, n_grid={self.n_grid}"
                    )
        crops = np.empty((n_crops,) + (self.crop_size,) * 3, dtype=np.float64)
        for i in range(n_crops):
            crops[i] = self.crop_at(accepted[i])
        centers = (
            accepted.astype(np.float64) + (self.crop_size - 1) / 2.0
        ) / float(self.n_grid)
        distances = distance_to_train_region(centers, self.scheme).astype(
            np.float32
        )
        return crops, accepted, distances
