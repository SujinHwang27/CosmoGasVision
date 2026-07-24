"""[U-04] Stage-1 R6 (A6): (input, target) pair dataset for the U-Net track.

Spec of record: ``experiments/unet-inversion/design/u04_stage1_ratification.md``
SS2(a)-(c), commit 58ac831. Yields:

* input  : torch.float32 (2, 64, 64, 64) — ch0 scaled delta_F on ray voxels,
  ch1 binary ray mask (R5 rasterizer, pinned DELTA_F_SCALE);
* target : torch.float32 (1, 64, 64, 64) — x = log10(max(rho/<rho>, 1e-3)),
  the exact [D-75] scoring variable via ``TruthCropProvider`` (float64
  transform, cast to float32 at emission only).

Determinism: per-index RNG ``np.random.default_rng([seed, index])`` — the
same (seed, index) always yields the same example regardless of access
order or epoch. Multi-physics interleaving: physics cycles with
``index % n_sources``.

Augmentation (spec SS2(a)): 90-degree transverse rotations + flips, drawn in
the SampleSpec and applied IDENTICALLY to input and target. Transverse =
spatial axes perpendicular to the [D-49] split axis (axis 0); the split axis
is never rotated or flipped.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.sightline_rasterizer import (
    DELTA_F_SCALE,
    flux_decrement,
    rasterize_crop,
)
from src.data.truth_crop_provider import TruthCropProvider
from src.data.unet_crop_sampler import (
    N_RAYS_RANGE_DEFAULT,
    RayCropSampler,
    SampleSpec,
    SightlineGeometry,
    geometry_from_sightlines,
)


@dataclass
class PhysicsSource:
    """One physics variant: truth provider + ray geometry + per-bin delta_F."""

    physics_id: int
    provider: TruthCropProvider
    geometry: SightlineGeometry
    delta_f: np.ndarray  # (n_rays, nbins) float32


def apply_transverse_aug(vol: np.ndarray, rot_k: int, flip: bool) -> np.ndarray:
    """Apply the SampleSpec augmentation to a (C, s, s, s) volume whose
    spatial axes are (1, 2, 3) with the split axis at spatial position 1.
    Rotation plane = transverse axes (2, 3); flip = transverse axis 2."""
    out = np.rot90(vol, k=rot_k, axes=(2, 3))
    if flip:
        out = np.flip(out, axis=2)
    return np.ascontiguousarray(out)


class UNetPairDataset(Dataset):
    """Map-style dataset over 1-4 ``PhysicsSource``s (spec SS2 pair factory).

    Parameters
    ----------
    sources : sequence of PhysicsSource (interleaved by ``index % len``).
    length : dataset __len__ (samples are drawn fresh per index; an "epoch"
        is a budget, not a pass over finite items).
    seed : int — determinism root.
    n_rays_fixed : Optional[int] — fix the requested ray count (eval/S4
        patterns); None = training log-uniform draw in ``n_rays_range``.
    augment : bool — apply transverse rot/flip (specs still carry the draws
        so the RNG stream is identical either way).
    """

    def __init__(
        self,
        sources: Sequence[PhysicsSource],
        length: int,
        seed: int = 42,
        n_rays_fixed: Optional[int] = None,
        n_rays_range: Tuple[int, int] = N_RAYS_RANGE_DEFAULT,
        scale: float = DELTA_F_SCALE,
        augment: bool = True,
    ) -> None:
        if len(sources) == 0:
            raise ValueError("need at least one PhysicsSource")
        if length <= 0:
            raise ValueError(f"length must be positive; got {length}")
        self.sources = list(sources)
        self.length = int(length)
        self.seed = int(seed)
        self.n_rays_fixed = n_rays_fixed
        self.scale = float(scale)
        self.augment = bool(augment)
        self.samplers = [
            RayCropSampler(s.provider, s.geometry, n_rays_range=n_rays_range)
            for s in self.sources
        ]

    # ----------------------------------------------------------------- api

    def __len__(self) -> int:
        return self.length

    def zero_ray_rejection_stats(self) -> Dict[str, float]:
        draws = sum(s.n_corner_draws for s in self.samplers)
        rej = sum(s.n_zero_ray_rejections for s in self.samplers)
        return {
            "n_corner_draws": draws,
            "n_zero_ray_rejections": rej,
            "rate": (rej / draws) if draws else 0.0,
        }

    def example(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, SampleSpec, int]:
        """(input, target, spec, physics_id) as numpy; used by tests/viz."""
        if not (0 <= index < self.length):
            raise IndexError(index)
        src_i = index % len(self.sources)
        src = self.sources[src_i]
        rng = np.random.default_rng([self.seed, index])
        spec = self.samplers[src_i].sample_spec(rng, n_rays=self.n_rays_fixed)
        inp = rasterize_crop(
            src.delta_f, src.geometry, spec.ray_indices, spec.corner,
            crop_size=src.provider.crop_size, scale=self.scale,
        )
        tgt = src.provider.crop_at(spec.corner).astype(np.float32)[None]
        if self.augment:
            inp = apply_transverse_aug(inp, spec.rot_k, spec.flip)
            tgt = apply_transverse_aug(tgt, spec.rot_k, spec.flip)
        return inp, tgt, spec, src.physics_id

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inp, tgt, _, _ = self.example(index)
        return torch.from_numpy(inp), torch.from_numpy(tgt)


def build_physics_source(
    data_root: str,
    physics_id: int,
    cube_path: str,
    redshift: float = 0.3,
    region: str = "train",
    n_grid: int = 192,
    provider_seed: int = 42,
) -> PhysicsSource:
    """Production builder: canonical loader -> geometry + delta_F (float32),
    truth cube -> TruthCropProvider. Frees the loader dict before returning
    (delta_f keeps ~134 MB/physics; the raw fields are dropped)."""
    from src.data.loader import SherwoodLoader  # deferred: heavy module

    sl = SherwoodLoader(data_root).load_sightlines(physics_id, redshift)
    geometry = geometry_from_sightlines(sl, n_grid=n_grid)
    delta_f = flux_decrement(sl["tau_h1"]).astype(np.float32)
    del sl
    gc.collect()
    provider = TruthCropProvider(cube_path, region=region, seed=provider_seed)
    return PhysicsSource(
        physics_id=physics_id,
        provider=provider,
        geometry=geometry,
        delta_f=delta_f,
    )
