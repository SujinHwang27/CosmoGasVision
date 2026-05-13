"""Cube-isometry augmentations for the sprint-4 [D-51] 3D classifier.

Sprint-4 design doc \xa75 specifies *only* isometries of the periodic
Sherwood box as valid augmentations (no scaling / cropping / noise —
sim-data is exact). The octahedral group on a cube has 24 orientation-
preserving rotations + 24 mirror-extended elements; the augmentations
implemented here sample from a representative subset:

  - Random axis flip on each of the 3 spatial axes (p=0.5 per axis,
    independent — 8 possible flip combinations including no-flip).
  - Random 90\xb0 rotation about a random axis (p=0.5 to apply; if
    applied, uniform choice over 3 rotation axes and over
    {1, 2, 3} quarter-turns).

**Determinism contract** (sprint-4 gate (c) + AD-4): the augmentation
RNG is seeded per ``(epoch, sample_index)`` pair, NOT per label. This
guarantees that:

  (i) repeated runs with the same seed produce the same augmented
      crops bit-by-bit, satisfying gate (c) split-determinism;
  (ii) the augmentation distribution is label-independent at every
       seeded (epoch, idx), foreclosing the AD-4 "augmentation RNG
       leaks label as a covariate" failure mode.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class AugmentConfig:
    """Configurable knobs for the cube-isometry augmentation pipeline.
    Defaults match the sprint-4 design doc \xa75 specification.
    """
    flip_p_per_axis: float = 0.5
    rotation_p: float = 0.5
    enabled: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.flip_p_per_axis <= 1.0):
            raise ValueError(
                f"flip_p_per_axis must be in [0,1], got {self.flip_p_per_axis}"
            )
        if not (0.0 <= self.rotation_p <= 1.0):
            raise ValueError(
                f"rotation_p must be in [0,1], got {self.rotation_p}"
            )


DEFAULT_AUGMENT = AugmentConfig()


def _per_sample_rng(epoch: int, sample_index: int, base_seed: int) -> np.random.Generator:
    """Deterministic per-sample RNG seeded by (epoch, sample_index).

    The seed mixing is intentionally label-independent: the label of the
    crop never enters the RNG state. This is the AD-4 anti-leakage
    discipline that the augmentation-label-independence unit test
    verifies.
    """
    # Mix epoch + sample_index + base_seed via a deterministic hash;
    # NumPy's default_rng with a SeedSequence + spawn keeps streams
    # independent across (epoch, idx) without correlation artifacts.
    ss = np.random.SeedSequence(
        entropy=base_seed, spawn_key=(int(epoch), int(sample_index)),
    )
    return np.random.default_rng(ss)


def augment_cube_isometry(
    crop: torch.Tensor,
    epoch: int,
    sample_index: int,
    base_seed: int = 0,
    config: AugmentConfig = DEFAULT_AUGMENT,
) -> torch.Tensor:
    """Apply cube-isometry augmentation to one crop tensor.

    Args:
        crop: shape ``(C, D, H, W)`` or ``(D, H, W)`` (channel-first or
            no-channel). The leading dim, if 4D, is preserved.
        epoch: training epoch index. Mixed into the per-sample seed.
        sample_index: stable sample identifier (e.g., the dataset index
            of this crop). Mixed into the per-sample seed.
        base_seed: global RNG seed (typically the training-seed knob).
        config: augmentation knobs.

    Returns:
        Augmented crop with the same shape as input.

    The same ``(epoch, sample_index, base_seed)`` triplet always
    produces the same augmented crop (bit-equivalent).
    """
    if not config.enabled:
        return crop
    if crop.ndim not in (3, 4):
        raise ValueError(
            f"crop must be 3D (D,H,W) or 4D (C,D,H,W), got shape {crop.shape}"
        )

    rng = _per_sample_rng(epoch, sample_index, base_seed)

    # Determine the spatial axis offset (3 for 4D, 0 for 3D).
    spatial_offset = crop.ndim - 3

    # 1) Per-axis flips
    flips = rng.random(3) < config.flip_p_per_axis
    flip_dims = [spatial_offset + a for a, do_flip in enumerate(flips) if do_flip]
    if flip_dims:
        crop = torch.flip(crop, dims=flip_dims)

    # 2) Optional 90\xb0 rotation about a random axis
    if rng.random() < config.rotation_p:
        # Pick a rotation axis (one of 3 pairs in the spatial volume)
        rot_axis_pair_id = int(rng.integers(0, 3))
        # Axis pairs: 0 -> (D,H), 1 -> (D,W), 2 -> (H,W)
        pair_lookup = {
            0: (spatial_offset + 0, spatial_offset + 1),
            1: (spatial_offset + 0, spatial_offset + 2),
            2: (spatial_offset + 1, spatial_offset + 2),
        }
        rot_dims = pair_lookup[rot_axis_pair_id]
        n_quarters = int(rng.integers(1, 4))  # 1, 2, or 3 quarter-turns
        crop = torch.rot90(crop, k=n_quarters, dims=rot_dims)

    return crop.contiguous()


def augment_batch(
    crops: torch.Tensor,
    epoch: int,
    sample_indices: torch.Tensor,
    base_seed: int = 0,
    config: AugmentConfig = DEFAULT_AUGMENT,
) -> torch.Tensor:
    """Apply ``augment_cube_isometry`` to each sample in a batch.

    Args:
        crops: shape ``(B, C, D, H, W)`` or ``(B, D, H, W)``.
        epoch: training epoch index.
        sample_indices: 1D tensor of length B with the stable dataset
            index of each sample in the batch. Used to seed
            per-sample augmentation RNG; same indices in different
            epochs produce different augmentations (epoch mixed in).
        base_seed: global RNG seed.
        config: augmentation knobs.
    """
    if crops.ndim not in (4, 5):
        raise ValueError(
            f"crops must be 4D (B,D,H,W) or 5D (B,C,D,H,W), got {crops.shape}"
        )
    if sample_indices.ndim != 1 or sample_indices.size(0) != crops.size(0):
        raise ValueError(
            f"sample_indices must be 1D of length B={crops.size(0)}, "
            f"got shape {sample_indices.shape}"
        )

    out = torch.empty_like(crops)
    for b in range(crops.size(0)):
        out[b] = augment_cube_isometry(
            crops[b], epoch=epoch, sample_index=int(sample_indices[b].item()),
            base_seed=base_seed, config=config,
        )
    return out
