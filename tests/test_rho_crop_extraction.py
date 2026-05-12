"""
Regression tests for Stage 3 infrastructure: 3D overdensity crop extraction
for the feedback classifier (parallel to [D-46]).

Pins per the PI dispatch:

  (1) Shape & dtype: `extract_rho_crops` returns
      `(n_crops, 1, D, H, W) float32` for crops and `(n_crops,) long`
      for labels, with `D == H == W == crop_size`.
  (2) Label correctness: every label entry equals the input `physics_id`.
  (3) Reproducibility: same `seed` -> byte-identical crops on repeat
      calls (no global RNG mutation).
  (4) Overdensity sanity: per-crop means cluster around 1.0 (the global
      mean of rho/<rho>) with non-zero cross-crop variance — proves
      crops are sampling the true 3D field rather than e.g. constant
      blocks.
  (5) Cross-physics shape consistency: P1..P4 all return the same shape
      at the same `crop_size`. Physics variants with missing local data
      are skipped, but the remaining ones are still compared.

The 3D field is materialized via `SherwoodIGMGalLoader.load_3d_field` —
CIC deposition of GADGET HDF5 particle data. Each `(physics, n_grid)`
deposition is amortized via the module-level `_RHO_FIELD_CACHE` in
`src.data.loader`. We use `n_grid=64` in tests (instead of the native
768) so the deposition stays fast (<10 s/physics on a laptop); the
crop-extraction math is independent of `n_grid`.

Run:
    PYTHONPATH=. uv run pytest tests/test_rho_crop_extraction.py -v
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from src.data.loader import SherwoodLoader

# --------------------------------------------------------------------- config
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IGM_GAL_ROOT = os.path.join(REPO_ROOT, "SherwoodIGM_gal", "extracted")
REDSHIFT = 0.300
# Small n_grid keeps CIC deposition under ~10 s per physics on a laptop.
# Crop-extraction logic is grid-resolution-independent.
N_GRID = 64
CROP_SIZE = 8

PHYSICS_DIRS = {
    1: "planck1_60_768_z0.300",
    2: "planck1_60_768_ps13_z0.300",
    3: "planck1_60_768_ps13agn_z0.300",
    4: "planck1_60_768_ps13agn_strong_z0.300",
}


def _have_physics(physics_id: int) -> bool:
    snap_dir = os.path.join(
        IGM_GAL_ROOT, PHYSICS_DIRS[physics_id], "snapdir_012"
    )
    return os.path.isdir(snap_dir) and any(
        f.startswith("snap_012.") and f.endswith(".hdf5")
        for f in (os.listdir(snap_dir) if os.path.isdir(snap_dir) else [])
    )


_skip_no_p1 = pytest.mark.skipif(
    not _have_physics(1),
    reason="P1 IGM_gal snapshot not available locally",
)


@pytest.fixture(scope="module")
def loader() -> SherwoodLoader:
    # `data_root` is required by `__init__` but is unused by
    # `extract_rho_crops`, which delegates to `SherwoodIGMGalLoader`
    # under its own default root (`SherwoodIGM_gal/extracted/`).
    return SherwoodLoader(data_root=os.path.join(REPO_ROOT, "Sherwood"))


# ------------------------------------------------------------------- tests

@_skip_no_p1
def test_shape_and_dtype(loader):
    """(1) Shape (n, 1, L, L, L) float32 + labels (n,) long."""
    n_crops = 4
    crops, labels = loader.extract_rho_crops(
        physics_id=1,
        redshift=REDSHIFT,
        crop_size=CROP_SIZE,
        n_crops=n_crops,
        seed=42,
        n_grid=N_GRID,
    )
    assert isinstance(crops, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert crops.shape == (n_crops, 1, CROP_SIZE, CROP_SIZE, CROP_SIZE), (
        f"crops shape {tuple(crops.shape)} != "
        f"(n_crops, 1, L, L, L) = ({n_crops}, 1, {CROP_SIZE}, {CROP_SIZE}, {CROP_SIZE})"
    )
    assert crops.dtype == torch.float32, f"crops dtype {crops.dtype} != float32"
    assert labels.shape == (n_crops,), f"labels shape {tuple(labels.shape)} != ({n_crops},)"
    assert labels.dtype == torch.long, f"labels dtype {labels.dtype} != long"


@_skip_no_p1
def test_label_correctness(loader):
    """(2) Every label equals the input physics_id."""
    for pid in (1,):  # only P1 has local data; loop kept for future
        if not _have_physics(pid):
            continue
        crops, labels = loader.extract_rho_crops(
            physics_id=pid,
            redshift=REDSHIFT,
            crop_size=CROP_SIZE,
            n_crops=8,
            seed=42,
            n_grid=N_GRID,
        )
        assert (labels == pid).all(), (
            f"labels for physics_id={pid} not all equal: "
            f"unique={labels.unique().tolist()}"
        )


@_skip_no_p1
def test_reproducibility(loader):
    """(3) Same seed -> byte-identical crops on repeat calls."""
    kw = dict(
        physics_id=1,
        redshift=REDSHIFT,
        crop_size=CROP_SIZE,
        n_crops=8,
        n_grid=N_GRID,
    )
    crops_a, labels_a = loader.extract_rho_crops(seed=42, **kw)
    crops_b, labels_b = loader.extract_rho_crops(seed=42, **kw)
    # Byte-exact equality on the tensor content.
    assert torch.equal(crops_a, crops_b), (
        "extract_rho_crops is not reproducible at the byte level with "
        "seed=42; max |a - b| = "
        f"{(crops_a - crops_b).abs().max().item():.3e}"
    )
    assert torch.equal(labels_a, labels_b)

    # Sanity: a different seed should give DIFFERENT crops (unless we
    # happened to land on the same corners, which is vanishingly unlikely).
    crops_c, _ = loader.extract_rho_crops(seed=43, **kw)
    assert not torch.equal(crops_a, crops_c), (
        "extract_rho_crops returned identical crops for seed=42 vs seed=43; "
        "RNG plumbing is broken."
    )


@_skip_no_p1
def test_overdensity_sanity(loader):
    """
    (4) Per-crop means cluster around 1.0 with non-zero variance.

    For an unbiased uniform sampling of rho/<rho>, the expectation of the
    per-crop mean is 1.0. We allow a generous tolerance (overall mean
    of the 16 crops within +/- 0.5 of 1.0) since 16 small (8^3 cells)
    crops on a coarse 64^3 grid still oversample individual high-density
    cells. Also assert non-zero cross-crop variance: a constant rho-field
    would produce zero variance and would be a deposition bug.
    """
    crops, _ = loader.extract_rho_crops(
        physics_id=1,
        redshift=REDSHIFT,
        crop_size=CROP_SIZE,
        n_crops=16,
        seed=42,
        n_grid=N_GRID,
    )
    per_crop_means = crops.mean(dim=(1, 2, 3, 4)).numpy()
    overall_mean = float(per_crop_means.mean())
    cross_crop_std = float(per_crop_means.std())

    assert 0.5 <= overall_mean <= 1.5, (
        f"Mean of per-crop means {overall_mean:.3f} not in [0.5, 1.5]; "
        f"deposition or sampling may be biased."
    )
    assert cross_crop_std > 0.01, (
        f"Cross-crop std of means {cross_crop_std:.3e} too small; "
        f"crops may be sampling a degenerate / constant field."
    )

    # Positivity & finiteness (defense in depth; _validate_rho_crops
    # already enforces this on the return path).
    assert torch.isfinite(crops).all()
    assert (crops > 0).all()


def test_cross_physics_shape_consistency(loader):
    """
    (5) P1..P4 all return the same shape at the same crop_size.
    Physics variants missing locally are skipped, but the remaining
    ones are still compared against the first available one.
    """
    available = [pid for pid in (1, 2, 3, 4) if _have_physics(pid)]
    if not available:
        pytest.skip("No IGM_gal physics variants available locally")

    shapes = {}
    label_shapes = {}
    for pid in available:
        crops, labels = loader.extract_rho_crops(
            physics_id=pid,
            redshift=REDSHIFT,
            crop_size=CROP_SIZE,
            n_crops=4,
            seed=42,
            n_grid=N_GRID,
        )
        shapes[pid] = tuple(crops.shape)
        label_shapes[pid] = tuple(labels.shape)

    ref_pid = available[0]
    ref_shape = shapes[ref_pid]
    for pid, sh in shapes.items():
        assert sh == ref_shape, (
            f"Shape mismatch: physics={pid} returned {sh}, "
            f"reference physics={ref_pid} returned {ref_shape}"
        )
        assert label_shapes[pid] == label_shapes[ref_pid]

    # Surface which physics actually participated so a maintainer reading
    # the test log can confirm coverage. In the local dev env only P1 is
    # present (P2/P3/P4 IGM_gal snapshots not yet extracted); CI / a fully
    # populated host should exercise all four.
    print(f"\ncross-physics check ran on: {available}")
