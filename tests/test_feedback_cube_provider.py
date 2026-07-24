"""Unit tests for src/data/feedback_cube_provider.py (A3, [F-02] §5 R28 R2).

Covers: [D-49] split boundaries exact, coord round-trip, x-transform matches the
d75 reference to <=1e-12, determinism under a seed, all-4-variants
load-or-graceful-skip. Run: PYTHONPATH=. .venv/bin/python -m pytest.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.d75_corrected_metric_rescore import x_transform as d75_x_transform
from src.data import feedback_cube_provider as fcp
from src.data.feedback_cube_provider import (
    BOX_KPC_H,
    N_GRID,
    FeedbackCubeProvider,
    apply_x_transform,
    region_voxel_bounds,
    voxel_index_to_siren,
    _siren_to_kpc_h,
    _voxel_center_kpc_h,
)


# --------------------------------------------------------------------------- #
# split boundaries — exact, no cube needed
# --------------------------------------------------------------------------- #
def test_d49_split_boundaries_exact():
    assert region_voxel_bounds("train") == (0, 134)
    assert region_voxel_bounds("val") == (134, 163)
    assert region_voxel_bounds("test") == (163, 192)


def test_d49_split_covers_box_no_overlap():
    bounds = [region_voxel_bounds(r) for r in ("train", "val", "test")]
    # contiguous, right-open, cover [0, 192)
    assert bounds[0][0] == 0 and bounds[-1][1] == N_GRID
    for (_, e), (s, _) in zip(bounds[:-1], bounds[1:]):
        assert e == s


def test_split_matches_loader_int_floor_rule():
    from src.data.loader import DEFAULT_SCHEME
    assert region_voxel_bounds("train")[1] == int(DEFAULT_SCHEME.train_x_max * N_GRID)
    assert region_voxel_bounds("val")[1] == int(DEFAULT_SCHEME.val_x_max * N_GRID)


def test_bad_region_raises():
    with pytest.raises(ValueError):
        region_voxel_bounds("heldout")


# --------------------------------------------------------------------------- #
# coordinate round-trip
# --------------------------------------------------------------------------- #
def test_coord_roundtrip_siren_to_voxel():
    idx = np.array([[0, 0, 0], [191, 191, 191], [134, 5, 163], [67, 100, 30]])
    siren = voxel_index_to_siren(idx)
    # SIREN range strictly inside [-1, 1] (cell centers, never on the wall)
    assert siren.min() > -1.0 and siren.max() < 1.0
    kpc = _siren_to_kpc_h(siren)
    recovered = np.floor(kpc / BOX_KPC_H * N_GRID).astype(int)
    assert np.array_equal(recovered, idx)


def test_cell_center_kpc_h_endpoints():
    # voxel 0 center = 0.5/192 * 60000; voxel 191 center = 191.5/192 * 60000
    c = _voxel_center_kpc_h(np.array([0, 191]))
    assert np.isclose(c[0], 0.5 / N_GRID * BOX_KPC_H)
    assert np.isclose(c[1], 191.5 / N_GRID * BOX_KPC_H)


# --------------------------------------------------------------------------- #
# x-transform parity with the d75 reference (<= 1e-12)
# --------------------------------------------------------------------------- #
def test_x_transform_matches_d75_reference():
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.0, 4000.0, size=(8, 8, 8))
    x_ref, clamp_ref = d75_x_transform(rho)
    x_ours, clamp_ours = apply_x_transform(rho)
    assert np.max(np.abs(x_ours - x_ref)) <= 1e-12
    assert clamp_ours == clamp_ref


def test_x_transform_floor():
    # values below 1e-3 clamp to log10(1e-3) = -3
    x, frac = apply_x_transform(np.array([0.0, 1e-9, 1.0, 100.0]))
    assert np.isclose(x[0], -3.0) and np.isclose(x[1], -3.0)
    assert frac == 0.5


# --------------------------------------------------------------------------- #
# provider — load-or-graceful-skip, all 4 variants
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def provider():
    return FeedbackCubeProvider()


def test_all_four_load_or_graceful_skip(provider):
    loaded = set(provider.loaded_variants)
    missing = set(provider.missing_variants)
    assert loaded | missing == {1, 2, 3, 4}
    assert loaded.isdisjoint(missing)
    # this machine: enumeration says all 4 present -> assert richer behavior
    assert loaded, "no cubes loaded at all; check CUBE_PATHS / dvc pull"


def test_missing_path_graceful_and_strict():
    # temporarily point one variant at a nonexistent path
    orig = fcp.CUBE_PATHS[2]
    fcp.CUBE_PATHS[2] = orig.parent / "does_not_exist.npy"
    try:
        p = FeedbackCubeProvider(variants=[2])
        assert p.loaded_variants == [] and p.missing_variants == [2]
        with pytest.raises(FileNotFoundError):
            FeedbackCubeProvider(variants=[2], strict=True)
    finally:
        fcp.CUBE_PATHS[2] = orig


def _skip_if_empty(provider):
    if not provider.loaded_variants:
        pytest.skip("no cubes on disk")


def test_validate_data_shapes(provider):
    _skip_if_empty(provider)
    p = provider.loaded_variants[0]
    cube = provider._require(p)
    assert cube.rho.shape == (N_GRID, N_GRID, N_GRID)
    assert cube.x.shape == (N_GRID, N_GRID, N_GRID)
    assert np.all(np.isfinite(cube.x))


# --------------------------------------------------------------------------- #
# sampling — determinism, region correctness, shapes, target parity
# --------------------------------------------------------------------------- #
def test_sample_determinism(provider):
    _skip_if_empty(provider)
    p = provider.loaded_variants[0]
    c1, x1 = provider.sample(p, "train", 256, seed=42)
    c2, x2 = provider.sample(p, "train", 256, seed=42)
    assert torch.equal(c1, c2) and torch.equal(x1, x2)
    c3, _ = provider.sample(p, "train", 256, seed=7)
    assert not torch.equal(c1, c3)


def test_sample_shapes_and_dtype(provider):
    _skip_if_empty(provider)
    p = provider.loaded_variants[0]
    coords, x = provider.sample(p, "val", 100, seed=1)
    assert coords.shape == (100, 3) and x.shape == (100,)
    assert coords.dtype == torch.float32 and x.dtype == torch.float32
    assert coords.min() >= -1.0 and coords.max() <= 1.0


def test_sample_stays_in_region_axis0(provider):
    _skip_if_empty(provider)
    p = provider.loaded_variants[0]
    start, end = region_voxel_bounds("test")
    coords, _ = provider.sample(p, "test", 500, seed=3)
    # recover axis-0 voxel index from SIREN coord
    kpc = _siren_to_kpc_h(coords[:, 0].numpy())
    i0 = np.floor(kpc / BOX_KPC_H * N_GRID).astype(int)
    assert i0.min() >= start and i0.max() < end


def test_sample_targets_match_cube(provider):
    _skip_if_empty(provider)
    p = provider.loaded_variants[0]
    cube = provider._require(p)
    coords, x = provider.sample(p, "train", 200, seed=9)
    kpc = _siren_to_kpc_h(coords.numpy())
    idx = np.floor(kpc / BOX_KPC_H * N_GRID).astype(int)
    expected = cube.x[idx[:, 0], idx[:, 1], idx[:, 2]]
    assert np.max(np.abs(x.numpy() - expected.astype(np.float32))) <= 1e-6


def test_coords_autograd_capable(provider):
    _skip_if_empty(provider)
    p = provider.loaded_variants[0]
    coords, _ = provider.sample(p, "train", 16, seed=0)
    coords.requires_grad_(True)
    y = (coords ** 2).sum()
    y.backward()
    assert coords.grad is not None and torch.all(torch.isfinite(coords.grad))


# --------------------------------------------------------------------------- #
# whole-region accessor for scoring
# --------------------------------------------------------------------------- #
def test_region_cube_shapes_and_alignment(provider):
    _skip_if_empty(provider)
    p = provider.loaded_variants[0]
    cube = provider._require(p)
    start, end = region_voxel_bounds("test")
    x_slab, coords = provider.region_cube(p, "test")
    assert x_slab.shape == (end - start, N_GRID, N_GRID)
    assert coords.shape == (end - start, N_GRID, N_GRID, 3)
    assert np.array_equal(x_slab, cube.x[start:end])
    # corner voxel (start,0,0) coord round-trips to its index
    kpc = _siren_to_kpc_h(coords[0, 0, 0])
    assert np.array_equal(
        np.floor(kpc / BOX_KPC_H * N_GRID).astype(int), [start, 0, 0]
    )


def test_region_flat_matches_region_cube(provider):
    _skip_if_empty(provider)
    p = provider.loaded_variants[0]
    x_slab, _ = provider.region_cube(p, "val")
    coords_t, x_t = provider.region_flat(p, "val")
    assert x_t.shape[0] == x_slab.size
    assert np.max(np.abs(x_t.numpy() - x_slab.reshape(-1).astype(np.float32))) <= 1e-6
