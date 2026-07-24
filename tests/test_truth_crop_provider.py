"""[U-04] Stage-1 R3 unit tests: TruthCropProvider over the [D-49] split at
n=192 (spec §2(d) S1 subset: containment, straddle rejection at 192 pitch,
x-transform <=1e-12 match vs the [D-75] scoring transform, determinism)."""

import numpy as np
import pytest

from src.data.loader import (
    DEFAULT_SCHEME,
    HeldoutSplitScheme,
    region_voxel_interval,
)
from src.data.truth_crop_provider import TruthCropProvider, X_FLOOR, x_transform

N192 = 192
CROP = 64


@pytest.fixture(scope="module")
def rho_cube():
    """Synthetic positive rho/<rho> cube, 192^3, with sub-floor voxels so
    the clamp path is exercised. Coordinate-coded so containment checks can
    recover the source voxels exactly."""
    rng = np.random.default_rng(1234)
    cube = rng.lognormal(mean=0.0, sigma=1.0, size=(N192, N192, N192))
    cube[rng.random(cube.shape) < 0.01] = 0.0  # zeros -> clamped at 1e-3
    return cube.astype(np.float32)


@pytest.fixture(scope="module")
def provider(rho_cube):
    return TruthCropProvider(rho_cube, region="train", crop_size=CROP, seed=42)


# --------------------------------------------------------------- geometry --

def test_region_intervals_at_192_and_768():
    """Documents the fraction->voxel mapping the provider inherits: the
    scheme is lattice-independent (fractions); int() truncation per lattice."""
    assert region_voxel_interval("train", 192) == (0, 134)   # 0.7*192=134.4
    assert region_voxel_interval("val", 192) == (134, 163)   # 0.85*192=163.2
    assert region_voxel_interval("test", 192) == (163, 192)
    assert region_voxel_interval("heldout", 192) == (134, 192)
    assert region_voxel_interval("train", 768) == (0, 537)
    assert region_voxel_interval("val", 768) == (537, 652)
    assert region_voxel_interval("test", 768) == (652, 768)


def test_train_axis_offset_diversity(provider):
    # corners in [0, 70] -> 71 distinct axis-0 offsets (~70 per spec §2(a))
    assert provider.corner_min == 0
    assert provider.corner_max_inclusive == 70
    assert provider.n_axis_offsets() == 71


# ------------------------------------------------------------ containment --

def test_sampled_crops_wholly_in_train_region(provider):
    crops, corners, distances = provider.sample(64)
    assert crops.shape == (64, CROP, CROP, CROP)
    assert corners.shape == (64, 3)
    ax = corners[:, DEFAULT_SCHEME.axis]
    assert (ax >= 0).all() and (ax + CROP <= 134).all()
    # train-region crops have zero distance to train
    assert (distances == 0.0).all()


def test_crop_values_match_source_voxels(provider, rho_cube):
    """crop_at returns exactly x_cube[np.ix_] with transverse wraparound."""
    corner = np.array([70, 190, 5])  # y wraps; x at the last legal offset
    crop = provider.crop_at(corner)
    off = np.arange(CROP)
    expected_raw = rho_cube[
        np.ix_((70 + off) % N192, (190 + off) % N192, (5 + off) % N192)
    ]
    expected = np.log10(np.maximum(expected_raw.astype(np.float64), X_FLOOR))
    np.testing.assert_array_equal(crop, expected)


# ----------------------------------------------- straddle rejection @ 192 --

def test_straddling_corner_rejected_never_relabelled(provider):
    for bad_axis0 in (71, 100, 133, 134, 163, 191):
        with pytest.raises(ValueError, match="straddle|exit"):
            provider.crop_at(np.array([bad_axis0, 0, 0]))


def test_val_test_heldout_cannot_hold_64_crops_at_192(rho_cube):
    """At 192 pitch, widths are val=29, test=29, heldout=58 < 64: the
    constructor must refuse (test-region eval is sliding-window, not crops)."""
    for region in ("val", "test", "heldout"):
        with pytest.raises(ValueError, match="too large for region"):
            TruthCropProvider(rho_cube, region=region, crop_size=CROP)


def test_smaller_crop_admitted_and_contained_in_val(rho_cube):
    p = TruthCropProvider(rho_cube, region="val", crop_size=16, seed=7)
    _, corners, distances = p.sample(32)
    ax = corners[:, DEFAULT_SCHEME.axis]
    assert (ax >= 134).all() and (ax + 16 <= 163).all()
    assert (distances > 0.0).all()


def test_nondefault_scheme_respected(rho_cube):
    scheme = HeldoutSplitScheme(train_x_max=0.5, val_x_max=0.75, axis=2)
    p = TruthCropProvider(
        rho_cube, region="train", crop_size=CROP, scheme=scheme, seed=3
    )
    assert p.corner_max_inclusive == int(0.5 * N192) - CROP  # 96 - 64 = 32
    _, corners, _ = p.sample(16)
    assert (corners[:, 2] + CROP <= 96).all()


# ------------------------------------------------------------ x-transform --

def test_x_transform_matches_d75_scoring_transform_1e12(rho_cube):
    """Contract: <=1e-12 vs scripts/d75_corrected_metric_rescore.py
    x_transform. Replicate the reference expression literally here."""
    r = np.asarray(rho_cube, dtype=np.float64)
    ref = np.log10(np.maximum(r, 1.0e-3))
    ref_clamped = float((r < 1.0e-3).mean())
    got, got_clamped = x_transform(rho_cube)
    assert got.dtype == np.float64
    assert np.max(np.abs(got - ref)) <= 1e-12  # bit-exact in practice
    assert got_clamped == ref_clamped
    # provider applies the same transform internally
    p = TruthCropProvider(rho_cube, region="train", crop_size=CROP)
    assert np.max(np.abs(p.x_cube - ref)) <= 1e-12
    assert p.clamped_fraction == ref_clamped


def test_floor_applied_before_log(rho_cube):
    p = TruthCropProvider(rho_cube, region="train", crop_size=CROP)
    assert np.isfinite(p.x_cube).all()
    assert float(p.x_cube.min()) >= np.log10(X_FLOOR) - 1e-15
    assert p.clamped_fraction > 0.0  # fixture plants sub-floor voxels


# ------------------------------------------------------------ determinism --

def test_determinism_under_seed(rho_cube):
    a = TruthCropProvider(rho_cube, region="train", crop_size=CROP, seed=42)
    b = TruthCropProvider(rho_cube, region="train", crop_size=CROP, seed=42)
    ca, ka, da = a.sample(32)
    cb, kb, db = b.sample(32)
    np.testing.assert_array_equal(ka, kb)
    np.testing.assert_array_equal(ca, cb)
    np.testing.assert_array_equal(da, db)
    # a second draw from the same provider advances the stream
    ca2, ka2, _ = a.sample(32)
    assert not np.array_equal(ka, ka2)
    # different seed -> different corners
    c = TruthCropProvider(rho_cube, region="train", crop_size=CROP, seed=43)
    _, kc, _ = c.sample(32)
    assert not np.array_equal(ka, kc)


# ------------------------------------------------------------- validation --

def test_rejects_nan_and_negative_cubes():
    bad = np.ones((8, 8, 8))
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        TruthCropProvider(bad, region="train", crop_size=2)
    neg = np.ones((8, 8, 8))
    neg[1, 1, 1] = -0.5
    with pytest.raises(ValueError, match="negative"):
        TruthCropProvider(neg, region="train", crop_size=2)
