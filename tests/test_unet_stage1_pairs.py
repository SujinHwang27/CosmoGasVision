"""[U-04] Stage-1 R7 (A7): unit tests for the pair factory (exit criteria
S1 + S2 of u04_stage1_ratification.md SS2(d), commit 58ac831).

Coverage map (spec S1):
* crop containment + straddle rejection — sampler-level here; provider-level
  coverage lives in tests/test_truth_crop_provider.py (12 tests, reused);
* ray-axis handling x3 axes;
* synthetic-ray round-trip (constant-F ray -> EXACT on-path value, EXACT
  zeros off-path);
* bin-to-voxel mean equivalence;
* mask correctness (multi-ray voxel: mean value, mask stays 1);
* target transform matches the d75 x_transform <= 1e-12;
* seeded determinism.
S2: P1 target-crop bit-exactness vs the pinned truth_real_192.npy
(sha256 pin asserted + sampled-crop equality).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.sightline_rasterizer import (
    DELTA_F_SCALE,
    flux_decrement,
    rasterize_crop,
)
from src.data.truth_crop_provider import TruthCropProvider, x_transform
from src.data.unet_crop_sampler import (
    RayCropSampler,
    geometry_from_sightlines,
    intersecting_rays,
)
from src.data.unet_pair_dataset import (
    PhysicsSource,
    UNetPairDataset,
    apply_transverse_aug,
)

REPO = Path(__file__).resolve().parents[1]
BOX = 60000.0
P1_CUBE = REPO / "experiments/nerf/artifacts/d75_rescore/cubes/truth_real_192.npy"
P1_SHA256_PIN = (
    "971a72ed5b1b872a972fd3ff8c35e99d6a0998a129afb822ebbd42417da34994"
)  # pin of record: r1_data_locality.json SSc


# --------------------------------------------------------------- fixtures

def make_sl(rays, nbins=2048, box=BOX):
    """Fake loader dict. rays = list of (iaxis, vx, vy, vz) with voxel
    coords on the 192 lattice (own-axis entry ignored)."""
    pitch = box / 192.0
    ia = np.array([r[0] for r in rays], dtype=np.int32)
    xyz = np.array([[r[1], r[2], r[3]] for r in rays], dtype=np.float64)
    coords = (xyz + 0.5) * pitch
    return {
        "header": {"box_kpc_h": box},
        "iaxis": ia,
        "xaxis": coords[:, 0],
        "yaxis": coords[:, 1],
        "zaxis": coords[:, 2],
        "pos_axis": (np.arange(nbins) + 0.5) * box / nbins,
    }


def make_random_source(seed=0, n_rays=400, nbins=2048, physics_id=1, cube=None):
    rng = np.random.default_rng(seed)
    rays = [
        (int(rng.integers(1, 4)),
         int(rng.integers(0, 192)),
         int(rng.integers(0, 192)),
         int(rng.integers(0, 192)))
        for _ in range(n_rays)
    ]
    geom = geometry_from_sightlines(make_sl(rays, nbins=nbins))
    if cube is None:
        cube = rng.uniform(0.05, 3.0, (192, 192, 192))
    provider = TruthCropProvider(cube, region="train", crop_size=64, seed=seed)
    delta_f = rng.uniform(0.0, 0.2, (n_rays, nbins)).astype(np.float32)
    return PhysicsSource(physics_id, provider, geom, delta_f)


@pytest.fixture(scope="module")
def rand_source():
    return make_random_source(seed=11)


# ------------------------------------------------ geometry / bin mapping

def test_bin_voxel_mapping_matches_d75_convention():
    # canonical mapping: floor(pos_axis / pitch) % 192 (d75_rescore.py:495)
    sl = make_sl([(1, 0, 5, 5)], nbins=2048)
    geom = geometry_from_sightlines(sl)
    cell = BOX / 192.0
    expected = np.floor(sl["pos_axis"] / cell).astype(int) % 192
    assert np.array_equal(geom.bin_voxel_idx, expected)
    counts = np.bincount(geom.bin_voxel_idx, minlength=192)
    assert counts.min() >= 10 and counts.max() <= 11  # ~10.7 bins/voxel


def test_intersecting_rays_all_three_axes_and_wrap():
    corner = np.array([10, 150, 150])  # transverse wrap: 150+64 > 192
    rays = [
        (1, 0, 160, 170),   # 0 along x, y/z inside -> IN
        (1, 0, 100, 170),   # 1 y outside -> OUT
        (2, 40, 0, 10),     # 2 along y, x in [10,74), z=10 in wrap [150,22) -> IN
        (2, 80, 0, 10),     # 3 x outside -> OUT
        (3, 20, 5, 0),      # 4 along z, x in, y=5 wraps in -> IN
        (3, 20, 60, 0),     # 5 y outside wrap window -> OUT
    ]
    geom = geometry_from_sightlines(make_sl(rays))
    got = intersecting_rays(geom, corner, 64)
    assert got.tolist() == [0, 2, 4]


# ----------------------------------------------------------- sampler (R4)

def test_sampler_containment_ray_counts_and_spec_ranges(rand_source):
    s = RayCropSampler(rand_source.provider, rand_source.geometry)
    rng = np.random.default_rng(3)
    for _ in range(100):
        spec = s.sample_spec(rng)
        # split-axis containment: [D-49] train at n=192 -> corners in [0, 70]
        assert 0 <= spec.corner[0] <= 70
        rand_source.provider.crop_at(spec.corner)  # must not raise
        assert 64 <= spec.n_rays_requested <= 1024
        assert len(spec.ray_indices) == min(
            spec.n_rays_requested, spec.n_rays_available
        )
        pool = intersecting_rays(rand_source.geometry, spec.corner, 64)
        assert np.isin(spec.ray_indices, pool).all()
        assert len(np.unique(spec.ray_indices)) == len(spec.ray_indices)
        assert spec.rot_k in (0, 1, 2, 3)


def test_sampler_straddle_rejection_reused_from_provider(rand_source):
    # corner 71 on the split axis would straddle train/val at n=192
    with pytest.raises(ValueError, match="straddle"):
        rand_source.provider.crop_at(np.array([71, 0, 0]))


def test_sampler_zero_ray_rejection_logged():
    # single ray -> most corners have an empty pool -> rejections logged
    src = make_random_source(seed=2, n_rays=1)
    s = RayCropSampler(src.provider, src.geometry)
    rng = np.random.default_rng(0)
    for _ in range(10):
        spec = s.sample_spec(rng)
        assert spec.n_rays_available >= 1
    assert s.n_corner_draws == 10 + s.n_zero_ray_rejections
    assert s.n_zero_ray_rejections > 0  # 1-ray pool: hit prob ~1/9 per draw
    assert 0.0 < s.zero_ray_rejection_rate < 1.0


# -------------------------------------------------------- rasterizer (R5)

def _single_ray_roundtrip(axis_iax, corner, tvox):
    """Constant-F ray -> exact on-path value, exact 0 off-path (spec S1)."""
    rays = {1: (1, 0, tvox[0], tvox[1]),
            2: (2, tvox[0], 0, tvox[1]),
            3: (3, tvox[0], tvox[1], 0)}[axis_iax]
    geom = geometry_from_sightlines(make_sl([rays]))
    tau0 = -np.log(0.75)  # delta_F = 1 - exp(-tau0) == 0.25 exactly (fp64)
    df = flux_decrement(np.full((1, 2048), tau0))
    assert df[0, 0] == 0.25
    idx = intersecting_rays(geom, corner, 64)
    assert idx.tolist() == [0]
    out = rasterize_crop(df, geom, idx, corner)
    expected = np.float32(DELTA_F_SCALE * 0.25)
    mask = out[1]
    assert mask.sum() == 64.0  # full line through the crop
    on = out[0][mask == 1.0]
    assert (on == expected).all()          # EXACT on-path
    assert (out[0][mask == 0.0] == 0.0).all()   # EXACT zeros off-path
    assert set(np.unique(mask)) <= {0.0, 1.0}
    return out


def test_synthetic_ray_roundtrip_axis_x():
    out = _single_ray_roundtrip(1, np.array([5, 100, 40]), (110, 60))
    assert (out[1][:, 10, 20] == 1.0).all()  # line at local (y=10, z=20)


def test_synthetic_ray_roundtrip_axis_y():
    out = _single_ray_roundtrip(2, np.array([5, 100, 40]), (30, 60))
    assert (out[1][25, :, 20] == 1.0).all()


def test_synthetic_ray_roundtrip_axis_z():
    out = _single_ray_roundtrip(3, np.array([5, 100, 40]), (30, 110))
    assert (out[1][25, 10, :] == 1.0).all()


def test_bin_to_voxel_mean_equivalence():
    # varying delta_F: each crop voxel = mean of the bins assigned to it
    geom = geometry_from_sightlines(make_sl([(1, 0, 100, 40)]))
    rng = np.random.default_rng(4)
    df = rng.uniform(0.0, 1.0, (1, 2048))
    corner = np.array([6, 90, 30])
    out = rasterize_crop(df, geom, np.array([0]), corner)
    loc = (geom.bin_voxel_idx - corner[0]) % 192
    for v in (0, 17, 63):
        ref = DELTA_F_SCALE * df[0, loc == v].mean()
        # float64 accumulation agrees to <=1e-12 relative; emission is a
        # float32 cast, so the observable bar is float32 rounding of ref
        assert out[0][v, 10, 10] == np.float32(ref)
        assert out[1][v, 10, 10] == 1.0


def test_multi_ray_voxel_mean_and_mask_stays_one():
    # two crossing rays (x-ray and y-ray sharing voxel [20,30,40])
    rays = [(1, 0, 30, 40), (2, 20, 0, 40)]
    geom = geometry_from_sightlines(make_sl(rays))
    rng = np.random.default_rng(5)
    df = rng.uniform(0.0, 1.0, (2, 2048))
    corner = np.array([4, 20, 30])
    out = rasterize_crop(df, geom, np.array([0, 1]), corner)
    cross = (20 - 4, 30 - 20, 40 - 30)
    m0 = df[0, (geom.bin_voxel_idx - corner[0]) % 192 == cross[0]].mean()
    m1 = df[1, (geom.bin_voxel_idx - corner[1]) % 192 == cross[1]].mean()
    assert out[0][cross] == np.float32(DELTA_F_SCALE * 0.5 * (m0 + m1))
    assert out[1][cross] == 1.0  # mask stays 1, not 2
    assert out[1].sum() == 64.0 + 64.0 - 1.0


def test_rasterizer_matches_bruteforce_reference(rand_source):
    src = rand_source
    corner = np.array([3, 170, 88])
    pool = intersecting_rays(src.geometry, corner, 64)
    assert pool.size > 3
    out = rasterize_crop(src.delta_f, src.geometry, pool, corner)
    # naive reference: per-ray voxel means, then mean across rays
    sums = np.zeros((64, 64, 64))
    cnts = np.zeros((64, 64, 64))
    for r in pool:
        a = src.geometry.axis[r]
        o0, o1 = [ax for ax in range(3) if ax != a]
        l0 = (src.geometry.voxel3[r, o0] - corner[o0]) % 192
        l1 = (src.geometry.voxel3[r, o1] - corner[o1]) % 192
        loc = (src.geometry.bin_voxel_idx - corner[a]) % 192
        for v in range(64):
            m = src.delta_f[r, loc == v].astype(np.float64).mean()
            sel = [0, 0, 0]
            sel[a], sel[o0], sel[o1] = v, l0, l1
            sums[tuple(sel)] += m
            cnts[tuple(sel)] += 1
    ref = np.where(cnts > 0, DELTA_F_SCALE * sums / np.maximum(cnts, 1), 0.0)
    assert np.allclose(out[0], ref.astype(np.float32), rtol=1e-6, atol=1e-7)
    assert np.array_equal(out[1], (cnts > 0).astype(np.float32))


# ------------------------------------------------------------ dataset (R6)

def test_target_transform_matches_d75_within_1e12(rand_source):
    ds = UNetPairDataset([rand_source], length=4, seed=9, augment=False)
    inp, tgt, spec, _ = ds.example(0)
    raw = np.array(10.0) ** rand_source.provider.x_cube  # invert for ref only
    # independent d75 expression on the raw crop (float64, floor-before-log)
    c = spec.corner
    idx = [(c[a] + np.arange(64)) % 192 for a in range(3)]
    ref64 = np.log10(np.maximum(raw[np.ix_(*idx)], 1.0e-3))
    assert np.max(np.abs(rand_source.provider.crop_at(c) - ref64)) <= 1e-12
    assert np.array_equal(tgt[0], ref64.astype(np.float32))


def test_dataset_shapes_dtypes_and_interleaving():
    s1 = make_random_source(seed=21, physics_id=1,
                            cube=np.full((192, 192, 192), 1.0))
    s2 = make_random_source(seed=22, physics_id=2,
                            cube=np.full((192, 192, 192), 2.0))
    ds = UNetPairDataset([s1, s2], length=6, seed=1)
    for i in range(4):
        inp, tgt = ds[i]
        assert tuple(inp.shape) == (2, 64, 64, 64)
        assert tuple(tgt.shape) == (1, 64, 64, 64)
        assert str(inp.dtype) == "torch.float32"
        assert str(tgt.dtype) == "torch.float32"
        want = np.float32(np.log10(1.0)) if i % 2 == 0 else np.float32(
            np.log10(2.0))
        assert float(tgt[0, 0, 0, 0]) == pytest.approx(float(want), abs=0)


def test_dataset_determinism_under_seed(rand_source):
    a = UNetPairDataset([rand_source], length=6, seed=7)
    b = UNetPairDataset([rand_source], length=6, seed=7)
    c = UNetPairDataset([rand_source], length=6, seed=8)
    diverged = False
    for i in (0, 1, 5):
        ia, ta = a[i]
        ib, tb = b[(i)]
        assert np.array_equal(ia.numpy(), ib.numpy())
        assert np.array_equal(ta.numpy(), tb.numpy())
        ic, tc = c[i]
        diverged = diverged or not np.array_equal(ia.numpy(), ic.numpy())
    assert diverged
    # access-order independence: fresh instance, reversed access
    d = UNetPairDataset([rand_source], length=6, seed=7)
    i5 = d[5]
    assert np.array_equal(i5[0].numpy(), a[5][0].numpy())


def test_augmentation_applied_identically_to_input_and_target(rand_source):
    aug = UNetPairDataset([rand_source], length=8, seed=13, augment=True)
    raw = UNetPairDataset([rand_source], length=8, seed=13, augment=False)
    saw_nontrivial = False
    for i in range(8):
        ia, ta, sa, _ = aug.example(i)
        ir, tr, sr, _ = raw.example(i)
        assert np.array_equal(sa.corner, sr.corner)  # same spec both modes
        assert (sa.rot_k, sa.flip) == (sr.rot_k, sr.flip)
        assert np.array_equal(ia, apply_transverse_aug(ir, sa.rot_k, sa.flip))
        assert np.array_equal(ta, apply_transverse_aug(tr, sa.rot_k, sa.flip))
        # split axis untouched: axis-0 profile sets must match
        assert np.array_equal(
            np.sort(ia[1].sum(axis=(1, 2))), np.sort(ir[1].sum(axis=(1, 2)))
        )
        saw_nontrivial = saw_nontrivial or sa.rot_k != 0 or sa.flip
    assert saw_nontrivial


# ------------------------------------------------------------------ S2

def test_s2_p1_target_crop_bitexact_vs_pinned_cube():
    if not P1_CUBE.exists():
        pytest.fail(f"S2 artifact missing: {P1_CUBE} (exit criterion S2 "
                    f"cannot be waived)")
    h = hashlib.sha256()
    with open(P1_CUBE, "rb") as f:
        for blk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(blk)
    assert h.hexdigest() == P1_SHA256_PIN  # source identity (S2)
    raw = np.load(P1_CUBE)
    provider = TruthCropProvider(P1_CUBE, region="train", seed=0)
    x_ref, _ = x_transform(raw)
    for corner in ([0, 0, 0], [70, 128, 128], [10, 60, 5]):
        crop = provider.crop_at(np.array(corner))
        sl = tuple(slice(c, c + 64) for c in corner)
        assert np.array_equal(crop, x_ref[sl])  # bit-exact sampled crops
    # and via the dataset path (float32 emission is a pure cast)
    src = PhysicsSource(1, provider,
                        make_random_source(seed=33).geometry,
                        make_random_source(seed=33).delta_f)
    ds = UNetPairDataset([src], length=1, seed=0, augment=False)
    _, tgt, spec, _ = ds.example(0)
    c = spec.corner
    idx = [(c[a] + np.arange(64)) % 192 for a in range(3)]
    assert np.array_equal(tgt[0], x_ref[np.ix_(*idx)].astype(np.float32))
