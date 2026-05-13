"""Numerical-equivalence + chunk-invariance + real-data regression for the
[D-50] chunked-scatter refactor of ``_cic_deposit_inplace``.

The pre-refactor implementation in ``src/data/igm_gal_loader.py`` allocated
one ``np.bincount(..., minlength=n_grid**3)`` per corner per call (~3.6 GB
at n_grid=768) and one ~165 MB int64 cast intermediate per corner on the
20.6M-particle P1 z=0.3 IGM_gal table. The refactor processes particles in
chunks of ``chunk_size`` (default 1M) and scatters into ``flat_view`` via
``np.add.at``, eliminating both allocations.

Gate per LEDGER \xa73 [D-50]:
  (b) np.allclose(rho_new, rho_old, rtol=1e-5, atol=1e-7) against a
      pre-refactor baseline at n_grid=64.

This file covers (b) via three independent paths:
  1. Synthetic against an inline reference (no Sherwood dependency).
  2. Chunk-size invariance (same inputs, varying chunk_size).
  3. Real-data against a stashed pre-refactor baseline at
     ``D:\\tmp\\cosmogasvision_d50\\baseline_p1_n64.npy`` (skipped if
     the snapshot is not present).

Run:
    PYTHONPATH=. uv run pytest tests/test_cic_chunked.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.data.igm_gal_loader import _cic_deposit_inplace


# ---------------------------------------------------------------- reference

def _cic_reference_per_corner_bincount(
    grid: np.ndarray,
    coords: np.ndarray,
    weights: np.ndarray,
    box: float,
    n_grid: int,
) -> None:
    """Pre-refactor implementation: one ``np.bincount(minlength=n_grid**3)``
    per corner, no chunking. Used as the equivalence reference. The full
    array sizes here are small (n_grid <= 32) so the n^3 bincount return
    is harmless.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    weights = np.ascontiguousarray(weights, dtype=np.float32)

    cell = np.float32(box / n_grid)
    fx = coords[:, 0] / cell
    fy = coords[:, 1] / cell
    fz = coords[:, 2] / cell

    ix = np.floor(fx).astype(np.int32)
    iy = np.floor(fy).astype(np.int32)
    iz = np.floor(fz).astype(np.int32)
    dx = fx - ix
    dy = fy - iy
    dz = fz - iz

    ix0 = ix % n_grid
    iy0 = iy % n_grid
    iz0 = iz % n_grid
    ix1 = (ix + 1) % n_grid
    iy1 = (iy + 1) % n_grid
    iz1 = (iz + 1) % n_grid

    n3 = n_grid * n_grid * n_grid
    flat_view = grid.reshape(n3)
    for ax, ay, az, wx, wy, wz in (
        (ix0, iy0, iz0, 1 - dx, 1 - dy, 1 - dz),
        (ix1, iy0, iz0,     dx, 1 - dy, 1 - dz),
        (ix0, iy1, iz0, 1 - dx,     dy, 1 - dz),
        (ix0, iy0, iz1, 1 - dx, 1 - dy,     dz),
        (ix1, iy1, iz0,     dx,     dy, 1 - dz),
        (ix1, iy0, iz1,     dx, 1 - dy,     dz),
        (ix0, iy1, iz1, 1 - dx,     dy,     dz),
        (ix1, iy1, iz1,     dx,     dy,     dz),
    ):
        idx = (
            ax.astype(np.int64) * n_grid * n_grid
            + ay.astype(np.int64) * n_grid
            + az
        )
        flat_view += np.bincount(
            idx, weights=weights * wx * wy * wz, minlength=n3
        )


# ----------------------------------------------------------------- fixtures

def _synthetic_particles(n: int, box: float, seed: int):
    """Deterministic float32 (coords, weights). Coords uniform in [0, box),
    weights drawn from a unit-mean LogNormal so that the per-cell sums
    are non-trivial (heterogeneous, with the long tail typical of a real
    density field)."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, box, size=(n, 3)).astype(np.float32)
    # Lognormal weights mimicking density-tracer mass spectrum: mean ~ 1
    weights = rng.lognormal(mean=0.0, sigma=0.5, size=n).astype(np.float32)
    return coords, weights


# ------------------------------------------------------------------- tests

@pytest.mark.parametrize("n_grid,n_particles,seed", [
    (16, 50_000, 0),
    (32, 200_000, 7),
    (32, 1_500_000, 13),  # spans multiple chunks at chunk_size=1M
])
def test_cic_chunked_numerical_equivalence(n_grid, n_particles, seed):
    """Gate (b) [D-50]: chunked scatter agrees with the per-corner-bincount
    reference within (rtol=1e-5, atol=1e-7) across small grid sizes and
    chunk-boundary-crossing particle counts.
    """
    box = 60.0  # kpc/h, scale-invariant for this test
    coords, weights = _synthetic_particles(n_particles, box, seed)

    grid_new = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
    _cic_deposit_inplace(grid_new, coords, weights, box, n_grid)

    grid_ref = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
    _cic_reference_per_corner_bincount(grid_ref, coords, weights, box, n_grid)

    # Mass-conservation sanity: both implementations conserve total weight
    # (modulo periodic wraparound, which both handle identically).
    assert np.isclose(
        grid_new.sum(), grid_ref.sum(), rtol=1e-10
    ), "mass conservation broken vs. reference"

    # The actual D-50 gate
    assert np.allclose(grid_new, grid_ref, rtol=1e-5, atol=1e-7), (
        f"chunked scatter deviates from reference: "
        f"max abs = {np.abs(grid_new - grid_ref).max():.3e}, "
        f"max rel = "
        f"{np.abs((grid_new - grid_ref) / np.where(grid_ref != 0, grid_ref, 1)).max():.3e}"
    )


@pytest.mark.parametrize("chunk_size", [1, 1_000, 100_000, 5_000_000])
def test_cic_chunk_invariance(chunk_size):
    """Same inputs, varying ``chunk_size`` must produce results that agree
    within (rtol=1e-5, atol=1e-7). chunk_size=1 is the worst case for
    summation-order drift; chunk_size > N collapses to a single chunk.
    """
    n_grid = 32
    box = 60.0
    coords, weights = _synthetic_particles(500_000, box, seed=42)

    grid_chunked = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
    _cic_deposit_inplace(
        grid_chunked, coords, weights, box, n_grid, chunk_size=chunk_size
    )

    grid_single = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
    _cic_deposit_inplace(
        grid_single, coords, weights, box, n_grid, chunk_size=10_000_000
    )

    assert np.isclose(grid_chunked.sum(), grid_single.sum(), rtol=1e-10)
    assert np.allclose(grid_chunked, grid_single, rtol=1e-5, atol=1e-7), (
        f"chunk_size={chunk_size} drift from single-chunk: "
        f"max abs = {np.abs(grid_chunked - grid_single).max():.3e}"
    )


def test_cic_empty_input_no_op():
    """N=0 must be a clean no-op (no IndexError, grid unchanged)."""
    n_grid = 8
    box = 60.0
    grid = np.full((n_grid, n_grid, n_grid), 3.14, dtype=np.float64)
    grid_before = grid.copy()
    _cic_deposit_inplace(
        grid,
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        box,
        n_grid,
    )
    assert np.array_equal(grid, grid_before)


def test_cic_invalid_chunk_size_raises():
    """chunk_size must be positive; 0 and negatives raise ValueError."""
    grid = np.zeros((4, 4, 4), dtype=np.float64)
    coords = np.zeros((1, 3), dtype=np.float32)
    weights = np.ones((1,), dtype=np.float32)
    with pytest.raises(ValueError, match="chunk_size"):
        _cic_deposit_inplace(grid, coords, weights, 1.0, 4, chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size"):
        _cic_deposit_inplace(grid, coords, weights, 1.0, 4, chunk_size=-5)


# ----------------------------------------- real-data baseline regression


_BASELINE_NPY = Path(
    os.environ.get(
        "COSMOGAS_D50_BASELINE_NPY",
        r"D:\tmp\cosmogasvision_d50\baseline_p1_n64.npy",
    )
)


@pytest.mark.skipif(
    not _BASELINE_NPY.is_file(),
    reason=(
        "pre-refactor baseline snapshot missing at "
        f"{_BASELINE_NPY} (run scripts/d50_snapshot_baseline.py to create)"
    ),
)
def test_cic_real_baseline_match(tmp_path, monkeypatch):
    """Gate (b) [D-50] against a real-data baseline. The snapshot at
    ``D:\\tmp\\cosmogasvision_d50\\baseline_p1_n64.npy`` was produced by
    the pre-refactor per-corner-bincount implementation on Sherwood P1
    z=0.300 n_grid=64; re-running the loader against a fresh cache dir
    triggers the new chunked-scatter CIC, and the resulting rho field
    must match the baseline within (rtol=1e-5, atol=1e-7).
    """
    # Late import: SherwoodLoader's transitive deps are heavy and we want
    # the synthetic tests above to remain importable without them resolved.
    import src.data.loader as loader_mod
    from src.data.loader import SherwoodLoader, _RHO_FIELD_CACHE

    repo_root = Path(__file__).resolve().parent.parent
    igm_gal_root = repo_root / "SherwoodIGM_gal" / "extracted"
    snap_dir = igm_gal_root / "planck1_60_768_z0.300" / "snapdir_012"
    if not snap_dir.is_dir() or not any(
        f.name.startswith("snap_012.") and f.name.endswith(".hdf5")
        for f in snap_dir.iterdir()
    ):
        pytest.skip("P1 IGM_gal snapshot not available locally")

    cache_dir = tmp_path / "rho_cache_d50"
    cache_dir.mkdir()
    monkeypatch.setenv("COSMOGAS_RHO_CACHE_DIR", str(cache_dir))
    _RHO_FIELD_CACHE.clear()

    loader = SherwoodLoader(data_root=str(repo_root / "Sherwood"))
    _ = loader.extract_rho_crops(
        physics_id=1,
        redshift=0.300,
        crop_size=8,
        n_crops=2,
        seed=42,
        n_grid=64,
    )

    new_npy_path = cache_dir / "rho_field_p1_z0.300_n64.npy"
    assert new_npy_path.is_file(), (
        f"refactored loader did not write the cache .npy at {new_npy_path}"
    )

    rho_new = np.load(new_npy_path)
    rho_old = np.load(_BASELINE_NPY)
    assert rho_new.shape == rho_old.shape == (64, 64, 64)
    assert rho_new.dtype == rho_old.dtype

    # Mass-conservation (ρ/<ρ>: mean must be ~1 in both)
    assert np.isclose(rho_new.mean(), rho_old.mean(), rtol=1e-6), (
        f"mean drift: new={rho_new.mean():.6f} old={rho_old.mean():.6f}"
    )

    assert np.allclose(rho_new, rho_old, rtol=1e-5, atol=1e-7), (
        f"real-data CIC field deviates from pre-refactor baseline: "
        f"max abs = {np.abs(rho_new - rho_old).max():.3e}, "
        f"max rel = "
        f"{np.abs((rho_new - rho_old) / np.where(rho_old != 0, rho_old, 1)).max():.3e}"
    )
