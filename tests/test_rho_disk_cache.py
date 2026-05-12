"""
Regression tests for the rho-field disk-cache layer in
`src.data.loader.SherwoodLoader.extract_rho_crops`.

Sprint-1 of the [D-46] / [D-47] Stage 3 infrastructure prep. Validates the
7 spec criteria from the dispatch:

  (1) Cache location: D:\\...\\Sherwood\\.rho_field_cache\\, env override
      `COSMOGAS_RHO_CACHE_DIR` honored. Covered via the env override the
      whole module uses (tmp_path); a separate D:-default sanity test is
      kept lightweight to avoid writing to the real cache dir in CI.
  (2) Cache key schema: filename `rho_field_p{pid}_z{z:.3f}_n{n}.npy` +
      sidecar manifest .json with the documented fields. Implicitly
      covered by the cold/warm tests.
  (3) Integration point: in-memory miss -> disk hit skips CIC; disk miss
      -> CIC runs and re-caches. Covered by
      `test_warm_disk_hit_skips_cic` + `test_cold_cache_populates_*`.
  (4) Cache-hit timing gate: <= 15 s for the production case. Covered by
      `test_cache_hit_timing_gate` (marked @pytest.mark.slow because the
      cold-call cost at n_grid=768 is ~9 min).
  (5) Corruption / mismatch fall-back: warns + falls through, no bubble.
      Covered by `test_mtime_mismatch_triggers_recompute` and
      `test_corrupted_npy_triggers_fallback`.
  (6) Test coverage: this file, 5 tests + the timing gate.
  (7) Gitignore / DVC hygiene: not test-covered — visual check at PR.

Run:
    PYTHONPATH=. uv run pytest tests/test_rho_disk_cache.py -v
    PYTHONPATH=. uv run pytest tests/test_rho_disk_cache.py -v -m slow
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch  # noqa: F401  # imported transitively by loader

import src.data.loader as loader_mod
from src.data.loader import (
    SherwoodLoader,
    _RHO_CACHE_SCHEMA_VERSION,
    _RHO_FIELD_CACHE,
    _rho_cache_paths,
    _sherwood_snapshot_mtime_utc,
    _write_rho_disk_cache,
)

# --------------------------------------------------------------------- config
REPO_ROOT = Path(__file__).resolve().parent.parent
IGM_GAL_ROOT = REPO_ROOT / "SherwoodIGM_gal" / "extracted"
REDSHIFT = 0.300
N_GRID_FAST = 64  # CIC ~ a few seconds, deposition cost is non-trivial
CROP_SIZE = 8
N_CROPS = 2

PHYSICS_DIRS = {
    1: "planck1_60_768_z0.300",
    2: "planck1_60_768_ps13_z0.300",
    3: "planck1_60_768_ps13agn_z0.300",
    4: "planck1_60_768_ps13agn_strong_z0.300",
}


def _have_physics(physics_id: int) -> bool:
    snap_dir = IGM_GAL_ROOT / PHYSICS_DIRS[physics_id] / "snapdir_012"
    if not snap_dir.is_dir():
        return False
    return any(
        f.name.startswith("snap_012.") and f.name.endswith(".hdf5")
        for f in snap_dir.iterdir()
    )


_skip_no_p1 = pytest.mark.skipif(
    not _have_physics(1),
    reason="P1 IGM_gal snapshot not available locally",
)


# ----------------------------------------------------------------------- fxts


@pytest.fixture(scope="session")
def _shared_cic_npy(tmp_path_factory) -> Path:
    """Materialize the CIC rho field ONCE per pytest session into a session-
    scoped tmp dir, then let each per-test fixture copy it into its own tmp
    cache dir. Without this, each test re-pays the n_grid=64 CIC cost (~8 min
    on this laptop), turning a 5-test run into a 40+ minute marathon. With
    it, the first test does the CIC, all subsequent ones copy ~1 MB.
    """
    if not _have_physics(1):
        pytest.skip("P1 IGM_gal snapshot not available locally")
    session_dir = tmp_path_factory.mktemp("rho_cache_shared")
    # Drive a single warm production-path write so we get both the .npy and
    # .json manifests, then return the directory holding them.
    import os as _os

    _os.environ["COSMOGAS_RHO_CACHE_DIR"] = str(session_dir)
    _RHO_FIELD_CACHE.clear()
    loader = SherwoodLoader(data_root=str(REPO_ROOT / "Sherwood"))
    _ = loader.extract_rho_crops(
        physics_id=1,
        redshift=REDSHIFT,
        crop_size=CROP_SIZE,
        n_crops=N_CROPS,
        seed=42,
        n_grid=N_GRID_FAST,
    )
    npy_path, json_path = _rho_cache_paths(
        1, REDSHIFT, N_GRID_FAST, cache_dir=session_dir
    )
    assert npy_path.exists() and json_path.exists()
    return session_dir


@pytest.fixture
def fresh_cache_dir(tmp_path, monkeypatch) -> Path:
    """Per-test EMPTY cache dir. Use this in tests that need to observe the
    cold-CIC -> disk-cache write path. Note: this will pay the CIC cost.
    Currently only ``test_cold_cache_populates_disk_and_memory`` needs it.
    """
    cache_dir = tmp_path / "rho_cache_empty"
    cache_dir.mkdir()
    monkeypatch.setenv("COSMOGAS_RHO_CACHE_DIR", str(cache_dir))
    yield cache_dir


@pytest.fixture
def prewarmed_cache_dir(tmp_path, monkeypatch, _shared_cic_npy) -> Path:
    """Per-test cache dir pre-seeded with a valid .npy+.json from the
    session-scoped CIC run. Tests using this fixture observe the disk-hit
    path WITHOUT having to re-run CIC.
    """
    import shutil

    cache_dir = tmp_path / "rho_cache_prewarmed"
    cache_dir.mkdir()
    base = (
        f"rho_field_p1_z{REDSHIFT:.3f}_n{N_GRID_FAST}"
    )
    shutil.copy2(_shared_cic_npy / f"{base}.npy", cache_dir / f"{base}.npy")
    shutil.copy2(_shared_cic_npy / f"{base}.json", cache_dir / f"{base}.json")
    monkeypatch.setenv("COSMOGAS_RHO_CACHE_DIR", str(cache_dir))
    yield cache_dir


@pytest.fixture(autouse=True)
def _clear_in_memory_cache():
    """Each test starts with a clean in-memory cache; otherwise a previous
    test's CIC-deposited field would short-circuit the disk-cache path."""
    _RHO_FIELD_CACHE.clear()
    yield
    _RHO_FIELD_CACHE.clear()


@pytest.fixture(scope="module")
def loader() -> SherwoodLoader:
    return SherwoodLoader(data_root=str(REPO_ROOT / "Sherwood"))


def _cache_paths(physics_id: int, redshift: float, n_grid: int):
    return _rho_cache_paths(physics_id, redshift, n_grid)


# ------------------------------------------------------------------- tests


@_skip_no_p1
def test_cold_cache_populates_disk_and_memory(loader, fresh_cache_dir):
    """Spec (1)+(2)+(3): cold call writes the .npy + manifest with the
    correct filename schema, populates in-memory, and a follow-up call
    finds both layers warm.
    """
    npy_path, json_path = _cache_paths(1, REDSHIFT, N_GRID_FAST)
    assert not npy_path.exists()
    assert not json_path.exists()

    crops, labels = loader.extract_rho_crops(
        physics_id=1,
        redshift=REDSHIFT,
        crop_size=CROP_SIZE,
        n_crops=N_CROPS,
        seed=42,
        n_grid=N_GRID_FAST,
    )
    # Output contract intact (regression vs existing extract_rho_crops tests)
    assert crops.shape == (N_CROPS, 1, CROP_SIZE, CROP_SIZE, CROP_SIZE)
    assert labels.shape == (N_CROPS,)

    # Disk-side: both files exist with the schema'd names
    assert npy_path.exists(), f"expected disk-cache .npy at {npy_path}"
    assert json_path.exists(), f"expected manifest at {json_path}"
    assert npy_path.name == f"rho_field_p1_z{REDSHIFT:.3f}_n{N_GRID_FAST}.npy"
    assert json_path.name == f"rho_field_p1_z{REDSHIFT:.3f}_n{N_GRID_FAST}.json"

    # Manifest content sanity
    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == _RHO_CACHE_SCHEMA_VERSION
    assert manifest["physics_id"] == 1
    assert manifest["n_grid"] == N_GRID_FAST
    assert round(float(manifest["redshift"]), 3) == round(REDSHIFT, 3)
    assert manifest["shape"] == [N_GRID_FAST, N_GRID_FAST, N_GRID_FAST]
    assert manifest["dtype"] == "float32"
    assert 0.95 <= float(manifest["mean"]) <= 1.05
    assert float(manifest["min"]) >= 0.0
    # First-MB hash should be a 64-char hex sha256
    assert isinstance(manifest["sha256_first_1MB"], str)
    assert len(manifest["sha256_first_1MB"]) == 64
    # mtime is either an ISO string or None (the IGM_gal snap is in-tree
    # for this test, so it should resolve to a real timestamp)
    assert manifest["sherwood_snapshot_mtime_utc"] is not None

    # In-memory cache populated
    in_mem_key = (1, round(REDSHIFT, 3), N_GRID_FAST)
    assert in_mem_key in _RHO_FIELD_CACHE


@_skip_no_p1
def test_warm_disk_hit_skips_cic(loader, prewarmed_cache_dir, monkeypatch):
    """Spec (3): on in-memory miss + valid disk hit, `load_3d_field` is
    NOT invoked. We monkeypatch `SherwoodIGMGalLoader.load_3d_field` to a
    sentinel that raises if called; a successful return therefore proves
    the disk path was hit.
    """
    # Sentinel: any call to load_3d_field is a failure
    from src.data import igm_gal_loader as igm_mod

    def _boom(self, *a, **kw):  # noqa: ARG001
        raise AssertionError(
            "load_3d_field was invoked but the disk cache should have "
            "satisfied the request."
        )

    monkeypatch.setattr(
        igm_mod.SherwoodIGMGalLoader, "load_3d_field", _boom, raising=True
    )

    # Warm disk call: must succeed without invoking the sentinel.
    crops, labels = loader.extract_rho_crops(
        physics_id=1,
        redshift=REDSHIFT,
        crop_size=CROP_SIZE,
        n_crops=N_CROPS,
        seed=42,
        n_grid=N_GRID_FAST,
    )
    assert crops.shape == (N_CROPS, 1, CROP_SIZE, CROP_SIZE, CROP_SIZE)
    assert (labels == 1).all()
    # And in-memory was repopulated
    assert (1, round(REDSHIFT, 3), N_GRID_FAST) in _RHO_FIELD_CACHE


@_skip_no_p1
def test_mtime_mismatch_triggers_recompute(loader, prewarmed_cache_dir):
    """Spec (5): if the manifest's `sherwood_snapshot_mtime_utc` does not
    match the current upstream snapshot mtime, the disk entry is treated
    as stale: it is removed and CIC re-runs.
    """
    npy_path, json_path = _cache_paths(1, REDSHIFT, N_GRID_FAST)

    # 1) Tamper with the pre-seeded manifest's mtime
    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    original_mtime = manifest["sherwood_snapshot_mtime_utc"]
    stale_mtime = "1970-01-01T00:00:00Z"
    assert original_mtime != stale_mtime
    manifest["sherwood_snapshot_mtime_utc"] = stale_mtime
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    # 2) Call: expect a RuntimeWarning naming the mtime field, then a fresh
    # CIC + re-cache.
    with pytest.warns(RuntimeWarning, match="sherwood_snapshot_mtime_utc"):
        _ = loader.extract_rho_crops(
            physics_id=1,
            redshift=REDSHIFT,
            crop_size=CROP_SIZE,
            n_crops=N_CROPS,
            seed=42,
            n_grid=N_GRID_FAST,
        )

    # 3) Disk-cache must be re-written with the correct (current) mtime
    assert npy_path.exists() and json_path.exists()
    manifest_fresh = json.loads(json_path.read_text(encoding="utf-8"))
    assert manifest_fresh["sherwood_snapshot_mtime_utc"] == original_mtime


@_skip_no_p1
def test_corrupted_npy_triggers_fallback(loader, prewarmed_cache_dir):
    """Spec (5): a truncated .npy must produce a `warnings.warn`, the
    corrupt entry is removed, CIC runs, and the disk cache is replaced
    with a valid entry.
    """
    npy_path, json_path = _cache_paths(1, REDSHIFT, N_GRID_FAST)
    original_size = npy_path.stat().st_size
    assert original_size > 4096

    # 1) Truncate .npy in place to ~1 KB (header survives, body is gone).
    #    This will surface either as a sha256_first_1MB mismatch (if the
    #    first 1 MB now differs) or as an np.load failure on shape readback.
    with open(npy_path, "rb+") as fh:
        fh.truncate(1024)

    # 2) Expect a RuntimeWarning + successful recovery via fresh CIC
    with pytest.warns(RuntimeWarning):
        crops, _ = loader.extract_rho_crops(
            physics_id=1,
            redshift=REDSHIFT,
            crop_size=CROP_SIZE,
            n_crops=N_CROPS,
            seed=42,
            n_grid=N_GRID_FAST,
        )
    assert crops.shape == (N_CROPS, 1, CROP_SIZE, CROP_SIZE, CROP_SIZE)

    # 3) Disk cache replaced with a healthy entry (same size as the original
    #    deterministic CIC output)
    assert npy_path.exists() and json_path.exists()
    assert npy_path.stat().st_size == original_size


@_skip_no_p1
def test_default_cache_dir_resolves_to_d_drive(monkeypatch):
    """Spec (1): when COSMOGAS_RHO_CACHE_DIR is unset, the resolved cache
    directory must live on D:\\, not C:\\ or under ~/. (Path-resolution
    sanity only — we do NOT write to it.)
    """
    monkeypatch.delenv("COSMOGAS_RHO_CACHE_DIR", raising=False)
    resolved = loader_mod._resolve_rho_cache_dir()
    assert resolved.drive.lower() == "d:", (
        f"default rho-cache dir must resolve to D: drive, got {resolved!r}"
    )
    # And the expected basename
    assert resolved.name == ".rho_field_cache"
    assert resolved.parent.name == "Sherwood"


# ------------------------------------------------------------------- slow gate


@_skip_no_p1
@pytest.mark.slow
def test_cache_hit_timing_gate(loader):
    """Spec (4): second call after fresh in-memory eviction must return in
    <= 15 s for the production case (physics_id=1, z=0.300, n_grid=768).
    The first call performs the full CIC deposition (~9 min); we allow up
    to 15 minutes for it.

    Marked `slow` so default test runs skip this. Invoke with:
        pytest tests/test_rho_disk_cache.py -v -m slow
    """
    n_grid = 768

    # Cold call: time-budgeted at 15 min (the spec ceiling); record actual.
    t0 = time.perf_counter()
    _ = loader.extract_rho_crops(
        physics_id=1,
        redshift=REDSHIFT,
        crop_size=CROP_SIZE,
        n_crops=N_CROPS,
        seed=42,
        n_grid=n_grid,
    )
    cold_seconds = time.perf_counter() - t0
    assert cold_seconds < 15 * 60, (
        f"cold CIC deposition took {cold_seconds:.1f} s, > 15 min ceiling"
    )

    # Evict in-memory only (disk cache persists)
    _RHO_FIELD_CACHE.clear()

    # Warm call: <= 15 s gate
    t0 = time.perf_counter()
    _ = loader.extract_rho_crops(
        physics_id=1,
        redshift=REDSHIFT,
        crop_size=CROP_SIZE,
        n_crops=N_CROPS,
        seed=42,
        n_grid=n_grid,
    )
    warm_seconds = time.perf_counter() - t0
    print(
        f"\n[timing gate] cold={cold_seconds:.2f} s   warm={warm_seconds:.2f} s"
    )
    assert warm_seconds <= 15.0, (
        f"warm disk-cache hit took {warm_seconds:.2f} s, > 15 s budget"
    )
