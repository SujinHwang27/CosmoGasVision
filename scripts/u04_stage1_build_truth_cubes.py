#!/usr/bin/env python
"""[U-04] Stage-1 R2: P2-P4 truth cubes at 192^3 for the unet-inversion track.

Replicates the [D-75] P1 truth-cube producer EXACTLY (spec of record:
experiments/unet-inversion/design/u04_stage1_ratification.md §2(c)):

  1. 768^3 rho/<rho> field via the loader's three-tier path — the identical
     [D-48] disk cache + [D-50] chunked-CIC machinery that produced
     ``Sherwood/.rho_field_cache/rho_field_p1_z0.300_n768.npy`` (the sha256-
     pinned source of ``truth_real_192.npy``). We call
     ``SherwoodLoader.extract_rho_crops`` (tiny throwaway crop) so the cache
     materialization + atomic cache write is byte-for-byte the production
     code path, not a fork.
  2. 768^3 -> 192^3 mean-pool, float64 — ``mean_pool`` replicated verbatim
     from ``scripts/d75_corrected_metric_rescore.py:174-179`` (am-6 §S rule),
     matching ``stage_truth`` (ibid.:341-375).
  3. Save + JSON provenance manifest per cube (producer commit, source
     snapshot path/mtime, sha256s, mean-in-[0.95,1.05] check, wall clock).

Run from repo root:  PYTHONPATH=. .venv/bin/python -u scripts/u04_stage1_build_truth_cubes.py
"""

import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import loader as loader_mod  # noqa: E402
from src.data.loader import SherwoodLoader, _rho_cache_paths  # noqa: E402
from src.data.igm_gal_loader import PHYSICS_DIRS  # noqa: E402

Z = 0.3
N768 = 768
N192 = 192
OUT = REPO / "experiments" / "unet-inversion" / "artifacts" / "stage1" / "cubes"
PHYSICS = (2, 3, 4)


def mean_pool(cube: np.ndarray, target: int) -> np.ndarray:
    """Verbatim replica of scripts/d75_corrected_metric_rescore.py:174-179."""
    n = cube.shape[0]
    f = n // target
    assert n % target == 0
    return cube.reshape(target, f, target, f, target, f).mean(
        axis=(1, 3, 5), dtype=np.float64)


def _sha256(path: Path, first_mb_only: bool = False) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        if first_mb_only:
            h.update(fh.read(1024 * 1024))
        else:
            for blk in iter(lambda: fh.read(1 << 24), b""):
                h.update(blk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "UNKNOWN"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    head = _git_head()
    sl = SherwoodLoader(str(REPO / "Sherwood"))

    for p in PHYSICS:
        t0 = time.time()
        npy_path, json_path = _rho_cache_paths(p, Z, N768)
        cache_pre_existing = npy_path.exists() and json_path.exists()

        # ---- tier-2/3 materialization through the EXACT production path.
        # crop_size=8/n_crops=1 is a throwaway; its only role is to drive
        # extract_rho_crops' cache logic (CIC + atomic [D-48] cache write).
        sl.extract_rho_crops(
            physics_id=p, redshift=Z, crop_size=8, n_crops=1, n_grid=N768)
        t_field = time.time() - t0
        if not (npy_path.exists() and json_path.exists()):
            raise RuntimeError(
                f"P{p}: rho cache not on disk after materialization: {npy_path}")

        # Drop the in-memory copy; re-read from the on-disk cache so the cube
        # provenance chain starts at the persisted, hashable artifact —
        # identical to stage_truth reading RHO768 (d75 script:341-352).
        loader_mod._RHO_FIELD_CACHE.clear()
        gc.collect()

        sidecar = json.loads(json_path.read_text())
        sha_full = _sha256(npy_path)
        sha_1mb = _sha256(npy_path, first_mb_only=True)

        t1 = time.time()
        cube = np.load(npy_path)  # float32, ~1.7 GB
        pooled = mean_pool(cube, N192)
        del cube
        gc.collect()
        cube_path = OUT / f"truth_real_192_p{p}.npy"
        np.save(cube_path, pooled)
        t_pool = time.time() - t1

        mean = float(pooled.mean())
        mean_ok = 0.95 <= mean <= 1.05
        snap0 = (REPO / "SherwoodIGM_gal" / "extracted" / PHYSICS_DIRS[p]
                 / "snapdir_012" / "snap_012.0.hdf5")
        manifest = {
            "deliverable": "[U-04] Stage-1 R2 truth cube (A2)",
            "spec": "experiments/unet-inversion/design/u04_stage1_ratification.md §2(c), commit 58ac831",
            "producer_script": "scripts/u04_stage1_build_truth_cubes.py",
            "producer_git_head": head,
            "producer_lineage": "identical to scripts/d75_corrected_metric_rescore.py stage_truth (mean_pool lines 174-179 verbatim); field via SherwoodLoader.extract_rho_crops [D-48] cache + [D-50] chunked CIC",
            "physics_id": p,
            "physics_name": PHYSICS_DIRS[p],
            "redshift": Z,
            "source_snapshot": {
                "path": str(snap0.relative_to(REPO)),
                "mtime_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(snap0.stat().st_mtime)),
            },
            "rho768_cache": {
                "path": str(npy_path),
                "pre_existing": cache_pre_existing,
                "sha256_full": sha_full,
                "sha256_first_1MB": sha_1mb,
                "sidecar_manifest": sidecar,
                "sidecar_sha256_first_1MB_match": bool(
                    sha_1mb == sidecar.get("sha256_first_1MB")),
            },
            "pool": "768^3 -> 192^3 mean-pool (am-6 §S rule)",
            "cube": {
                "path": str(cube_path.relative_to(REPO)),
                "sha256": _sha256(cube_path),
                "dtype": "float64",
                "shape": [N192, N192, N192],
                "mean": mean,
                "std": float(pooled.std()),
                "min": float(pooled.min()),
                "max": float(pooled.max()),
                "mean_in_0.95_1.05": mean_ok,
            },
            "wall_clock_s": {
                "field_materialization": t_field,
                "pool_and_save": t_pool,
                "total": time.time() - t0,
            },
        }
        man_path = OUT / f"truth_real_192_p{p}.json"
        man_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[R2] P{p}: cache_pre_existing={cache_pre_existing} "
              f"field={t_field:.1f}s pool={t_pool:.1f}s mean={mean:.6f} "
              f"mean_ok={mean_ok} sha256={manifest['cube']['sha256'][:16]}...",
              flush=True)
        if not mean_ok:
            raise RuntimeError(f"P{p}: mean {mean} outside [0.95, 1.05]")
        del pooled
        gc.collect()

    print("[R2] DONE", flush=True)


if __name__ == "__main__":
    main()
