#!/usr/bin/env python
"""[U-06] A5 follow-up (B1 panel amendments K1/K2), part 2: Monte Carlo error
of the null-band 97.5th-percentile edges.

For BOTH banked bands (test: null_band_n200.json, val: null_band_val_n200.json):

  (i)  Bootstrap the banked N=200 values (10,000 resamples, seeded) ->
       bootstrap SE + 95% CI on pct_97p5 per (frame, sigma, metric).
  (ii) N=1000 extension at sigma=2 (both frames, both metrics, BOTH masks in
       one pass): realizations i = 200..999 with the same seed family
       default_rng([20260726, i]); combined with the banked 200 values ->
       N=1000 median / [2.5, 97.5] edges + bootstrap CI on the N=1000 edge.

Both blocks are APPENDED to the respective band JSONs under new keys
("mc_error_97p5_edge", "n1000_extension_sigma2"); existing content preserved.

Machinery identical to scripts/u06_a5_null_band.py (R9 conventions imported:
smooth full periodic cube first, then mask; masks loader-derived).

Usage: PYTHONPATH=. .venv/bin/python -u scripts/u06_a5_mc_error.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.analysis import nccf as NC                                 # noqa: E402
from scripts.d75_corrected_metric_rescore import (                  # noqa: E402
    BOX_MPC_H, N192, SEED_PHASE, x_transform, load_cube, _js)
from scripts.u04_r9_heldout_rescore import (                        # noqa: E402
    masked_pearson, masked_spearman)
from src.data.loader import DEFAULT_SCHEME, region_voxel_interval   # noqa: E402

OUT_DIR = REPO / "experiments" / "unet-inversion" / "artifacts" / "stage2"
BAND_JSON = {"test": OUT_DIR / "null_band_n200.json",
             "val": OUT_DIR / "null_band_val_n200.json"}
PARTIAL = OUT_DIR / "n1000_sigma2.partial.npz"

FRAMES = ("real", "zspace")
METRICS = ("pearson", "spearman")
SIGMA2 = 2.0
N_BASE, N_EXT = 200, 1000
CHUNK = 20
N_BOOT = 10_000
BOOT_SEED = {"test": [SEED_PHASE, 777001], "val": [SEED_PHASE, 777002]}
BOOT_SEED_N1000 = {"test": [SEED_PHASE, 777011], "val": [SEED_PHASE, 777012]}


def boot_edge(values: np.ndarray, seed, n_boot=N_BOOT):
    """Bootstrap SE + 95% CI of the 97.5th-percentile edge."""
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    edges = np.percentile(values[idx], 97.5, axis=1)
    return {"edge": float(np.percentile(values, 97.5)),
            "boot_se": float(edges.std(ddof=1)),
            "boot_ci95": [float(np.percentile(edges, 2.5)),
                          float(np.percentile(edges, 97.5))],
            "n": int(n)}


def main():
    t0 = time.time()

    bands = {}
    for reg, p in BAND_JSON.items():
        bands[reg] = json.loads(p.read_text())
        assert "band" in bands[reg], f"{p} missing band block"

    # -------- (i) bootstrap MC error on the banked N=200 edges --------------
    for reg in ("test", "val"):
        mc = {"method": f"nonparametric bootstrap of the banked N=200 "
                        f"values, {N_BOOT} resamples, 97.5th percentile per "
                        "resample; SE = std(ddof=1), CI95 = [2.5, 97.5] "
                        "percentiles of the bootstrap distribution",
              "seed": BOOT_SEED[reg], "n_boot": N_BOOT}
        for fr in FRAMES:
            mc[fr] = {}
            for sg in ("1", "2", "4"):
                mc[fr][sg] = {}
                for m in METRICS:
                    v = np.asarray(bands[reg]["band"][fr][sg][m]["values"])
                    assert len(v) == N_BASE
                    mc[fr][sg][m] = boot_edge(v, BOOT_SEED[reg])
        bands[reg]["mc_error_97p5_edge"] = mc
        e = mc["real"]["2"]["pearson"]
        print(f"[a5mc] {reg}: n200 edge(real,s2,pearson)={e['edge']:+.4f} "
              f"se={e['boot_se']:.4f} ci95=[{e['boot_ci95'][0]:+.4f},"
              f"{e['boot_ci95'][1]:+.4f}]", flush=True)

    # -------- (ii) N=1000 extension, sigma=2, both frames, both masks -------
    masks = {}
    for reg in ("test", "val"):
        lo, hi = region_voxel_interval(reg, N192, DEFAULT_SCHEME)
        assert DEFAULT_SCHEME.axis == 0 and hi - lo == 29
        m = np.zeros((N192, N192, N192), dtype=bool)
        m[lo:hi, :, :] = True
        masks[reg] = m
        print(f"[a5mc] {reg} mask=[{lo},{hi}) (loader-derived)", flush=True)

    x_real, _ = x_transform(load_cube("truth_real_192.npy"))
    x_z, _ = x_transform(load_cube("truth_zspace_192.npy"))
    st = {"real": NC.gaussian_smooth_periodic(x_real, BOX_MPC_H, SIGMA2),
          "zspace": NC.gaussian_smooth_periodic(x_z, BOX_MPC_H, SIGMA2)}
    del x_z
    print(f"[a5mc] truth loaded + smoothed s2 ({time.time()-t0:.0f}s)",
          flush=True)

    # ext values: (region, frame, metric, realization 200..999)
    ext = np.full((2, len(FRAMES), len(METRICS), N_EXT - N_BASE), np.nan)
    start = N_BASE
    if PARTIAL.exists():
        ck = np.load(PARTIAL)
        ext, start = ck["ext"], int(ck["done"])
        print(f"[a5mc] resuming: {start} done", flush=True)

    regs = ("test", "val")
    for i in range(start, N_EXT):
        pr = NC.phase_randomized(x_real, [SEED_PHASE, i])
        prs = NC.gaussian_smooth_periodic(pr, BOX_MPC_H, SIGMA2)
        j = i - N_BASE
        for ri, reg in enumerate(regs):
            for fi, fr in enumerate(FRAMES):
                ext[ri, fi, 0, j] = masked_pearson(st[fr], prs, masks[reg])
                ext[ri, fi, 1, j] = masked_spearman(st[fr], prs, masks[reg])
        if (i + 1) % CHUNK == 0 or i == N_EXT - 1:
            np.savez(PARTIAL, ext=ext, done=i + 1)
            el = time.time() - t0
            print(f"[a5mc] {i+1}/{N_EXT} ({el:.0f}s, "
                  f"{el/(i+1-start):.1f}s/real)", flush=True)

    for ri, reg in enumerate(regs):
        blk = {"seeds": f"default_rng([{SEED_PHASE}, i]) i=0..{N_EXT-1}; "
                        f"i=0..{N_BASE-1} reused from the banked N=200 band "
                        "(identical construction), i=200..999 computed here",
               "sigma": SIGMA2}
        for fi, fr in enumerate(FRAMES):
            blk[fr] = {}
            for mi, m in enumerate(METRICS):
                v200 = np.asarray(bands[reg]["band"][fr]["2"][m]["values"])
                v1000 = np.concatenate([v200, ext[ri, fi, mi, :]])
                assert len(v1000) == N_EXT and np.all(np.isfinite(v1000))
                be = boot_edge(v1000, BOOT_SEED_N1000[reg])
                blk[fr][m] = {
                    "pct_97p5_n200": float(np.percentile(v200, 97.5)),
                    "pct_97p5_n1000": be["edge"],
                    "median_n1000": float(np.median(v1000)),
                    "pct_2p5_n1000": float(np.percentile(v1000, 2.5)),
                    "min_n1000": float(v1000.min()),
                    "max_n1000": float(v1000.max()),
                    "boot_se_edge_n1000": be["boot_se"],
                    "boot_ci95_edge_n1000": be["boot_ci95"],
                    "boot_seed": BOOT_SEED_N1000[reg],
                    "values_i200_to_999": ext[ri, fi, mi, :].tolist(),
                }
        bands[reg]["n1000_extension_sigma2"] = blk
        for fr in FRAMES:
            b = blk[fr]["pearson"]
            print(f"[a5mc] {reg:4s} {fr:6s} s2 pearson edge: "
                  f"n200={b['pct_97p5_n200']:+.4f} -> "
                  f"n1000={b['pct_97p5_n1000']:+.4f} "
                  f"ci95=[{b['boot_ci95_edge_n1000'][0]:+.4f},"
                  f"{b['boot_ci95_edge_n1000'][1]:+.4f}]", flush=True)

    for reg, p in BAND_JSON.items():
        bands[reg].setdefault("amendments", []).append(
            "B1 panel K1/K2 follow-up (2026-07-24): appended "
            "mc_error_97p5_edge + n1000_extension_sigma2 blocks "
            "(scripts/u06_a5_mc_error.py); pre-existing content unchanged")
        bands[reg]["mc_wall_clock_s"] = time.time() - t0
        p.write_text(json.dumps(_js(bands[reg]), indent=2))
        print(f"[a5mc] wrote {p}", flush=True)
    PARTIAL.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
