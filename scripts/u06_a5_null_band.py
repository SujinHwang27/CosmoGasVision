#!/usr/bin/env python
"""[U-06] Stage-2 rung A5: pre-registered N=200 phase-randomized null band.

Spec of record: experiments/unet-inversion/design/u06_stage2_spec.md §(c)
null-band protocol (branch HEAD 489a0d3).

Protocol:
  - N = 200 phase-randomized realizations of truth_real_192 (identical |FFT|
    amplitudes, random phases; NC.phase_randomized, imported — the exact d75
    controls construction).
  - Seeds: numpy.random.default_rng([20260726, i]) for i = 0..199. Verified
    property (recorded below): default_rng([20260726, 0]) yields the identical
    stream to default_rng(20260726), so realization 0 IS the banked
    seed-20260726 d75/R9 control draw — lineage check asserts its masked
    r_s(sigma=2, real) reproduces the banked 0.1185251247959566 to <= 1e-6.
  - Scoring: R9 conventions IMPORTED from scripts/u04_r9_heldout_rescore.py
    (masked_pearson / masked_spearman) — smooth the FULL periodic cube FIRST
    (NC.gaussian_smooth_periodic, sigma in {1,2,4} h^-1 Mpc), then restrict
    to the [D-49] test mask region_voxel_interval('test', 192) = axis-0
    slab [163, 192), runtime-asserted.
  - Both frames (real primary, zspace column; truth_zspace md5-pinned),
    Pearson + Spearman.

Output: experiments/unet-inversion/artifacts/stage2/null_band_n200.json
  per (frame, sigma, metric): median, [2.5, 97.5] percentiles, min/max,
  all 200 values; realization-0 lineage check; identity hashes; wall-clock.

Checkpointing: partial results appended to null_band_n200.partial.npz every
CHUNK realizations; restart resumes from the last completed chunk.

Usage: PYTHONPATH=. .venv/bin/python -u scripts/u06_a5_null_band.py
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
    CUBES, BOX_MPC_H, N192, SIGMAS, SEED_PHASE, x_transform, load_cube,
    _sha256, _md5, _js)
from scripts.u04_r9_heldout_rescore import (                        # noqa: E402
    masked_pearson, masked_spearman,
    PIN_TRUTH_REAL_SHA256, PIN_TRUTH_ZSPACE_MD5)
from src.data.loader import DEFAULT_SCHEME, region_voxel_interval   # noqa: E402

N_REAL = 200
CHUNK = 20
FRAMES = ("real", "zspace")
METRICS = ("pearson", "spearman")
LINEAGE_BANKED = 0.11852512479595659   # r9_heldout_bars.json
LINEAGE_TOL = 1e-6

OUT_DIR = REPO / "experiments" / "unet-inversion" / "artifacts" / "stage2"
OUT = OUT_DIR / "null_band_n200.json"
PARTIAL = OUT_DIR / "null_band_n200.partial.npz"


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- identity checks at load (r1 pins of record) -----------------------
    identity = []
    for fname, kind, pin, fn in (
            ("truth_real_192.npy", "sha256_full", PIN_TRUTH_REAL_SHA256,
             _sha256),
            ("truth_zspace_192.npy", "md5", PIN_TRUTH_ZSPACE_MD5, _md5)):
        obs = fn(CUBES / fname)
        v = "MATCH" if obs == pin else "MISMATCH"
        identity.append({"file": fname, "kind": kind, "pinned": pin,
                         "observed": obs, "verdict": v})
        print(f"[a5] identity {fname} {kind}: {v}", flush=True)
        if v == "MISMATCH":
            raise SystemExit(f"IDENTITY MISMATCH: {fname}")

    # ---- mask (runtime-asserted, per R9) ------------------------------------
    lo, hi = region_voxel_interval("test", N192, DEFAULT_SCHEME)
    assert (lo, hi) == (163, 192), f"unexpected test interval [{lo},{hi})"
    assert DEFAULT_SCHEME.axis == 0
    mask = np.zeros((N192, N192, N192), dtype=bool)
    mask[lo:hi, :, :] = True

    # ---- truth fields, smoothed FIRST on the full periodic cube ------------
    x_real, _ = x_transform(load_cube("truth_real_192.npy"))
    x_z, _ = x_transform(load_cube("truth_zspace_192.npy"))
    smoothed_truth = {
        "real": {s: NC.gaussian_smooth_periodic(x_real, BOX_MPC_H, s)
                 for s in SIGMAS},
        "zspace": {s: NC.gaussian_smooth_periodic(x_z, BOX_MPC_H, s)
                   for s in SIGMAS},
    }
    del x_z
    print(f"[a5] truth loaded + smoothed ({time.time()-t0:.0f}s)", flush=True)

    # ---- seed-equivalence record (realization-0 lineage precondition) ------
    seed0_equiv = bool(np.array_equal(
        np.random.default_rng(SEED_PHASE).standard_normal(8),
        np.random.default_rng([SEED_PHASE, 0]).standard_normal(8)))
    print(f"[a5] default_rng([{SEED_PHASE},0]) == default_rng({SEED_PHASE}): "
          f"{seed0_equiv}", flush=True)

    # ---- values array: (frame, sigma, metric, realization) -----------------
    vals = np.full((len(FRAMES), len(SIGMAS), len(METRICS), N_REAL), np.nan)
    start = 0
    if PARTIAL.exists():
        ck = np.load(PARTIAL)
        vals, start = ck["vals"], int(ck["done"])
        print(f"[a5] resuming from checkpoint: {start} done", flush=True)

    lineage = None
    for i in range(start, N_REAL):
        pr = NC.phase_randomized(x_real, [SEED_PHASE, i])
        for si, s in enumerate(SIGMAS):
            prs = NC.gaussian_smooth_periodic(pr, BOX_MPC_H, s)
            for fi, fr in enumerate(FRAMES):
                xt = smoothed_truth[fr][s]
                vals[fi, si, 0, i] = masked_pearson(xt, prs, mask)
                vals[fi, si, 1, i] = masked_spearman(xt, prs, mask)
        if i == 0:
            r0 = vals[0, list(SIGMAS).index(2.0), 0, 0]
            dev = abs(r0 - LINEAGE_BANKED)
            lineage_pass = bool(dev <= LINEAGE_TOL)
            print(f"[a5] LINEAGE realization 0: r_s(sig=2, real, masked) = "
                  f"{r0:.10f} vs banked {LINEAGE_BANKED:.10f} "
                  f"(dev {dev:.2e}) -> "
                  f"{'PASS' if lineage_pass else 'FAIL'}", flush=True)
            if not lineage_pass:
                raise SystemExit("LINEAGE CHECK FAILED — aborting; the seed "
                                 "convention does not reproduce the banked "
                                 "d75/R9 control draw.")
        if (i + 1) % CHUNK == 0 or i == N_REAL - 1:
            np.savez(PARTIAL, vals=vals, done=i + 1)
            el = time.time() - t0
            print(f"[a5] {i+1}/{N_REAL} done ({el:.0f}s, "
                  f"{el/(i+1-start):.1f}s/real)", flush=True)

    # lineage record (recompute cheaply if resumed past 0)
    r0 = float(vals[0, list(SIGMAS).index(2.0), 0, 0])
    lineage = {
        "realization": 0,
        "seed": [SEED_PHASE, 0],
        "seed_equivalence": {
            "statement": f"default_rng([{SEED_PHASE}, 0]) yields the "
                         f"identical stream to default_rng({SEED_PHASE}) "
                         "(trailing-zero SeedSequence entropy word), so "
                         "realization 0 IS the banked seed-20260726 draw",
            "verified": seed0_equiv,
        },
        "banked_value": LINEAGE_BANKED,
        "banked_source": "experiments/unet-inversion/artifacts/stage1/"
                         "r9_heldout_bars.json g1_clause_c.phase_rand_null"
                         ".r_s_vs_truth_real['2'].pearson_masked",
        "observed": r0,
        "abs_dev": abs(r0 - LINEAGE_BANKED),
        "tolerance": LINEAGE_TOL,
        "verdict": "PASS" if abs(r0 - LINEAGE_BANKED) <= LINEAGE_TOL
                   else "FAIL",
    }

    # ---- band statistics ----------------------------------------------------
    band = {}
    for fi, fr in enumerate(FRAMES):
        band[fr] = {}
        for si, s in enumerate(SIGMAS):
            band[fr][f"{s:g}"] = {}
            for mi, m in enumerate(METRICS):
                v = vals[fi, si, mi, :]
                assert np.all(np.isfinite(v)), (fr, s, m)
                band[fr][f"{s:g}"][m] = {
                    "median": float(np.median(v)),
                    "pct_2p5": float(np.percentile(v, 2.5)),
                    "pct_97p5": float(np.percentile(v, 97.5)),
                    "min": float(v.min()),
                    "max": float(v.max()),
                    "values": v.tolist(),
                }

    payload = {
        "rung": "A5 (R7) — pre-registered N=200 phase-randomized null band",
        "spec": "experiments/unet-inversion/design/u06_stage2_spec.md §(c) "
                "null-band protocol, branch HEAD 489a0d3",
        "session_utc": "2026-07-24",
        "protocol": {
            "n_realizations": N_REAL,
            "construction": "NC.phase_randomized(x_truth_real, seed) — "
                            "identical |FFT| amplitudes, random phases, mean "
                            "restored (imported; d75 controls construction, "
                            "byte-identical)",
            "seeds": f"numpy.random.default_rng([{SEED_PHASE}, i]), "
                     f"i = 0..{N_REAL-1}",
            "scoring": "R9 conventions imported from "
                       "scripts/u04_r9_heldout_rescore.py: smooth the FULL "
                       "periodic cube FIRST (sigma in {1,2,4} h^-1 Mpc), "
                       "then restrict Pearson/Spearman to the test mask",
            "mask": "region_voxel_interval('test', 192) = axis-0 slab "
                    "[163, 192), runtime-asserted",
            "frames": "real (primary) + zspace (column); the randomized "
                      "field is always built from x_truth_real and scored "
                      "against each frame's smoothed truth",
            "threshold_rule": "'above null' = above the 97.5th percentile "
                              "(spec §(c)); the single-draw 0.1185 is "
                              "RETIRED as a null level",
        },
        "identity_checks": identity,
        "lineage_check_realization_0": lineage,
        "band": band,
        "wall_clock_s": time.time() - t0,
    }
    OUT.write_text(json.dumps(_js(payload), indent=2))
    print(f"[a5] wrote {OUT} ({time.time()-t0:.0f}s)", flush=True)
    PARTIAL.unlink(missing_ok=True)

    # console band table
    for fr in FRAMES:
        for s in SIGMAS:
            for m in METRICS:
                b = band[fr][f"{s:g}"][m]
                print(f"[a5] {fr:6s} sig={s:g} {m:8s} median={b['median']:+.4f} "
                      f"band=[{b['pct_2p5']:+.4f}, {b['pct_97p5']:+.4f}] "
                      f"min/max=[{b['min']:+.4f}, {b['max']:+.4f}]",
                      flush=True)


if __name__ == "__main__":
    main()
