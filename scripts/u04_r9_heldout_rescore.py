#!/usr/bin/env python
"""[U-04] Stage-1 R9 (A9): held-out-region re-scoring of the banked [D-75]
baseline cubes -> OPERATIVE G2 bars.

Spec of record: experiments/unet-inversion/design/u04_stage1_ratification.md
S1 (G1 clause (c) + G2 operative-bar adjustment), commit 58ac831 lineage.

Conventions (byte-identical reuse of the [D-75] machinery):
  - scoring variable x = log10(max(rho/<rho>, 1e-3)), float64
    (scripts/d75_corrected_metric_rescore.py:x_transform, imported)
  - smoothing: periodic FFT Gaussian exp(-k^2 sigma^2 / 2), sigma in {1,2,4}
    h^-1 Mpc (src/analysis/nccf.py:gaussian_smooth_periodic, imported)
  - MASKING ORDER (stated convention): smoothing is applied to the FULL
    periodic 192^3 cube FIRST; the Pearson/Spearman is then restricted to
    the test-slab voxels. Smoothing after masking would corrupt the slab
    boundaries and is NOT done anywhere in this script.
  - held-out mask: [D-49] split at n=192 via
    src.data.loader.region_voxel_interval("test", 192, DEFAULT_SCHEME)
    -> axis-0 slab [163, 192), asserted at runtime.

Output: experiments/unet-inversion/artifacts/stage1/r9_heldout_bars.json

Usage: PYTHONPATH=. .venv/bin/python -u scripts/u04_r9_heldout_rescore.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.analysis import nccf as NC                                # noqa: E402
from scripts.d75_corrected_metric_rescore import (                 # noqa: E402
    CUBES, BOX_MPC_H, N192, SIGMAS, SEED_PHASE, x_transform, load_cube,
    _sha256, _md5, _js)
from src.data.loader import DEFAULT_SCHEME, region_voxel_interval  # noqa: E402

OUT = (REPO / "experiments" / "unet-inversion" / "artifacts" / "stage1"
       / "r9_heldout_bars.json")
D75_SCORES = (REPO / "experiments" / "nerf" / "artifacts" / "d75_rescore"
              / "d75_scores.json")

# identity pins (sources recorded in the JSON)
PIN_TRUTH_REAL_SHA256 = ("971a72ed5b1b872a972fd3ff8c35e99d"
                         "6a0998a129afb822ebbd42417da34994")   # r1 pin of record
PIN_TRUTH_ZSPACE_MD5 = "3a7af286ddff91f26f00459c11577d29"      # d75_scores.json
PIN_GRID_MD5 = "299f09b84c6f747a2ffbe800de9be51c"              # r1 cube-level pin
PIN_MLP_MD5 = "8357a9b7384af0d5e721c096648174e9"               # r1 cube-level pin
PIN_WIENER_MD5 = {                                             # d75_scores.json
    "wiener_rec192_L1.npy": "2efeb390aeedd329dddfa58e0acc60f7",
    "wiener_rec192_L2.npy": "e3a391e7e8180926d30a215ebc6443d1",
    "wiener_rec192_L3.npy": "52fd6ae0fecfc69485ce0f4c14929707",
}

TOL_G1 = 1e-6


def masked_pearson(a, b, mask):
    return NC.pearson(a[mask], b[mask])


def masked_spearman(a, b, mask):
    return NC.spearman(a[mask], b[mask])


def main():
    t0 = time.time()

    # ---------------- mask geometry (confirmed from source, not assumed) ----
    lo, hi = region_voxel_interval("test", N192, DEFAULT_SCHEME)
    assert (lo, hi) == (163, 192), f"unexpected test interval [{lo},{hi})"
    assert DEFAULT_SCHEME.axis == 0
    mask = np.zeros((N192, N192, N192), dtype=bool)
    mask[lo:hi, :, :] = True
    n_mask = int(mask.sum())

    # 8 congruent sub-blocks: 2 axis-0 x 2x2 transverse ([U-04] G2 cond 2).
    # 29 voxels is odd -> trim ONE voxel at the high-index end (use [163,191),
    # 28 thick -> two congruent 14-voxel halves). Deviation recorded.
    blk_lo, blk_hi = lo, hi - 1
    half = (blk_hi - blk_lo) // 2                     # 14
    ht = N192 // 2                                    # 96
    block_slices, block_names = [], []
    for i0, (a0, a1) in enumerate([(blk_lo, blk_lo + half),
                                   (blk_lo + half, blk_hi)]):
        for j in (0, 1):
            for k in (0, 1):
                block_slices.append((slice(a0, a1),
                                     slice(j * ht, (j + 1) * ht),
                                     slice(k * ht, (k + 1) * ht)))
                block_names.append(f"ax0[{a0},{a1})_y{j}_z{k}")

    def per_block_pearson(a, b):
        return np.array([NC.pearson(a[s], b[s]) for s in block_slices])

    # ---------------- identity checks at load -------------------------------
    identity = []

    def check(fname, kind, pinned, observed):
        v = "MATCH" if observed == pinned else "MISMATCH"
        identity.append({"file": fname, "kind": kind, "pinned": pinned,
                         "observed": observed, "verdict": v})
        print(f"[r9] identity {fname} {kind}: {v}", flush=True)
        if v == "MISMATCH":
            raise SystemExit(f"IDENTITY MISMATCH: {fname}")

    check("truth_real_192.npy", "sha256_full", PIN_TRUTH_REAL_SHA256,
          _sha256(CUBES / "truth_real_192.npy"))
    check("truth_zspace_192.npy", "md5", PIN_TRUTH_ZSPACE_MD5,
          _md5(CUBES / "truth_zspace_192.npy"))
    check("grid_192.npy", "md5", PIN_GRID_MD5, _md5(CUBES / "grid_192.npy"))
    check("mlp_192.npy", "md5", PIN_MLP_MD5, _md5(CUBES / "mlp_192.npy"))
    for f, pin in PIN_WIENER_MD5.items():
        check(f, "md5", pin, _md5(CUBES / f))

    # ---------------- fields -------------------------------------------------
    x_truth = {}
    x_truth["real"], _ = x_transform(load_cube("truth_real_192.npy"))
    x_truth["zspace"], _ = x_transform(load_cube("truth_zspace_192.npy"))
    smoothed_truth = {fr: {s: NC.gaussian_smooth_periodic(x_truth[fr],
                                                          BOX_MPC_H, s)
                           for s in SIGMAS} for fr in x_truth}

    objects = {
        "grid": x_transform(load_cube("grid_192.npy"))[0],
        "mlp": x_transform(load_cube("mlp_192.npy"))[0],
    }
    for L in (1, 2, 3):
        objects[f"wiener_L{L}"] = x_transform(
            1.0 + load_cube(f"wiener_rec192_L{L}.npy"))[0]

    # ---------------- G1 clause (c): masked acceptance ----------------------
    g1c = {"tolerance": TOL_G1, "truth_vs_truth": {}, "phase_rand_null": {}}
    ok_all = True
    for fr in ("real", "zspace"):
        vals = {}
        for s in SIGMAS:
            xs = smoothed_truth[fr][s]
            r = masked_pearson(xs, xs, mask)
            vals[f"{s:g}"] = {"r_s": r, "abs_dev_from_1": abs(r - 1.0),
                              "pass": bool(abs(r - 1.0) <= TOL_G1)}
            ok_all &= abs(r - 1.0) <= TOL_G1
        g1c["truth_vs_truth"][fr] = vals

    # phase-randomized-truth null (seed 20260726). Banked d75 control cube
    # exists; verify it equals a fresh reconstruction from the pinned truth.
    pr_path = CUBES / "control_phase_rand_x.npy"
    pr_banked = np.load(pr_path)                      # float32 on disk
    pr_fresh = NC.phase_randomized(x_truth["real"], SEED_PHASE)  # float64,
    # identical construction to the d75 score stage (which scored the fresh
    # float64 field; the banked cube is its float32 export)
    pr_maxdev = float(np.max(np.abs(pr_banked.astype(np.float64) - pr_fresh)))
    pr_f32_identical = bool(np.array_equal(pr_banked,
                                           pr_fresh.astype(np.float32)))
    pr = pr_fresh
    d75_null = json.loads(D75_SCORES.read_text())[
        "objects"]["real"]["phase_rand"]["r_s"]
    null_scores = {}
    for s in SIGMAS:
        prs = NC.gaussian_smooth_periodic(pr, BOX_MPC_H, s)
        fb_d75 = d75_null[f"{s:.1f}"]["pearson"]
        fb_here = NC.pearson(smoothed_truth["real"][s], prs)
        null_scores[f"{s:g}"] = {
            "pearson_masked": masked_pearson(smoothed_truth["real"][s],
                                             prs, mask),
            "pearson_fullbox_recomputed": fb_here,
            "pearson_fullbox_d75": fb_d75,
            "fullbox_reproduction_abs_dev": abs(fb_here - fb_d75),
        }
    g1c["phase_rand_null"] = {
        "seed": SEED_PHASE,
        "scored_object": "fresh float64 NC.phase_randomized(x_truth_real, "
                         "20260726) — identical construction to the d75 "
                         "score stage",
        "banked_cube_identity_check": {
            "cube": str(pr_path.relative_to(REPO)),
            "banked_dtype": str(pr_banked.dtype),
            "banked_equals_float32_cast_of_fresh": pr_f32_identical,
            "max_abs_dev_banked_vs_fresh_float64": pr_maxdev,
        },
        "r_s_vs_truth_real": null_scores,
    }
    g1c["pass_truth_identity"] = bool(ok_all)
    print(f"[r9] G1(c) truth-identity pass={ok_all} "
          f"phase-rand banked==f32(fresh)={pr_f32_identical} "
          f"(maxdev {pr_maxdev:.2e})", flush=True)

    # ---------------- held-out scores + sub-blocks + deltas -----------------
    d75 = json.loads(D75_SCORES.read_text())

    def fullbox_rs(fr, name, s):
        try:
            return d75["objects"][fr][name]["r_s"][f"{s:.1f}"]["pearson"]
        except KeyError:
            return None

    results = {"real": {}, "zspace": {}}
    for fr in ("real", "zspace"):
        for name, x_o in objects.items():
            entry = {}
            for s in SIGMAS:
                xo_s = NC.gaussian_smooth_periodic(x_o, BOX_MPC_H, s)
                xt_s = smoothed_truth[fr][s]
                p = masked_pearson(xt_s, xo_s, mask)
                sp = masked_spearman(xt_s, xo_s, mask)
                blocks = per_block_pearson(xt_s, xo_s)
                fb = fullbox_rs(fr, name, s)
                entry[f"{s:g}"] = {
                    "pearson_heldout": p,
                    "spearman_heldout": sp,
                    "block_r": blocks,
                    "block_mean": float(np.nanmean(blocks)),
                    "block_se": float(np.nanstd(blocks, ddof=1)
                                      / np.sqrt(len(blocks))),
                    "pearson_fullbox_d75": fb,
                    "delta_heldout_minus_fullbox": (None if fb is None
                                                    else p - fb),
                }
            results[fr][name] = entry
            print(f"[r9] {fr}/{name}: r_s(2) heldout="
                  f"{entry['2']['pearson_heldout']:.5f} "
                  f"(fullbox {entry['2']['pearson_fullbox_d75']}) "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # best-L re-selected ON THE MASK, per frame, by r_s(sigma=2) pearson
    best_L = {}
    for fr in ("real", "zspace"):
        Ls = ["wiener_L1", "wiener_L2", "wiener_L3"]
        best = max(Ls, key=lambda n: results[fr][n]["2"]["pearson_heldout"])
        best_L[fr] = {
            "best": best,
            "r_s2_heldout_by_L": {n: results[fr][n]["2"]["pearson_heldout"]
                                  for n in Ls},
            "d75_fullbox_best": d75["wiener_best_L_per_frame"][fr],
        }

    payload = {
        "rung": "R9 (A9) — held-out re-scoring of banked baseline cubes -> "
                "operative G2 bars",
        "spec": "experiments/unet-inversion/design/u04_stage1_ratification.md "
                "S1 G1(c) + G2 operative-bar adjustment; S2(e) item (3)",
        "session_utc": "2026-07-23",
        "role_note": "measurements only; no cells, no interpretation ([D-37])",
        "conventions": {
            "scoring_variable": "x = log10(max(rho/<rho>, 1e-3)), float64 "
                                "(imported x_transform, byte-identical)",
            "smoothing": "periodic FFT Gaussian exp(-k^2 sigma^2/2), sigma in "
                         "{1,2,4} h^-1 Mpc (imported "
                         "NC.gaussian_smooth_periodic, byte-identical)",
            "masking_order": "smoothing applied to the FULL periodic cube "
                             "FIRST, Pearson/Spearman then restricted to "
                             "test-slab voxels; smoothing-after-masking is "
                             "never performed (would corrupt slab boundaries)",
            "wiener_rho_convention": "rho/<rho> = 1 + rec (d75 convention, "
                                     "unchanged)",
            "spearman": "average-tie ranks over masked voxels only",
            "block_se": "std(block r, ddof=1)/sqrt(8), same estimator as the "
                        "d75 octant_se",
        },
        "mask": {
            "source": "src.data.loader.region_voxel_interval('test', 192, "
                      "DEFAULT_SCHEME) — [D-49] split, runtime-asserted",
            "axis": 0,
            "interval_right_open": [lo, hi],
            "thickness_voxels": hi - lo,
            "n_voxels": n_mask,
            "fraction_of_box": n_mask / N192 ** 3,
        },
        "sub_blocks": {
            "geometry": "2 axis-0 x 2x2 transverse = 8 congruent blocks of "
                        "14x96x96 voxels ([U-04] G2 condition 2 geometry)",
            "names": block_names,
            "deviation": "test slab is 29 voxels thick (odd); ONE voxel "
                         "trimmed at the high-index end (blocks span "
                         "[163,191); voxel 191 excluded from blocks only, "
                         "NOT from the primary masked scores)",
        },
        "identity_checks": identity,
        "g1_clause_c": g1c,
        "heldout_scores": results,
        "wiener_best_L_on_mask": best_L,
        "fullbox_reference": {
            "file": str(D75_SCORES.relative_to(REPO)),
            "note": "pearson_fullbox_d75 / delta columns pull the banked "
                    "full-box r_s from d75_scores.json objects block",
        },
        "deviations": [
            "sub-block congruence trim: 29 -> 28 voxels along axis 0 "
            "(see sub_blocks.deviation)",
            "G1 clause (c) executed as the task-scoped pair: masked "
            "truth-vs-truth r_s + phase-randomized null on the mask; the "
            "masked NCCF re-run is not part of this rung's deliverable "
            "(FFT shell estimator is full-box periodic; slab-masked NCCF "
            "path does not exist and was not fabricated)",
        ],
        "wall_clock_s": time.time() - t0,
    }
    OUT.write_text(json.dumps(_js(payload), indent=2))
    print(f"[r9] wrote {OUT} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
