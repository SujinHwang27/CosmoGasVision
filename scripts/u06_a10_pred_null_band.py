#!/usr/bin/env python
"""[U-06] A10 (spec v2 S3): prediction-matched null band on the banked s3 read.

Spec of record: u06_stage2_spec.md v2 amendment block S3 — per scored
prediction, N=200 phase-randomizations OF THE PREDICTION cube (seeds
default_rng([20260728, i]) i=0..199), scored vs truth on the identical
mask/frame/sigma. Every "above null" quote must clear BOTH bands
(truth-spectrum AND prediction-matched). Retro-application: the s3 quick-eval
"above null" quote is PROVISIONAL until this both-band clearance lands.

Procedure:
  1. The s3 prediction cube was not banked -> regenerate: load the s3
     checkpoint (s3_model_step500.pt, DVC-tracked, arch per
     s3_minirun_record.json), run the pipeline's sliding_window_predict on P1
     with the [0,1024) file-order pattern. UNIFORM overlap averaging — the
     banked s3 convention (pipeline.py unchanged since the banked run,
     commit 3082ee1; the later Hann-taper ratification is NOT applied here:
     banked numbers are never restated).
  2. Reproduction check: VAL-masked r_s(sigma=2, real) of the regenerated
     cube must match the banked 0.8824201230101358 to ~1e-3 (device
     nondeterminism tolerance); recorded, hard-fail otherwise.
  3. N=200 phase-randomizations of the prediction cube, scored vs truth on
     the VAL mask (region_voxel_interval('val', 192), loader-derived),
     both frames, sigma {1,2,4}, Pearson + Spearman (R9 conventions imported:
     smooth full periodic cube FIRST, then mask).
  4. Band stats per A5 format + 10k-bootstrap MC error on the 97.5 edge.
  5. Verdict: does the banked s3 actual (0.8824) clear the
     prediction-matched 97.5th percentile at sigma=2 real? Plus the
     both-band (truth-spectrum K2 edge AND prediction-matched) clearance.

Output: experiments/unet-inversion/artifacts/stage2/pred_null_bands_s3.json
        (+ the regenerated prediction cube, float32, DVC-tracked)

Usage: PYTHONPATH=. .venv/bin/python -u scripts/u06_a10_pred_null_band.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "unet-inversion"))

from src.analysis import nccf as NC                                 # noqa: E402
from scripts.d75_corrected_metric_rescore import (                  # noqa: E402
    CUBES as D75_CUBES, BOX_MPC_H, N192, x_transform, load_cube,
    _sha256, _md5, _js)
from scripts.u04_r9_heldout_rescore import (                        # noqa: E402
    masked_pearson, masked_spearman,
    PIN_TRUTH_REAL_SHA256, PIN_TRUTH_ZSPACE_MD5)
from src.data.loader import DEFAULT_SCHEME, region_voxel_interval   # noqa: E402
import pipeline as PL                                               # noqa: E402

SEED_PRED_PHASE = 20260728          # spec v2 S3 seed family (NOT 20260726)
N_REAL = 200
CHUNK = 20
FRAMES = ("real", "zspace")
METRICS = ("pearson", "spearman")
SIGMAS = (1.0, 2.0, 4.0)
N_BOOT = 10_000
BOOT_SEED = [SEED_PRED_PHASE, 777001]

BANKED_ACTUAL_S2 = 0.8824201230101358    # s3_minirun_record.json
REPRO_TOL = 1e-3

OUT_DIR = REPO / "experiments" / "unet-inversion" / "artifacts" / "stage2"
OUT = OUT_DIR / "pred_null_bands_s3.json"
PRED_NPY = OUT_DIR / "s3_pred_p1_rays1024.npy"
PARTIAL = OUT_DIR / "pred_null_bands_s3.partial.npz"
CKPT = OUT_DIR / "s3_model_step500.pt"
S3_RECORD = OUT_DIR / "s3_minirun_record.json"


def boot_edge(values, seed, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    edges = np.percentile(values[idx], 97.5, axis=1)
    return {"edge": float(np.percentile(values, 97.5)),
            "boot_se": float(edges.std(ddof=1)),
            "boot_ci95": [float(np.percentile(edges, 2.5)),
                          float(np.percentile(edges, 97.5))],
            "n": int(len(values))}


def main():
    t0 = time.time()

    # ---- identity checks -----------------------------------------------
    identity = []
    for fname, kind, pin, fn in (
            ("truth_real_192.npy", "sha256_full", PIN_TRUTH_REAL_SHA256,
             _sha256),
            ("truth_zspace_192.npy", "md5", PIN_TRUTH_ZSPACE_MD5, _md5)):
        obs = fn(D75_CUBES / fname)
        v = "MATCH" if obs == pin else "MISMATCH"
        identity.append({"file": fname, "kind": kind, "pinned": pin,
                         "observed": obs, "verdict": v})
        print(f"[a10] identity {fname} {kind}: {v}", flush=True)
        if v == "MISMATCH":
            raise SystemExit(f"IDENTITY MISMATCH: {fname}")
    ckpt_md5 = _md5(CKPT)
    identity.append({"file": "s3_model_step500.pt", "kind": "md5",
                     "pinned": None, "observed": ckpt_md5,
                     "verdict": "RECORDED (DVC-tracked checkpoint)"})

    # ---- VAL mask (loader-derived) --------------------------------------
    lo, hi = region_voxel_interval("val", N192, DEFAULT_SCHEME)
    assert (lo, hi) == (134, 163), f"unexpected val interval [{lo},{hi})"
    assert DEFAULT_SCHEME.axis == 0
    mask = np.zeros((N192, N192, N192), dtype=bool)
    mask[lo:hi] = True

    # ---- truth, smoothed first ------------------------------------------
    x_real, _ = x_transform(load_cube("truth_real_192.npy"))
    x_z, _ = x_transform(load_cube("truth_zspace_192.npy"))
    st = {"real": {s: NC.gaussian_smooth_periodic(x_real, BOX_MPC_H, s)
                   for s in SIGMAS},
          "zspace": {s: NC.gaussian_smooth_periodic(x_z, BOX_MPC_H, s)
                     for s in SIGMAS}}
    del x_z

    # ---- prediction cube: load if regenerated already, else regenerate --
    if PRED_NPY.exists():
        pred = np.load(PRED_NPY).astype(np.float64)
        pred_provenance = f"loaded banked regeneration {PRED_NPY.name}"
        print(f"[a10] loaded existing prediction cube {PRED_NPY.name}",
              flush=True)
    else:
        device = PL.pick_device()
        model = PL.UNet3D().to(device)
        sd = torch.load(CKPT, map_location=device)
        model.load_state_dict(sd)
        n_params = model.n_parameters()
        rec = json.loads(S3_RECORD.read_text())
        assert n_params == rec["model"]["n_params_exact"], (
            n_params, rec["model"]["n_params_exact"])
        print(f"[a10] checkpoint loaded ({n_params} params, {device}); "
              f"building P1 source ...", flush=True)
        source = PL.build_sources([1])[0]
        # scoring truth == provider truth (exact [D-75] x), asserted
        assert np.allclose(source.provider.x_cube, x_real), (
            "provider x_cube != d75 truth_real x_transform")
        rays = np.arange(1024, dtype=np.int64)   # [0,1024) file-order
        ti = time.time()
        pred = PL.sliding_window_predict(model, source, rays, device)
        pred_provenance = (
            "regenerated: PL.sliding_window_predict(s3 checkpoint, P1, "
            "[0,1024) file-order), periodic stride-32 windows, UNIFORM "
            "averaging (banked s3 convention; pipeline.py unchanged since "
            "3082ee1), S7 8x-coverage assert inside")
        print(f"[a10] inference done ({time.time()-ti:.0f}s)", flush=True)
        np.save(PRED_NPY, pred.astype(np.float32))
        print(f"[a10] saved {PRED_NPY.name} (float32)", flush=True)

    # ---- reproduction check vs banked 0.8824 -----------------------------
    pred_s2 = NC.gaussian_smooth_periodic(pred, BOX_MPC_H, 2.0)
    r_repro = masked_pearson(st["real"][2.0], pred_s2, mask)
    dev = abs(r_repro - BANKED_ACTUAL_S2)
    repro = {
        "banked_value": BANKED_ACTUAL_S2,
        "banked_source": "s3_minirun_record.json quick_masked_eval."
                         "scores_real_frame.actual['2'].pearson_masked",
        "observed": float(r_repro),
        "abs_dev": float(dev),
        "tolerance": REPRO_TOL,
        "tolerance_basis": "~1e-3 per coordinator instruction (device "
                           "nondeterminism; MPS forward passes are not "
                           "bit-reproducible)",
        "verdict": "PASS" if dev <= REPRO_TOL else "FAIL",
        "prediction_provenance": pred_provenance,
    }
    print(f"[a10] REPRODUCTION: r_s(2,real,VAL) = {r_repro:.10f} vs banked "
          f"{BANKED_ACTUAL_S2:.10f} (dev {dev:.2e}) -> {repro['verdict']}",
          flush=True)
    if repro["verdict"] == "FAIL":
        raise SystemExit("REPRODUCTION CHECK FAILED — the regenerated "
                         "prediction does not match the banked s3 readout; "
                         "banked numbers may not be restated. Aborting.")

    # ---- N=200 phase-randomizations OF THE PREDICTION --------------------
    vals = np.full((len(FRAMES), len(SIGMAS), len(METRICS), N_REAL), np.nan)
    start = 0
    if PARTIAL.exists():
        ck = np.load(PARTIAL)
        vals, start = ck["vals"], int(ck["done"])
        print(f"[a10] resuming: {start} done", flush=True)
    for i in range(start, N_REAL):
        pr = NC.phase_randomized(pred, [SEED_PRED_PHASE, i])
        for si, s in enumerate(SIGMAS):
            prs = NC.gaussian_smooth_periodic(pr, BOX_MPC_H, s)
            for fi, fr in enumerate(FRAMES):
                vals[fi, si, 0, i] = masked_pearson(st[fr][s], prs, mask)
                vals[fi, si, 1, i] = masked_spearman(st[fr][s], prs, mask)
        if (i + 1) % CHUNK == 0 or i == N_REAL - 1:
            np.savez(PARTIAL, vals=vals, done=i + 1)
            el = time.time() - t0
            print(f"[a10] {i+1}/{N_REAL} ({el:.0f}s)", flush=True)

    band, mc = {}, {}
    for fi, fr in enumerate(FRAMES):
        band[fr], mc[fr] = {}, {}
        for si, s in enumerate(SIGMAS):
            band[fr][f"{s:g}"], mc[fr][f"{s:g}"] = {}, {}
            for mi, m in enumerate(METRICS):
                v = vals[fi, si, mi, :]
                assert np.all(np.isfinite(v))
                band[fr][f"{s:g}"][m] = {
                    "median": float(np.median(v)),
                    "pct_2p5": float(np.percentile(v, 2.5)),
                    "pct_97p5": float(np.percentile(v, 97.5)),
                    "min": float(v.min()), "max": float(v.max()),
                    "values": v.tolist(),
                }
                mc[fr][f"{s:g}"][m] = boot_edge(v, BOOT_SEED)

    # ---- verdict: both-band clearance of the banked s3 actual ------------
    pred_edge_s2 = band["real"]["2"]["pearson"]["pct_97p5"]
    truth_edge, truth_m, truth_prov = PL.read_val_band_edge()
    s3 = json.loads(S3_RECORD.read_text())
    actual = {sg: s3["quick_masked_eval"]["scores_real_frame"]["actual"][sg]
              ["pearson_masked"] for sg in ("1", "2", "4")}
    verdict = {
        "question": "does the banked s3 actual clear the prediction-matched "
                    "97.5th percentile at sigma=2 real (VAL mask)?",
        "banked_actual_r_s2_real_val": actual["2"],
        "pred_matched_null_97p5_s2_real": pred_edge_s2,
        "pred_matched_edge_mc": mc["real"]["2"]["pearson"],
        "clears_pred_matched": bool(actual["2"] > pred_edge_s2),
        "truth_spectrum_band": {
            "governing_edge_sigma2": truth_edge, "m": truth_m,
            "provenance": truth_prov,
            "clears": bool(actual["2"] > truth_edge + truth_m),
        },
        "both_band_clearance_s2_real": bool(
            actual["2"] > pred_edge_s2
            and actual["2"] > truth_edge + truth_m),
        "per_sigma_pred_edges_real_pearson": {
            sg: band["real"][sg]["pearson"]["pct_97p5"]
            for sg in ("1", "2", "4")},
        "per_sigma_actual_clears_pred_edge": {
            sg: bool(actual[sg] > band["real"][sg]["pearson"]["pct_97p5"])
            for sg in ("1", "2", "4")},
    }

    payload = {
        "rung": "A10 (R10) — spec v2 S3 prediction-matched null band on the "
                "banked s3 readout",
        "spec": "experiments/unet-inversion/design/u06_stage2_spec.md v2 "
                "amendment block S3 + retro-application clause",
        "session_utc": "2026-07-24",
        "protocol": {
            "n_realizations": N_REAL,
            "construction": "NC.phase_randomized(PREDICTION cube, seed) — "
                            "identical |FFT| amplitudes to the s3 prediction, "
                            "random phases, mean restored",
            "seeds": f"numpy.random.default_rng([{SEED_PRED_PHASE}, i]), "
                     f"i = 0..{N_REAL-1} (S3 seed family 20260728)",
            "scoring": "R9 conventions imported "
                       "(scripts/u04_r9_heldout_rescore.py): smooth the FULL "
                       "periodic cube FIRST (sigma {1,2,4} h^-1 Mpc), then "
                       "restrict Pearson/Spearman to the VAL mask",
            "mask": f"region_voxel_interval('val', 192) = axis-0 slab "
                    f"[{lo}, {hi}), loader-derived, runtime-asserted (K1)",
            "frames": "real (primary) + zspace (column); randomized field "
                      "built from the s3 prediction, scored against each "
                      "frame's smoothed truth",
        },
        "identity_checks": identity,
        "prediction_cube": {
            "file": str(PRED_NPY.relative_to(REPO)),
            "dtype_on_disk": "float32",
            "md5": _md5(PRED_NPY),
            "provenance": pred_provenance,
        },
        "reproduction_check": repro,
        "band": band,
        "mc_error_97p5_edge": {"method": f"nonparametric bootstrap, {N_BOOT} "
                                         "resamples, seeded",
                               "seed": BOOT_SEED, **mc},
        "verdict": verdict,
        "wall_clock_s": time.time() - t0,
    }
    OUT.write_text(json.dumps(_js(payload), indent=2))
    print(f"[a10] wrote {OUT} ({time.time()-t0:.0f}s)", flush=True)
    PARTIAL.unlink(missing_ok=True)

    for fr in FRAMES:
        for s in SIGMAS:
            b = band[fr][f"{s:g}"]["pearson"]
            print(f"[a10] {fr:6s} sig={s:g} pearson median={b['median']:+.4f} "
                  f"band=[{b['pct_2p5']:+.4f}, {b['pct_97p5']:+.4f}]",
                  flush=True)
    print(f"[a10] VERDICT: actual {actual['2']:.4f} vs pred-matched 97.5th "
          f"{pred_edge_s2:+.4f} -> clears={verdict['clears_pred_matched']}; "
          f"both-band={verdict['both_band_clearance_s2_real']}", flush=True)


if __name__ == "__main__":
    main()
