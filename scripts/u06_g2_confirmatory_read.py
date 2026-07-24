#!/usr/bin/env python
"""[U-06] G2 CONFIRMATORY READ — the single sanctioned test-slab evaluation.

Spec of record: experiments/unet-inversion/design/u06_stage2_spec.md
([U-04] G2 five-condition machinery + v2 amendments K1/S3/S4/S5/S7 +
micro-cycle A1/A2/R1/R2 riders). This IS one of the exactly-two sanctioned
test-mask touches (K1: "G2 close and G3").

Mechanical, complete, no interpretation. Every number, condition verdict and
deviation lands in
  experiments/unet-inversion/artifacts/stage2/g2_confirmatory_read.json
plus the per-readout prediction-matched null artifact (A10 shape)
  experiments/unet-inversion/artifacts/stage2/pred_null_bands_g2.json
and the prediction cube (float32, DVC-tracked)
  experiments/unet-inversion/artifacts/stage2/g2_pred_test_p1_rays1024_hann.npy

Checkpoint-of-record (K1 best-VAL rule, never test-informed):
  cloud_runs/UNetS2-S42-39804b8-1784883256-e5b664/checkpoints/best_val.pt

Usage: PYTHONPATH=. .venv/bin/python -u scripts/u06_g2_confirmatory_read.py
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
    PIN_TRUTH_REAL_SHA256, PIN_TRUTH_ZSPACE_MD5, PIN_GRID_MD5,
    PIN_MLP_MD5, PIN_WIENER_MD5)
from src.data.loader import (                                       # noqa: E402
    DEFAULT_SCHEME, region_voxel_interval, distance_to_train_region)
import pipeline as PL                                               # noqa: E402

# ----------------------------------------------------------------- constants
CKPT = (REPO / "cloud_runs" / "UNetS2-S42-39804b8-1784883256-e5b664"
        / "checkpoints" / "best_val.pt")
TRAIN_RECORD = CKPT.parent.parent / "train_full_record.json"
STAGE2 = REPO / "experiments" / "unet-inversion" / "artifacts" / "stage2"
R9_BARS = (REPO / "experiments" / "unet-inversion" / "artifacts" / "stage1"
           / "r9_heldout_bars.json")
SEALED_BAND = STAGE2 / "null_band_n200.json"
OUT = STAGE2 / "g2_confirmatory_read.json"
OUT_PRED_NULL = STAGE2 / "pred_null_bands_g2.json"
PRED_NPY = STAGE2 / "g2_pred_test_p1_rays1024_hann.npy"
PARTIAL = STAGE2 / "pred_null_bands_g2.partial.npz"

N_PARAMS_PIN = 5839713                     # train_full_record.json n_params
SEALED_TEST_NULL975_S2_REAL = 0.22108341828682537   # SEALED band, quoted 0.22108
SIGMAS = (1.0, 2.0, 4.0)
FRAMES = ("real", "zspace")
SEED_Z3_DERANGE = 20260729                 # S5 pinned
SEED_RANDOM_PATTERN = 20260730             # v2 P3
SEED_PRED_PHASE = 20260728                 # S3 seed family (shared lineage
#                                            with the s3 pred-null: same
#                                            family, different cube)
SEED_BOOT_PAIR = 20260723                  # d75 block-bootstrap convention
N_PRED_NULL = 200
N_BOOT_EDGE = 10_000
BOOT_EDGE_SEED = [SEED_PRED_PHASE, 777002]
N_BOOT_PAIR = 1000
T_CRIT = 2.365                             # df=7, 0.975
BARS_QUOTED = {"grid": 0.5758, "wiener_L3": 0.0863, "mlp": 0.0727}

TEST_LO, TEST_HI = 163, 192


# ------------------------------------------------------------------ helpers
def r9_block_slices():
    """The 8 congruent R9 sub-blocks of the test slab: [163,191) split into
    two 14-voxel halves x 2x2 transverse (96). Banked R9 deviation reused:
    voxel 191 excluded from blocks only, NOT from the primary masked scores."""
    lo, hi = TEST_LO, TEST_HI - 1
    half = (hi - lo) // 2
    ht = N192 // 2
    slices, names = [], []
    for a0, a1 in ((lo, lo + half), (lo + half, hi)):
        for j in (0, 1):
            for k in (0, 1):
                slices.append((slice(a0, a1), slice(j * ht, (j + 1) * ht),
                               slice(k * ht, (k + 1) * ht)))
                names.append(f"ax0[{a0},{a1})_y{j}_z{k}")
    return slices, names


BLOCK_SLICES, BLOCK_NAMES = r9_block_slices()


def per_block_pearson(a, b):
    return np.array([NC.pearson(a[s], b[s]) for s in BLOCK_SLICES])


def column_block_sums(truth_slab, obj_slab):
    """[U-04] G2 cond-3 held-out-geometry adaptation (disclosed): the block
    unit is a FULL-THICKNESS transverse column of the test slab,
    29 x 24 x 24 voxels, 8 x 8 = 64 blocks tiling the mask exactly."""
    a = truth_slab.reshape(29, 8, 24, 8, 24)
    b = obj_slab.reshape(29, 8, 24, 8, 24)
    ax = (0, 2, 4)
    return {"n_cell": float(29 * 24 * 24),
            "sa": a.sum(axis=ax).ravel(), "sb": b.sum(axis=ax).ravel(),
            "saa": (a ** 2).sum(axis=ax).ravel(),
            "sbb": (b ** 2).sum(axis=ax).ravel(),
            "sab": (a * b).sum(axis=ax).ravel()}


def column_block_bootstrap(truth_slab, slab_a, slab_b,
                           n_boot=N_BOOT_PAIR, seed=SEED_BOOT_PAIR):
    """Paired block bootstrap of Delta r = r(a,truth) - r(b,truth) over the
    64 test-slab columns; same resample for both objects; CI95 vs zero."""
    st_a = column_block_sums(truth_slab, slab_a)
    st_b = column_block_sums(truth_slab, slab_b)
    nb = 64
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, nb, nb)
        deltas[i] = (NC._pearson_from_sums(st_a, idx)
                     - NC._pearson_from_sums(st_b, idx))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"n_boot": n_boot, "n_blocks": nb,
            "block_shape_voxels": [29, 24, 24],
            "seed": seed,
            "delta_mean": float(np.nanmean(deltas)),
            "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def shell_pk(x):
    """Isotropic shell-averaged power of the mean-subtracted field; bins are
    integer multiples of the fundamental kf = 2*pi/60."""
    xf = np.asarray(x, dtype=np.float64)
    F = np.fft.rfftn(xf - xf.mean())
    p3 = np.abs(F) ** 2
    kx, ky, kz = NC.k_grids(N192, BOX_MPC_H)
    kf = 2.0 * np.pi / BOX_MPC_H
    kmag = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2
                   + kz[None, None, :] ** 2)
    idx = np.rint(kmag / kf).astype(np.int64)
    nmax = N192 // 2
    counts = np.bincount(idx.ravel(), minlength=nmax + 1)[:nmax + 1]
    sums = np.bincount(idx.ravel(), weights=p3.ravel(),
                       minlength=nmax + 1)[:nmax + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        pk = sums / counts
    return pk, kf


def boot_edge(values, seed, n_boot=N_BOOT_EDGE):
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
    dev_notes = []      # deviations / read-time mechanical rulings, all listed

    # ------------------------------------------------ identity checks (U0)
    identity = []

    def check(fname, kind, pinned, observed):
        v = "MATCH" if observed == pinned else "MISMATCH"
        identity.append({"file": fname, "kind": kind, "pinned": pinned,
                         "observed": observed, "verdict": v})
        print(f"[g2] identity {fname} {kind}: {v}", flush=True)
        if v == "MISMATCH":
            raise SystemExit(f"IDENTITY MISMATCH: {fname} (U0 fail-closed)")

    check("truth_real_192.npy", "sha256_full", PIN_TRUTH_REAL_SHA256,
          _sha256(D75_CUBES / "truth_real_192.npy"))
    check("truth_zspace_192.npy", "md5", PIN_TRUTH_ZSPACE_MD5,
          _md5(D75_CUBES / "truth_zspace_192.npy"))
    check("grid_192.npy", "md5", PIN_GRID_MD5, _md5(D75_CUBES / "grid_192.npy"))
    check("mlp_192.npy", "md5", PIN_MLP_MD5, _md5(D75_CUBES / "mlp_192.npy"))
    check("wiener_rec192_L3.npy", "md5",
          PIN_WIENER_MD5["wiener_rec192_L3.npy"],
          _md5(D75_CUBES / "wiener_rec192_L3.npy"))
    ckpt_md5 = _md5(CKPT)
    identity.append({"file": str(CKPT.relative_to(REPO)), "kind": "md5",
                     "pinned": None, "observed": ckpt_md5,
                     "verdict": "RECORDED (checkpoint-of-record, K1 "
                                "best-VAL rule)"})
    print(f"[g2] checkpoint md5 {ckpt_md5}", flush=True)

    train_rec = json.loads(TRAIN_RECORD.read_text())
    assert train_rec["n_params"] == N_PARAMS_PIN, (
        train_rec["n_params"], N_PARAMS_PIN)
    assert train_rec["checkpoint_of_record"].startswith("best_val.pt"), \
        train_rec["checkpoint_of_record"]

    # ------------------------------------------------ sealed band (K1 seal)
    sealed = json.loads(SEALED_BAND.read_text())
    sb = sealed["band"]
    assert abs(sb["real"]["2"]["pearson"]["pct_97p5"]
               - SEALED_TEST_NULL975_S2_REAL) < 1e-12
    sealed_975 = {fr: {sg: {m: sb[fr][sg][m]["pct_97p5"]
                            for m in ("pearson", "spearman")}
                       for sg in ("1", "2", "4")} for fr in FRAMES}

    # ------------------------------------------------ TEST mask (sanctioned)
    lo, hi = region_voxel_interval("test", N192, DEFAULT_SCHEME)
    assert (lo, hi) == (TEST_LO, TEST_HI), f"unexpected test interval " \
                                           f"[{lo},{hi})"
    assert DEFAULT_SCHEME.axis == 0
    mask = np.zeros((N192,) * 3, dtype=bool)
    mask[lo:hi] = True
    print(f"[g2] TEST mask [163,192) — sanctioned touch (K1: G2 close)",
          flush=True)

    # ------------------------------------------------ truth + G1(c) re-assert
    x_truth = {}
    x_truth["real"], _ = x_transform(load_cube("truth_real_192.npy"))
    x_truth["zspace"], _ = x_transform(load_cube("truth_zspace_192.npy"))
    st = {fr: {s: NC.gaussian_smooth_periodic(x_truth[fr], BOX_MPC_H, s)
               for s in SIGMAS} for fr in FRAMES}
    g1c = {"tolerance": 1e-6, "truth_vs_truth_masked": {}}
    g1_ok = True
    for fr in FRAMES:
        vals = {}
        for s in SIGMAS:
            r = masked_pearson(st[fr][s], st[fr][s], mask)
            vals[f"{s:g}"] = {"r_s": r, "abs_dev_from_1": abs(r - 1.0),
                              "pass": bool(abs(r - 1.0) <= 1e-6)}
            g1_ok &= abs(r - 1.0) <= 1e-6
        g1c["truth_vs_truth_masked"][fr] = vals
    g1c["pass"] = bool(g1_ok)
    print(f"[g2] G1(c) scoring-path re-assert pass={g1_ok}", flush=True)

    # ------------------------------------------------ R9 bars (baselines)
    r9 = json.loads(R9_BARS.read_text())
    hs = r9["heldout_scores"]
    assert r9["sub_blocks"]["names"] == BLOCK_NAMES
    for name, bar in BARS_QUOTED.items():
        v = hs["real"][name]["2"]["pearson_heldout"]
        assert abs(v - bar) < 5e-5, (name, v, bar)
    assert r9["wiener_best_L_on_mask"]["real"]["best"] == "wiener_L3"
    assert r9["wiener_best_L_on_mask"]["zspace"]["best"] == "wiener_L3"

    # baseline x-fields (for cond-3 bootstrap), smoothed per sigma
    x_base = {
        "grid": x_transform(load_cube("grid_192.npy"))[0],
        "mlp": x_transform(load_cube("mlp_192.npy"))[0],
        "wiener_L3": x_transform(1.0 + load_cube("wiener_rec192_L3.npy"))[0],
    }
    base_s = {n: {s: NC.gaussian_smooth_periodic(x, BOX_MPC_H, s)
                  for s in SIGMAS} for n, x in x_base.items()}
    # cond-3 sanity: banked masked scores reproduce from the smoothed fields
    for n in x_base:
        rep = masked_pearson(st["real"][2.0], base_s[n][2.0], mask)
        assert abs(rep - hs["real"][n]["2"]["pearson_heldout"]) < 1e-9, n

    # ------------------------------------------------ model + sources
    device = PL.pick_device()
    model = PL.UNet3D().to(device)
    sd = torch.load(CKPT, map_location=device)
    model.load_state_dict(sd)          # strict — arch cross-check
    n_params = model.n_parameters()
    assert n_params == N_PARAMS_PIN, (n_params, N_PARAMS_PIN)
    print(f"[g2] checkpoint loaded; n_params={n_params} == pinned "
          f"{N_PARAMS_PIN}; device={device}", flush=True)

    sources = PL.build_sources([1, 2])
    p1, p2 = sources
    assert np.allclose(p1.provider.x_cube, x_truth["real"]), \
        "provider x_cube != d75 truth_real x_transform"
    assert np.array_equal(p1.geometry.voxel3, p2.geometry.voxel3)
    assert np.array_equal(p1.geometry.axis, p2.geometry.axis)
    n_rays_total = int(p1.delta_f.shape[0])
    assert n_rays_total == 16384

    delta_f_p3 = PL.build_untrained_delta_f(3)      # z5 (R2), local data
    assert delta_f_p3.shape == p1.delta_f.shape

    rays_primary = np.arange(1024, dtype=np.int64)
    rays_secondary = np.arange(64, dtype=np.int64)
    rng_pat = np.random.default_rng(SEED_RANDOM_PATTERN)
    rays_random = np.sort(rng_pat.choice(n_rays_total, size=1024,
                                         replace=False)).astype(np.int64)
    dev_notes.append(
        "random-pattern column (v2 P3): size pinned = primary size 1024, "
        f"drawn without replacement from all {n_rays_total} file sightlines, "
        f"default_rng({SEED_RANDOM_PATTERN}).choice, sorted ascending; the "
        "spec pinned only the seed — size/pool are read-time mechanical "
        "rulings, disclosed here")

    # z3: pinned global derangement of whole-ray flux profiles (seed 20260729)
    z3_rng = np.random.default_rng(SEED_Z3_DERANGE)
    mlen = rays_primary.size
    perm = z3_rng.permutation(mlen)
    n_redraws = 0
    while (perm == np.arange(mlen)).any():
        perm = z3_rng.permutation(mlen)
        n_redraws += 1
    df_derange = p1.delta_f.copy()
    df_derange[rays_primary] = p1.delta_f[rays_primary[perm]]
    df_swap_p2 = p1.delta_f.copy()
    df_swap_p2[rays_primary] = p2.delta_f[rays_primary]
    df_swap_p3 = p1.delta_f.copy()
    df_swap_p3[rays_primary] = delta_f_p3[rays_primary]

    # ------------------------------------------------ inference columns
    runs = [
        ("actual", dict(ray_ids=rays_primary)),
        ("uniform_co_report", dict(ray_ids=rays_primary, taper="uniform")),
        ("secondary_64", dict(ray_ids=rays_secondary)),
        ("random_pattern_1024", dict(ray_ids=rays_random)),
        ("z1_all_zero_input", dict(
            ray_ids=rays_primary,
            input_transform=lambda a: np.zeros_like(a))),
        ("z2_mask_only", dict(
            ray_ids=rays_primary,
            input_transform=lambda a: np.stack(
                [np.zeros_like(a[0]), a[1]]))),
        ("z3_deranged_ray", dict(ray_ids=rays_primary, delta_f=df_derange)),
        ("z4_cross_physics_swap_P2", dict(ray_ids=rays_primary,
                                          delta_f=df_swap_p2)),
        ("z5_untrained_swap_P3", dict(ray_ids=rays_primary,
                                      delta_f=df_swap_p3)),
    ]
    cubes = {}
    for name, kw in runs:
        ti = time.time()
        ray_ids = kw.pop("ray_ids")
        cubes[name] = PL.sliding_window_predict(model, p1, ray_ids, device,
                                                **kw)
        print(f"[g2] inference {name}: done ({time.time()-ti:.0f}s)",
              flush=True)

    np.save(PRED_NPY, cubes["actual"].astype(np.float32))
    pred_md5 = _md5(PRED_NPY)
    print(f"[g2] prediction cube saved {PRED_NPY.name} md5={pred_md5}",
          flush=True)

    # ------------------------------------------------ scoring, all columns
    scores = {}
    smoothed_cache = {}          # (name, sigma) -> smoothed cube (float64)
    for name in cubes:
        entry = {fr: {} for fr in FRAMES}
        for s in SIGMAS:
            cs = NC.gaussian_smooth_periodic(cubes[name], BOX_MPC_H, s)
            smoothed_cache[(name, s)] = cs
            for fr in FRAMES:
                blk = per_block_pearson(st[fr][s], cs)
                entry[fr][f"{s:g}"] = {
                    "pearson_masked": masked_pearson(st[fr][s], cs, mask),
                    "spearman_masked": masked_spearman(st[fr][s], cs, mask),
                    "block_r": [float(v) for v in blk],
                    "block_mean": float(np.mean(blk)),
                    "block_se": float(np.std(blk, ddof=1) / np.sqrt(len(blk))),
                }
        scores[name] = entry
        print(f"[g2] score {name}: r_s(2,real,test)="
              f"{entry['real']['2']['pearson_masked']:.4f} "
              f"(zspace {entry['zspace']['2']['pearson_masked']:.4f})",
              flush=True)
    # drop cached smoothed control cubes not needed downstream (keep actual,
    # z4, z5, baselines handled separately)
    keep = {("actual", s) for s in SIGMAS} | {
        ("z4_cross_physics_swap_P2", 2.0), ("z5_untrained_swap_P3", 2.0)}
    for k in list(smoothed_cache):
        if k not in keep:
            del smoothed_cache[k]

    r2_actual = scores["actual"]["real"]["2"]["pearson_masked"]

    # ------------------------------------------------ seam + uniform columns
    interior = PL.window_interior_mask(N192)
    pred_s2 = smoothed_cache[("actual", 2.0)]
    r2_interior = masked_pearson(st["real"][2.0], pred_s2, mask & interior)
    seam = {
        "r_s2_test_mask_hann": r2_actual,
        "r_s2_test_mask_window_interior_hann": r2_interior,
        "interior_definition": "(coord mod 32) in [8,24) on all axes — >=8 "
                               "from every face of every covering window",
        "n_interior_test_voxels": int((mask & interior).sum()),
        "divergence": abs(r2_interior - r2_actual),
        "threshold": 0.02,
        "seam_flag": bool(abs(r2_interior - r2_actual) > 0.02),
        "a2_rule": "a tapered read flagging >0.02 REOPENS the seam issue "
                   "(micro-cycle A2)",
    }
    uniform_co = {
        "r_s2_test_mask_uniform": scores["uniform_co_report"]["real"]["2"]
                                  ["pearson_masked"],
        "hann_minus_uniform_s2_real": r2_actual
        - scores["uniform_co_report"]["real"]["2"]["pearson_masked"],
        "note": "diagnostic co-report only; hann column governs (A1, "
                "irreversible)",
    }

    # ------------------------------------------------ descriptives (actual)
    pred = cubes["actual"]
    pcts = [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5]
    pk_pred, kf = shell_pk(pred)
    pk_truth, _ = shell_pk(x_truth["real"])
    with np.errstate(invalid="ignore", divide="ignore"):
        pk_ratio = pk_pred / pk_truth
    j_taper = 6            # k_taper = 2*pi/(32 vox * 0.3125 Mpc/h) = 6 * kf
    neigh = [4, 5, 7, 8]
    taper_check = {
        "k_taper_h_mpc": 2.0 * np.pi / 10.0,
        "k_fundamental_h_mpc": kf,
        "bin_index_taper": j_taper,
        "pk_ratio_at_taper_bin": float(pk_ratio[j_taper]),
        "pk_ratio_neighbor_bins": {str(j): float(pk_ratio[j]) for j in neigh},
        "excursion_vs_neighbor_mean": float(
            pk_ratio[j_taper] / np.mean([pk_ratio[j] for j in neigh])),
        "note": "PROBE column (micro-cycle): detection channel for possible "
                "k~2pi/32vox modulation from periodic Hann weights; "
                "descriptive, no pre-registered threshold",
    }
    descriptive = {
        "var_ratio_unsmoothed_test_mask": float(np.var(pred[mask])
                                                / np.var(x_truth["real"][mask])),
        "var_ratio_sigma2_test_mask": float(np.var(pred_s2[mask])
                                            / np.var(st["real"][2.0][mask])),
        "x_pdf_percentiles": pcts,
        "x_pdf_pred_test_mask": [float(v) for v in
                                 np.percentile(pred[mask], pcts)],
        "x_pdf_truth_test_mask": [float(v) for v in
                                  np.percentile(x_truth["real"][mask], pcts)],
        "pk_ratio_pred_over_truth_fullbox": [
            None if not np.isfinite(v) else float(v) for v in pk_ratio],
        "pk_bin_k_h_mpc": [float(j * kf) for j in range(len(pk_ratio))],
        "taper_modulation_check": taper_check,
        "standing_disclosure": "posterior-mean variance compression rides "
                               "every recovery quote (PI ruling 3)",
    }

    # ------------------------------------------------ distance stratification
    planes = np.arange(TEST_LO, TEST_HI)
    dists = np.array([distance_to_train_region((i + 0.5) / N192)
                      for i in planes])
    order = np.argsort(dists)
    group_sizes = [8, 7, 7, 7]
    strata = []
    pos = 0
    for gs in group_sizes:
        sel = planes[order[pos:pos + gs]]
        pos += gs
        smask = np.zeros((N192,) * 3, dtype=bool)
        smask[sel] = True
        strata.append({
            "planes_axis0": [int(v) for v in np.sort(sel)],
            "mean_distance_norm": float(np.mean(
                [distance_to_train_region((i + 0.5) / N192) for i in sel])),
            "mean_distance_mpc_h": float(np.mean(
                [distance_to_train_region((i + 0.5) / N192) for i in sel])
                * BOX_MPC_H),
            "r_s2_real_pearson": masked_pearson(st["real"][2.0], pred_s2,
                                                smask),
        })
    distance_strat = {
        "rule": "29 test planes sorted by loader distance_to_train_region "
                "(periodic), grouped 8/7/7/7 near->far is by sort order "
                "(mechanical read-time ruling, disclosed)",
        "strata": strata,
    }
    dev_notes.append(
        "distance-to-train stratification binning (4 distance-sorted plane "
        "groups 8/7/7/7) is a read-time mechanical ruling; the [U-04] "
        "mandatory column pinned the variable, not the binning")

    # ------------------------------------------------ controls: S4 / z4 / z5
    trigger_keys = ("z1_all_zero_input", "z2_mask_only", "z3_deranged_ray")
    controls_r2 = {k: scores[k]["real"]["2"]["pearson_masked"]
                   for k in ("z1_all_zero_input", "z2_mask_only",
                             "z3_deranged_ray", "z4_cross_physics_swap_P2",
                             "z5_untrained_swap_P3")}
    ug_details = {
        k: {"r_s2": controls_r2[k],
            "ge_half_actual": bool(controls_r2[k] >= 0.5 * r2_actual),
            "gt_sealed_test_null975": bool(
                controls_r2[k] > SEALED_TEST_NULL975_S2_REAL)}
        for k in trigger_keys}
    u_g_fired = any(v["ge_half_actual"] or v["gt_sealed_test_null975"]
                    for v in ug_details.values())
    collapse = descriptive["var_ratio_unsmoothed_test_mask"] < 0.01

    def swap_delta(label):
        cs = smoothed_cache[(label, 2.0)]
        d_blk = per_block_pearson(st["real"][2.0], pred_s2) \
            - per_block_pearson(st["real"][2.0], cs)
        d_se = float(np.std(d_blk, ddof=1) / np.sqrt(len(d_blk)))
        d_val = r2_actual - scores[label]["real"]["2"]["pearson_masked"]
        return d_blk, d_se, d_val

    z4_blk, z4_se, z4_val = swap_delta("z4_cross_physics_swap_P2")
    delta_phys = {
        "delta_phys_sigma2_real": z4_val,
        "block_deltas": [float(v) for v in z4_blk],
        "block_se": z4_se,
        "per_sigma_delta_real": {
            f"{s:g}": scores["actual"]["real"][f"{s:g}"]["pearson_masked"]
            - scores["z4_cross_physics_swap_P2"]["real"][f"{s:g}"]
            ["pearson_masked"] for s in SIGMAS},
        "r1_anomaly_flag_two_sided": bool(z4_val < -2.0 * z4_se),
        "within_2se_of_zero": bool(abs(z4_val) <= 2.0 * z4_se),
        "scope_rule": "while Delta_phys ~ 0 (within SE) every recovery claim "
                      "is scoped 'suite-common (shared-IC) structure "
                      "recovered from flux' (3ccf6a2; G3's question)",
    }
    z5_blk, z5_se, z5_val = swap_delta("z5_untrained_swap_P3")
    z5_col = {
        "delta_untrained_sigma2_real": z5_val,
        "block_deltas": [float(v) for v in z5_blk],
        "block_se": z5_se,
        "per_sigma_delta_real": {
            f"{s:g}": scores["actual"]["real"][f"{s:g}"]["pearson_masked"]
            - scores["z5_untrained_swap_P3"]["real"][f"{s:g}"]
            ["pearson_masked"] for s in SIGMAS},
        "rule": "R2 rider (50de53a): untrained-physics swap column — "
                "diagnostic, NOT a trigger; P3 flux into the P1 eval pattern",
    }

    # ------------------------------------------------ prediction-matched null
    vals = np.full((len(FRAMES), len(SIGMAS), 2, N_PRED_NULL), np.nan)
    start = 0
    if PARTIAL.exists():
        ck = np.load(PARTIAL)
        vals, start = ck["vals"], int(ck["done"])
        print(f"[g2] pred-null resuming: {start} done", flush=True)
    for i in range(start, N_PRED_NULL):
        pr = NC.phase_randomized(pred, [SEED_PRED_PHASE, i])
        for si, s in enumerate(SIGMAS):
            prs = NC.gaussian_smooth_periodic(pr, BOX_MPC_H, s)
            for fi, fr in enumerate(FRAMES):
                vals[fi, si, 0, i] = masked_pearson(st[fr][s], prs, mask)
                vals[fi, si, 1, i] = masked_spearman(st[fr][s], prs, mask)
        if (i + 1) % 20 == 0 or i == N_PRED_NULL - 1:
            np.savez(PARTIAL, vals=vals, done=i + 1)
            print(f"[g2] pred-null {i+1}/{N_PRED_NULL} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    band, mc = {}, {}
    for fi, fr in enumerate(FRAMES):
        band[fr], mc[fr] = {}, {}
        for si, s in enumerate(SIGMAS):
            band[fr][f"{s:g}"], mc[fr][f"{s:g}"] = {}, {}
            for mi, m in enumerate(("pearson", "spearman")):
                v = vals[fi, si, mi, :]
                assert np.all(np.isfinite(v))
                band[fr][f"{s:g}"][m] = {
                    "median": float(np.median(v)),
                    "pct_2p5": float(np.percentile(v, 2.5)),
                    "pct_97p5": float(np.percentile(v, 97.5)),
                    "min": float(v.min()), "max": float(v.max()),
                    "values": v.tolist(),
                }
                mc[fr][f"{s:g}"][m] = boot_edge(v, BOOT_EDGE_SEED)

    # both-band clearance table (S3: must clear BOTH the SEALED
    # truth-spectrum band and this prediction-matched band)
    clearance = {}
    for fr in FRAMES:
        clearance[fr] = {}
        for sg in ("1", "2", "4"):
            clearance[fr][sg] = {}
            for m in ("pearson", "spearman"):
                a = scores["actual"][fr][sg][f"{m}_masked"]
                te = sealed_975[fr][sg][m]
                pe = band[fr][sg][m]["pct_97p5"]
                clearance[fr][sg][m] = {
                    "actual": a,
                    "sealed_truth_null_97p5": te,
                    "pred_matched_null_97p5": pe,
                    "clears_sealed": bool(a > te),
                    "clears_pred_matched": bool(a > pe),
                    "clears_both": bool(a > te and a > pe),
                }

    pred_null_payload = {
        "rung": "G2 confirmatory read — per-readout prediction-matched null "
                "band (A10 shape, spec v2 S3)",
        "session_utc": "2026-07-24",
        "protocol": {
            "n_realizations": N_PRED_NULL,
            "construction": "NC.phase_randomized(G2 TEST-read prediction "
                            "cube, seed) — identical |FFT| amplitudes, "
                            "random phases, mean restored",
            "seeds": f"numpy.random.default_rng([{SEED_PRED_PHASE}, i]), "
                     f"i = 0..{N_PRED_NULL-1}",
            "seed_lineage_disclosure": "seed family 20260728 is SHARED with "
                                       "the s3 pred-null "
                                       "(pred_null_bands_s3.json) per the "
                                       "pinned S3 protocol; the randomized "
                                       "cube differs (this read's test "
                                       "prediction, not the s3 prediction)",
            "mask": "region_voxel_interval('test', 192) = [163,192) — the "
                    "sanctioned G2 touch",
            "scoring": "R9 conventions imported: smooth full periodic cube "
                       "FIRST, then mask; both frames; sigma {1,2,4}; "
                       "Pearson + Spearman",
        },
        "prediction_cube": {"file": str(PRED_NPY.relative_to(REPO)),
                            "md5": pred_md5, "dtype_on_disk": "float32"},
        "band": band,
        "mc_error_97p5_edge": {"method": f"nonparametric bootstrap, "
                                         f"{N_BOOT_EDGE} resamples, seeded",
                               "seed": BOOT_EDGE_SEED, **mc},
        "wall_clock_s": time.time() - t0,
    }
    OUT_PRED_NULL.write_text(json.dumps(_js(pred_null_payload), indent=2))
    print(f"[g2] wrote {OUT_PRED_NULL}", flush=True)

    # ------------------------------------------------ G2 pair tests
    unet_slabs = {s: smoothed_cache[("actual", s)][TEST_LO:TEST_HI]
                  for s in SIGMAS}
    truth_slabs = {s: st["real"][s][TEST_LO:TEST_HI] for s in SIGMAS}
    base_slabs = {n: {s: base_s[n][s][TEST_LO:TEST_HI] for s in SIGMAS}
                  for n in x_base}

    def rs(fr, name, s):
        if name == "unet":
            return scores["actual"][fr][f"{s:g}"]["pearson_masked"]
        return hs[fr][name][f"{s:g}"]["pearson_heldout"]

    def blocks_of(fr, name, s):
        if name == "unet":
            return np.array(scores["actual"][fr][f"{s:g}"]["block_r"])
        return np.array(hs[fr][name][f"{s:g}"]["block_r"])

    def slab_of(name, s):
        return unet_slabs[s] if name == "unet" else base_slabs[name][s]

    def pair_test(a, b, s):
        """Ordered pair a > b at smoothing s, real frame primary; the [D-75]
        §7 B-ii five conditions with the two disclosed held-out adaptations
        (8 R9 sub-blocks replace octants; 64 slab-column blocks replace the
        full-box 8^3 block bootstrap)."""
        delta = rs("real", a, s) - rs("real", b, s)
        ttest = NC.paired_fisher_t(blocks_of("real", a, s),
                                   blocks_of("real", b, s))
        boot = column_block_bootstrap(truth_slabs[s], slab_of(a, s),
                                      slab_of(b, s))
        wiener_in_pair = a.startswith("wiener") or b.startswith("wiener")
        signs = {fr: float(np.sign(rs(fr, a, s) - rs(fr, b, s)))
                 for fr in FRAMES}
        sign_consistent = len(set(signs.values())) == 1
        cond = {
            "1_delta_ge_0.10": bool(delta >= 0.10),
            "2_fisher_t": ttest,
            "3_block_bootstrap": boot,
            "4_wiener_sign_both_frames": (bool(sign_consistent)
                                          if wiener_in_pair else None),
            "5_no_frame_sign_reversal": bool(sign_consistent),
        }
        fires = (cond["1_delta_ge_0.10"] and ttest["pass"]
                 and boot["excludes_zero"] and boot["delta_mean"] > 0
                 and delta > 0)
        if wiener_in_pair:
            fires = fires and sign_consistent
        if not sign_consistent:
            fires = False
        all_but_5 = (cond["1_delta_ge_0.10"] and ttest["pass"]
                     and boot["excludes_zero"] and boot["delta_mean"] > 0
                     and delta > 0)
        return {"r_s_a_real": rs("real", a, s), "r_s_b_real": rs("real", b, s),
                "delta_rs": float(delta), "conditions": cond,
                "frame_signs": signs, "fires": bool(fires),
                "all_conditions_except_5": bool(all_but_5)}

    pair_defs = [("unet", "grid"), ("grid", "unet"),
                 ("unet", "wiener_L3"), ("wiener_L3", "unet"),
                 ("unet", "mlp"), ("mlp", "unet")]
    pairs = {}
    for s in SIGMAS:
        key = f"sigma{s:g}"
        pairs[key] = {}
        for a, b in pair_defs:
            pairs[key][f"{a}>{b}"] = pair_test(a, b, s)
            print(f"[g2] pair {a}>{b} sigma={s:g}: "
                  f"delta={pairs[key][f'{a}>{b}']['delta_rs']:+.4f} "
                  f"fires={pairs[key][f'{a}>{b}']['fires']}", flush=True)
    dev_notes.append(
        "pair tests at sigma=1 and sigma=4 are the mechanical extension "
        "needed solely to evaluate cell U-E (win only at sigma=4); the "
        "five-condition machinery of record is defined at sigma=2 (primary "
        "metric); condition-1 threshold 0.10 applied unchanged at the other "
        "sigmas")
    dev_notes.append(
        "condition-3 block bootstrap: [U-04]-disclosed held-out-geometry "
        "adaptation — 64 full-thickness transverse column blocks "
        "(29x24x24) tiling the test slab, paired resample, n_boot=1000, "
        "seed 20260723 (d75 convention); replaces the full-box 8^3 "
        "NC.block_bootstrap_delta_rs geometry")
    dev_notes.append(
        "condition-2 baselines use the BANKED R9 per-block values "
        "(r9_heldout_bars.json block_r) against freshly computed unet "
        "per-block values on the SAME 8 blocks ([163,191) trim, banked R9 "
        "deviation reused)")
    dev_notes.append(
        "z3 derangement (seed 20260729) and z4/z5 swaps operate on the "
        "primary [0,1024) pattern only, matching the banked s3/VAL-read "
        "construction")
    dev_notes.append(
        "MPS float32 forward passes are not bit-reproducible; all quoted "
        "numbers are from this single read of the pinned checkpoint")

    p2s = pairs["sigma2"]

    # ------------------------------------------------ cells U0..U-I
    fires_ug = p2s["unet>grid"]["fires"]
    fires_gu = p2s["grid>unet"]["fires"]
    fires_uw = p2s["unet>wiener_L3"]["fires"]
    fires_um = p2s["unet>mlp"]["fires"]
    fires_mu = p2s["mlp>unet"]["fires"]
    cells = {
        "U0": {"fires": bool(not g1_ok
                             or any(e["verdict"] == "MISMATCH"
                                    for e in identity)),
               "condition": "G1 fails / scoring-path acceptance fails / "
                            "artifact-identity mismatch"},
        "U-A": {"fires": bool(fires_ug and fires_uw and fires_um),
                "condition": "unet>grid AND unet>wiener_L3 AND unet>mlp all "
                             "fire (sigma=2, five conditions)"},
        "U-A'": {"fires": bool(fires_ug and not (fires_uw and fires_um)),
                 "condition": "unet>grid fires but unet>wiener_L3 or "
                              "unet>mlp does not (transitivity anomaly -> "
                              "fail-closed audit)"},
        "U-B": {"fires": bool(fires_um and fires_uw and not fires_ug
                              and not fires_gu),
                "condition": "unet>mlp + unet>wiener_L3 fire; tie with grid "
                             "(neither unet>grid nor grid>unet fires)"},
        "U-C": {"fires": bool(fires_um and fires_gu),
                "condition": "unet>mlp fires AND grid>unet fires"},
        "U-D": {"fires": bool((not fires_um) or fires_mu),
                "condition": "unet>mlp does not fire, or mlp>unet fires"},
        "U-E": {"fires": bool((not p2s["unet>grid"]["fires"])
                              and pairs["sigma4"]["unet>grid"]["fires"]),
                "condition": "unet>grid does not fire at sigma=2 but fires "
                             "at sigma=4 (mechanical extension, see "
                             "deviations)"},
        "U-F": {"fires": bool(any(
                    p2s[f"unet>{b}"]["all_conditions_except_5"]
                    and not p2s[f"unet>{b}"]["conditions"]
                    ["5_no_frame_sign_reversal"]
                    for b in ("grid", "wiener_L3", "mlp"))),
                "condition": "condition-5 frame sign reversal is the sole "
                             "failure on a primary-direction pair"},
        "U-G": {"fires": bool(u_g_fired),
                "condition": "S4 over z1-z3: any control r_s2 >= 0.5 x "
                             f"actual ({0.5 * r2_actual:.5f}) OR > sealed "
                             f"test null97.5 ({SEALED_TEST_NULL975_S2_REAL})",
                "details": ug_details},
        "U-H": {"fires": None,
                "condition": "held-out-physics drop > 0.05 — NOT EVALUABLE "
                             "on this read (no held-out-physics-trained "
                             "model exists; G3 scope)"},
        "U-I": {"fires": False,
                "condition": "trainability/overfit process failure pre-eval",
                "evidence": {"stop_reason": train_rec["stop_reason"],
                             "steps_run": train_rec["steps_run"],
                             "best_val_step": train_rec["best_val_step"],
                             "best_val_mask_mse_smoothed":
                                 train_rec["best_val_mask_mse_smoothed"]}},
    }
    fired = [k for k, v in cells.items() if v["fires"]]

    # ------------------------------------------------ payload
    payload = {
        "rung": "G2 CONFIRMATORY READ — the single sanctioned test-slab "
                "evaluation ([U-04] five-condition machinery + [U-06] v2 "
                "amendments; mechanical, no interpretation)",
        "spec": "experiments/unet-inversion/design/u06_stage2_spec.md "
                "(v1 S(c) + v2 K1/S3/S4/S5/S7 + micro-cycle A1/A2/R1/R2)",
        "session_utc": "2026-07-24",
        "sanctioned_touch": "K1: the test mask is touched exactly twice "
                            "ever — G2 close (THIS read) and G3",
        "checkpoint_of_record": {
            "file": str(CKPT.relative_to(REPO)),
            "md5": ckpt_md5,
            "rule": "best-VAL-MSE primary (K1), never test-informed",
            "n_params_loaded": n_params,
            "n_params_pinned_train_record": N_PARAMS_PIN,
            "arch": "UNet3D 4-level base-32 GN8+SiLU (strict state_dict "
                    "load)",
            "train_record": str(TRAIN_RECORD.relative_to(REPO)),
            "train_stop_reason": train_rec["stop_reason"],
            "train_steps_run": train_rec["steps_run"],
            "best_val_step": train_rec["best_val_step"],
        },
        "identity_checks": identity,
        "g1_clause_c_reassert_test_mask": g1c,
        "mask": {"interval_right_open": [lo, hi], "axis": 0,
                 "region": "test", "n_voxels": int(mask.sum()),
                 "source": "region_voxel_interval('test', 192) — "
                           "runtime-asserted"},
        "ray_patterns": {
            "primary": "[0, 1024) file-order",
            "secondary": "[0, 64) file-order",
            "random_pattern": {
                "seed": SEED_RANDOM_PATTERN,
                "rule": "default_rng(20260730).choice(16384, 1024, "
                        "replace=False), sorted",
                "first_8_ids": [int(v) for v in rays_random[:8]],
                "n_rays": 1024},
        },
        "inference": {
            "taper_of_record": "hann (micro-cycle A1, irreversible); "
                               "uniform demoted to diagnostic co-report",
            "windows": "periodic stride-32, 216 windows, COLA "
                       "accumulated-weight assert <1e-6 (S7)",
            "device": str(device),
        },
        "scores_test_mask": scores,
        "seam_diagnostic": seam,
        "uniform_co_report": uniform_co,
        "descriptive": descriptive,
        "distance_to_train_stratification": distance_strat,
        "controls": {
            "r_s2_real": controls_r2,
            "u_g_S4": {"rule": "z1-z3 only (z4 reclassified 3ccf6a2): any "
                               ">= 0.5 x actual OR > sealed test null97.5",
                       "half_actual": 0.5 * r2_actual,
                       "sealed_test_null975_s2_real":
                           SEALED_TEST_NULL975_S2_REAL,
                       "details": ug_details,
                       "fired": bool(u_g_fired)},
            "variance_collapse_lt_0.01": bool(collapse),
            "z3_derangement": {"seed": SEED_Z3_DERANGE,
                               "n_redraws": n_redraws,
                               "rule": "global derangement of whole-ray "
                                       "flux profiles across the primary "
                                       "pattern, geometry fixed, no "
                                       "self-maps"},
            "delta_phys_z4": delta_phys,
            "z5_untrained_swap": z5_col,
        },
        "null_bands": {
            "sealed_truth_spectrum_band": {
                "file": str(SEALED_BAND.relative_to(REPO)),
                "status": "SEALED test-mask band (K1); provenance of the "
                          "banked 0.22108",
                "pct_97p5": sealed_975,
                "seeds": "default_rng([20260726, i]) i=0..199 (truth "
                         "phase-rand)",
            },
            "prediction_matched_band": {
                "file": str(OUT_PRED_NULL.relative_to(REPO)),
                "pct_97p5": {fr: {sg: {m: band[fr][sg][m]["pct_97p5"]
                                       for m in ("pearson", "spearman")}
                                  for sg in ("1", "2", "4")}
                             for fr in FRAMES},
                "mc_error_97p5_edge_s2_real": mc["real"]["2"]["pearson"],
                "seeds": f"default_rng([{SEED_PRED_PHASE}, i]) i=0..199 — "
                         "seed family shared with the s3 pred-null per the "
                         "pinned S3 protocol (different cube)",
            },
            "both_band_clearance": clearance,
        },
        "g2_pairs": {
            "machinery": "[D-75] §7 B-ii five conditions; pairs ordered "
                         "both directions; real frame primary; "
                         "t_crit(df=7)=2.365; n_boot=1000; operative bars "
                         "from r9_heldout_bars.json",
            "operative_bars_real_s2": {
                n: hs["real"][n]["2"]["pearson_heldout"]
                for n in ("grid", "wiener_L3", "mlp")},
            "bars_quoted_check": BARS_QUOTED,
            "wiener_best_L_on_mask": {"real": "wiener_L3",
                                      "zspace": "wiener_L3"},
            "pairs": pairs,
        },
        "cells": cells,
        "fired_cells": fired,
        "deviations": dev_notes,
        "artifacts": {
            "prediction_cube": {"file": str(PRED_NPY.relative_to(REPO)),
                                "md5": pred_md5,
                                "dtype_on_disk": "float32",
                                "dvc": "tracked post-run"},
            "pred_null_band": str(OUT_PRED_NULL.relative_to(REPO)),
            "this_record": str(OUT.relative_to(REPO)),
        },
        "wall_clock_s": time.time() - t0,
    }
    OUT.write_text(json.dumps(_js(payload), indent=2))
    PARTIAL.unlink(missing_ok=True)
    print(f"[g2] wrote {OUT} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[g2] FIRED CELLS: {fired}", flush=True)
    for sg in ("1", "2", "4"):
        e = scores["actual"]["real"][sg]
        z = scores["actual"]["zspace"][sg]
        print(f"[g2] actual sigma={sg}: real p={e['pearson_masked']:.4f} "
              f"sp={e['spearman_masked']:.4f} | zspace "
              f"p={z['pearson_masked']:.4f} sp={z['spearman_masked']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
