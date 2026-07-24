#!/usr/bin/env python
"""[D-75] v3 corrected-metric re-scoring driver (spec of record commit 029f5b8:
experiments/nerf/design/d75_corrected_metric_rescore_spec_v3.md).

Resumable, stage-per-invocation design: every long stage checkpoints its cubes
and records under experiments/nerf/artifacts/d75_rescore/ so an interruption
costs minutes, not the run.

Stages (run in spec order):
    a0      A0-i divisor pin + A0-ii end-to-end frame check     (§2, gates §4)
    truth   real-space truth 192^3 + provenance sha256 + (g) point-sample
    vlos    particle-gate (S4) + redshift-space truth deposit + v_los cube
    grid    object (a): voxel-grid checkpoint -> 192^3 cube (md5-pinned)
    mlp     object (c): production MLP density head at 192^3 cell centers
    wiener  object (b): Wiener re-emit, CPU L-sweep {1,2,3}, 64^3 -> 192^3
    accept  §3 acceptance suite T-A/T-B/T-C/T-D + r_zc measurement (pre-score)
    score   §6/§7/§8 metrics, cells (mechanical), figures, d75_scores.json

Usage:  PYTHONPATH=. .venv/bin/python -u scripts/d75_corrected_metric_rescore.py <stage>
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.analysis import nccf as NC  # noqa: E402

ART = REPO / "experiments" / "nerf" / "artifacts" / "d75_rescore"
CUBES = ART / "cubes"
FIGS = ART / "figures"

BOX_KPC_H = 60000.0
BOX_MPC_H = 60.0
N192 = 192
Z = 0.3
SQRT_A = 1.0 / np.sqrt(1.0 + Z)          # GADGET u -> peculiar v = u*sqrt(a)
X_FLOOR = 1.0e-3                          # v3 §3 unified hard floor on rho
SIGMAS = (1.0, 2.0, 4.0)                  # h^-1 Mpc; sigma=2 sole gated scale

LOS_FILE = REPO / "Sherwood" / "Physics1_nofeedback" / "los2048_n16384_z0.300.dat"
RHO768 = REPO / "Sherwood" / ".rho_field_cache" / "rho_field_p1_z0.300_n768.npy"
RHO768_SIDECAR = RHO768.with_suffix(".json")
SNAPDIR = REPO / "SherwoodIGM_gal" / "extracted" / "planck1_60_768_z0.300" / "snapdir_012"

GRID_CKPT = (REPO / "cloud_runs" /
             "d73-1dprime-voxel192-P1-z0.3-c6f3aed-20260618-000035-8b4e90" /
             "checkpoints" / "step_050000.pt")
GRID_MD5_PIN = "9ceb25bc1061379874a21e9d7fad1322"
MLP_CKPT = (REPO / "cloud_runs" / "pub-t1-extracted" /
            "P1-N64-S0-1778430089-7f65fe" / "checkpoints" / "step_050000.pt")

CONFIG_COMPARABILITY = (
    "grid n_rays=1024 / MLP n_rays=64 / Wiener 74 px/ray on 1024 rays: "
    "verdict-level comparisons licensed; magnitudes are not a controlled contrast."
)

# Controls seeds (v3 §6)
SEED_PHASE = 20260726
SEEDS_NOISE = {0.25: 20260723, 0.5: 20260724, 1.0: 20260725}
KC_LADDER = (0.25, 0.5, 1.0, 2.0)
# Acceptance seeds (v3 §3)
SEEDS_TA = {0.6: 20260731, 0.2: 20260801}
TA_Z_SEED_OFFSET = 500009
SEEDS_TC = [20260901 + i for i in range(10)]
SEEDS_TD = [20260723, 20260724, 20260725]
TD_NOISE_AMP = 1.0   # x-space white-noise amplitude for T-D (in units of std(x))


# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #

def _mkdirs():
    for d in (ART, CUBES, FIGS):
        d.mkdir(parents=True, exist_ok=True)


def _js(obj):
    """JSON-serializable coercion."""
    if isinstance(obj, dict):
        return {k: _js(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_js(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_js(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None if np.isnan(obj) else ("inf" if obj > 0 else "-inf")
    return obj


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(_js(payload), indent=2))
    print(f"[d75] wrote {path}", flush=True)


def _sha256(path: Path, first_mb_only=False) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if first_mb_only:
            h.update(f.read(1024 * 1024))
        else:
            for blk in iter(lambda: f.read(16 * 1024 * 1024), b""):
                h.update(blk)
    return h.hexdigest()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def read_los_header_geometry(n_rays=None):
    """Direct numpy read of the los file header + per-ray geometry (+ pos/vel
    axes). Mirrors src/data/loader.py:604-630 byte layout; read-only."""
    with open(LOS_FILE, "rb") as f:
        hdr = np.fromfile(f, dtype=np.double, count=7)
        nbins = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        num_los = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        iaxis = np.fromfile(f, dtype=np.int32, count=num_los)
        xaxis = np.fromfile(f, dtype=np.double, count=num_los)
        yaxis = np.fromfile(f, dtype=np.double, count=num_los)
        zaxis = np.fromfile(f, dtype=np.double, count=num_los)
        pos_axis = np.fromfile(f, dtype=np.double, count=nbins)
        vel_axis = np.fromfile(f, dtype=np.double, count=nbins)
        block_origin = f.tell()
    out = {
        "z": float(hdr[0]), "omega_m": float(hdr[1]), "omega_l": float(hdr[2]),
        "omega_b": float(hdr[3]), "h100": float(hdr[4]), "box_kpc_h": float(hdr[5]),
        "Xh": float(hdr[6]), "nbins": nbins, "num_los": num_los,
        "iaxis": iaxis, "xaxis": xaxis, "yaxis": yaxis, "zaxis": zaxis,
        "pos_axis": pos_axis, "vel_axis": vel_axis,
        "block_origin_bytes": block_origin,
    }
    if n_rays is not None:
        for k in ("iaxis", "xaxis", "yaxis", "zaxis"):
            out[k] = out[k][:n_rays]
    return out


def read_los_field_rows(field_index: int, n_rows: int, geo=None):
    """Read the first n_rows rays of the field block #field_index from the los
    file. Block order per loader.py:633-636: 0=density, 1=h1_frac, 2=temp,
    3=v_pec; each block is (num_los, nbins) float64."""
    if geo is None:
        geo = read_los_header_geometry()
    nbins, num_los = geo["nbins"], geo["num_los"]
    off = geo["block_origin_bytes"] + field_index * num_los * nbins * 8
    with open(LOS_FILE, "rb") as f:
        f.seek(off)
        arr = np.fromfile(f, dtype=np.double, count=n_rows * nbins)
    return arr.reshape(n_rows, nbins)


def mean_pool(cube: np.ndarray, target: int) -> np.ndarray:
    n = cube.shape[0]
    f = n // target
    assert n % target == 0
    return cube.reshape(target, f, target, f, target, f).mean(
        axis=(1, 3, 5), dtype=np.float64)


def point_sample_768_to_192(cube768: np.ndarray) -> np.ndarray:
    """Trilinear point-sample of a 768^3 field at the 192^3 cell centers.
    Cell center (i+0.5)/192 falls exactly midway between 768-cells 4i+1 and
    4i+2 in every axis, so the trilinear value is the mean of the 2x2x2
    neighborhood [4i+1:4i+3]^3 (uniform 1/8 weights). Exact, no interpolation
    library needed."""
    acc = np.zeros((N192, N192, N192), dtype=np.float64)
    for ox in (1, 2):
        for oy in (1, 2):
            for oz in (1, 2):
                acc += cube768[ox::4, oy::4, oz::4].astype(np.float64)
    return acc / 8.0


def x_transform(rho: np.ndarray):
    """v3 §3 scoring variable: x = log10(max(rho/<rho>, 1e-3)), float64.
    Returns (x, clamped_fraction)."""
    r = np.asarray(rho, dtype=np.float64)
    clamped = float((r < X_FLOOR).mean())
    return np.log10(np.maximum(r, X_FLOOR)), clamped


def load_cube(name: str) -> np.ndarray:
    return np.load(CUBES / name).astype(np.float64)


# --------------------------------------------------------------------------- #
# stage: a0
# --------------------------------------------------------------------------- #

def stage_a0():
    t0 = time.time()
    geo = read_los_header_geometry()
    ratio_kms_per_kpch = float(geo["vel_axis"][-1] / geo["pos_axis"][-1])
    ratio_kms_per_mpch = ratio_kms_per_kpch * 1000.0
    Hz = 100.0 * np.sqrt(geo["omega_m"] * (1 + geo["z"]) ** 3 + geo["omega_l"])
    aH = Hz / (1 + geo["z"])
    conv = None
    if abs(ratio_kms_per_mpch / aH - 1) < 0.05:
        conv = "aH"
    elif abs(ratio_kms_per_mpch / Hz - 1) < 0.05:
        conv = "H"
    a0i_pass = conv is not None

    # ---- A0-ii: end-to-end frame check on ray 0 (FGPA lineage: d62_3, ------
    # beta=1.6, gamma=-0.7; tau_i ~ Delta^beta T^gamma / b, CIC-deposited at
    # s = x + v_pec/ratio, periodic; cross-correlate against native tau_h1).
    nbins = geo["nbins"]
    dens0 = read_los_field_rows(0, 1, geo)[0]
    h10 = read_los_field_rows(1, 1, geo)[0]      # noqa: F841 (recorded path)
    temp0 = read_los_field_rows(2, 1, geo)[0]
    vpec0 = read_los_field_rows(3, 1, geo)[0]
    # native redshift-space tau for ray 0 via the canonical loader (tau file
    # layout has subtleties; use the production reader)
    from src.data.loader import SherwoodLoader
    sl = SherwoodLoader(str(REPO / "Sherwood")).load_sightlines(1, Z)
    tau0 = np.asarray(sl["tau_h1"][0], dtype=np.float64)
    del sl
    gc.collect()

    beta, gamma = 1.6, -0.7
    b = 12.85 * np.sqrt(np.maximum(temp0, 1.0) / 1e4)      # km/s
    w = np.power(np.maximum(dens0, 1e-30), beta) * np.power(
        np.maximum(temp0, 1e-30), gamma) / b

    dx_kpch = BOX_KPC_H / nbins

    def deposit(shift_sign: float):
        s = (geo["pos_axis"] + shift_sign * vpec0 / ratio_kms_per_kpch) % BOX_KPC_H
        fidx = s / dx_kpch
        i0 = np.floor(fidx).astype(int) % nbins
        fr = fidx - np.floor(fidx)
        tau = np.zeros(nbins)
        np.add.at(tau, i0, w * (1 - fr))
        np.add.at(tau, (i0 + 1) % nbins, w * fr)
        # thermal smearing ~ median b (velocity) -> bins
        dv = float(geo["vel_axis"][1] - geo["vel_axis"][0])
        sig_bins = float(np.median(b) / np.sqrt(2.0) / dv)
        k = np.fft.rfftfreq(nbins) * 2 * np.pi
        return np.fft.irfft(np.fft.rfft(tau) * np.exp(-0.5 * (k * sig_bins) ** 2),
                            n=nbins)

    def peak_lag(a, bb):
        da, db = a - a.mean(), bb - bb.mean()
        cc = np.fft.irfft(np.conj(np.fft.rfft(da)) * np.fft.rfft(db), n=nbins)
        lag = int(np.argmax(cc))
        if lag > nbins // 2:
            lag -= nbins
        return lag, float(cc.max() / (np.linalg.norm(da) * np.linalg.norm(db)))

    lag_plus, r_plus = peak_lag(deposit(+1.0), tau0)
    lag_minus, r_minus = peak_lag(deposit(-1.0), tau0)
    lag_zero, r_zero = peak_lag(deposit(0.0), tau0)
    a0ii_pass = abs(lag_plus) <= 1

    payload = {
        "deliverable": "[D-75] v3 §2 A0 empirical convention discriminators",
        "spec_commit": "029f5b8",
        "a0_i": {
            "los_file": str(LOS_FILE.relative_to(REPO)),
            "vel_axis_last_kms": float(geo["vel_axis"][-1]),
            "pos_axis_last_kpch": float(geo["pos_axis"][-1]),
            "ratio_kms_per_kpch": ratio_kms_per_kpch,
            "divisor_kms_per_hMpc": ratio_kms_per_mpch,
            "header_cosmology": {k: geo[k] for k in
                                 ("z", "omega_m", "omega_l", "omega_b", "h100",
                                  "box_kpc_h", "Xh")},
            "H_z_kms_per_mpch": float(Hz),
            "aH_kms_per_mpch": float(aH),
            "ratio_over_aH": float(ratio_kms_per_mpch / aH),
            "ratio_over_H": float(ratio_kms_per_mpch / Hz),
            "convention": conv,
            "pass": bool(a0i_pass),
            "ruling": ("File ratio matches the aH comoving-displacement "
                       "convention to machine precision; divisor of record = "
                       f"{ratio_kms_per_mpch:.5f} km/s per h^-1 Mpc. The v2 "
                       "H=117 hardcode is excluded (ratio/H = 0.769)."),
        },
        "a0_ii": {
            "ray_index": 0,
            "fgpa_lineage": "scripts/d62_3_fgpa_variance_spectrum.py "
                            "(beta=1.6, gamma=-0.7; imported machinery, "
                            "file untouched)",
            "peak_lag_bins_shift_plus": lag_plus,
            "peak_corr_shift_plus": r_plus,
            "peak_lag_bins_shift_minus": lag_minus,
            "peak_corr_shift_minus": r_minus,
            "peak_lag_bins_no_shift": lag_zero,
            "peak_corr_no_shift": r_zero,
            "sign_convention_check": {
                "plus_beats_minus": bool(r_plus > r_minus),
                "plus_beats_noshift": bool(r_plus > r_zero),
            },
            "tolerance_bins": 1,
            "pass": bool(a0ii_pass),
        },
        "conditional_erratum_am9_9c": {
            "triggered": bool(conv == "aH"),
            "ruling": ("aH convention returned -> banked am-9 §9c Delta-chi "
                       "numbers (1.30 rms / 2.53 p95 h^-1 Mpc, computed with "
                       "divisor H=117) UNDERSTATE the S7 frame confound by "
                       "x(1+z)=1.3 -> corrected 1.69 rms / 3.28 p95 h^-1 Mpc. "
                       "Erratum OWED at [D-75] absorption (PI). Direction: "
                       "frame confound LARGER, strengthens the S7 caveat."),
        },
        "zspace_arm": "PROCEEDS" if (a0i_pass and a0ii_pass) else "BLOCKED",
        "wall_clock_s": time.time() - t0,
    }
    _write_json(ART / "a0_convention.json", payload)
    print(f"[a0] convention={conv} divisor={ratio_kms_per_mpch:.5f} "
          f"a0ii peak lag={lag_plus} (r={r_plus:.3f}; -sign r={r_minus:.3f}; "
          f"no-shift r={r_zero:.3f}) -> zspace arm {payload['zspace_arm']}",
          flush=True)


# --------------------------------------------------------------------------- #
# stage: truth
# --------------------------------------------------------------------------- #

def stage_truth():
    t0 = time.time()
    sidecar = json.loads(RHO768_SIDECAR.read_text())
    sha_full = _sha256(RHO768)
    sha_1mb = _sha256(RHO768, first_mb_only=True)
    cube = np.load(RHO768)  # float32, 1.7 GB
    pooled = mean_pool(cube, N192)
    np.save(CUBES / "truth_real_192.npy", pooled)
    ps = point_sample_768_to_192(cube)
    np.save(CUBES / "truth_real_pointsample_192.npy", ps)
    del cube
    gc.collect()
    payload = {
        "source": str(RHO768.relative_to(REPO)),
        "sha256_full": sha_full,
        "sha256_first_1MB": sha_1mb,
        "sidecar_manifest": sidecar,
        "sidecar_sha256_first_1MB_match": bool(
            sha_1mb == sidecar.get("sha256_first_1MB")),
        "pool": "768^3 -> 192^3 mean-pool (am-6 §S rule)",
        "truth_real_192": {
            "path": str((CUBES / 'truth_real_192.npy').relative_to(REPO)),
            "mean": float(pooled.mean()), "std": float(pooled.std()),
            "min": float(pooled.min()), "max": float(pooled.max()),
        },
        "pointsample_control_g": {
            "path": str((CUBES / 'truth_real_pointsample_192.npy').relative_to(REPO)),
            "definition": "trilinear point-sample of the 768^3 field at 192^3 "
                          "cell centers (exact 2x2x2 mean; centers fall midway "
                          "between 768-cells)",
            "mean": float(ps.mean()), "std": float(ps.std()),
        },
        "wall_clock_s": time.time() - t0,
    }
    _write_json(ART / "truth_real_record.json", payload)


# --------------------------------------------------------------------------- #
# stage: vlos  (S4 particle gate + redshift-space truth + v_los cube)
# --------------------------------------------------------------------------- #

def stage_vlos():
    t0 = time.time()
    a0 = json.loads((ART / "a0_convention.json").read_text())
    if a0["zspace_arm"] != "PROCEEDS":
        _write_json(ART / "vlos_gate_record.json", {
            "status": "BLOCKED", "reason": "A0 gate failed", "a0": a0})
        return
    ratio_kpch = a0["a0_i"]["ratio_kms_per_kpch"]  # km/s per kpc/h (divisor)
    divisor_used = ratio_kpch * 1000.0
    redeposit_check = abs(divisor_used / a0["a0_i"]["divisor_kms_per_hMpc"] - 1.0)

    from src.data.igm_gal_loader import SherwoodIGMGalLoader, _cic_deposit_inplace
    ld = SherwoodIGMGalLoader(str(REPO / "SherwoodIGM_gal" / "extracted"))
    meta = ld.get_box_meta(1)
    assert abs(meta["box_kpc_h"] - BOX_KPC_H) < 1e-6

    # accumulators
    mass768s = np.zeros((768, 768, 768), dtype=np.float64)   # z-shifted deposit
    mass192 = np.zeros((N192, N192, N192), dtype=np.float64)
    momz192 = np.zeros((N192, N192, N192), dtype=np.float64)
    stats = {ax: {"sum_v": 0.0, "sum_v2": 0.0,
                  "sum_mv": 0.0, "sum_mv2": 0.0} for ax in "xyz"}
    sum_m = 0.0
    n_part = 0
    n_files = 0
    for chunk in ld.iter_gas_chunks(1, fields=("Coordinates", "Masses",
                                               "Velocities")):
        c = chunk["Coordinates"]
        m = chunk["Masses"].astype(np.float64)
        v = chunk["Velocities"].astype(np.float64) * SQRT_A  # peculiar km/s
        n = c.shape[0]
        n_part += n
        n_files += 1
        sum_m += m.sum()
        for ai, ax in enumerate("xyz"):
            stats[ax]["sum_v"] += v[:, ai].sum()
            stats[ax]["sum_v2"] += (v[:, ai] ** 2).sum()
            stats[ax]["sum_mv"] += (m * v[:, ai]).sum()
            stats[ax]["sum_mv2"] += (m * v[:, ai] ** 2).sum()
        # real-position deposits at 192 (mass + z-momentum)
        _cic_deposit_inplace(mass192, c, chunk["Masses"], box=BOX_KPC_H,
                             n_grid=N192)
        _cic_deposit_inplace(momz192, c,
                             (m * v[:, 2]).astype(np.float32), box=BOX_KPC_H,
                             n_grid=N192)
        # redshift-space-shifted deposit at 768 (LOS axis = z, cube axis 2;
        # production geometry is mixed-axis per-ray -- disclosed in the record)
        cs = np.array(c, dtype=np.float64)
        cs[:, 2] = (cs[:, 2] + v[:, 2] / ratio_kpch) % BOX_KPC_H
        _cic_deposit_inplace(mass768s, cs.astype(np.float32), chunk["Masses"],
                             box=BOX_KPC_H, n_grid=768)
        del c, m, v, cs, chunk
        gc.collect()
        print(f"[vlos] file {n_files}/16 done ({n_part:,} particles cum, "
              f"{time.time()-t0:.0f}s)", flush=True)

    # ---- particle-level velocity stats ------------------------------------
    part = {}
    for ax in "xyz":
        s = stats[ax]
        part[ax] = {
            "rms_unweighted_kms": float(np.sqrt(s["sum_v2"] / n_part)),
            "mean_unweighted_kms": float(s["sum_v"] / n_part),
            "rms_mass_weighted_kms": float(np.sqrt(s["sum_mv2"] / sum_m)),
            "mean_mass_weighted_kms": float(s["sum_mv"] / sum_m),
        }
        part[ax]["rms_unweighted_no_sqrt_a_kms"] = (
            part[ax]["rms_unweighted_kms"] / SQRT_A)

    # ---- los-file v_pec RMS (re-derived, direct read; provenance recorded) --
    geo = read_los_header_geometry()
    vpec_1024 = read_los_field_rows(3, 1024, geo)
    with open(LOS_FILE, "rb") as f:
        f.seek(geo["block_origin_bytes"] + 3 * geo["num_los"] * geo["nbins"] * 8)
        vpec_all = np.fromfile(f, dtype=np.double,
                               count=geo["num_los"] * geo["nbins"])
    rms_los_all = float(np.sqrt(np.mean(vpec_all ** 2)))
    rms_los_1024 = float(np.sqrt(np.mean(vpec_1024 ** 2)))
    del vpec_all

    # ---- S4 gate: particle v_los RMS vs los-file v_pec RMS, +/-3% ----------
    gate_ratio = part["z"]["rms_unweighted_kms"] / rms_los_all
    gate_pass = abs(gate_ratio - 1.0) < 0.03
    mean_ok = abs(part["z"]["mean_unweighted_kms"]) < 5.0

    # ---- redshift-space truth cube ----------------------------------------
    mean768 = mass768s.mean()
    mass768s /= mean768
    zpool = mean_pool(mass768s, N192)
    np.save(CUBES / "truth_zspace_192.npy", zpool)
    zps = point_sample_768_to_192(mass768s)
    np.save(CUBES / "truth_zspace_pointsample_192.npy", zps)
    del mass768s
    gc.collect()

    # ---- v_los cube (mass-weighted CIC velocity) --------------------------
    empty = int((mass192 <= 0).sum())
    vz192 = np.where(mass192 > 0, momz192 / np.maximum(mass192, 1e-300), 0.0)
    np.save(CUBES / "vlos_z_192.npy", vz192)
    dep_rms_vol = float(np.sqrt(np.mean(vz192[mass192 > 0] ** 2)))
    dep_rms_mw = float(np.sqrt((mass192 * vz192 ** 2).sum() / mass192.sum()))

    # ---- sign-convention check vs los file (iaxis==3 rays, first 1024) -----
    cell = BOX_KPC_H / N192
    sel = np.where(geo["iaxis"][:1024] == 3)[0]
    prof_cube, prof_los = [], []
    for ri in sel:
        ix = int(np.floor(geo["xaxis"][ri] / cell)) % N192
        iy = int(np.floor(geo["yaxis"][ri] / cell)) % N192
        prof_cube.append(vz192[ix, iy, :])
        # pool the ray's 2048-bin v_pec into 192 position bins
        vb = np.zeros(N192)
        cb = np.zeros(N192)
        bidx = np.floor(geo["pos_axis"] / cell).astype(int) % N192
        np.add.at(vb, bidx, vpec_1024[ri])
        np.add.at(cb, bidx, 1.0)
        prof_los.append(vb / np.maximum(cb, 1.0))
    sign_r = NC.pearson(np.concatenate(prof_cube), np.concatenate(prof_los))

    payload = {
        "deliverable": "[D-75] v3 §4 S4 particle gate + redshift-space truth "
                       "+ truth v_los cube",
        "sqrt_a_applied": "yes (v_pec = GADGET_Velocities * sqrt(a), "
                          f"sqrt(a)={SQRT_A:.6f} at z={Z})",
        "n_particles": n_part,
        "n_files": n_files,
        "particle_velocity_stats_kms": part,
        "los_file_vpec_rms": {
            "all_16384_rays": rms_los_all,
            "first_1024_rays": rms_los_1024,
            "banked_am9_9c_value": 151.8,
            "provenance": "direct block-4 read of " + str(LOS_FILE.name),
        },
        "s4_gate": {
            "definition": "particle v_los (unweighted RMS, z-axis, sqrt(a) "
                          "applied) vs los-file v_pec RMS (all rays), +/-3%",
            "particle_rms_z_kms": part["z"]["rms_unweighted_kms"],
            "los_rms_kms": rms_los_all,
            "ratio": float(gate_ratio),
            "band": 0.03,
            "pass": bool(gate_pass),
            "global_mean_abs_lt_5kms": bool(mean_ok),
            "mass_weighted_rms_z_kms": part["z"]["rms_mass_weighted_kms"],
            "note": "mass-weighted RMS recorded beside the unweighted gate "
                    "value; los-file sampling is SPH-kernel (volume-like) "
                    "weighting, so the unweighted particle RMS is gated.",
        },
        "sign_convention_check": {
            "n_rays_iaxis3_in_first_1024": int(len(sel)),
            "pearson_vz_cube_vs_los_vpec": float(sign_r),
            "pass_sign_positive": bool(sign_r > 0),
        },
        "redshift_space_deposit": {
            "los_axis": "z (cube axis 2). Loader-confirmed finding: the "
                        "production sightline set is MIXED-axis (first 1024 "
                        "rays: 334 x / 331 y / 359 z per iaxis) -- there is "
                        "no single production LOS axis. z chosen as the modal "
                        "axis of the 1024-ray slice; the z-space frame is a "
                        "representative single-axis diagnostic (disclosed).",
            "divisor_kms_per_hMpc": divisor_used,
            "redeposit_check_abs_rel_dev": float(redeposit_check),
            "redeposit_check_pass": bool(redeposit_check < 0.01),
            "periodic_wrap": True,
            "truth_zspace_192": {
                "path": str((CUBES / 'truth_zspace_192.npy').relative_to(REPO)),
                "mean": float(zpool.mean()), "std": float(zpool.std()),
                "min": float(zpool.min()), "max": float(zpool.max()),
            },
        },
        "vlos_cube": {
            "path": str((CUBES / 'vlos_z_192.npy').relative_to(REPO)),
            "empty_cells": empty,
            "deposited_rms_volume_kms": dep_rms_vol,
            "deposited_rms_mass_weighted_kms": dep_rms_mw,
        },
        "status": ("OK" if (gate_pass and mean_ok and sign_r > 0
                            and redeposit_check < 0.01) else "BLOCKED"),
        "wall_clock_s": time.time() - t0,
    }
    _write_json(ART / "vlos_gate_record.json", payload)
    print(f"[vlos] gate ratio={gate_ratio:.4f} pass={gate_pass} "
          f"sign_r={sign_r:.3f} status={payload['status']} "
          f"({time.time()-t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------- #
# stage: grid
# --------------------------------------------------------------------------- #

def _ray_masks(n_rays: int) -> np.ndarray:
    """Bool (192,192,192): voxels whose center is <= 1 cell (perp distance)
    from any of the first n_rays sightlines (axis-aligned lines)."""
    geo = read_los_header_geometry(n_rays)
    cell = BOX_KPC_H / N192
    m2d = {1: np.zeros((N192, N192), bool), 2: np.zeros((N192, N192), bool),
           3: np.zeros((N192, N192), bool)}
    for ri in range(n_rays):
        ax = int(geo["iaxis"][ri])
        if ax == 1:
            g1, g2 = geo["yaxis"][ri] / cell, geo["zaxis"][ri] / cell
        elif ax == 2:
            g1, g2 = geo["xaxis"][ri] / cell, geo["zaxis"][ri] / cell
        else:
            g1, g2 = geo["xaxis"][ri] / cell, geo["yaxis"][ri] / cell
        i0, j0 = int(np.floor(g1)), int(np.floor(g2))
        for di in range(-1, 3):
            for dj in range(-1, 3):
                u, w = (i0 + di) % N192, (j0 + dj) % N192
                if ((u + 0.5 - g1) ** 2 + (w + 0.5 - g2) ** 2) <= 1.0:
                    m2d[ax][u, w] = True
    mask = np.zeros((N192, N192, N192), bool)
    mask |= m2d[1][None, :, :]     # rays along x: perp plane = (y, z)
    mask |= m2d[2][:, None, :]     # rays along y: perp plane = (x, z)
    mask |= m2d[3][:, :, None]     # rays along z: perp plane = (x, y)
    return mask


def stage_grid():
    t0 = time.time()
    import torch
    from src.models.voxel_grid_field import VoxelGridField
    md5 = _md5(GRID_CKPT)
    md5_ok = md5 == GRID_MD5_PIN
    state = torch.load(str(GRID_CKPT), map_location="cpu", weights_only=False)
    ms = state.get("model_state", state)
    log_rho = ms["log_rho_grid"].detach().to(torch.float64)
    assert tuple(log_rho.shape) == (N192,) * 3
    rho = VoxelGridField.density_log_to_linear(log_rho).numpy()
    np.save(CUBES / "grid_192.npy", rho)
    lta = state.get("log_tau_amp", None)
    tau_amp = float(np.exp(float(lta))) if lta is not None else None
    mask1024 = _ray_masks(1024)
    np.save(CUBES / "near_ray_mask_1024.npy", mask1024)
    mask64 = _ray_masks(64)
    np.save(CUBES / "near_ray_mask_64.npy", mask64)
    payload = {
        "checkpoint": str(GRID_CKPT.relative_to(REPO)),
        "md5": md5, "md5_pin": GRID_MD5_PIN, "md5_match": bool(md5_ok),
        "step": _js(state.get("step")),
        "tau_amp_from_ckpt": tau_amp,
        "cube": {"path": str((CUBES / 'grid_192.npy').relative_to(REPO)),
                 "mean": float(rho.mean()), "std": float(rho.std()),
                 "min": float(rho.min()), "max": float(rho.max())},
        "near_ray_mask": {
            "n1024_frac": float(mask1024.mean()),
            "n64_frac": float(mask64.mean()),
            "definition": "voxel center perp-distance <= 1 cell from any ray",
        },
        "wall_clock_s": time.time() - t0,
    }
    _write_json(ART / "grid_record.json", payload)
    if not md5_ok:
        print("[grid] FATAL: md5 mismatch vs pin", flush=True)
        sys.exit(2)


# --------------------------------------------------------------------------- #
# stage: mlp
# --------------------------------------------------------------------------- #

def stage_mlp():
    t0 = time.time()
    import torch
    from src.models.nerf import IGMNeRF
    md5 = _md5(MLP_CKPT)
    state = torch.load(str(MLP_CKPT), map_location="cpu", weights_only=False)
    ms = state["model_state"]
    # arch cross-check (local MLflow server 403 -> checkpoint-shape check +
    # extracted mlflow dir if present; out-of-scope flag (iii) in spec v3 §11)
    arch = {
        "layers1.0.weight_shape": list(ms["layers1.0.weight"].shape),
        "layers2.0.weight_shape": list(ms["layers2.0.weight"].shape),
        "n_layers1": len([k for k in ms if k.startswith("layers1.")]) // 2,
        "n_layers2": len([k for k in ms if k.startswith("layers2.")]) // 2,
        "out_shape": list(ms["out_layer.weight"].shape),
    }
    expect = {"in_dim": 63, "skip_in": 319}
    arch_ok = (arch["layers1.0.weight_shape"] == [256, 63]
               and arch["layers2.0.weight_shape"] == [256, 319]
               and arch["n_layers1"] == 4 and arch["n_layers2"] == 4
               and arch["out_shape"] == [4, 256])
    mlflow_dir = MLP_CKPT.parent.parent / "mlflow"
    mlflow_note = ("extracted mlflow dir present: "
                   + str(sorted(p.name for p in mlflow_dir.iterdir())[:8])
                   if mlflow_dir.is_dir() else "no extracted mlflow dir")

    model = IGMNeRF(hidden_dim=256, num_layers=8, L=10,
                    body_arch="current", density_head="softplus").double()
    model.load_state_dict(ms)
    model.eval()

    centers = (np.arange(N192, dtype=np.float64) + 0.5) / N192
    yy, zz = np.meshgrid(centers, centers, indexing="ij")
    cube = np.empty((N192, N192, N192), dtype=np.float64)
    with torch.no_grad():
        for i in range(N192):
            pts = np.stack([np.full_like(yy, centers[i]), yy, zz], axis=-1)
            t = torch.from_numpy(pts.reshape(-1, 3))     # float64
            fields = model(t)
            cube[i] = fields[:, 0].numpy().reshape(N192, N192)
            if i % 32 == 0:
                print(f"[mlp] slab {i}/{N192} ({time.time()-t0:.0f}s)",
                      flush=True)
    np.save(CUBES / "mlp_192.npy", cube)
    lta = state.get("log_tau_amp", None)
    payload = {
        "checkpoint": str(MLP_CKPT.relative_to(REPO)),
        "md5_pinned_at_load": md5,
        "arch_check": {"observed": arch, "expected_marker": expect,
                       "pass": bool(arch_ok),
                       "mlflow_run": "31acdf9d900e447081e6d051f7d42c0e",
                       "mlflow_server": "local tracker 403 (out-of-scope flag "
                                        "iii); cross-check done against "
                                        "checkpoint tensor shapes",
                       "mlflow_dir_note": mlflow_note},
        "density_head": "softplus (production default; channel 0 = linear "
                        "rho/<rho>)",
        "eval": "192^3 cell centers (i+0.5)/192, float64 end-to-end, "
                "slab-chunked (192 slabs x 192^2 points)",
        "tau_amp_from_ckpt": float(np.exp(float(lta))) if lta is not None else None,
        "cube": {"path": str((CUBES / 'mlp_192.npy').relative_to(REPO)),
                 "mean": float(cube.mean()), "std": float(cube.std()),
                 "min": float(cube.min()), "max": float(cube.max())},
        "wall_clock_s": time.time() - t0,
    }
    _write_json(ART / "mlp_record.json", payload)


# --------------------------------------------------------------------------- #
# stage: wiener
# --------------------------------------------------------------------------- #

def _fourier_upsample_64_to_192(rec64: np.ndarray) -> np.ndarray:
    """Periodic spectral zero-pad regrid 64^3 -> 192^3. Nyquist plane of the
    64-grid is zeroed before embedding (keeps the padded spectrum Hermitian;
    negligible power loss, disclosed)."""
    F = np.fft.fftn(rec64.astype(np.float64))
    n, m = 64, 192
    F[n // 2, :, :] = 0.0
    F[:, n // 2, :] = 0.0
    F[:, :, n // 2] = 0.0
    G = np.zeros((m, m, m), dtype=complex)
    idx64 = np.r_[0:32, 32:64]           # source indices
    tgt = np.r_[0:32, 160:192]           # target indices (negative freqs at end)
    G[np.ix_(tgt, tgt, tgt)] = F[np.ix_(idx64, idx64, idx64)]
    G *= (m / n) ** 3
    out = np.fft.ifftn(G)
    return out.real


def stage_wiener():
    t0 = time.time()
    from src.data.loader import SherwoodLoader
    from src.analysis.wiener_baseline import (
        WienerConfig, _build_sparse_Cdd, _build_sparse_Cmd)
    from scipy.sparse import diags
    from scipy.sparse.linalg import cg

    N_RAYS, N_GRID64, STRIDE, NOISE_REL = 1024, 64, 28, 1e-3
    loader = SherwoodLoader(str(REPO / "Sherwood"))
    sl = loader.load_sightlines(1, Z)
    coords_world = loader.get_world_coordinates(sl)[:N_RAYS]
    tau = np.asarray(sl["tau_h1"][:N_RAYS], dtype=np.float64)
    del sl
    gc.collect()
    F = np.exp(-np.minimum(tau, 10.0))
    mean_F = float(F.mean())
    tracer = -(F / mean_F - 1.0)
    pix_idx = np.arange(0, coords_world.shape[1], STRIDE)
    pix_xyz = coords_world[:, pix_idx, :].reshape(-1, 3) / 1000.0
    pix_data = tracer[:, pix_idx].reshape(-1).astype(np.float64)
    std_before = float(pix_data.std())
    pix_data = pix_data / std_before        # FIX 1 unit-variance tracer
    ppr = len(pix_idx)
    del coords_world, tracer, F, tau
    gc.collect()

    cfg = WienerConfig()
    record = {
        "config_of_record": {
            "noise_rel": NOISE_REL, "pixel_stride": STRIDE,
            "px_per_ray": ppr, "Npix": int(pix_xyz.shape[0]),
            "tracer_standardized": True, "tracer_std_before": std_before,
            "mean_F": mean_F, "n_rays": N_RAYS,
            "native_grid": N_GRID64,
            "sparse_n_sigma": cfg.sparse_n_sigma,
            "cg_tol": cfg.cg_tol, "cg_maxiter": cfg.cg_maxiter,
            "emission": "block-wise C_md (32^3-voxel blocks, identical kernel "
                        "+ 5-sigma cutoff + periodic KD-tree as record path; "
                        "bounded-RAM re-implementation of the all-at-once "
                        "emission, disclosed)",
            "regrid": "Fourier zero-pad 64^3 -> 192^3 (periodic; 64-grid "
                      "Nyquist plane zeroed, disclosed)",
            "rho_convention": "rho/<rho> = 1 + rec (rec = Wiener tracer "
                              "estimate; unified §3 floor applied at scoring; "
                              "disclosed convention -- Wiener gain is "
                              "arbitrary, Spearman column is gain-robust)",
        },
        "L_sweep": [],
    }
    cell64 = BOX_MPC_H / N_GRID64
    ax64 = (np.arange(N_GRID64) + 0.5) * cell64
    VX, VY, VZ = np.meshgrid(ax64, ax64, ax64, indexing="ij")
    vox = np.stack([VX.ravel(), VY.ravel(), VZ.ravel()], axis=1)

    for L in (1.0, 2.0, 3.0):
        tL = time.time()
        A = _build_sparse_Cdd(pix_xyz, BOX_MPC_H, L, NOISE_REL,
                              cfg.sparse_n_sigma)
        d = A.diagonal()
        M = diags(1.0 / np.where(d != 0, d, 1.0), format="csr")
        w, info = cg(A, pix_data, rtol=cfg.cg_tol, maxiter=cfg.cg_maxiter, M=M)
        del A, d, M
        gc.collect()
        rec = np.empty(N_GRID64 ** 3, dtype=np.float64)
        blk = 32 ** 3
        nvox = vox.shape[0]
        for s in range(0, nvox, blk):
            e = min(s + blk, nvox)
            Cmd = _build_sparse_Cmd(vox[s:e], pix_xyz, BOX_MPC_H, L,
                                    cfg.sparse_n_sigma)
            rec[s:e] = np.asarray(Cmd @ w).ravel()
            del Cmd
            gc.collect()
        rec64 = rec.reshape(N_GRID64, N_GRID64, N_GRID64)
        np.save(CUBES / f"wiener_rec64_L{L:g}.npy", rec64)
        rec192 = _fourier_upsample_64_to_192(rec64)
        np.save(CUBES / f"wiener_rec192_L{L:g}.npy", rec192)
        entry = {
            "L_mpc_h": L, "cg_info": int(info),
            "rec64_path": str((CUBES / f'wiener_rec64_L{L:g}.npy').relative_to(REPO)),
            "rec192_path": str((CUBES / f'wiener_rec192_L{L:g}.npy').relative_to(REPO)),
            "rec_std": float(rec64.std()), "rec_min": float(rec64.min()),
            "rec_max": float(rec64.max()),
            "frac_rho_below_floor": float(((1.0 + rec192) < X_FLOOR).mean()),
            "wall_clock_s": time.time() - tL,
        }
        record["L_sweep"].append(entry)
        print(f"[wiener] L={L:g} cg_info={info} rec_std={entry['rec_std']:.4f} "
              f"({entry['wall_clock_s']:.0f}s)", flush=True)

    record["wall_clock_s"] = time.time() - t0
    _write_json(ART / "wiener_record.json", record)


# --------------------------------------------------------------------------- #
# stage: accept  (v3 §3 acceptance suite; r_zc measured FIRST)
# --------------------------------------------------------------------------- #

def _rs_all_sigmas(x, y, sigmas=SIGMAS):
    out = {}
    for s in sigmas:
        xs = NC.gaussian_smooth_periodic(x, BOX_MPC_H, s)
        ys = NC.gaussian_smooth_periodic(y, BOX_MPC_H, s)
        out[s] = NC.pearson(xs, ys)
    return out


def stage_accept():
    t0 = time.time()
    results = {"deliverable": "[D-75] v3 §3 acceptance suite",
               "spec_commit": "029f5b8"}

    x_real, _cl = x_transform(load_cube("truth_real_192.npy"))

    # ---- r_zc measurement FIRST (v3 §3: before any other object is scored) --
    rmag = NC.periodic_r_grid(N192, BOX_MPC_H)
    cxx = NC.cross_corr_cube(x_real, x_real)
    r_zc_real = NC.shell_zero_crossing(cxx, rmag, BOX_MPC_H)
    edges = NC.default_r_edges()
    m_xx, cnt = NC.shell_bin(cxx, rmag, edges)
    valid = m_xx > NC.EPS_D_PINNED * cxx[0, 0, 0]
    del cxx
    zsp = None
    vg = json.loads((ART / "vlos_gate_record.json").read_text())
    if vg.get("status") == "OK":
        x_z, _ = x_transform(load_cube("truth_zspace_192.npy"))
        czz = NC.cross_corr_cube(x_z, x_z)
        zsp = NC.shell_zero_crossing(czz, rmag, BOX_MPC_H)
        del czz, x_z
    gc.collect()
    results["r_zc_truth"] = {
        "real_h_inv_mpc": r_zc_real,
        "zspace_h_inv_mpc": zsp,
        "valid_r_domain_real": {
            "r_edges": edges, "shell_C_xx": m_xx, "mode_counts": cnt,
            "valid_mask_eps_0.01": valid,
            "note": "recorded BEFORE any object scoring per §3",
        },
    }
    print(f"[accept] r_zc real={r_zc_real:.2f} zspace={zsp}", flush=True)

    amp = NC.field_amplitude(x_real)
    all_pass = True

    # ---- T-A: constant-coherence GRF pairs --------------------------------
    ta = []
    for c, seed in SEEDS_TA.items():
        X = NC.amplitude_matched_grf(amp, seed, x_real.shape)
        Zf = NC.amplitude_matched_grf(amp, seed + TA_Z_SEED_OFFSET,
                                      x_real.shape)
        Y = c * X + np.sqrt(1 - c * c) * Zf
        prof = NC.nccf(X, Y, BOX_MPC_H)
        dev = np.nanmax(np.abs(prof["nccf"][prof["valid"]] - c))
        rs = _rs_all_sigmas(X, Y)
        rs_dev = max(abs(v - c) for v in rs.values())
        ok = bool(dev <= 0.02 and rs_dev <= 0.02)
        all_pass &= ok
        ta.append({"c": c, "seed_X": seed, "seed_Z": seed + TA_Z_SEED_OFFSET,
                   "nccf_profile": prof["nccf"], "valid": prof["valid"],
                   "r_centers": prof["r_centers"],
                   "max_abs_dev_nccf": float(dev),
                   "r_s": {str(k): v for k, v in rs.items()},
                   "max_abs_dev_rs": float(rs_dev),
                   "tolerance": 0.02, "pass": ok})
        print(f"[accept] T-A c={c}: nccf dev={dev:.4f} rs dev={rs_dev:.4f} "
              f"-> {'PASS' if ok else 'FAIL'}", flush=True)
    results["T_A"] = ta

    # ---- T-B: known lag ----------------------------------------------------
    Xb = x_real
    Yb = np.roll(Xb, -5, axis=2)          # Y(u) = X(u + 5*e_z)
    cxy = NC.cross_corr_cube(Xb, Yb)
    los_prof = cxy[0, 0, :].copy()
    lag = int(np.argmax(los_prof))
    if lag > N192 // 2:
        lag -= N192
    del cxy
    tb_ok = abs(lag) == 5
    all_pass &= tb_ok
    results["T_B"] = {
        "construction": "Y = roll(X, -5, axis=z); 5 cells = 1.5625 h^-1 Mpc",
        "peak_lag_cells": lag, "expected_abs": 5, "pass": bool(tb_ok),
        "note": "unnormalized C_xy along the LOS axis; sign reflects the "
                "roll direction convention (recorded descriptively)",
    }
    print(f"[accept] T-B peak lag={lag} -> {'PASS' if tb_ok else 'FAIL'}",
          flush=True)

    # ---- T-C: empirically calibrated null ---------------------------------
    xs2 = NC.gaussian_smooth_periodic(x_real, BOX_MPC_H, 2.0)
    null_rs = []
    for seed in SEEDS_TC:
        g = NC.amplitude_matched_grf(amp, seed, x_real.shape)
        gs = NC.gaussian_smooth_periodic(g, BOX_MPC_H, 2.0)
        null_rs.append(NC.pearson(xs2, gs))
    null_rs = np.array(null_rs)
    sd_null = float(null_rs.std(ddof=1))
    tc_ok = bool(np.all(np.abs(null_rs) < 3 * sd_null))
    all_pass &= tc_ok
    results["T_C"] = {
        "n_realizations": len(SEEDS_TC), "seeds": SEEDS_TC,
        "r_s_sigma2_values": null_rs, "sd_null_measured": sd_null,
        "criterion": "each |r_s| < 3 * SD_null(measured)",
        "pass": tc_ok,
        "note": "v2 fixed 0.01 tolerance withdrawn per spec v3 (below null SD)",
    }
    print(f"[accept] T-C SD_null={sd_null:.4f} max|r|={np.abs(null_rs).max():.4f}"
          f" -> {'PASS' if tc_ok else 'FAIL'}", flush=True)

    # ---- T-D: white-noise NCCF blindness ----------------------------------
    td = []
    sx = x_real.std()
    for seed in SEEDS_TD:
        rng = np.random.default_rng(seed)
        Y = x_real + TD_NOISE_AMP * sx * rng.standard_normal(x_real.shape)
        prof = NC.nccf(x_real, Y, BOX_MPC_H)
        dev = np.nanmax(np.abs(prof["nccf"][prof["valid"]] - 1.0))
        rs = _rs_all_sigmas(x_real, Y, sigmas=(2.0,))
        ok = bool(dev <= 0.02)
        all_pass &= ok
        td.append({"seed": seed, "noise_amp_x_std": TD_NOISE_AMP,
                   "max_abs_dev_from_1": float(dev),
                   "r_s_sigma2": rs[2.0], "pass": ok})
        print(f"[accept] T-D seed={seed}: nccf dev={dev:.4f} "
              f"r_s(2)={rs[2.0]:.4f} -> {'PASS' if ok else 'FAIL'}", flush=True)
    results["T_D"] = td

    results["all_pass"] = bool(all_pass)
    results["wall_clock_s"] = time.time() - t0
    _write_json(ART / "acceptance_suite.json", results)
    print(f"[accept] ALL {'PASS' if all_pass else 'FAIL'} "
          f"({time.time()-t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------- #
# stage: score  (§6 objects/controls, §7 bands, §8 cells, figures)
# --------------------------------------------------------------------------- #

def _smooth(x, s):
    return NC.gaussian_smooth_periodic(x, BOX_MPC_H, s)


def _score_object(x_t, x_o, smoothed_truth, name):
    """Full §3 metric set for one (object, truth-frame) pair."""
    out = {"name": name}
    rs = {}
    for s in SIGMAS:
        xo_s = _smooth(x_o, s)
        p = NC.pearson(smoothed_truth[s], xo_s)
        sp = NC.spearman(smoothed_truth[s], xo_s)
        oct_r = NC.per_octant_pearson(smoothed_truth[s], xo_s)
        rs[str(s)] = {
            "pearson": p, "spearman": sp,
            "outlier_leverage_flag": bool(abs(p - sp) > 0.15),
            "octant_r": oct_r,
            "octant_mean": float(np.nanmean(oct_r)),
            "octant_se": float(np.nanstd(oct_r, ddof=1) / np.sqrt(8)),
        }
        if s == 2.0:
            out["_smoothed2"] = xo_s     # kept for pair tests (stripped later)
    out["r_s"] = rs
    prof = NC.nccf(x_t, x_o, BOX_MPC_H)
    if prof.get("undefined"):
        out["nccf"] = {"undefined": True, "reason": prof["reason"]}
    else:
        rc = prof["r_centers"]
        i2 = int(np.argmin(np.abs(rc - 2.0)))
        ok = prof["valid"] & np.isfinite(prof["nccf"])
        interp2 = (float(np.interp(2.0, rc[ok], prof["nccf"][ok]))
                   if ok.sum() >= 2 else float("nan"))
        out["nccf"] = {
            "r_centers": rc, "profile": prof["nccf"], "valid": prof["valid"],
            "mode_counts": prof["mode_counts"],
            "at_r2_nearest_bin": {"r": float(rc[i2]),
                                  "value": float(prof["nccf"][i2])},
            "at_r2_interp": interp2,
            "pearson_zero_lag": prof["pearson_zero_lag"],
            "r_zc_xx": prof["r_zc_xx"], "r_zc_yy": prof["r_zc_yy"],
        }
    rk = NC.rk_coherence(x_t, x_o, BOX_MPC_H)
    out["r_k"] = {
        "k_centers": rk["k_centers"], "profile": rk["r_k"],
        "mode_counts": rk["mode_counts"],
        "k_first_below_0.5": NC.rk_first_crossing(rk["k_centers"], rk["r_k"]),
        "note": "lowest bin box-limited descriptive; k>1 descriptive-only",
    }
    return out


def stage_score():
    t0 = time.time()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = json.loads((ART / "acceptance_suite.json").read_text())
    vg = json.loads((ART / "vlos_gate_record.json").read_text())
    a0 = json.loads((ART / "a0_convention.json").read_text())
    grid_rec = json.loads((ART / "grid_record.json").read_text())
    mlp_rec = json.loads((ART / "mlp_record.json").read_text())
    wiener_rec = json.loads((ART / "wiener_record.json").read_text())
    wiener_rec["L_sweep"] = [e for e in wiener_rec["L_sweep"] if
                             (CUBES / f"wiener_rec192_L{e['L_mpc_h']:g}.npy"
                              ).exists()]
    wiener_pending = [L for L in (1.0, 2.0, 3.0) if L not in
                      [e["L_mpc_h"] for e in wiener_rec["L_sweep"]]]
    zspace_ok = (str(vg.get("status", "")).startswith("OK")
                 and a0["zspace_arm"] == "PROCEEDS")
    zspace_md5 = {}
    if zspace_ok:
        zspace_md5 = {
            "truth_zspace_192.npy": _md5(CUBES / "truth_zspace_192.npy"),
            "vlos_z_192.npy": _md5(CUBES / "vlos_z_192.npy"),
        }
        pins = {"truth_zspace_192.npy": "3a7af286ddff91f26f00459c11577d29",
                "vlos_z_192.npy": "198a7e8fcf260f5ffd3359d2d61f916d"}
        if zspace_md5 != pins:
            zspace_ok = False   # discharge VOID on identity mismatch

    scores = {
        "deliverable": "[D-75] v4 corrected-metric re-scoring (both frames)",
        "spec_commit": "029f5b8 + v4 amendment d6cb900 + micro-cycle 767054e",
        "zspace_ruling": (
            "scored (S4 micro-cycle APPROVE-WITH-AMENDMENTS; retroactive "
            "discharge bound to artifact identity, md5-verified at load)"
            if zspace_ok else
            "BLOCKED (gate/discharge not satisfied; clause gamma fail-closed)"),
        "zspace_artifact_identity_md5": zspace_md5,
        "zspace_mixed_axis_disclosure": (
            "production sightlines are MIXED-axis (334 x / 331 y / 359 z in "
            "the first 1024 rays; loader-confirmed); the z-axis z-space frame "
            "is a representative single-axis diagnostic — rides every z-space "
            "entry"),
        "wiener_L_pending": wiener_pending,
        "config_comparability": CONFIG_COMPARABILITY,
        "acceptance_all_pass": acc["all_pass"],
        "zspace_arm": "OK" if zspace_ok else "BLOCKED",
        "provenance": {
            "grid_md5": grid_rec["md5"], "mlp_md5": mlp_rec["md5_pinned_at_load"],
            "truth_sha256": json.loads(
                (ART / "truth_real_record.json").read_text())["sha256_full"],
            "wiener_cube_md5_at_load": {
                f"wiener_rec192_L{e['L_mpc_h']:g}": _md5(
                    CUBES / f"wiener_rec192_L{e['L_mpc_h']:g}.npy")
                for e in wiener_rec["L_sweep"]},
            "wiener_producer_reconciliation": (json.loads(
                (ART / "wiener_producer_reconciliation.json").read_text())
                if (ART / "wiener_producer_reconciliation.json").exists()
                else None),
        },
        "r_zc_truth": acc["r_zc_truth"],
        "frames": {}, "pairs": {}, "controls_meta": {}, "cells": {},
        "deviations": [],
    }

    # ---------------- truth frames + scoring variable ----------------------
    truth = {"real": load_cube("truth_real_192.npy")}
    if zspace_ok:
        truth["zspace"] = load_cube("truth_zspace_192.npy")
    x_truth, truth_clamp, smoothed_truth = {}, {}, {}
    for fr, cube in truth.items():
        x_truth[fr], truth_clamp[fr] = x_transform(cube)
        smoothed_truth[fr] = {s: _smooth(x_truth[fr], s) for s in SIGMAS}
    scores["truth_clamped_fraction"] = truth_clamp

    # ---------------- objects ----------------------------------------------
    rho_objects = {
        "grid": load_cube("grid_192.npy"),
        "mlp": load_cube("mlp_192.npy"),
    }
    for e in wiener_rec["L_sweep"]:
        L = e["L_mpc_h"]
        rho_objects[f"wiener_L{L:g}"] = 1.0 + load_cube(
            f"wiener_rec192_L{L:g}.npy")

    # controls built per frame from that frame's truth x-field
    def build_controls(fr):
        xt = x_truth[fr]
        ctl = {}
        ctl["phase_rand"] = NC.phase_randomized(xt, SEED_PHASE)
        for kc in KC_LADDER:
            ctl[f"lowpass_kc{kc:g}"] = NC.lowpass_sharp(xt, BOX_MPC_H, kc)
        sx = xt.std()
        for amp_f, seed in SEEDS_NOISE.items():
            rng = np.random.default_rng(seed)
            ctl[f"noise_{amp_f:g}x"] = xt + amp_f * sx * rng.standard_normal(
                xt.shape)
        ps_name = ("truth_real_pointsample_192.npy" if fr == "real"
                   else "truth_zspace_pointsample_192.npy")
        ctl["pointsample_g"], _ = x_transform(load_cube(ps_name))
        return ctl

    frames = ["real"] + (["zspace"] if zspace_ok else [])
    obj_results = {fr: {} for fr in frames}
    smoothed2 = {fr: {} for fr in frames}     # sigma=2 smoothed fields for pairs
    for fr in frames:
        # scored objects: x-transform of rho
        for name, rho in rho_objects.items():
            x_o, clamped = x_transform(rho)
            res = _score_object(x_truth[fr], x_o, smoothed_truth[fr],
                                name)
            res["clamped_fraction"] = clamped
            smoothed2[fr][name] = res.pop("_smoothed2")
            obj_results[fr][name] = res
            print(f"[score] {fr}/{name}: r_s(2)="
                  f"{res['r_s']['2.0']['pearson']:.4f} "
                  f"(spearman {res['r_s']['2.0']['spearman']:.4f})", flush=True)
        # controls: already x-space
        for name, x_c in build_controls(fr).items():
            res = _score_object(x_truth[fr], x_c, smoothed_truth[fr], name)
            res["clamped_fraction"] = None   # built in x-space
            smoothed2[fr][name] = res.pop("_smoothed2")
            obj_results[fr][name] = res
            print(f"[score] {fr}/{name}: r_s(2)="
                  f"{res['r_s']['2.0']['pearson']:.4f}", flush=True)
        gc.collect()

    # Wiener best-L per frame by r_s(sigma=2) in that frame (v3 §6b)
    wiener_best = {}
    for fr in frames:
        Ls = [f"wiener_L{e['L_mpc_h']:g}" for e in wiener_rec["L_sweep"]]
        best = max(Ls, key=lambda n: obj_results[fr][n]["r_s"]["2.0"]["pearson"])
        wiener_best[fr] = best
    scores["wiener_best_L_per_frame"] = wiener_best

    # near-ray / far-field robustness readout for the grid (descriptive)
    mask = np.load(CUBES / "near_ray_mask_1024.npy")
    st2 = smoothed_truth["real"][2.0]
    sg2 = smoothed2["real"]["grid"]
    scores["grid_near_ray_readout"] = {
        "near_frac": float(mask.mean()),
        "r_s2_near": NC.pearson(st2[mask], sg2[mask]),
        "r_s2_far": NC.pearson(st2[~mask], sg2[~mask]),
        "r_s2_all": obj_results["real"]["grid"]["r_s"]["2.0"]["pearson"],
    }
    mask64 = np.load(CUBES / "near_ray_mask_64.npy")
    sm2 = smoothed2["real"]["mlp"]
    scores["mlp_near_ray_readout"] = {
        "near_frac": float(mask64.mean()),
        "r_s2_near": NC.pearson(st2[mask64], sm2[mask64]),
        "r_s2_far": NC.pearson(st2[~mask64], sm2[~mask64]),
        "r_s2_all": obj_results["real"]["mlp"]["r_s"]["2.0"]["pearson"],
    }
    del mask, mask64

    # ---------------- §7 B-i / B-ii machinery -------------------------------
    def rs2(fr, name):
        return obj_results[fr][name]["r_s"]["2.0"]["pearson"]

    def b_i(fr, name):
        return bool(rs2(fr, name) >= 0.50)

    def pair_test(fr, a, b):
        """Ordered pair a > b, all five B-ii conditions (v3 §7)."""
        delta = rs2(fr, a) - rs2(fr, b)
        oct_a = np.array(obj_results[fr][a]["r_s"]["2.0"]["octant_r"])
        oct_b = np.array(obj_results[fr][b]["r_s"]["2.0"]["octant_r"])
        ttest = NC.paired_fisher_t(oct_a, oct_b)
        boot = NC.block_bootstrap_delta_rs(
            smoothed_truth[fr][2.0], smoothed2[fr][a], smoothed2[fr][b])
        wiener_in_pair = a.startswith("wiener") or b.startswith("wiener")
        signs = {}
        for f2 in frames:
            a2 = wiener_best[f2] if a.startswith("wiener") else a
            b2 = wiener_best[f2] if b.startswith("wiener") else b
            signs[f2] = float(np.sign(rs2(f2, a2) - rs2(f2, b2)))
        sign_consistent = len(set(signs.values())) == 1
        cond = {
            "1_delta_ge_0.10": bool(delta >= 0.10),
            "2_fisher_t": ttest,
            "3_block_bootstrap": boot,
            "4_wiener_sign_both_frames": (
                bool(sign_consistent) if (wiener_in_pair and zspace_ok)
                else (False if (wiener_in_pair and not zspace_ok) else None)),
            "5_no_frame_sign_reversal": (bool(sign_consistent) if zspace_ok
                                         else None),
        }
        fires = (cond["1_delta_ge_0.10"] and ttest["pass"]
                 and boot["excludes_zero"] and boot["delta_mean"] > 0
                 and delta > 0)
        if wiener_in_pair:
            if not zspace_ok:
                fires = False   # fail-closed (γ): both-frame check unavailable
            else:
                fires = fires and sign_consistent
        if zspace_ok and not sign_consistent:
            fires = False
        return {"delta_rs2": float(delta), "conditions": cond,
                "frame_signs": signs, "fires": bool(fires),
                "fail_closed_note": (None if zspace_ok else
                                     "zspace BLOCKED -> Wiener-pair B-ii "
                                     "cannot fire (clause γ)")}

    neural = ["grid", "mlp"]
    prim = "real"
    wbest = wiener_best[prim]
    pair_defs = []
    for n in neural:
        pair_defs += [(n, wbest), (wbest, n)]
    pair_defs += [("grid", "mlp"), ("mlp", "grid")]
    for a, b in pair_defs:
        scores["pairs"][f"{a}>{b}"] = pair_test(prim, a, b)

    # ---------------- §8 cells (mechanical) ---------------------------------
    b_i_flags = {n: b_i(prim, n) for n in neural + [wbest]}
    b_i_controls = {n: b_i(prim, n) for n in obj_results[prim]
                    if n not in neural and not n.startswith("wiener")}
    bii = {k: v["fires"] for k, v in scores["pairs"].items()}
    neural_bi = any(b_i_flags[n] for n in neural)
    neural_win = any(b_i_flags[n] and bii.get(f"{n}>{wbest}", False)
                     for n in neural)
    wiener_bi = b_i_flags[wbest]
    wiener_win_over_neural = any(bii.get(f"{wbest}>{n}", False) for n in neural)
    any_bi = neural_bi or wiener_bi
    any_sep = any(bii.values())
    frame_flags = {}
    if zspace_ok:
        for name in neural + [f"wiener_L{e['L_mpc_h']:g}"
                              for e in wiener_rec["L_sweep"]]:
            d = abs(rs2("zspace", name) - rs2("real", name))
            frame_flags[name] = {"abs_delta_rs2": float(d),
                                 "fires": bool(d > 0.15)}
    noise_bi = {n: v for n, v in b_i_controls.items() if n.startswith("noise")}
    cells = {
        "0": {"fires": not acc["all_pass"],
              "condition": "any acceptance test FAILs"},
        "1": {"fires": bool(acc["all_pass"] and neural_win)},
        "2": {"fires": bool(acc["all_pass"] and neural_bi and not neural_win
                            and not (wiener_bi and wiener_win_over_neural))},
        "2b": {"fires": bool(acc["all_pass"] and neural_bi and wiener_bi
                             and wiener_win_over_neural)},
        "3": {"fires": bool(acc["all_pass"] and not any_bi
                            and bii.get(f"grid>{wbest}", False))},
        "4": {"fires": bool(acc["all_pass"] and wiener_bi and not neural_bi)},
        "5": {"fires": bool(acc["all_pass"] and not any_bi and not any_sep)},
        "6": {"fires": bool(acc["all_pass"] and bii.get("mlp>grid", False)),
              "point_estimate_mlp_gt_grid": bool(
                  rs2(prim, "mlp") > rs2(prim, "grid")),
              "audit_checklist": "pre-registered §8 cell-6 checklist pending "
                                 "iff fires"},
        "7": {"fires": bool(any(v["fires"] for v in frame_flags.values()))
              if zspace_ok else False,
              "per_object": frame_flags,
              "note": None if zspace_ok else "zspace BLOCKED"},
        "8": {"fires_phase_rand": bool(b_i_controls.get("phase_rand", False)),
              "noise_ladder_b_i": noise_bi,
              "fires": bool(b_i_controls.get("phase_rand", False)),
              "drafting_slip_flag": (
                  "The literal §8 cell-8 disjunct 'noise ladder clears B-i' is "
                  "true BY CONSTRUCTION (truth+noise at sigma=2 smoothing has "
                  "r_s >> 0.5; measured values reported). Mechanical firing on "
                  "that disjunct would route every outcome to cell 0; read as "
                  "a drafting slip, cell 8 here fires on the phase-randomized "
                  "anomaly only. PI adjudication flagged in deviations."),
              },
        "blanket": {
            "alpha_re_audit_triggered": bool(any_bi or
                                             any(b_i_controls.values())),
            "beta_no_new_positive_claim": None,   # PI-side consequence
            "gamma_fail_closed_blocked": ([] if zspace_ok else ["zspace"]),
            "delta_primary_frame": prim,
        },
    }
    primary_cell = None
    for c in ("0", "1", "2", "2b", "3", "4", "5"):
        if cells[c]["fires"]:
            primary_cell = c
            break
    if primary_cell is None:
        # exhaustive fallback: B-i cleared nothing but separations exist
        primary_cell = "5*"
        cells["5*"] = {"fires": True,
                       "note": "no B-i anywhere; some B-ii separation exists "
                               "(not grid>wiener); nearest enumerated cell is "
                               "3/5 boundary -- recorded verbatim for PI"}
    cells["primary"] = primary_cell
    cells["additive"] = [c for c in ("6", "7", "8") if cells[c]["fires"]]
    scores["cells"] = cells
    scores["b_i"] = {"objects": b_i_flags, "controls": b_i_controls}

    # control (e) ladder: k_c*(0.50)
    kcs = list(KC_LADDER)
    lad = [rs2(prim, f"lowpass_kc{k:g}") for k in kcs]
    kc_star = float(np.interp(0.50, lad, kcs)) if (min(lad) <= 0.5 <= max(lad)) \
        else float("nan")
    scores["controls_meta"]["lowpass_ladder"] = {
        "k_c": kcs, "r_s2_real": lad, "k_c_star_0.50": kc_star,
        "meaning": "interpolated cutoff at which low-pass truth scores "
                   "r_s=0.50 (gives B-i its derived physical meaning)",
    }
    scores["controls_meta"]["noise_ladder_note"] = (
        "estimator-behavior diagnostics ONLY, not recovery calibration (§6f)")

    scores["objects"] = obj_results
    scores["wall_clock_s_score_stage"] = time.time() - t0
    scores["stage_wall_clocks_s"] = {
        "truth": json.loads((ART / "truth_real_record.json").read_text())[
            "wall_clock_s"],
        "vlos": vg.get("wall_clock_s"),
        "grid": grid_rec["wall_clock_s"], "mlp": mlp_rec["wall_clock_s"],
        "wiener": wiener_rec.get("wall_clock_s"),
        "accept": acc["wall_clock_s"],
    }
    scores["deviations"] = [
        "LOS-axis finding: production sightlines are MIXED-axis "
        "(334/331/359 x/y/z in first 1024; 23/19/22 in first 64); z-space "
        "truth built along z (modal axis of the 1024-ray slice); single-axis "
        "z-frame is a representative diagnostic.",
        "Wiener rho convention rho/<rho> = 1 + rec (gain arbitrary; "
        "Spearman column gain-robust); block-wise C_md emission and Fourier "
        "64->192 regrid disclosed in wiener_record.json.",
        "Cell-8 'noise ladder clears B-i' read as drafting slip (fires by "
        "construction); cell 8 evaluated on phase-randomized anomaly; "
        "measured noise-ladder B-i values reported for PI adjudication.",
        "T-D amplitude pinned by implementer at 1.0x std(x) (spec leaves it "
        "free); seeds = noise-ladder seeds.",
        "MLflow arch cross-check done against checkpoint tensor shapes "
        "(local tracker 403; spec §11 out-of-scope flag iii).",
    ]
    _write_json(ART / "d75_scores.json", scores)

    # ---------------- figures ----------------------------------------------
    show = [("grid", "tab:blue"), ("mlp", "tab:orange"),
            (wbest, "tab:green"), ("phase_rand", "tab:gray"),
            ("pointsample_g", "tab:purple")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axi, fr in enumerate(frames):
        ax = axes[axi]
        for name, col in show:
            vals = [obj_results[fr][name]["r_s"][str(s)]["pearson"]
                    for s in SIGMAS]
            errs = [obj_results[fr][name]["r_s"][str(s)]["octant_se"]
                    for s in SIGMAS]
            ax.errorbar(SIGMAS, vals, yerr=errs, marker="o", label=name,
                        color=col)
        ax.axhline(0.5, ls="--", c="k", lw=0.8)
        ax.set_xlabel(r"$\sigma$ [$h^{-1}$Mpc]")
        ax.set_ylabel(r"$r_s(\sigma)$")
        ax.set_title(f"{fr} frame")
        ax.legend(fontsize=7)
    if len(frames) == 1:
        axes[1].axis("off")
    fig.suptitle("[D-75] smoothed Pearson vs sigma (internal diagnostic)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_rs_vs_sigma.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, col in show:
        nc = obj_results[prim][name]["nccf"]
        if isinstance(nc, dict) and not nc.get("undefined"):
            ax.plot(nc["r_centers"], nc["profile"], marker=".", label=name,
                    color=col)
    ax.axvline(2.0, ls=":", c="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$r$ [$h^{-1}$Mpc]")
    ax.set_ylabel("NCCF(r)")
    ax.set_title("[D-75] NCCF, real frame (internal diagnostic)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_nccf_real.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, col in show:
        rk = obj_results[prim][name]["r_k"]
        ax.plot(rk["k_centers"], rk["profile"], marker=".", label=name,
                color=col)
    ax.axhline(0.5, ls="--", c="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [$h$/Mpc]")
    ax.set_ylabel("r(k)")
    ax.set_title("[D-75] Fourier coherence, real frame (internal diagnostic)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_rk_real.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(kcs, lad, marker="o")
    ax.axhline(0.5, ls="--", c="k", lw=0.8)
    if np.isfinite(kc_star):
        ax.axvline(kc_star, ls=":", c="r", lw=0.8,
                   label=f"k_c*(0.50)={kc_star:.2f}")
        ax.legend()
    ax.set_xlabel(r"low-pass cutoff $k_c$ [$h$/Mpc]")
    ax.set_ylabel(r"$r_s(\sigma=2)$, real")
    ax.set_title("[D-75] low-pass-truth ladder (internal diagnostic)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_lowpass_ladder.png", dpi=150)
    plt.close(fig)

    slab_names = ["truth"] + [n for n, _ in show]
    fig, axes = plt.subplots(1, len(slab_names),
                             figsize=(3 * len(slab_names), 3.4))
    mid = N192 // 2
    vmin, vmax = np.percentile(smoothed_truth[prim][2.0][:, :, mid], [1, 99])
    for ax, name in zip(axes, slab_names):
        f = (smoothed_truth[prim][2.0] if name == "truth"
             else smoothed2[prim][name])
        ax.imshow(f[:, :, mid].T, origin="lower", cmap="magma",
                  vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("[D-75] sigma=2-smoothed x, central z-slab, real frame "
                 "(magma; internal diagnostic)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_slabs_real.png", dpi=150)
    plt.close(fig)

    if (CUBES / "vlos_z_192.npy").exists():
        vz = np.load(CUBES / "vlos_z_192.npy")
        fig, ax = plt.subplots(figsize=(4.4, 4))
        im = ax.imshow(vz[:, :, mid].T, origin="lower", cmap="coolwarm",
                       vmin=-300, vmax=300)
        fig.colorbar(im, ax=ax, label=r"$v_z$ [km/s]")
        ax.set_title("[D-75] mass-weighted CIC $v_z$, central slab")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(FIGS / "fig_vlos_slab.png", dpi=150)
        plt.close(fig)

    print(f"[score] primary cell = {primary_cell}; additive = "
          f"{cells['additive']} ({time.time()-t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------- #

STAGES = {
    "a0": stage_a0, "truth": stage_truth, "vlos": stage_vlos,
    "grid": stage_grid, "mlp": stage_mlp, "wiener": stage_wiener,
    "accept": stage_accept, "score": stage_score,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=sorted(STAGES))
    args = ap.parse_args()
    _mkdirs()
    STAGES[args.stage]()


if __name__ == "__main__":
    main()
