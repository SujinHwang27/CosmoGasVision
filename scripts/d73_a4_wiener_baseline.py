"""[D-73] A4 — Stark+2015 / CLAMATO-style Wiener density baseline (read-only).

Classical (non-neural) reconstruction datum for the close-out paper (panel
KILLER-4: the paper needs at least one classical comparison number). Runs a
Wiener-filter density reconstruction from the production P1 z=0.3 Lya
sightlines on the SAME geometry the production NeRF used (n_rays=1024,
[D-13] fiducial), and scores it on the [D-13] metric set to the extent
feasible.

Scoring (per the deliverable, [D-13] metric defs):
  - xi_{rho_hat,rho}(r=2 h^-1 Mpc): BOTH the 3D FFT-shell Pearson gate
    (src/analysis/cross_corr.compute_xi_pearson, the literal [D-13] gate) AND
    the production 1D-along-ray r_rho^log surrogate (per [D-58]), so the
    Wiener number is directly comparable to the NeRF on whichever estimator
    the paper cites.
  - density-power ratio over the inertial range: P_rec(k)/P_truth(k) from the
    isotropic 3D density power (src/analysis/density_power.compute_Pdelta_iso).
  - |Delta P_F / P_F|: the Wiener tracer is a DENSITY estimate, not a flux
    field; forward-modelling it back to flux requires the full FGPA+RSD+Voigt
    integrator on a 3D x_HI/T/v field the Wiener method does not produce.
    NOT computed; documented as such.

This is a baseline datum, NOT a gated PASS/FAIL.

Read-only / analysis only. Writes JSON to
experiments/nerf/artifacts/wiener_baseline/.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import SherwoodLoader  # noqa: E402
from src.analysis.cross_corr import compute_xi_pearson  # noqa: E402
from src.analysis.density_power import compute_Pdelta_iso  # noqa: E402
from src.analysis.wiener_baseline import (  # noqa: E402
    WienerConfig, wiener_reconstruct,
)

PHYSICS_ID = 1
REDSHIFT = 0.3
N_RAYS = 1024
N_GRID = 64
RHO_CUBE = REPO_ROOT / "Sherwood" / ".rho_field_cache" / "rho_field_p1_z0.300_n64.npy"
OUT_DIR = REPO_ROOT / "experiments" / "nerf" / "artifacts" / "wiener_baseline"

# [D-13] inertial range edges (s/km for P_F; here we report a density-power
# inertial band in h/Mpc chosen to overlap the reconstructable scales).
XI_GATE_R_MPC_H = 2.0


def _pearson_per_row(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean(axis=1, keepdims=True)
    num = (a_c * b_c).sum(axis=1)
    den = np.sqrt((a_c ** 2).sum(axis=1) * (b_c ** 2).sum(axis=1))
    out = np.full(a.shape[0], np.nan)
    valid = den > 0
    out[valid] = num[valid] / den[valid]
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--L-mpc-h", type=float, default=2.0,
                   help="Gaussian correlation length (Mpc/h), Stark+2015 ~few Mpc.")
    p.add_argument("--noise-rel", type=float, default=0.05)
    p.add_argument("--pixel-stride", type=int, default=64,
                   help="Subsample pixels along each ray for the Npix x Npix solve.")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RHO_CUBE.exists():
        print(f"FATAL: ground-truth rho cube missing: {RHO_CUBE}")
        return 1

    loader = SherwoodLoader(str(REPO_ROOT / "Sherwood"))
    sl = loader.load_sightlines(PHYSICS_ID, REDSHIFT)
    box_kpc_h = float(sl["header"]["box_kpc_h"])
    box_mpc_h = box_kpc_h / 1000.0

    coords_world = loader.get_world_coordinates(sl)  # (N, nbins, 3) kpc/h
    coords = coords_world[:N_RAYS]                    # production first-N slice
    tau = np.asarray(sl["tau_h1"][:N_RAYS], dtype=np.float64)
    F = np.exp(-np.minimum(tau, 10.0))
    mean_F = float(F.mean())
    delta_F = F / mean_F - 1.0                        # flux overdensity

    # Density tracer (Gunn-Peterson sign): high rho -> high tau -> low F ->
    # negative delta_F, so the density tracer is -delta_F.
    tracer = -delta_F

    # Subsample pixels along each ray to keep the Npix x Npix solve tractable.
    stride = args.pixel_stride
    nbins = coords.shape[1]
    pix_idx = np.arange(0, nbins, stride)
    pix_xyz = (coords[:, pix_idx, :].reshape(-1, 3) / 1000.0)   # Mpc/h
    pix_data = tracer[:, pix_idx].reshape(-1)
    Npix = pix_xyz.shape[0]
    print(f"[A4] Npix = {Npix} (= {N_RAYS} rays x {len(pix_idx)} pixels/ray, "
          f"stride {stride})", flush=True)

    # Map-voxel centers: the 64^3 grid cell centers in Mpc/h.
    cell = box_mpc_h / N_GRID
    ax = (np.arange(N_GRID) + 0.5) * cell
    VX, VY, VZ = np.meshgrid(ax, ax, ax, indexing="ij")
    vox_xyz = np.stack([VX.ravel(), VY.ravel(), VZ.ravel()], axis=1)

    cfg = WienerConfig(
        L_perp_mpc_h=args.L_mpc_h, L_para_mpc_h=args.L_mpc_h,
        noise_rel=args.noise_rel, pixel_stride=stride,
    )
    print(f"[A4] solving Wiener system ({Npix}x{Npix}) ...", flush=True)
    rec_flat = wiener_reconstruct(pix_xyz, pix_data, vox_xyz, box_mpc_h, cfg)
    rec_cube = rec_flat.reshape(N_GRID, N_GRID, N_GRID)   # Wiener density tracer

    # Ground-truth rho/<rho> cube.
    rho_truth = np.load(RHO_CUBE).astype(np.float64)       # (64,64,64), mean=1
    assert rho_truth.shape == (N_GRID, N_GRID, N_GRID)

    # The Wiener tracer is delta-like (mean ~0); convert to a rho/<rho>-like
    # field for the estimators that expect overdensity fields. compute_xi_pearson
    # and compute_Pdelta_iso both mean-subtract internally, so the absolute
    # offset/gain does not affect Pearson xi (scale-free). For the power RATIO
    # we fit a single global linear gain b minimizing ||b*tracer - delta_truth||
    # so the recovered amplitude is the best-case linear-gain power.
    delta_truth = rho_truth - rho_truth.mean()
    rec_delta = rec_cube - rec_cube.mean()
    denom = float((rec_delta * rec_delta).sum())
    gain = float((rec_delta * delta_truth).sum() / denom) if denom > 0 else 0.0
    rec_scaled = gain * rec_delta + rho_truth.mean()       # rho/<rho>-like

    # ----- [D-13] gate (literal): 3D FFT-shell Pearson xi(r=2 Mpc/h) -----
    # Bin edges bracketing r=2 Mpc/h.
    r_bins = np.array([0.0, 1.0, 1.5, 2.5, 3.5, 5.0, 8.0, 15.0, 30.0])
    r_centers, xi = compute_xi_pearson(rec_scaled, rho_truth, box_kpc_h, r_bins)
    # interpolate xi at r=2 from the shell centers
    finite = np.isfinite(xi)
    xi_at_2 = float(np.interp(XI_GATE_R_MPC_H, r_centers[finite], xi[finite]))
    # exact-shell value (the bin whose center is closest to 2)
    j = int(np.argmin(np.abs(r_centers - XI_GATE_R_MPC_H)))
    xi_shell = float(xi[j]); xi_shell_r = float(r_centers[j])

    # ----- production 1D-along-ray r_rho^log surrogate (per [D-58]) -----
    # Sample the Wiener cube at the sightline pixel positions via nearest-grid
    # (the Wiener field is grid-defined; sample at each ray's bin centers).
    rho_gt_rays = np.asarray(sl["density"][:N_RAYS], dtype=np.float64)
    # voxel index of each ray pixel
    cidx = np.minimum((coords / 1000.0 / cell).astype(int), N_GRID - 1)
    rec_rho_rays = rec_scaled[cidx[..., 0], cidx[..., 1], cidx[..., 2]]
    floor = 1e-6
    r_log = _pearson_per_row(
        np.log10(np.maximum(rec_rho_rays, floor)),
        np.log10(np.maximum(rho_gt_rays, floor)),
    )
    r_lin = _pearson_per_row(rec_rho_rays, rho_gt_rays)
    f_log = np.isfinite(r_log)
    f_lin = np.isfinite(r_lin)

    # ----- density-power ratio over an inertial range -----
    k_c, P_rec = compute_Pdelta_iso(rec_scaled, box_kpc_h)
    _, P_truth = compute_Pdelta_iso(rho_truth, box_kpc_h)
    # inertial band: scales the 1024-ray map can resolve; ~[0.1, 1.0] h/Mpc.
    band = (k_c >= 0.1) & (k_c <= 1.0) & np.isfinite(P_rec) & np.isfinite(P_truth)
    ratio = P_rec[band] / P_truth[band]
    power_ratio_mean = float(np.nanmean(ratio)) if band.any() else float("nan")

    out = {
        "deliverable": "[D-73] A4 Stark+2015/CLAMATO Wiener density baseline",
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/d73_a4_wiener_baseline.py",
            "wiener_module": "src/analysis/wiener_baseline.py",
            "inputs": {
                "sightlines": "Sherwood/Physics1_nofeedback/los2048_n16384_z0.300.dat "
                              "(via SherwoodLoader.load_sightlines(1, 0.3))",
                "tau": "Sherwood/Physics1_nofeedback/tauH1_2048_n16384_z0.300.dat",
                "rho_truth_cube": str(RHO_CUBE.relative_to(REPO_ROOT)),
            },
            "geometry": {
                "physics_id": PHYSICS_ID, "redshift": REDSHIFT,
                "n_rays": N_RAYS, "ray_selection": "first-N (production slice)",
                "n_grid": N_GRID, "box_mpc_h": box_mpc_h,
                "pixels_per_ray_used": int(len(pix_idx)),
                "pixel_stride": stride, "Npix_total": Npix,
            },
            "wiener_config": {
                "L_corr_mpc_h": args.L_mpc_h,
                "noise_rel_sigma2": args.noise_rel,
                "kernel": "isotropic 3D Gaussian exp(-r^2/2L^2), periodic min-image",
            },
            "metric_code_paths": {
                "xi_3d_gate": "src/analysis/cross_corr.py:compute_xi_pearson (line 18)",
                "xi_1d_surrogate": "scripts/proxy_xi_1d_sample.py:_pearson_per_row "
                                   "(median over rays); [D-58] production metric",
                "density_power": "src/analysis/density_power.py:compute_Pdelta_iso (line 120)",
            },
            "command": "PYTHONPATH=. ~/.venvs/cosmogasvision/bin/python -u "
                       f"scripts/d73_a4_wiener_baseline.py --L-mpc-h {args.L_mpc_h} "
                       f"--noise-rel {args.noise_rel} --pixel-stride {stride}",
        },
        "idealizations": [
            "Noiseless mock: noise term is a 1e-3*signal regularizer, not a "
            "survey continuum-fit+photon noise model -> best-case classical recon.",
            "Gaussian-correlation prior (prescribed L), not a measured nonlinear P(k).",
            "No redshift-space-distortion deconvolution (maps redshift-space flux "
            "directly, CLAMATO convention).",
            "Single global linear gain fit to truth for the power-ratio amplitude "
            "(Pearson xi metrics are gain-invariant and need no such fit).",
            "rho-truth cube is the 64^3 CIC field; reconstruction is on the same grid.",
        ],
        "results": {
            "mean_F": mean_F,
            "global_linear_gain_b": gain,
            "xi_3d_pearson_gate": {
                "estimator": "src/analysis/cross_corr.compute_xi_pearson (3D FFT shell)",
                "xi_at_r2_interp": xi_at_2,
                "xi_nearest_shell": {"r_mpc_h": xi_shell_r, "xi": xi_shell},
                "r_centers": [float(v) for v in r_centers],
                "xi_profile": [float(v) for v in xi],
                "d13_gate_bar": 0.6,
                "passes_d13_bar": bool(xi_at_2 > 0.6),
            },
            "xi_1d_surrogate_r_rho": {
                "estimator": "r_rho^log per [D-58] (median over rays)",
                "r_rho_log_median": float(np.median(r_log[f_log])),
                "r_rho_log_q16": float(np.quantile(r_log[f_log], 0.16)),
                "r_rho_log_q84": float(np.quantile(r_log[f_log], 0.84)),
                "r_rho_lin_median": float(np.median(r_lin[f_lin])),
                "n_valid": int(f_log.sum()),
            },
            "density_power_ratio": {
                "band_h_per_mpc": [0.1, 1.0],
                "P_rec_over_P_truth_mean": power_ratio_mean,
                "k_centers": [float(v) for v in k_c],
                "P_rec": [float(v) for v in P_rec],
                "P_truth": [float(v) for v in P_truth],
                "note": "Computed AFTER the global linear-gain fit; absolute "
                        "amplitude is gain-calibrated, so this ratio measures "
                        "SHAPE fidelity of the recovered power, not blind amplitude.",
            },
            "delta_pf_over_pf": {
                "computed": False,
                "reason": "The Wiener method outputs a 3D DENSITY estimate, not a "
                          "flux field. Forward-modelling it to P_F needs the full "
                          "FGPA+RSD+Voigt integrator on 3D x_HI/T/v_pec fields that "
                          "the linear Wiener filter does not produce. Out of scope "
                          "for a classical density baseline; documented per [D-37].",
            },
        },
    }
    out_path = OUT_DIR / "a4_wiener_baseline.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[A4] xi_3d(r=2 Mpc/h) interp = {xi_at_2:+.4f} "
          f"(nearest shell r={xi_shell_r:.2f}: {xi_shell:+.4f})", flush=True)
    print(f"[A4] r_rho^log median = {np.median(r_log[f_log]):+.4f}", flush=True)
    print(f"[A4] density power ratio (shape, gain-cal) = {power_ratio_mean:.3f}", flush=True)
    print(f"[A4] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
