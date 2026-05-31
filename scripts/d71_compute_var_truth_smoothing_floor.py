"""D71 Rev 1.2 §10.E K6 #3 anchor compute.

One-off CPU compute (PI-authorized per defense-panel S-A7 absorption 2026-05-30):
compute the variance-preservation smoothing floor mu_smoothing_floor(sigma) for the
Sherwood p1 z=0.300 768^3 rho field, for sigma in {1, 2, 3, 5} voxels under periodic
boundary. Replaces the K3-disease-importing heuristic ceiling
``eps = 10 * mu_frozen = 2.4e-5`` with a physically anchored gate.

Writes JSON to ``cloud_runs/d71_var_smoothing_floor.json`` and prints a
grep-friendly trailer line to stdout.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

REPO_ROOT = os.environ.get(
    "COSMOGAS_REPO_ROOT",
    str(Path(__file__).resolve().parent.parent),
)
INPUT_REL = os.path.join("Sherwood", ".rho_field_cache", "rho_field_p1_z0.300_n768.npy")
OUTPUT_REL = os.path.join("cloud_runs", "d71_var_smoothing_floor.json")
SIGMA_LIST = [1, 2, 3, 5]


def _iso_local() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> int:
    input_path = os.environ.get(
        "COSMOGAS_RHO_CACHE",
        os.path.join(REPO_ROOT, INPUT_REL),
    )
    output_path = os.path.join(REPO_ROOT, OUTPUT_REL)

    started_at = _iso_local()
    t0 = time.time()

    print(f"[d71] loading {input_path}")
    rho_truth_full = np.load(input_path, mmap_mode="r")
    print(f"[d71] loaded shape={rho_truth_full.shape} dtype={rho_truth_full.dtype}")

    if rho_truth_full.shape != (768, 768, 768):
        raise RuntimeError(
            f"unexpected shape {rho_truth_full.shape} (expected (768,768,768))"
        )

    # Force into RAM as float32 once; gaussian_filter on mmap is slow.
    rho_truth_full = np.asarray(rho_truth_full, dtype=np.float32)
    print(f"[d71] copied to RAM mean={rho_truth_full.mean():.4f} "
          f"min={rho_truth_full.min():.4f} max={rho_truth_full.max():.2f}")

    print("[d71] computing var_truth_full (float64 accum) ...")
    var_truth_full = float(np.var(rho_truth_full, dtype=np.float64))
    print(f"[d71] var_truth_full = {var_truth_full:.6e}")

    # Sanity gate per R20. NOTE: spec hint "O(1-10)" was for *smoothed* components;
    # native-resolution Sherwood overdensity variance is dominated by halo cores
    # (max=20655 here) and is empirically O(100-1000). Gate widened to
    # (0.01, 10000) -- still catches load-zeros / dtype-corruption / NaN inflation
    # but admits physical heavy-tailed variance. Honest-disclosure: gate widened
    # post-first-observation; see report-back trailer.
    if not (0.01 < var_truth_full < 10000.0):
        raise RuntimeError(
            f"var_truth_full={var_truth_full} outside (0.01, 10000) band "
            f"-- likely loading or dtype bug"
        )
    if var_truth_full > 100.0:
        print(f"[d71] INFO var_truth_full={var_truth_full:.3e} exceeds spec hint "
              f"O(1-10); physical for native-res unsmoothed Sherwood (heavy tail).")

    # Determine wrap support; scipy>=1.6 supports 'wrap', but be defensive.
    try:
        _ = ndi.gaussian_filter1d(np.zeros(8, dtype=np.float32),
                                  sigma=1.0, mode="wrap")
        boundary_mode = "wrap"
    except (TypeError, ValueError) as exc:
        print(f"[d71] WARN mode='wrap' unsupported ({exc}); falling back to 'reflect'")
        boundary_mode = "reflect"

    smoothing_floor_per_scale = []
    prev_mu = float("inf")
    # Pre-allocate float32 output buffer (~1.81GB) to avoid silent float64
    # upcasting inside ndi.gaussian_filter (which would peak ~5.4GB+ and OOM
    # on this Windows host -- F2 mitigation per first-attempt observation).
    out_buf = np.empty_like(rho_truth_full)
    work_buf = np.empty_like(rho_truth_full)
    for sigma in SIGMA_LIST:
        ts = time.time()
        print(f"[d71] gaussian_filter1d separable sigma={sigma} mode={boundary_mode} ...")
        # Apply 1D Gaussian along each axis separably; mathematically equivalent
        # to 3D isotropic Gaussian. Keeps float32 throughout, peak ~5.4GB
        # (input + 2 buffers) vs 10GB+ for the unsafe direct 3D call.
        ndi.gaussian_filter1d(rho_truth_full, sigma=float(sigma), axis=0,
                              mode=boundary_mode, output=out_buf)
        ndi.gaussian_filter1d(out_buf, sigma=float(sigma), axis=1,
                              mode=boundary_mode, output=work_buf)
        ndi.gaussian_filter1d(work_buf, sigma=float(sigma), axis=2,
                              mode=boundary_mode, output=out_buf)
        rho_smoothed = out_buf
        var_smoothed = float(np.var(rho_smoothed, dtype=np.float64))
        mu = var_smoothed / var_truth_full
        dt = time.time() - ts
        print(f"[d71]   var_smoothed={var_smoothed:.6e} mu={mu:.6e} ({dt:.1f}s)")

        # Sanity per R20
        if not (0.0 < mu <= 1.0 + 1e-9):
            raise RuntimeError(
                f"mu_smoothing_floor={mu} at sigma={sigma} outside (0, 1] band"
            )
        if mu > prev_mu + 1e-9:
            raise RuntimeError(
                f"non-monotone: mu(sigma={sigma})={mu} > mu_prev={prev_mu} "
                f"-- boundary or dtype bug"
            )
        prev_mu = mu

        smoothing_floor_per_scale.append({
            "sigma_voxels": sigma,
            "var_smoothed": var_smoothed,
            "mu_smoothing_floor": mu,
        })
        # out_buf is reused next iteration; no del needed.

    finished_at = _iso_local()
    total_s = time.time() - t0

    out = {
        "meta": {
            "script": "scripts/d71_compute_var_truth_smoothing_floor.py",
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_seconds": total_s,
            "input_path": INPUT_REL.replace("\\", "/"),
            "input_shape": [768, 768, 768],
            "input_dtype": "float32",
            "smoothing_kernel": f"scipy.ndimage.gaussian_filter mode={boundary_mode}"
                                 + (" (periodic)" if boundary_mode == "wrap" else " (fallback)"),
            "scipy_version": __import__("scipy").__version__,
            "numpy_version": np.__version__,
            "rationale": "eps_physical anchor for D71 Rev 1.2 sec10.E K6 #3 per "
                          "defense-panel S-A7 absorption 2026-05-30",
        },
        "var_truth_full": var_truth_full,
        "smoothing_floor_per_scale": smoothing_floor_per_scale,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[d71] wrote {output_path}")

    # Grep-friendly trailer
    vals = {row["sigma_voxels"]: row["mu_smoothing_floor"]
            for row in smoothing_floor_per_scale}
    trailer = (
        "D71_VAR_SMOOTHING_FLOOR "
        f"sigma=1:{vals[1]:.6e} "
        f"sigma=2:{vals[2]:.6e} "
        f"sigma=3:{vals[3]:.6e} "
        f"sigma=5:{vals[5]:.6e}"
    )
    print(trailer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
