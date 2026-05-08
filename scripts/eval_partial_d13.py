"""Partial [D-13] cosmological evaluator: P_F(k_||) and F-PDF KS-distance only.

Skips the chi cross-correlation and slice/anisotropy figures because those
require the SherwoodIGM_gal HDF5 particle snapshots (40 GB) processed via
``_chunked_cic_rho``, which currently segfaults on the host. The two gates
this driver produces are themselves [D-13] gates — they don't need the 3D
density grid, only tau predictions and tau truth on the held-out sightlines.

Authored 2026-05-08 to keep the paper writable without the third (xi) gate
or the T4 row of the degradation matrix.

Usage::

    PYTHONPATH=. uv run python scripts/eval_partial_d13.py \\
        --run-id <mlflow-run-id> \\
        --output-dir experiments/nerf/artifacts/reports/ \\
        [--n-rays-eval 1024]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the canonical figure helpers + model builder from stage2b_report.
from src.analysis.stage2b_report import (  # noqa: E402
    _build_model_from_run,
    _fig_flux_pdf,
    _fig_pf_compare,
    _load_mlflow_run,
    _render_tau_for_model,
)
from src.data.loader import SherwoodLoader  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-dir", default="experiments/nerf/artifacts/reports/")
    p.add_argument("--physics-id", type=int, default=1)
    p.add_argument("--redshift", type=float, default=0.3)
    p.add_argument("--n-rays-eval", type=int, default=1024)
    p.add_argument("--mlflow-uri", default=None,
                   help="Override MLFLOW_TRACKING_URI (e.g. a local file:// "
                        "store path on a cluster node).")
    p.add_argument("--ckpt-path", default=None,
                   help="Explicit checkpoint path (bypasses MLflow artifact "
                        "lookup; useful when the run's meta.yaml artifact_uri "
                        "points at a no-longer-existent scratch path).")
    args = p.parse_args()
    if args.mlflow_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri

    out = Path(args.output_dir) / f"{args.run_id}_partial"
    out.mkdir(parents=True, exist_ok=True)

    # Truth tau + vel_axis (no IGM_gal).
    sherwood = SherwoodLoader("Sherwood")
    sl_full = sherwood.load_sightlines(args.physics_id, args.redshift)
    tau_truth_full = np.asarray(sl_full["tau_h1"], dtype=np.float64)
    vel_axis = np.asarray(sl_full["vel_axis"], dtype=np.float64)
    box_kpc_h = float(sl_full["header"]["box_kpc_h"])
    n_rays_avail = tau_truth_full.shape[0]
    n_rays = min(args.n_rays_eval, n_rays_avail)
    rng = np.random.default_rng(seed=42)
    sel = rng.choice(n_rays_avail, size=n_rays, replace=False)
    sel.sort()
    tau_truth = tau_truth_full[sel]

    # Model from MLflow. Defensive against silent random-init: refuse to
    # proceed if the lookup failed or no checkpoint resolved. (Random-init
    # would yield meaningless [D-13] gates; this path produced bogus numerics
    # on 2026-05-08 before the fix.)
    run, ckpt = _load_mlflow_run(args.run_id)
    if run is None:
        sys.exit(f"FATAL: MLflow lookup for run_id={args.run_id!r} returned None. "
                 f"Check MLFLOW_TRACKING_URI ({os.environ.get('MLFLOW_TRACKING_URI', '<unset>')}) "
                 f"and that the run exists in that store.")
    # Explicit --ckpt-path overrides MLflow's artifact lookup. Necessary when
    # the file-store's meta.yaml artifact_uri points at a no-longer-existent
    # scratch path (training cleanup wiped the directory).
    if args.ckpt_path:
        if not os.path.exists(args.ckpt_path):
            sys.exit(f"FATAL: --ckpt-path {args.ckpt_path!r} does not exist.")
        ckpt = args.ckpt_path
        print(f"[eval-partial] using explicit --ckpt-path: {ckpt}")
    if ckpt is None:
        sys.exit(f"FATAL: run {args.run_id!r} has no checkpoint artifact (*.pt) "
                 f"and no --ckpt-path was supplied. Refusing to evaluate on random init.")
    model = _build_model_from_run(run, ckpt)

    # Sightline coords -> tau_pred.
    coords_world = sherwood.get_world_coordinates(sl_full)
    coords = torch.tensor(coords_world[sel] / box_kpc_h, dtype=torch.float32)
    vel_axis_t = torch.tensor(vel_axis, dtype=torch.float32)
    tau_pred = _render_tau_for_model(model, coords, vel_axis_t)

    # Two of three [D-13] gates.
    metrics = {}
    metrics.update(_fig_pf_compare(out, tau_pred, tau_truth, vel_axis))
    metrics.update(_fig_flux_pdf(out, tau_pred, tau_truth))

    print()
    print("=== [D-13] partial gates (P_F + flux-PDF only; xi skipped, needs IGM_gal) ===")
    for k, v in sorted(metrics.items()):
        try:
            print(f"  {k} = {float(v):.6g}")
        except (TypeError, ValueError):
            print(f"  {k} = {v}")
    print(f"\nFigures + numerics in {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
