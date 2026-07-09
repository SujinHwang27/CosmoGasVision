"""Thin wrapper: selements-website the-neural-field Figure 1 (pf-miss).

RE-RUN P_F(k_||) MLP vs truth over the measured k_|| range at P1 (+ optional
P2-P4 band scalars). No logic here — see src.export.export_pf_miss.

    PYTHONPATH=. uv run python scripts/export/fig1-pf-miss.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.export import EXPORT_ROOT, export_pf_miss  # noqa: E402

_DEFAULT_OUT = f"{EXPORT_ROOT}/selements-website/the-neural-field"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default=_DEFAULT_OUT)
    p.add_argument(
        "--cells", nargs="+", default=["P1"],
        help="Physics cells; P1 mandatory (first is the fiducial curve cell). "
             "Pass 'P1 P2 P3 P4' to also emit the P2-P4 band scalars.",
    )
    p.add_argument("--n-rays-eval", type=int, default=1024)
    p.add_argument("--pf-tolerance", type=float, default=0.05)
    args = p.parse_args()
    res = export_pf_miss(
        out_dir=args.out_dir,
        cells=args.cells,
        n_rays_eval=args.n_rays_eval,
        pf_tolerance=args.pf_tolerance,
    )
    print("[fig1-pf-miss] wrote:", res["artifact"])
    print("[fig1-pf-miss] sidecar:", res["sidecar"])
    print("[fig1-pf-miss] band table:", res["band_table_artifact"])
    print(f"[fig1-pf-miss] curve rows: {res['n_curve_rows']}")
    print(
        f"[fig1-pf-miss] P1 reproduced |dP_F/P_F|={res['p1_band_residual']:.4f} "
        f"vs banked {res['banked_p1_residual']:.4f}"
    )
    for cell, sc in res["band_scalars"].items():
        print(f"[fig1-pf-miss]   {cell}: band |dP_F/P_F| = "
              f"{sc['abs_delta_PF_over_PF_in_band']:.4f} "
              f"(banked seed42 {sc['banked_seed42_P_F_residual']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
