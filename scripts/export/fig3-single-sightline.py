"""Thin wrapper: selements-website the-neural-field Figure 3 (single sightline).

ONE representative P1 sightline (sel[0] of the sorted seed=42 selection):
v (km/s), F_mlp, F_truth. No logic here — see src.export.export_single_sightline.

    PYTHONPATH=. uv run python scripts/export/fig3-single-sightline.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.export import EXPORT_ROOT, export_single_sightline  # noqa: E402

_DEFAULT_OUT = f"{EXPORT_ROOT}/selements-website/the-neural-field"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default=_DEFAULT_OUT)
    p.add_argument("--cell", default="P1")
    p.add_argument("--n-rays-eval", type=int, default=1024)
    args = p.parse_args()
    res = export_single_sightline(
        out_dir=args.out_dir,
        cell=args.cell,
        n_rays_eval=args.n_rays_eval,
    )
    print("[fig3-single-sightline] wrote:", res["artifact"])
    print("[fig3-single-sightline] sidecar:", res["sidecar"])
    print(f"[fig3-single-sightline] bins: {res['n_bins']}  "
          f"representative ray global index: "
          f"{res['representative_ray_global_index']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
