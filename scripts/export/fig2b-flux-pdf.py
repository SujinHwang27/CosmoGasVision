"""Thin wrapper: selements-website the-neural-field Figure 2b (flux-pdf).

RE-RUN p(F) MLP vs truth histograms at P1 (F in [0.05,0.95]) + banked KS scalars
P1-P4. No logic here — see src.export.export_flux_pdf.

    PYTHONPATH=. uv run python scripts/export/fig2b-flux-pdf.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.export import (  # noqa: E402
    D44_BOOTSTRAP_JSON,
    EXPORT_ROOT,
    export_flux_pdf,
)

_DEFAULT_OUT = f"{EXPORT_ROOT}/selements-website/the-neural-field"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default=_DEFAULT_OUT)
    p.add_argument("--cell", default="P1")
    p.add_argument("--n-rays-eval", type=int, default=1024)
    p.add_argument("--bootstrap-json", default=D44_BOOTSTRAP_JSON)
    args = p.parse_args()
    res = export_flux_pdf(
        out_dir=args.out_dir,
        cell=args.cell,
        n_rays_eval=args.n_rays_eval,
        bootstrap_json=args.bootstrap_json,
    )
    print("[fig2b-flux-pdf] wrote:", res["artifact"])
    print("[fig2b-flux-pdf] sidecar:", res["sidecar"])
    print(f"[fig2b-flux-pdf] hist bins: {res['n_hist_bins']}")
    print(f"[fig2b-flux-pdf] KS (exported cell {args.cell}): "
          f"{res['ks_exported_cell']:.4f}")
    for cell, ks in res["ks_banked"].items():
        print(f"[fig2b-flux-pdf]   {cell}: KS_seed42={ks['KS_seed42']:.4f} "
              f"KS_bootstrap_mean={ks['KS_bootstrap_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
