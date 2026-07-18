"""Thin wrapper: selements-website the-grid-probe batch (ep10, all five exports).

fig1/fig2/fig3/fig4/spec all RE-READ from banked run artifacts (P_F sharpener,
the run's DVC-preserved metric store, verification battery, in-job xi profile,
Wiener L-sweep, healthy A7 control); nothing is recomputed from checkpoints.
No logic here — see the five src.export.export_d73grid_* functions.

    PYTHONPATH=. uv run python scripts/export/the-grid-probe.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.export import (  # noqa: E402
    D73_A7_CONTROL,
    D73_GRID_BATTERY,
    D73_GRID_METRIC_FILE,
    D73_GRID_RUN_META,
    D73_GRID_SHARPENER,
    D73_GRID_XI_INJOB,
    D73_WIENER_LSWEEP,
    EXPORT_ROOT,
    export_d73grid_flux_gates,
    export_d73grid_k2,
    export_d73grid_probe_config,
    export_d73grid_trainability,
    export_d73grid_xi,
)

_DEFAULT_OUT = f"{EXPORT_ROOT}/selements-website/the-grid-probe"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default=_DEFAULT_OUT)
    p.add_argument("--sharpener-json", default=D73_GRID_SHARPENER)
    p.add_argument("--battery-json", default=D73_GRID_BATTERY)
    p.add_argument("--xi-injob-json", default=D73_GRID_XI_INJOB)
    p.add_argument("--metric-file", default=D73_GRID_METRIC_FILE)
    p.add_argument("--run-meta-yaml", default=D73_GRID_RUN_META)
    p.add_argument("--a7-control-json", default=D73_A7_CONTROL)
    p.add_argument("--wiener-lsweep-json", default=D73_WIENER_LSWEEP)
    args = p.parse_args()

    r1 = export_d73grid_flux_gates(out_dir=args.out_dir, sharpener_json=args.sharpener_json)
    print(f"[fig1-flux-gates] wrote: {r1['artifact']} ({r1['n_rows']} rows)")
    r2 = export_d73grid_trainability(out_dir=args.out_dir, metric_file=args.metric_file,
                                     a7_control_json=args.a7_control_json)
    print(f"[fig2-trainability-trace] wrote: {r2['artifact']} ({r2['n_rows']} rows)")
    r3 = export_d73grid_k2(out_dir=args.out_dir, battery_json=args.battery_json)
    print(f"[fig3-truth-vs-grid] wrote: {r3['artifact']} ({r3['n_rows']} rows, margin ~{r3['margin']:.2f}x)")
    r4 = export_d73grid_xi(out_dir=args.out_dir, xi_injob_json=args.xi_injob_json,
                           battery_json=args.battery_json,
                           wiener_lsweep_json=args.wiener_lsweep_json)
    print(f"[fig4-xi-ceiling-relative] wrote: {r4['artifact']} ({r4['n_rows']} rows)")
    r5 = export_d73grid_probe_config(out_dir=args.out_dir, sharpener_json=args.sharpener_json,
                                     run_meta_yaml=args.run_meta_yaml)
    print(f"[spec-probe-config] wrote: {r5['artifact']} ({r5['n_rows']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
