"""Thin wrapper: selements-website removing-the-clamp batch (ep09, all four exports).

fig1/fig2 RE-READ from the on-disk head-probe artifacts (per-cell finals +
median trajectories); fig3/spec BANKED. No logic here — see the four
src.export.export_d73a1_* functions.

    PYTHONPATH=. uv run python scripts/export/removing-the-clamp.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.export import (  # noqa: E402
    D73_A1_DIR,
    D73_A1_SUMMARY,
    EXPORT_ROOT,
    export_d73a1_gate_spec,
    export_d73a1_median_trajectories,
    export_d73a1_per_cell_verdicts,
    export_d73a1_probe_config,
)

_DEFAULT_OUT = f"{EXPORT_ROOT}/selements-website/removing-the-clamp"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default=_DEFAULT_OUT)
    p.add_argument("--a1-dir", default=D73_A1_DIR)
    p.add_argument("--summary-json", default=D73_A1_SUMMARY)
    args = p.parse_args()

    r1 = export_d73a1_per_cell_verdicts(out_dir=args.out_dir, a1_dir=args.a1_dir)
    print(f"[fig1-per-cell-verdicts] wrote: {r1['artifact']} ({r1['n_rows']} rows)")
    r2 = export_d73a1_median_trajectories(out_dir=args.out_dir, summary_json=args.summary_json)
    print(f"[fig2-median-trajectories] wrote: {r2['artifact']} ({r2['n_rows']} rows)")
    r3 = export_d73a1_gate_spec(out_dir=args.out_dir)
    print(f"[fig3-gate-spec] wrote: {r3['artifact']} ({r3['n_rows']} rows)")
    r4 = export_d73a1_probe_config(out_dir=args.out_dir)
    print(f"[spec-probe-config] wrote: {r4['artifact']} ({r4['n_rows']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
