"""Thin wrapper: selements-website changing-the-target batch (ep08, all four exports).

fig1/fig2/spec BANKED from the decision record; fig3 RE-READ from the three
on-disk verdict artifacts. No logic here — see the src.export.export_d60_*/
export_d63_*/export_d69_* functions.

    PYTHONPATH=. uv run python scripts/export/changing-the-target.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.export import (  # noqa: E402
    D69_FGPA_VERDICT_JSON,
    D69_LRPROBE_SUMMARY_JSON,
    D71_SKIPRICH_VERDICT_JSON,
    EXPORT_ROOT,
    export_d60_campaign_config,
    export_d60_direct_target_arc,
    export_d63_collapsed_basin_cluster,
    export_d69_closing_probes,
)

_DEFAULT_OUT = f"{EXPORT_ROOT}/selements-website/changing-the-target"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default=_DEFAULT_OUT)
    p.add_argument("--fgpa-json", default=D69_FGPA_VERDICT_JSON)
    p.add_argument("--lrprobe-json", default=D69_LRPROBE_SUMMARY_JSON)
    p.add_argument("--skiprich-json", default=D71_SKIPRICH_VERDICT_JSON)
    args = p.parse_args()

    r1 = export_d60_direct_target_arc(out_dir=args.out_dir)
    print(f"[fig1-direct-target-arc] wrote: {r1['artifact']} ({r1['n_rows']} rows)")
    r2 = export_d63_collapsed_basin_cluster(out_dir=args.out_dir)
    print(f"[fig2-collapsed-basin-cluster] wrote: {r2['artifact']} ({r2['n_rows']} rows; band {r2['band_decades']:.2f} decades)")
    r3 = export_d69_closing_probes(out_dir=args.out_dir, fgpa_json=args.fgpa_json,
                                   lrprobe_json=args.lrprobe_json, skiprich_json=args.skiprich_json)
    print(f"[fig3-closing-probes] wrote: {r3['artifact']} ({r3['n_rows']} rows)")
    r4 = export_d60_campaign_config(out_dir=args.out_dir)
    print(f"[spec-campaign-config] wrote: {r4['artifact']} ({r4['n_rows']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
