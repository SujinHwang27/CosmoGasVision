"""Thin wrapper: selements-website the-direct-attack batch (ep04, all four exports).

ASSEMBLE FROM BANKED — [D-40] verdict-of-record scalars + the Addendum-1 per-bin
diagnostic JSONs. No recompute, no logic here — see the four
src.export.export_d40_* functions.

    PYTHONPATH=. uv run python scripts/export/the-direct-attack.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.export import (  # noqa: E402
    D40_BASELINE_PER_BIN_JSON,
    D40_PF_PER_BIN_JSON,
    EXPORT_ROOT,
    export_d40_pf_per_bin,
    export_d40_run_config,
    export_d40_shape_amplitude_summary,
    export_d40_verdict_table,
)

_DEFAULT_OUT = f"{EXPORT_ROOT}/selements-website/the-direct-attack"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default=_DEFAULT_OUT)
    p.add_argument("--per-bin-json", default=D40_PF_PER_BIN_JSON)
    p.add_argument("--baseline-json", default=D40_BASELINE_PER_BIN_JSON)
    args = p.parse_args()

    r1 = export_d40_verdict_table(out_dir=args.out_dir)
    print("[fig1-verdict-table] wrote:", r1["artifact"])
    print(f"[fig1-verdict-table] rows: {r1['n_rows']}; "
          f"P_F worsening vs baseline: +{r1['pf_worsening_pct']:.1f}%")

    r2 = export_d40_pf_per_bin(out_dir=args.out_dir, per_bin_json=args.per_bin_json)
    print("[fig2-pf-per-bin] wrote:", r2["artifact"])
    print(f"[fig2-pf-per-bin] rows: {r2['n_rows']}; "
          f"in-band log-Pearson re-derived: {r2['pearson_log_rederived']:.4f}")

    r3 = export_d40_shape_amplitude_summary(
        out_dir=args.out_dir,
        per_bin_json=args.per_bin_json,
        baseline_json=args.baseline_json,
    )
    print("[fig2-shape-amplitude-summary] wrote:", r3["artifact"])
    print(f"[fig2-shape-amplitude-summary] rows: {r3['n_rows']}")

    r4 = export_d40_run_config(out_dir=args.out_dir)
    print("[spec-run-config] wrote:", r4["artifact"])
    print(f"[spec-run-config] rows: {r4['n_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
