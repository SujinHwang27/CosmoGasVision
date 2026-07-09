"""Thin wrapper: selements-website the-neural-field Figure 2a (mean-flux table).

ASSEMBLE FROM BANKED — seed=42 anchor + [D-44] 5-seed bootstrap CI. No recompute,
no logic here — see src.export.export_mean_flux_table.

    PYTHONPATH=. uv run python scripts/export/fig2a-mean-flux-table.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.export import (  # noqa: E402
    D44_BOOTSTRAP_JSON,
    EXPORT_ROOT,
    export_mean_flux_table,
)

_DEFAULT_OUT = f"{EXPORT_ROOT}/selements-website/the-neural-field"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default=_DEFAULT_OUT)
    p.add_argument("--bootstrap-json", default=D44_BOOTSTRAP_JSON)
    args = p.parse_args()
    res = export_mean_flux_table(
        out_dir=args.out_dir,
        bootstrap_json=args.bootstrap_json,
    )
    print("[fig2a-mean-flux-table] wrote:", res["artifact"])
    print("[fig2a-mean-flux-table] sidecar:", res["sidecar"])
    print(f"[fig2a-mean-flux-table] rows: {res['n_rows']}")
    print(f"[fig2a-mean-flux-table] bootstrap PASS vs [D-13] band [0.974,0.984]: "
          f"{res['n_bootstrap_pass_of_4']}/4")
    print(f"[fig2a-mean-flux-table] cells w/ CI entirely below gate band: "
          f"{res['cells_ci_below_gate_band']}")
    print(f"[fig2a-mean-flux-table] cells w/ CI q84 below point anchor 0.979: "
          f"{res['cells_ci_below_point_anchor']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
