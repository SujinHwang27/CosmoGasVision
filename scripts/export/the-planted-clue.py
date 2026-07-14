"""Thin wrapper: selements-website the-planted-clue batch (ep06, all four exports).

fig1/fig2/spec BANKED from the [D-42] Addendum authoritative readout (the on-disk
gates JSON is header-only); fig3 RE-READ from the local MLflow store (the scalar
smoke trace survived). No logic here — see the four src.export.export_d42_* functions.

    PYTHONPATH=. uv run python scripts/export/the-planted-clue.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.export import (  # noqa: E402
    D42_SMOKE_RUN_ID,
    EXPORT_ROOT,
    MLFLOW_DB,
    export_d42_gate_table,
    export_d42_head_asymmetry,
    export_d42_run_config,
    export_d42_smoke_trace,
)

_DEFAULT_OUT = f"{EXPORT_ROOT}/selements-website/the-planted-clue"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default=_DEFAULT_OUT)
    p.add_argument("--db-path", default=MLFLOW_DB)
    p.add_argument("--run-id", default=D42_SMOKE_RUN_ID)
    args = p.parse_args()

    r1 = export_d42_gate_table(out_dir=args.out_dir)
    print("[fig1-gate-table] wrote:", r1["artifact"])
    print(f"[fig1-gate-table] gates: {r1['n_pass']} PASS / {r1['n_fail']} FAIL")

    r2 = export_d42_head_asymmetry(out_dir=args.out_dir)
    print("[fig2-head-asymmetry] wrote:", r2["artifact"])
    print(f"[fig2-head-asymmetry] rows: {r2['n_rows']}")

    r3 = export_d42_smoke_trace(out_dir=args.out_dir, db_path=args.db_path, run_id=args.run_id)
    print("[fig3-smoke-trace] wrote:", r3["artifact"])
    print(f"[fig3-smoke-trace] rows: {r3['n_rows']}")

    r4 = export_d42_run_config(out_dir=args.out_dir)
    print("[spec-run-config] wrote:", r4["artifact"])
    print(f"[spec-run-config] rows: {r4['n_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
