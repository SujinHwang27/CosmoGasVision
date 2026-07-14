"""Thin wrapper: selements-website the-physics-constraint batch (ep05, all four exports).

fig1 RE-READ from the local MLflow store (the smoke ran on this host); fig2/fig3/spec
BANKED from the retrospective diagnostic JSON + decision record. No logic here — see
the four src.export.export_d41_* functions.

    PYTHONPATH=. uv run python scripts/export/the-physics-constraint.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.export import (  # noqa: E402
    D41_COLLAPSE_JSON,
    D41_SMOKE_RUN_ID,
    EXPORT_ROOT,
    MLFLOW_DB,
    export_d41_collapse_signature,
    export_d41_run_config,
    export_d41_smoke_trace,
    export_d41_verdict_table,
)

_DEFAULT_OUT = f"{EXPORT_ROOT}/selements-website/the-physics-constraint"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default=_DEFAULT_OUT)
    p.add_argument("--db-path", default=MLFLOW_DB)
    p.add_argument("--run-id", default=D41_SMOKE_RUN_ID)
    p.add_argument("--collapse-json", default=D41_COLLAPSE_JSON)
    args = p.parse_args()

    r1 = export_d41_smoke_trace(out_dir=args.out_dir, db_path=args.db_path, run_id=args.run_id)
    print("[fig1-smoke-trace] wrote:", r1["artifact"])
    print(f"[fig1-smoke-trace] rows: {r1['n_rows']}; regularizer descent factor: {r1['descent_factor']:.1f}x")

    r2 = export_d41_collapse_signature(out_dir=args.out_dir, collapse_json=args.collapse_json)
    print("[fig2-collapse-signature] wrote:", r2["artifact"])
    print(f"[fig2-collapse-signature] rows: {r2['n_rows']}")

    r3 = export_d41_verdict_table(out_dir=args.out_dir)
    print("[fig3-verdict-table] wrote:", r3["artifact"])
    print(f"[fig3-verdict-table] rows: {r3['n_rows']}")

    r4 = export_d41_run_config(out_dir=args.out_dir)
    print("[spec-run-config] wrote:", r4["artifact"])
    print(f"[spec-run-config] rows: {r4['n_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
