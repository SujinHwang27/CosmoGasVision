"""Thin wrapper: selements-website all-four-at-once batch (ep07, all five exports).

fig1/fig2a/fig2b/fig3 RE-READ from the healthy on-disk [D-46] gates JSON
(consistency-asserted against the decision record); spec BANKED. No logic here —
see the five src.export.export_d46_* functions.

    PYTHONPATH=. uv run python scripts/export/all-four-at-once.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.export import (  # noqa: E402
    D46_GATES_JSON,
    EXPORT_ROOT,
    export_d46_d4_signature,
    export_d46_embedding_distances,
    export_d46_gate_table,
    export_d46_loss_trace,
    export_d46_run_config,
)

_DEFAULT_OUT = f"{EXPORT_ROOT}/selements-website/all-four-at-once"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default=_DEFAULT_OUT)
    p.add_argument("--gates-json", default=D46_GATES_JSON)
    args = p.parse_args()

    for fn, name in (
        (export_d46_gate_table, "fig1-gate-table"),
        (export_d46_d4_signature, "fig2a-d4-signature"),
        (export_d46_embedding_distances, "fig2b-embedding-distances"),
        (export_d46_loss_trace, "fig3-loss-trace"),
    ):
        r = fn(out_dir=args.out_dir, gates_json=args.gates_json)
        print(f"[{name}] wrote: {r['artifact']} ({r['n_rows']} rows)")

    r = export_d46_run_config(out_dir=args.out_dir)
    print(f"[spec-run-config] wrote: {r['artifact']} ({r['n_rows']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
