"""Replay a Juno Stage 2b batch (multiple cells, each with its own MLflow
file-store) into the local tracker.

The Juno launcher (``scripts/submit_juno_stage2b.sh``) writes per-cell stores at
``${RUN_DIR}/mlflow/`` and the babysitter tarballs all four cell dirs into one
``batch2-<timestamp>.tar.gz``. This wrapper reuses the SageMaker importer's
``_replay_run`` helper (it's source-agnostic) and walks each cell's file-store
in turn, tagging every imported run with ``compute=juno``.

Usage::

    uv run python scripts/juno_stage2b_import_mlflow.py \\
        cloud_runs/batch2-extracted \\
        --batch-tag batch2 \\
        [--mlflow_uri http://127.0.0.1:5000] \\
        [--latest-only]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Reuse the SageMaker importer's helpers verbatim — same replay semantics.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sagemaker_stage2b_import_mlflow import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    DEFAULT_LOCAL_URI,
    _ensure_experiment,
    _format_run_url,
    _replay_run,
)

try:
    from mlflow.tracking import MlflowClient  # type: ignore
except ImportError:
    sys.exit("ERROR: mlflow is not installed. `uv add mlflow` first.")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("extracted_dir", help="Directory containing per-cell subdirs each with mlflow/.")
    p.add_argument("--mlflow_uri", default=DEFAULT_LOCAL_URI)
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    p.add_argument("--batch-tag", default="batch2",
                   help="Value for the 'juno_batch' tag on every imported run.")
    p.add_argument("--latest-only", action="store_true",
                   help="Skip per-step metric history; log final value only.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    root = Path(args.extracted_dir).resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: not a directory: {root}")

    cell_stores = sorted(p.parent for p in root.glob("*/mlflow") if p.is_dir())
    if not cell_stores:
        sys.exit(f"ERROR: no <cell>/mlflow stores under {root}")

    dst_client = MlflowClient(tracking_uri=args.mlflow_uri)
    dst_experiment_id = _ensure_experiment(dst_client, args.experiment)

    total_runs = 0
    for cell_dir in cell_stores:
        mlflow_dir = cell_dir / "mlflow"
        src_uri = mlflow_dir.as_uri()
        print(f"[juno-import] cell={cell_dir.name} src={src_uri}")
        src_client = MlflowClient(tracking_uri=src_uri)

        for src_exp in src_client.search_experiments():
            for src_run in src_client.search_runs(
                experiment_ids=[src_exp.experiment_id], max_results=1000
            ):
                dst_run_id = _replay_run(
                    src_client, dst_client, src_run.info.run_id,
                    dst_experiment_id, src_store_root=mlflow_dir,
                    latest_only=args.latest_only,
                )
                # Add Juno-specific tags post-replay.
                dst_client.set_tag(dst_run_id, "compute", "juno")
                dst_client.set_tag(dst_run_id, "juno_batch", args.batch_tag)
                dst_client.set_tag(dst_run_id, "juno_cell_dir", cell_dir.name)
                total_runs += 1
                print(f"  src={src_run.info.run_id} -> dst={dst_run_id}")
                print(f"    {_format_run_url(args.mlflow_uri, dst_experiment_id, dst_run_id)}")

    print(f"\n[juno-import] imported {total_runs} run(s) into {args.mlflow_uri} -> {args.experiment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
