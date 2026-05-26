"""D70 Stage 1a file-store replay -> unified MLflow tracker.

Per D70 Rev 5.1 PI memo absorption (F2-α resolution): the Stage 1a (1b)
Juno sbatch (`scripts/submit_juno_stage1a_1b.sh`) sets

    MLFLOW_TRACKING_URI=file://${RUN_DIR}/mlflow

per seed. Post-PCV copy-out produces n=10 isolated file-stores under the
host-side results root, one per RUN_TAG sub-directory::

    <input_dir>/<RUN_TAG_seed_0>/mlflow/<experiment_id>/<run_id>/...
    <input_dir>/<RUN_TAG_seed_1>/mlflow/<experiment_id>/<run_id>/...
    ...
    <input_dir>/<RUN_TAG_seed_9>/mlflow/<experiment_id>/<run_id>/...

The Wilcoxon harness at `scripts/d70_wilcoxon_gate.py` reads via
`mlflow.search_runs` against ONE tracking URI; it cannot see n=10 isolated
stores at once. This script replays every source run into the unified
host tracker (default ``http://127.0.0.1:5000``) so the harness can
operate on a single search.

Mirrors the structural precedent of
`scripts/sagemaker_stage2b_import_mlflow.py` (S3 tarball -> local replay)
but is host-side and accepts an already-extracted directory tree of
n>=1 file-stores rather than a single tarball.

Idempotency:
    Two source runs are considered "the same" if they share
    ``(tag juno_batch, tag seed)``. A second invocation on the same input
    skips runs already imported (logged as ``[skip]``); the trailer
    surfaces FOUND / IMPORTED / SKIPPED for downstream sbatch-trailer
    grep equivalence.

R20 twin-gate (binding):
    - Empty input dir -> loud ``AssertionError`` (never silent return).
    - Per-run import failure -> raise with source path + cause
      (no swallow-and-continue).

Usage::

    uv run python scripts/d70_stage1a_import_mlflow.py \\
        --input-dir /path/to/stage1a_results \\
        [--tracking-uri http://127.0.0.1:5000] \\
        [--experiment CosmoGasVision/NeRF] \\
        [--dry-run] \\
        [--juno-batch stage1a-1b-skiprich]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# MLflow's set_terminated() prints a "🏃 View run ..." line that crashes on
# non-UTF-8 stdout encodings (e.g. cp949 on Windows-Korean locales).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and getattr(_stream, "encoding", "").lower() != "utf-8":
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover
            pass

try:
    import mlflow  # type: ignore
    from mlflow.tracking import MlflowClient  # type: ignore
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
DEFAULT_EXPERIMENT = "CosmoGasVision/NeRF"
DEFAULT_JUNO_BATCH = "stage1a-1b-skiprich"

TAG_JUNO_BATCH = "juno_batch"
TAG_SEED = "seed"

# Trailer keys (mandatory; grep-able from sbatch / harness chain).
TRAILER_FOUND = "D70_IMPORT_REPLAY_RUNS_FOUND"
TRAILER_IMPORTED = "D70_IMPORT_REPLAY_RUNS_IMPORTED"
TRAILER_SKIPPED = "D70_IMPORT_REPLAY_RUNS_SKIPPED"
TRAILER_STATUS = "D70_IMPORT_REPLAY_STATUS"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay n>=1 Stage 1a file-store MLflow runs into a unified tracker."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing per-seed RUN_TAG subdirs, each holding mlflow/ file-store.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help=f"Destination MLflow tracker. Default: {DEFAULT_TRACKING_URI}.",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"Destination MLflow experiment. Default: {DEFAULT_EXPERIMENT}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan + report what would be imported; do NOT write to the destination.",
    )
    parser.add_argument(
        "--juno-batch",
        default=None,
        help=(
            "Override the juno_batch tag (otherwise read from each source run). "
            f"Idempotency key is (juno_batch, seed). Default tag value: {DEFAULT_JUNO_BATCH}."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------

def _discover_file_stores(input_dir: Path) -> List[Path]:
    """Return every mlflow/ file-store under ``input_dir``.

    Per the Juno PCV layout, expected path is
    ``<input_dir>/<RUN_TAG>/mlflow/``. We accept anywhere under input_dir to
    tolerate operator variation (e.g. flat dump, nested archive extract).
    A directory qualifies as a file-store if it contains at least one
    experiment subdir with ``meta.yaml`` inside (the file-store sentinel).
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise AssertionError(
            f"No MLflow stores found at {input_dir}; refusing silent no-op "
            "(input-dir does not exist or is not a directory)."
        )
    stores: List[Path] = []
    for cand in input_dir.rglob("mlflow"):
        if not cand.is_dir():
            continue
        # Sentinel: at least one immediate child with experiment meta.yaml.
        for child in cand.iterdir():
            if child.is_dir() and (child / "meta.yaml").is_file():
                stores.append(cand)
                break
    if not stores:
        raise AssertionError(
            f"No MLflow stores found at {input_dir}; refusing silent no-op "
            "(rglob found no mlflow/ dir containing an experiment with meta.yaml)."
        )
    stores.sort()
    return stores


# ---------------------------------------------------------------------------
# Destination helpers
# ---------------------------------------------------------------------------

def _ensure_experiment(client, name: str) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is None:
        exp_id = client.create_experiment(name)
        print(f"[mlflow] created destination experiment {name} (id={exp_id})")
        return exp_id
    return exp.experiment_id


def _collect_existing_keys(client, experiment_id: str) -> Set[Tuple[str, str]]:
    """Return set of (juno_batch, seed) tag pairs already present in dst."""
    keys: Set[Tuple[str, str]] = set()
    page_token = None
    while True:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=1000,
            page_token=page_token,
        )
        for r in runs:
            jb = r.data.tags.get(TAG_JUNO_BATCH)
            sd = r.data.tags.get(TAG_SEED)
            if jb is not None and sd is not None:
                keys.add((jb, sd))
        page_token = getattr(runs, "token", None)
        if not page_token:
            break
    return keys


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def _replay_run(
    src_client,
    dst_client,
    src_run_id: str,
    dst_experiment_id: str,
    src_store_root: Path,
    juno_batch_override: Optional[str] = None,
) -> str:
    """Recreate one source run in the destination tracker. Transactional:
    on any failure, the partially-created dst run is set to FAILED status
    and the exception is re-raised so the caller fails loud."""
    src_run = src_client.get_run(src_run_id)
    src_data = src_run.data
    src_info = src_run.info

    run_name = src_data.tags.get("mlflow.runName", src_run_id)
    src_tags = {k: v for k, v in src_data.tags.items() if not k.startswith("mlflow.")}
    if juno_batch_override is not None:
        src_tags[TAG_JUNO_BATCH] = juno_batch_override

    dst_run = dst_client.create_run(
        experiment_id=dst_experiment_id,
        start_time=src_info.start_time,
        run_name=run_name,
        tags={
            **src_tags,
            "imported_from_juno": "true",
            "source_run_id": src_run_id,
        },
    )
    dst_run_id = dst_run.info.run_id

    try:
        # Params.
        for k, v in src_data.params.items():
            dst_client.log_param(dst_run_id, k, v)

        # Metrics: full per-step history (Stage 1a runs are short — 500 steps —
        # so full replay is cheap and preserves time-series view in the UI).
        for metric_key in src_data.metrics:
            history = src_client.get_metric_history(src_run_id, metric_key)
            for m in history:
                dst_client.log_metric(
                    dst_run_id, m.key, m.value, timestamp=m.timestamp, step=m.step,
                )

        # Artifacts: walk <store>/<exp_id>/<run_id>/artifacts/.
        src_artifact_path = (
            src_store_root / src_info.experiment_id / src_run_id / "artifacts"
        )
        if src_artifact_path.exists():
            for fp in src_artifact_path.rglob("*"):
                if fp.is_file():
                    rel = fp.relative_to(src_artifact_path).parent.as_posix()
                    rel = None if rel in ("", ".") else rel
                    dst_client.log_artifact(dst_run_id, str(fp), artifact_path=rel)

        dst_client.set_terminated(
            dst_run_id,
            status=src_info.status,
            end_time=src_info.end_time,
        )
    except Exception:
        try:
            dst_client.set_terminated(dst_run_id, status="FAILED")
        except Exception:  # pragma: no cover
            pass
        raise

    return dst_run_id


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _emit_trailer(found: int, imported: int, skipped: int, status: str) -> None:
    print(f"{TRAILER_FOUND}={found}", flush=True)
    print(f"{TRAILER_IMPORTED}={imported}", flush=True)
    print(f"{TRAILER_SKIPPED}={skipped}", flush=True)
    print(f"{TRAILER_STATUS}={status}", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if mlflow is None or MlflowClient is None:
        sys.exit("ERROR: mlflow is not installed. `uv add mlflow` first.")

    input_dir = Path(args.input_dir).resolve()
    stores = _discover_file_stores(input_dir)
    print(f"[scan] {len(stores)} file-store(s) under {input_dir}")
    for s in stores:
        print(f"  store: {s}")

    # Enumerate source runs without writing anything.
    pending: List[Tuple[Path, str, str, Dict[str, str]]] = []
    # tuples: (store_root, src_run_id, idempotency_key (jb,seed) joined, src_tags-summary)
    found_total = 0
    for store_root in stores:
        src_uri = store_root.resolve().as_uri()
        src_client = MlflowClient(tracking_uri=src_uri)
        src_experiments = src_client.search_experiments()
        for exp in src_experiments:
            src_runs = src_client.search_runs(
                experiment_ids=[exp.experiment_id], max_results=1000,
            )
            for r in src_runs:
                found_total += 1
                jb = (
                    args.juno_batch
                    if args.juno_batch is not None
                    else r.data.tags.get(TAG_JUNO_BATCH, "")
                )
                sd = r.data.tags.get(TAG_SEED, "")
                pending.append((store_root, r.info.run_id, f"{jb}|{sd}",
                                {"juno_batch": jb, "seed": sd,
                                 "run_name": r.data.tags.get("mlflow.runName", "")}))

    if args.dry_run:
        print(f"[dry-run] would consider {found_total} run(s) across {len(stores)} store(s):")
        for store_root, rid, key, summary in pending:
            print(f"  [dry-run] store={store_root.name} run_id={rid} "
                  f"juno_batch={summary['juno_batch']!r} seed={summary['seed']!r} "
                  f"run_name={summary['run_name']!r}")
        _emit_trailer(found=found_total, imported=0, skipped=0, status="OK")
        return 0

    dst_client = MlflowClient(tracking_uri=args.tracking_uri)
    dst_experiment_id = _ensure_experiment(dst_client, args.experiment)
    existing_keys = _collect_existing_keys(dst_client, dst_experiment_id)
    print(f"[dst] {len(existing_keys)} existing (juno_batch,seed) keys in dst experiment")

    imported = 0
    skipped = 0
    for store_root, src_run_id, key, summary in pending:
        jb, sd = summary["juno_batch"], summary["seed"]
        if jb and sd and (jb, sd) in existing_keys:
            print(f"[skip] store={store_root.name} run={src_run_id} key=({jb},{sd}) already imported")
            skipped += 1
            continue
        src_uri = store_root.resolve().as_uri()
        src_client = MlflowClient(tracking_uri=src_uri)
        try:
            dst_run_id = _replay_run(
                src_client, dst_client, src_run_id, dst_experiment_id,
                src_store_root=store_root,
                juno_batch_override=args.juno_batch,
            )
        except Exception as exc:
            _emit_trailer(found=found_total, imported=imported, skipped=skipped, status="FAIL")
            raise RuntimeError(
                f"FATAL: failed to import src run {src_run_id} from store "
                f"{store_root}: {exc}"
            ) from exc
        print(f"[ok] store={store_root.name} src={src_run_id} -> dst={dst_run_id} "
              f"(juno_batch={jb!r}, seed={sd!r})")
        imported += 1
        if jb and sd:
            existing_keys.add((jb, sd))

    _emit_trailer(found=found_total, imported=imported, skipped=skipped, status="OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
