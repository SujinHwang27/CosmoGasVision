"""Replay MLflow runs from a finished SageMaker training job into the local
tracker.

Stage 2b cloud trainers write MLflow under ``file:///opt/ml/model/mlflow`` in
the container (see ``scripts/sagemaker_stage2b_launch.py``); SageMaker tarballs
``/opt/ml/model/`` into ``model.tar.gz`` and uploads it to
``s3://cosmo-gas-vision-storage/stage2b-output/<JOBNAME>/output/model.tar.gz``
at job exit. This script:

  1. Downloads that tarball.
  2. Extracts the embedded ``mlflow/`` file-store.
  3. For every source run, creates a fresh run in the local tracker (default
     ``http://127.0.0.1:5000``) under experiment ``CosmoGasVision/NeRF`` and
     replays params, tags, metrics (per-step), and artifacts.
  4. Prints the new local run URLs.

The file-store ``run_id`` is opaque, so the source -> dest run_id mapping is
not preserved (the new run gets a fresh dest id). All science-relevant
metadata is preserved.

Usage::

    uv run python scripts/sagemaker_stage2b_import_mlflow.py <JOBNAME>
    uv run python scripts/sagemaker_stage2b_import_mlflow.py <JOBNAME> \\
        --mlflow_uri http://127.0.0.1:5000 \\
        --bucket cosmo-gas-vision-storage \\
        --prefix stage2b-output \\
        --experiment CosmoGasVision/NeRF
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# boto3 + mlflow are runtime deps; soft-import for clearer error messages.
try:
    import boto3  # type: ignore
except ImportError:  # pragma: no cover
    boto3 = None  # noqa: N816

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


DEFAULT_BUCKET = "cosmo-gas-vision-storage"
DEFAULT_PREFIX = "stage2b-output"
DEFAULT_LOCAL_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
DEFAULT_EXPERIMENT = "CosmoGasVision/NeRF"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job_name", help="SageMaker training job name.")
    parser.add_argument(
        "--mlflow_uri",
        default=DEFAULT_LOCAL_URI,
        help=f"Destination tracker. Default: {DEFAULT_LOCAL_URI}.",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"S3 bucket holding the job output. Default: {DEFAULT_BUCKET}.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"S3 key prefix under bucket. Default: {DEFAULT_PREFIX}.",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"Destination MLflow experiment. Default: {DEFAULT_EXPERIMENT}.",
    )
    parser.add_argument(
        "--source_tarball",
        default=None,
        help="Optional local path to model.tar.gz. Skips the S3 download "
             "(useful for offline replays / unit tests).",
    )
    parser.add_argument(
        "--keep_tempdir",
        action="store_true",
        help="Don't delete the extracted tarball directory (debugging).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# S3 download + extract
# ---------------------------------------------------------------------------

def _download_tarball(bucket: str, prefix: str, job_name: str, dest: Path) -> Path:
    if boto3 is None:
        sys.exit("ERROR: boto3 is not installed. `uv add boto3` first.")
    key = f"{prefix}/{job_name}/output/model.tar.gz"
    print(f"[s3] downloading s3://{bucket}/{key} -> {dest}")
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(dest))
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"[s3] downloaded {size_mb:.1f} MB")
    return dest


def _extract_tarball(tarball: Path, dest_dir: Path) -> Path:
    """Extract the tarball; return the path to the embedded mlflow/ directory."""
    print(f"[tar] extracting {tarball} -> {dest_dir}")
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(dest_dir)
    # The mlflow store may be at <dest>/mlflow/ or anywhere under <dest>.
    # Look for the file-store sentinel: a directory containing experiment
    # subdirectories which contain meta.yaml. Be permissive about layout.
    candidates: List[Path] = []
    for d in dest_dir.rglob("mlflow"):
        if d.is_dir():
            candidates.append(d)
    if not candidates:
        sys.exit(f"ERROR: no mlflow/ directory found in tarball under {dest_dir}")
    if len(candidates) > 1:
        # Prefer the shallowest match.
        candidates.sort(key=lambda p: len(p.parts))
    mlflow_dir = candidates[0]
    print(f"[tar] mlflow store at {mlflow_dir}")
    return mlflow_dir


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def _ensure_experiment(client: "MlflowClient", name: str) -> str:
    """Get or create destination experiment; returns its id."""
    exp = client.get_experiment_by_name(name)
    if exp is None:
        exp_id = client.create_experiment(name)
        print(f"[mlflow] created destination experiment {name} (id={exp_id})")
        return exp_id
    return exp.experiment_id


def _replay_run(
    src_client: "MlflowClient",
    dst_client: "MlflowClient",
    src_run_id: str,
    dst_experiment_id: str,
    src_store_root: Path,
) -> str:
    """Recreate one source run in the destination tracker.

    Returns the destination run id.
    """
    src_run = src_client.get_run(src_run_id)
    src_data = src_run.data
    src_info = src_run.info

    # Create the destination run with the source run name as a tag so the
    # opaque file-store id stays auditable.
    run_name = src_data.tags.get("mlflow.runName", src_run_id)
    dst_run = dst_client.create_run(
        experiment_id=dst_experiment_id,
        start_time=src_info.start_time,
        run_name=run_name,
        tags={
            **{k: v for k, v in src_data.tags.items()
               if not k.startswith("mlflow.")},
            # Preserve provenance of the original SageMaker run.
            "imported_from_sagemaker": "true",
            "source_run_id": src_run_id,
        },
    )
    dst_run_id = dst_run.info.run_id

    # Params (one-shot).
    for k, v in src_data.params.items():
        dst_client.log_param(dst_run_id, k, v)

    # Metrics: replay the full per-step history so charts in the UI look right.
    for metric_key in src_data.metrics:
        history = src_client.get_metric_history(src_run_id, metric_key)
        for m in history:
            dst_client.log_metric(
                dst_run_id, m.key, m.value, timestamp=m.timestamp, step=m.step,
            )

    # Artifacts: the artifact_uri stored in meta.yaml points to the path used
    # when the source store was first created (e.g. /opt/ml/model/mlflow/...
    # inside the SageMaker container), which no longer exists locally. Resolve
    # the actual on-disk artifact directory by walking the extracted store
    # layout: <store_root>/<experiment_id>/<run_id>/artifacts/.
    src_artifact_path = (
        src_store_root / src_info.experiment_id / src_run_id / "artifacts"
    )
    if src_artifact_path.exists():
        for fp in src_artifact_path.rglob("*"):
            if fp.is_file():
                rel = fp.relative_to(src_artifact_path).parent.as_posix()
                rel = None if rel in ("", ".") else rel
                dst_client.log_artifact(dst_run_id, str(fp), artifact_path=rel)

    # Mirror terminal status.
    dst_client.set_terminated(
        dst_run_id,
        status=src_info.status,
        end_time=src_info.end_time,
    )

    return dst_run_id


def _format_run_url(tracking_uri: str, experiment_id: str, run_id: str) -> str:
    if tracking_uri.startswith(("http://", "https://")):
        base = tracking_uri.rstrip("/")
        return f"{base}/#/experiments/{experiment_id}/runs/{run_id}"
    return f"{tracking_uri}  experiment={experiment_id}  run={run_id}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if mlflow is None or MlflowClient is None:
        sys.exit("ERROR: mlflow is not installed. `uv add mlflow` first.")

    workdir = Path(tempfile.mkdtemp(prefix="sagemaker-mlflow-import-"))
    try:
        # 1. Get the tarball.
        if args.source_tarball:
            tarball = Path(args.source_tarball)
            if not tarball.exists():
                sys.exit(f"ERROR: --source_tarball path does not exist: {tarball}")
            print(f"[s3] using local tarball {tarball}")
        else:
            tarball = _download_tarball(args.bucket, args.prefix, args.job_name,
                                        workdir / "model.tar.gz")

        # 2. Extract and locate the mlflow file-store.
        extract_dir = workdir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        mlflow_dir = _extract_tarball(tarball, extract_dir)
        # Path.as_uri() handles Windows drive letters correctly (file:///C:/...)
        # which the bare f-string form does not.
        src_uri = mlflow_dir.resolve().as_uri()

        # 3. Walk all source runs in all source experiments.
        src_client = MlflowClient(tracking_uri=src_uri)
        dst_client = MlflowClient(tracking_uri=args.mlflow_uri)
        dst_experiment_id = _ensure_experiment(dst_client, args.experiment)

        all_src_experiments = src_client.search_experiments()
        if not all_src_experiments:
            print("[mlflow] WARNING: source store has no experiments — nothing to import.")
            return 0

        replayed: Dict[str, str] = {}
        for src_exp in all_src_experiments:
            src_runs = src_client.search_runs(
                experiment_ids=[src_exp.experiment_id], max_results=1000,
            )
            print(f"[mlflow] source experiment {src_exp.name!r}: {len(src_runs)} runs")
            for src_run in src_runs:
                dst_run_id = _replay_run(
                    src_client, dst_client, src_run.info.run_id, dst_experiment_id,
                    src_store_root=mlflow_dir,
                )
                replayed[src_run.info.run_id] = dst_run_id
                print(f"  replayed src={src_run.info.run_id} -> dst={dst_run_id}")
                print(f"    {_format_run_url(args.mlflow_uri, dst_experiment_id, dst_run_id)}")

        if not replayed:
            print("[mlflow] WARNING: no runs replayed (source experiments were empty).")
        else:
            print(f"\n[mlflow] imported {len(replayed)} run(s) into "
                  f"{args.mlflow_uri} -> experiment {args.experiment}")
        return 0
    finally:
        if args.keep_tempdir:
            print(f"[cleanup] keeping {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
