"""SageMaker training-job launcher for the Stage 2b ablation matrix.

Submits one managed-spot training job on ``ml.g5.xlarge`` (NVIDIA A10G, 24 GB)
per invocation. The full Stage 2b matrix is the 4x4 cartesian product of
``--n_rays {16384, 1024, 256, 64}`` x ``--physics {1, 2, 3, 4}`` (see [D-13]);
this launcher submits **one** point — the orchestrator (or a wrapper shell
script) is responsible for sweeping the matrix.

Pinned by [D-14]:
  - instance_type        = ml.g5.xlarge
  - EnableManagedSpot    = True
  - MaxRuntimeInSeconds  = 18000   (5 hr)
  - MaxWaitTimeInSeconds = 36000
  - checkpoint S3 sync   = s3://cosmo-gas-vision-storage/stage2b-checkpoints/<run_id>/
  - run_name format      = Stage2b-Ablation-P<n>-N<m>-S<s>
  - entry point          = experiments/nerf/pipeline.py (parametrized in C1)
  - MLFLOW_TRACKING_URI is forwarded to the container as an env var.

This file does NOT launch the job by default. Importing it produces no side
effect; running ``python scripts/sagemaker_stage2b_launch.py --launch ...``
is required to actually call ``boto3.client('sagemaker').create_training_job``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from typing import Any, Dict

# boto3 is a soft dependency — the script must import cleanly so reviewers can
# inspect the payload even on a machine without AWS SDK installed.
try:
    import boto3  # type: ignore
except ImportError:  # pragma: no cover
    boto3 = None  # noqa: N816


# ---------------------------------------------------------------------------
# Pinned constants (do not parameterize without a LEDGER decision update).
# ---------------------------------------------------------------------------

S3_BUCKET = "cosmo-gas-vision-storage"
CHECKPOINT_PREFIX = "stage2b-checkpoints"
INSTANCE_TYPE = "ml.g5.xlarge"            # A10G 24 GB, [D-14]
MAX_RUNTIME_S = 18000                      # 5 hr, [D-14]
MAX_WAIT_S = 36000                         # 10 hr, [D-14]
CHECKPOINT_INTERVAL_STEPS = 10000          # checkpoint sync cadence, [D-14]
DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# Sherwood data mirror in our bucket. Mirrors only the Physics<N>/los*.dat and
# tauH1*.dat files actually consumed by SherwoodLoader; the upstream
# s3://sherwood-raw/ prefix is too large to ingest in full.
# SageMaker auto-syncs this to /opt/ml/input/data/sherwood/ on the worker.
SHERWOOD_S3_PREFIX = f"s3://{S3_BUCKET}/sherwood/"
SHERWOOD_CONTAINER_PATH = "/opt/ml/input/data/sherwood"

# In-container MLflow store. file:// scheme avoids the 127.0.0.1:5000 retry
# storm — the local tracker isn't reachable from a SageMaker worker. Output
# under /opt/ml/model/ is auto-tarballed to S3 at job exit; the post-job
# importer (scripts/sagemaker_stage2b_import_mlflow.py) replays runs into
# the local tracker.
SAGEMAKER_MLFLOW_URI = "file:///opt/ml/model/mlflow"

# These two are deployment-environment dependent. They MUST be set as env vars
# (or via the matching CLI flags) before launching.
#   SAGEMAKER_ROLE_ARN    : IAM role the training job assumes.
#   SAGEMAKER_IMAGE_URI   : ECR URI of the container with project deps installed.
#   SAGEMAKER_CODE_S3_URI : s3:// path containing the source tarball that
#                           SageMaker stages into /opt/ml/code/ on the worker.
ROLE_ARN_ENV = "SAGEMAKER_ROLE_ARN"
IMAGE_URI_ENV = "SAGEMAKER_IMAGE_URI"
CODE_S3_URI_ENV = "SAGEMAKER_CODE_S3_URI"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit one Stage 2b ablation training job to SageMaker.",
    )
    parser.add_argument(
        "--n_rays",
        type=int,
        required=True,
        choices=[16384, 1024, 256, 64],
        help="Sightline-density ablation point (see [D-13]).",
    )
    parser.add_argument(
        "--physics",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="Sherwood physics variant (1=no-fb, 2=stellar, 3=wind+AGN, 4=strong-AGN).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed (logged as MLflow tag).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50000,
        help="Total optimizer steps. Default 50000 per [D-14].",
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        required=True,
        help="MLflow tracking URI for THIS host (not the container). Logged to "
             "stdout for audit; the container always uses the file:// store at "
             f"{SAGEMAKER_MLFLOW_URI}, which is replayed back to this URI by "
             "scripts/sagemaker_stage2b_import_mlflow.py after the job finishes.",
    )
    parser.add_argument(
        "--role_arn",
        type=str,
        default=os.environ.get(ROLE_ARN_ENV),
        help=f"IAM role ARN. Defaults to ${ROLE_ARN_ENV}.",
    )
    parser.add_argument(
        "--image_uri",
        type=str,
        default=os.environ.get(IMAGE_URI_ENV),
        help=f"Training container ECR URI. Defaults to ${IMAGE_URI_ENV}.",
    )
    parser.add_argument(
        "--code_s3_uri",
        type=str,
        default=os.environ.get(CODE_S3_URI_ENV),
        help=f"S3 URI of source tarball staged at /opt/ml/code/. Defaults to ${CODE_S3_URI_ENV}. "
             "Optional when source is already baked into the image at /opt/ml/code/.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=DEFAULT_REGION,
        help="AWS region. Default from AWS_DEFAULT_REGION env var.",
    )
    parser.add_argument(
        "--no_spot",
        action="store_true",
        help="Use on-demand instead of managed spot training. Default is spot "
             "(70%% cheaper, can be preempted). Use on-demand when spot quota "
             "is unavailable or for time-sensitive runs.",
    )
    parser.add_argument(
        "--stage_tag",
        type=str,
        default="2b",
        help="Value for the MLflow `stage` tag (and SageMaker job tag). Default "
             "'2b' for production matrix runs; pass '2b-microsweep' for the "
             "[D-23] 16-cell pre-flight grid so it is filterable from the main "
             "ablation in MLflow.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Actually call create_training_job. Without this flag the script "
             "only prints the payload (dry-run).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

def _build_run_id(physics: int, n_rays: int, seed: int) -> str:
    """Stable, human-readable run id.

    Format ``Stage2b-Ablation-P<n>-N<m>-S<s>-<unix_ts>-<short_uuid>`` so the
    same matrix point can be relaunched (e.g. spot interruption) without a
    SageMaker job-name collision while still grouping cleanly in MLflow.
    """
    base = f"Stage2b-Ablation-P{physics}-N{n_rays}-S{seed}"
    suffix = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    return f"{base}-{suffix}"


def _build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.role_arn:
        sys.exit(f"ERROR: --role_arn or ${ROLE_ARN_ENV} is required.")
    if not args.image_uri:
        sys.exit(f"ERROR: --image_uri or ${IMAGE_URI_ENV} is required.")
    # code_s3_uri is optional when the image bakes /opt/ml/code/ at build time.

    run_id = _build_run_id(args.physics, args.n_rays, args.seed)
    checkpoint_s3 = f"s3://{S3_BUCKET}/{CHECKPOINT_PREFIX}/{run_id}/"

    # Hyperparameters land in /opt/ml/input/config/hyperparameters.json on the
    # worker; the entry-point script must parse and forward them. They are the
    # CLI surface that ``experiments/nerf/pipeline.py`` will be parametrized
    # against in C1 (see immediate-next-steps item 4 in the LEDGER).
    hyperparameters: Dict[str, str] = {
        "n_rays": str(args.n_rays),
        "physics": str(args.physics),
        "seed": str(args.seed),
        "max_steps": str(args.max_steps),
        "checkpoint_interval": str(CHECKPOINT_INTERVAL_STEPS),
        "run_name": run_id,
        # Entry-point script path inside /opt/ml/code/.
        "sagemaker_program": "experiments/nerf/pipeline.py",
    }
    if args.code_s3_uri:
        # SageMaker SDK convention: pull source tarball from S3 to /opt/ml/code/
        # at job start. Without this, /opt/ml/code/ must already exist in the
        # training image (the Dockerfile in this repo bakes it).
        hyperparameters["sagemaker_submit_directory"] = args.code_s3_uri

    payload: Dict[str, Any] = {
        "TrainingJobName": run_id,
        "RoleArn": args.role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": args.image_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": [
                "python",
                "-u",
                "/opt/ml/code/experiments/nerf/pipeline.py",
            ],
            "ContainerArguments": [
                "--n_rays", str(args.n_rays),
                "--physics", str(args.physics),
                "--seed", str(args.seed),
                "--max_steps", str(args.max_steps),
                "--run_name", run_id,
                "--checkpoint_dir", "/opt/ml/checkpoints",
                "--checkpoint_interval", str(CHECKPOINT_INTERVAL_STEPS),
                # Sherwood mirror is mounted at /opt/ml/input/data/sherwood/
                # by the InputDataConfig channel below. SherwoodLoader expects
                # data_root such that data_root/Physics<N>_<name>/los*.dat
                # resolves, so we pass the channel mount point directly.
                "--data_root", SHERWOOD_CONTAINER_PATH,
            ],
        },
        "InputDataConfig": [
            {
                "ChannelName": "sherwood",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": SHERWOOD_S3_PREFIX,
                        # File mode: full prefix is downloaded to
                        # /opt/ml/input/data/sherwood/ before container start.
                        # Stage 2b file sizes (~1.5 GB / Physics) are well
                        # within EBS budget set in ResourceConfig.
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "InputMode": "File",
            },
        ],
        "HyperParameters": hyperparameters,
        "ResourceConfig": {
            "InstanceType": INSTANCE_TYPE,
            "InstanceCount": 1,
            "VolumeSizeInGB": 200,  # holds Sherwood snapshot + checkpoints
        },
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{S3_BUCKET}/stage2b-output/",
        },
        # Spot-friendly checkpointing: SageMaker syncs everything written to
        # /opt/ml/checkpoints/ to the S3 path below at the configured cadence,
        # so an interrupted job can resume from the last 10k-step checkpoint.
        "CheckpointConfig": {
            "S3Uri": checkpoint_s3,
            "LocalPath": "/opt/ml/checkpoints",
        },
        "StoppingCondition": (
            {"MaxRuntimeInSeconds": MAX_RUNTIME_S}
            if args.no_spot
            else {"MaxRuntimeInSeconds": MAX_RUNTIME_S, "MaxWaitTimeInSeconds": MAX_WAIT_S}
        ),
        "EnableManagedSpotTraining": not args.no_spot,
        "Environment": {
            # In-container MLflow goes to a local file store under
            # /opt/ml/model/mlflow/. Anything under /opt/ml/output/ is
            # auto-tarballed by SageMaker into model.tar.gz at job exit, then
            # replayed into the host MLflow tracker (args.mlflow_uri) by
            # scripts/sagemaker_stage2b_import_mlflow.py. This avoids the
            # 127.0.0.1:5000 retry storm that hangs the container ~4 min and
            # silently records nothing (Run id: None in the B-2 smoke).
            "MLFLOW_TRACKING_URI": SAGEMAKER_MLFLOW_URI,
            "MLFLOW_EXPERIMENT_NAME": "CosmoGasVision/NeRF",
            "STAGE2B_RUN_NAME": run_id,
            "PYTHONUNBUFFERED": "1",
            # Harmless for file:// scheme but kept as belt-and-suspenders in
            # case any code path falls through to an HTTP backend.
            "MLFLOW_HTTP_REQUEST_TIMEOUT": "10",
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "1",
        },
        "Tags": [
            {"Key": "model_type", "Value": "nerf"},
            {"Key": "stage", "Value": args.stage_tag},
            {"Key": "physics_id", "Value": str(args.physics)},
            {"Key": "n_rays", "Value": str(args.n_rays)},
            {"Key": "seed", "Value": str(args.seed)},
            {"Key": "redshift", "Value": "0.3"},
            {"Key": "ablation_matrix", "Value": "stage2b-4x4"},
        ],
    }
    return payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = _build_payload(args)

    # Always print the resolved payload — this doubles as the dry-run output
    # and as the audit trail when --launch is set.
    import json
    print(json.dumps(payload, indent=2, default=str))

    if not args.launch:
        print("\nDry run — pass --launch to submit. No SageMaker API call made.")
        return 0

    if boto3 is None:
        sys.exit("ERROR: boto3 is not installed. `uv add boto3` first.")

    client = boto3.client("sagemaker", region_name=args.region)
    response = client.create_training_job(**payload)
    print(f"\nSubmitted: {response['TrainingJobArn']}")
    # In-container MLflow writes to file:///opt/ml/model/mlflow which lands in
    # model.tar.gz at job exit. Replay into the host tracker with:
    print(
        f"==> When job completes: uv run python scripts/sagemaker_stage2b_import_mlflow.py "
        f"{payload['TrainingJobName']} --mlflow_uri {args.mlflow_uri}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
