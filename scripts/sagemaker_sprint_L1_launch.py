"""SageMaker training-job launcher for sprint-L1 (direct P_F MSE loss; [D-60]).

Submits one managed-spot training job on ``ml.g5.xlarge`` (NVIDIA A10G, 24 GB)
for the sprint-L1 single-point experiment: direct P_F MSE loss + GradNorm
combination with the [D-24] tau-MSE, evaluated at the tier-3 (P1, n_rays=1024,
seed=0) cost-survey schedule. See ``experiments/nerf/design/sprint_L1_*`` for
the design doc and gate-6 PI rulings.

Adapted from ``scripts/sagemaker_stage2b_launch.py`` (Stage 2b precedent):
  - instance_type        = ml.g5.xlarge          ([D-14], unchanged)
  - EnableManagedSpot    = True                  (PI gate-6 ruling #4)
  - MaxRuntimeInSeconds  = 144000   (40 hr)      (PI gate-6 ruling #5; 18% margin)
  - MaxWaitTimeInSeconds = 288000   (80 hr)      (Stage 2b 2x runtime precedent)
  - run_name format      = SprintL1-PFloss-P1-N1024-S0-<unix_ts>-<short_uuid>
  - entry point          = experiments/nerf/pipeline.py
  - MLFLOW_TRACKING_URI is forwarded as the in-container file:// store URI.

This file does NOT launch the job by default. Importing it produces no side
effect; running ``python scripts/sagemaker_sprint_L1_launch.py --launch ...``
is required to actually call ``boto3.client('sagemaker').create_training_job``.

PI gate-6 rulings absorbed (sprint-L1 specific knobs):
  1. Dockerfile patch landed pre-build (commit ad9d93b on exp/nerf).
  2. --l1-d24-baseline-tau-mse 0.01 (cost-survey T3 200-step value 0.0121,
     conservative upper bound; documented via MLflow tag
     r_c_baseline_source=cost_survey_t3_200step_conservative_upper_bound).
  3. microbatch=256 + accum_steps=4 (cost-survey T3 schedule; pipeline auto-
     derives accum from ceil(n_rays/microbatch) = ceil(1024/256) = 4).
  4. Managed spot.
  5. MaxRuntimeInSeconds=144000.
  6. Run name SprintL1-PFloss-P1-N1024-S0-<ts>-<uuid6>.

CLI flag corrections from prior gate-6a session:
  - --physics 1 (not --physics_id 1)
  - --mean_flux_obs 0.979 (not --anchor_mean_F 0.979)
  - --enable-l1-pf-loss (pipeline.py:206)
  - Other sprint-L1 sub-flags use pipeline defaults.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from typing import Any, Dict

# UTF-8 stdout/stderr — payload includes em-dash etc.; default cp949 on
# Windows-Korean crashes on print. Same fix as in the Stage 2b launcher.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and getattr(_stream, "encoding", "").lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="replace")

try:
    import boto3  # type: ignore
except ImportError:  # pragma: no cover
    boto3 = None  # noqa: N816


# ---------------------------------------------------------------------------
# Pinned constants — sprint-L1 specific (do not parameterize without a LEDGER
# decision update or another PI ruling).
# ---------------------------------------------------------------------------

S3_BUCKET = "cosmo-gas-vision-storage"
CHECKPOINT_PREFIX = "sprintL1-checkpoints"
OUTPUT_PREFIX = "sprintL1-output"
INSTANCE_TYPE = "ml.g5.xlarge"            # A10G 24 GB, [D-14] precedent
MAX_RUNTIME_S = 144000                     # 40 hr, PI gate-6 ruling #5
MAX_WAIT_S = 288000                        # 80 hr, 2x runtime per Stage 2b precedent
CHECKPOINT_INTERVAL_STEPS = 10000          # cadence (40k + 50k checkpoints land in 50k run)
DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# Sherwood data mirror in our bucket. Identical to Stage 2b launcher.
SHERWOOD_S3_PREFIX = f"s3://{S3_BUCKET}/sherwood/"
SHERWOOD_CONTAINER_PATH = "/opt/ml/input/data/sherwood"

# In-container MLflow file-store. Replayed to host MLflow tracker post-job
# via scripts/sagemaker_stage2b_import_mlflow.py (run-name agnostic).
SAGEMAKER_MLFLOW_URI = "file:///opt/ml/model/mlflow"

# Sprint-L1 fixed experimental point (PI gate-6 ruling #6 + design doc v2).
SPRINT_L1_PHYSICS = 1
SPRINT_L1_N_RAYS = 1024
SPRINT_L1_SEED = 0
SPRINT_L1_MICROBATCH = 256
SPRINT_L1_MAX_STEPS = 50000
SPRINT_L1_MEAN_FLUX_OBS = 0.979            # Kirkman+2007 anchor at z=0.3 ([D-11]/[D-34])
SPRINT_L1_D24_BASELINE_TAU_MSE = 0.01      # PI gate-6 ruling #2 (cost-survey T3 200-step upper bound)

# Env-var names mirroring Stage 2b launcher conventions.
ROLE_ARN_ENV = "SAGEMAKER_ROLE_ARN"
IMAGE_URI_ENV = "SAGEMAKER_IMAGE_URI"
CODE_S3_URI_ENV = "SAGEMAKER_CODE_S3_URI"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit the sprint-L1 (direct P_F MSE loss; [D-60]) "
                    "training job to SageMaker.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=SPRINT_L1_MAX_STEPS,
        help=f"Total optimizer steps. Default {SPRINT_L1_MAX_STEPS} per design v2 / PI "
             "gate-6 ruling. Override to 5 for the on-cloud memory smoke "
             "(~135 s billable, ~$0.04 spot).",
    )
    parser.add_argument(
        "--commit_sha",
        type=str,
        required=True,
        help="Full git SHA at HEAD when the ECR image was built. Logged as "
             "MLflow tag commit_sha for run<->code traceability ([D-37]-ext R10).",
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
             "per PI gate-6 ruling #4 (~$10 vs ~$34 on-demand ceiling).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tag the run as a memory smoke (stage=2b_sprint_L1_smoke). Use "
             "with --max_steps 5 to validate Dockerfile patch + image rebuild "
             "+ IAM permissions before the full 50k-step dispatch. ~135 s "
             "billable, ~$0.04 spot.",
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

def _build_run_id(smoke: bool) -> str:
    """Stable, human-readable run id.

    Format ``SprintL1-PFloss-P1-N1024-S0-<unix_ts>-<short_uuid>`` per PI
    gate-6 ruling #6. Smoke variant prefixes ``Smoke-`` so it sorts apart
    from production runs in the SageMaker console + MLflow UI.
    """
    base = f"SprintL1-PFloss-P{SPRINT_L1_PHYSICS}-N{SPRINT_L1_N_RAYS}-S{SPRINT_L1_SEED}"
    if smoke:
        base = f"Smoke-{base}"
    suffix = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    return f"{base}-{suffix}"


def _build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.role_arn:
        sys.exit(f"ERROR: --role_arn or ${ROLE_ARN_ENV} is required.")
    if not args.image_uri:
        sys.exit(f"ERROR: --image_uri or ${IMAGE_URI_ENV} is required.")

    run_id = _build_run_id(args.smoke)
    checkpoint_s3 = f"s3://{S3_BUCKET}/{CHECKPOINT_PREFIX}/{run_id}/"
    stage_tag = "2b_sprint_L1_smoke" if args.smoke else "2b_sprint_L1"

    # Hyperparameters land in /opt/ml/input/config/hyperparameters.json on the
    # worker; pipeline.py parses them via _parse_args() argparse.
    hyperparameters: Dict[str, str] = {
        "n_rays": str(SPRINT_L1_N_RAYS),
        "physics": str(SPRINT_L1_PHYSICS),
        "seed": str(SPRINT_L1_SEED),
        "max_steps": str(args.max_steps),
        "microbatch": str(SPRINT_L1_MICROBATCH),
        "checkpoint_interval": str(CHECKPOINT_INTERVAL_STEPS),
        "mean_flux_obs": str(SPRINT_L1_MEAN_FLUX_OBS),
        "enable_l1_pf_loss": "true",
        "l1_d24_baseline_tau_mse": str(SPRINT_L1_D24_BASELINE_TAU_MSE),
        "run_name": run_id,
        "sagemaker_program": "experiments/nerf/pipeline.py",
    }
    if args.code_s3_uri:
        hyperparameters["sagemaker_submit_directory"] = args.code_s3_uri

    # ContainerArguments is the authoritative flag list — SageMaker honors the
    # Dockerfile ENTRYPOINT + ContainerEntrypoint + ContainerArguments; the
    # HyperParameters dict above is documentation/MLflow-readable only.
    container_args = [
        "--n_rays", str(SPRINT_L1_N_RAYS),
        "--physics", str(SPRINT_L1_PHYSICS),
        "--seed", str(SPRINT_L1_SEED),
        "--max_steps", str(args.max_steps),
        "--microbatch", str(SPRINT_L1_MICROBATCH),
        "--mean_flux_obs", str(SPRINT_L1_MEAN_FLUX_OBS),
        "--run_name", run_id,
        "--checkpoint_dir", "/opt/ml/checkpoints",
        "--checkpoint_interval", str(CHECKPOINT_INTERVAL_STEPS),
        "--data_root", SHERWOOD_CONTAINER_PATH,
        # Sprint-L1 sub-flags — only the non-default baseline is set; the
        # other 4 (gradnorm-alpha 0.12, burnin-tau-mse 1000, burnin-var-f 500,
        # retire-dir = checkpoint_dir) take pipeline.py defaults per PI ruling.
        "--enable-l1-pf-loss",
        "--l1-d24-baseline-tau-mse", str(SPRINT_L1_D24_BASELINE_TAU_MSE),
    ]

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
            "ContainerArguments": container_args,
        },
        "InputDataConfig": [
            {
                "ChannelName": "sherwood",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": SHERWOOD_S3_PREFIX,
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
            "VolumeSizeInGB": 200,
        },
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{S3_BUCKET}/{OUTPUT_PREFIX}/",
        },
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
            "MLFLOW_TRACKING_URI": SAGEMAKER_MLFLOW_URI,
            "MLFLOW_EXPERIMENT_NAME": "CosmoGasVision/NeRF",
            "STAGE2B_RUN_NAME": run_id,
            "PYTHONUNBUFFERED": "1",
            "MLFLOW_HTTP_REQUEST_TIMEOUT": "10",
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "1",
        },
        "Tags": [
            # Mandatory project-wide tags (CLAUDE.md "Experiment workflow").
            {"Key": "model_type", "Value": "nerf"},
            {"Key": "stage", "Value": stage_tag},
            {"Key": "physics_id", "Value": str(SPRINT_L1_PHYSICS)},
            {"Key": "redshift", "Value": "0.3"},
            # Sprint-L1 specific (PI gate-6 ruling #6).
            {"Key": "n_rays", "Value": str(SPRINT_L1_N_RAYS)},
            {"Key": "seed", "Value": str(SPRINT_L1_SEED)},
            {"Key": "loss_variant", "Value": "L1_direct_pf"},
            {"Key": "gate", "Value": "6_dispatch"},
            {"Key": "commit_sha", "Value": args.commit_sha},
            {"Key": "design_doc", "Value": "sprint_L1_direct_pf_loss_v2"},
            {"Key": "r_c_baseline_source",
             "Value": "cost_survey_t3_200step_conservative_upper_bound"},
        ],
    }
    return payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = _build_payload(args)

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
    print(
        f"==> When job completes: uv run python scripts/sagemaker_stage2b_import_mlflow.py "
        f"{payload['TrainingJobName']} --mlflow_uri {args.mlflow_uri} "
        f"--prefix {OUTPUT_PREFIX}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
