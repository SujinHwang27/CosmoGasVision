#!/usr/bin/env bash
# Build and push the Stage 2b training container to ECR.
#
# All AWS account / region / repository config is loaded from .env (gitignored).
# After a successful push, prints the ECR URI in the form expected by .env's
# SAGEMAKER_IMAGE_URI key — copy-paste it back into .env to keep the launcher
# in sync.
#
# Usage:
#   bash scripts/build_and_push_ecr.sh
#
# Required .env keys (no defaults — error out if missing):
#   AWS_ACCOUNT_ID       — 12-digit AWS account number
#   AWS_DEFAULT_REGION   — e.g. us-east-1
#   ECR_REPOSITORY       — e.g. cosmo-gas-vision/nerf-trainer
#
# Optional:
#   ECR_TAG_PREFIX       — defaults to "stage2b". Final tag is "<prefix>-<git_sha>".

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

# --- Load .env ---------------------------------------------------------------
if [[ ! -f .env ]]; then
    echo "ERROR: .env not found at ${ROOT}/.env" >&2
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

# --- Validate required keys --------------------------------------------------
: "${AWS_ACCOUNT_ID:?ERROR: AWS_ACCOUNT_ID missing in .env}"
: "${AWS_DEFAULT_REGION:?ERROR: AWS_DEFAULT_REGION missing in .env}"
: "${ECR_REPOSITORY:?ERROR: ECR_REPOSITORY missing in .env}"

REGION="${AWS_DEFAULT_REGION}"
ACCOUNT_ID="${AWS_ACCOUNT_ID}"
REPOSITORY="${ECR_REPOSITORY}"
TAG_PREFIX="${ECR_TAG_PREFIX:-stage2b}"
GIT_SHA="$(git rev-parse --short HEAD)"
TAG="${TAG_PREFIX}-${GIT_SHA}"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY}:${TAG}"

echo "==> Build / push plan"
echo "    Repository: ${REPOSITORY}"
echo "    Tag       : ${TAG}"
echo "    URI       : ${ECR_URI}"
echo

# --- Authenticate Docker to ECR ----------------------------------------------
echo "==> Authenticating Docker to ECR (${REGION}) ..."
aws ecr get-login-password --region "${REGION}" \
    | docker login \
        --username AWS \
        --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# --- Ensure the repository exists --------------------------------------------
if ! aws ecr describe-repositories --repository-names "${REPOSITORY}" --region "${REGION}" >/dev/null 2>&1; then
    echo "==> ECR repository ${REPOSITORY} not found; creating ..."
    aws ecr create-repository \
        --repository-name "${REPOSITORY}" \
        --region "${REGION}" \
        --image-scanning-configuration scanOnPush=true >/dev/null
fi

# --- Build, tag, push --------------------------------------------------------
echo "==> Building image ${REPOSITORY}:${TAG} ..."
docker build -t "${REPOSITORY}:${TAG}" .

echo "==> Tagging for ECR ..."
docker tag "${REPOSITORY}:${TAG}" "${ECR_URI}"

echo "==> Pushing ${ECR_URI} ..."
docker push "${ECR_URI}"

echo
echo "==> Done. Update .env if needed:"
echo "    SAGEMAKER_IMAGE_URI=\"${ECR_URI}\""
