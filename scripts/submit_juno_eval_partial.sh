#!/bin/bash
#SBATCH --job-name=Stage2b-Juno-Eval-Partial
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --output=stage2b-eval-partial-%j.out
#SBATCH --error=stage2b-eval-partial-%j.err

# Partial [D-13] cosmological evaluator: P_F(k_||) and F-PDF KS-distance only.
# Skips the 3D cross-correlation (xi) figure because that requires the
# SherwoodIGM_gal HDF5 particle snapshots (40 GB) which are not on Juno.
#
# Required env: RESULT_DIR (per-cell result dir on /work containing
# both mlflow/ and checkpoints/ subdirs).

set -euo pipefail

if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=/work/sxh240010/CosmoGasVision}"
: "${JUNO_SCRATCH:=/scratch/juno/sxh240010}"
: "${RESULT_DIR:?RESULT_DIR must be exported}"

# Resolve source-side mlflow run id (file-store: mlflow/<exp_id>/<run_id>/).
# Filter strictly to 32-hex-char run-id dirs to avoid stray pollution (literal
# `*` dirs from earlier botched dispatches, partial mkdir state, etc.) per
# infrastructure-manager.md PCV section.
SRC_RUN_DIR=$(find "${RESULT_DIR}/mlflow" -mindepth 2 -maxdepth 2 -type d \
    -regextype posix-extended -regex '.*/[a-f0-9]{32}$' | head -1)
[[ -n "${SRC_RUN_DIR}" ]] || { echo "FATAL: no run dir under ${RESULT_DIR}/mlflow matching /[a-f0-9]{32}$/"; exit 2; }
SRC_RUN_ID=$(basename "${SRC_RUN_DIR}")
[[ "${SRC_RUN_ID}" =~ ^[a-f0-9]{32}$ ]] || { echo "FATAL: resolved run_id '${SRC_RUN_ID}' is not a 32-hex UUID"; exit 2; }

# Stage checkpoint as MLflow artifact so _load_mlflow_run picks it up.
LATEST_CKPT=$(ls -1 "${RESULT_DIR}/checkpoints"/*.pt 2>/dev/null | sort -V | tail -1)
[[ -n "${LATEST_CKPT}" ]] || { echo "FATAL: no checkpoint at ${RESULT_DIR}/checkpoints"; exit 3; }
mkdir -p "${SRC_RUN_DIR}/artifacts"
cp -n "${LATEST_CKPT}" "${SRC_RUN_DIR}/artifacts/"

echo "=== eval-partial config ==="
echo "RESULT_DIR=${RESULT_DIR}"
echo "SRC_RUN_ID=${SRC_RUN_ID}"
echo "LATEST_CKPT=${LATEST_CKPT}"

RUN_DIR="${JUNO_SCRATCH}/stage2b_eval_partial/${SRC_RUN_ID}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
cp -r "${JUNO_WORK}"/{src,experiments,scripts,pyproject.toml,uv.lock} .
ln -sfn "${JUNO_SCRATCH}/sherwood" Sherwood

source "${JUNO_WORK}/.venv/bin/activate"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export GIT_PYTHON_REFRESH=quiet

REPORT_OUT="${JUNO_WORK%/CosmoGasVision}/stage2b_results/eval-partial-${SRC_RUN_ID:0:8}"
mkdir -p "${REPORT_OUT}"

python -u scripts/eval_partial_d13.py \
    --run-id "${SRC_RUN_ID}" \
    --output-dir "${REPORT_OUT}/" \
    --physics-id "${PHYSICS_ID:-1}" \
    --redshift "${REDSHIFT:-0.3}" \
    --n-rays-eval "${N_RAYS_EVAL:-1024}" \
    --mlflow-uri "file://${RESULT_DIR}/mlflow" \
    --ckpt-path "${LATEST_CKPT}"

echo "=== Report at ${REPORT_OUT}/${SRC_RUN_ID}_partial/ ==="
ls -la "${REPORT_OUT}/${SRC_RUN_ID}_partial/"

cd "${HOME}"
rm -rf "${RUN_DIR}"
echo "=== eval-partial done ==="
