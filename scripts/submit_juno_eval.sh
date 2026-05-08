#!/bin/bash
#SBATCH --job-name=Stage2b-Juno-Eval
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --output=stage2b-eval-%j.out
#SBATCH --error=stage2b-eval-%j.err

# Run the [D-13] cosmological evaluator (src/analysis/stage2b_report.generate_report)
# on Juno against a previously-trained checkpoint. Outputs 5 PNGs + index.html
# under experiments/nerf/artifacts/reports/<run_id>/.
#
# Required env: RESULT_DIR (path to the per-cell result dir on /work that has
# both mlflow/ and checkpoints/ subdirs, e.g.
#   /work/sxh240010/stage2b_results/P1-N1024-S0-1778229084-c08848)

set -euo pipefail

if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=/work/sxh240010/CosmoGasVision}"
: "${JUNO_SCRATCH:=/scratch/juno/sxh240010}"
: "${RESULT_DIR:?RESULT_DIR must be exported (e.g. /work/.../stage2b_results/P1-...)}"

# Resolve the source-side mlflow run id: file-store has one run dir per run
# under <result>/mlflow/<exp_id>/<run_id>/. Skip the empty exp_id=0 dir.
SRC_RUN_DIR=$(find "${RESULT_DIR}/mlflow" -mindepth 2 -maxdepth 2 -type d ! -path '*/0/*' ! -path '*/.trash/*' | head -1)
[[ -n "${SRC_RUN_DIR}" ]] || { echo "FATAL: no run dir under ${RESULT_DIR}/mlflow"; exit 2; }
SRC_RUN_ID=$(basename "${SRC_RUN_DIR}")
EXP_ID=$(basename "$(dirname "${SRC_RUN_DIR}")")

# Stage the latest checkpoint as an MLflow artifact in the file-store so
# _load_mlflow_run picks it up via list_artifacts.
LATEST_CKPT=$(ls -1 "${RESULT_DIR}/checkpoints"/*.pt 2>/dev/null | sort -V | tail -1)
[[ -n "${LATEST_CKPT}" ]] || { echo "FATAL: no checkpoint at ${RESULT_DIR}/checkpoints"; exit 3; }

ART_DIR="${SRC_RUN_DIR}/artifacts"
mkdir -p "${ART_DIR}"
cp "${LATEST_CKPT}" "${ART_DIR}/$(basename ${LATEST_CKPT})"

echo "=== eval config ==="
echo "RESULT_DIR=${RESULT_DIR}"
echo "SRC_RUN_ID=${SRC_RUN_ID}"
echo "LATEST_CKPT=${LATEST_CKPT}"
echo "ART_DIR=${ART_DIR}"

# Run from a fresh scratch workdir with a Sherwood symlink, mirroring the
# training contract. Use the file-store URI as tracking URI so Juno doesn't
# need to reach the host tracker.
RUN_DIR="${JUNO_SCRATCH}/stage2b_eval/${SRC_RUN_ID}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
cp -r "${JUNO_WORK}"/{src,experiments,scripts,pyproject.toml,uv.lock} .
ln -sfn "${JUNO_SCRATCH}/sherwood" Sherwood

source "${JUNO_WORK}/.venv/bin/activate"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${RESULT_DIR}/mlflow"
export GIT_PYTHON_REFRESH=quiet

REPORT_OUT="${JUNO_WORK%/CosmoGasVision}/stage2b_results/eval-${SRC_RUN_ID:0:8}"
mkdir -p "${REPORT_OUT}"

python -u -m src.analysis.stage2b_report \
    --run_id "${SRC_RUN_ID}" \
    --output_dir "${REPORT_OUT}/" \
    --physics_id "${PHYSICS_ID:-1}" \
    --redshift "${REDSHIFT:-0.3}" \
    --n_grid "${N_GRID:-192}" \
    --n_rays_eval "${N_RAYS_EVAL:-1024}"

echo "=== Report at ${REPORT_OUT}/${SRC_RUN_ID}/ ==="
ls -la "${REPORT_OUT}/${SRC_RUN_ID}/"

cd "${HOME}"
rm -rf "${RUN_DIR}"
echo "=== eval done ==="
