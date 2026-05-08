#!/bin/bash
#SBATCH --job-name=Stage2b-Juno
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=stage2b-juno-%j.out
#SBATCH --error=stage2b-juno-%j.err

# Stage 2b training cell on UTD Juno HPC.
# See `.claude/skills/juno-hpc/SKILL.md` for the contract; [D-25] documents
# the cu124 torch override; PCV (Producer-Consumer Verification) section
# below was added 2026-05-08 after the Batch-2/3 checkpoint-loss incident
# (the prior copy-out line silently dropped checkpoints; see infrastructure-
# manager agent definition for the framework).

set -euo pipefail

if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=/work/sxh240010/CosmoGasVision}"
: "${JUNO_SCRATCH:=/scratch/juno/sxh240010}"

RUN_TAG="P${PHYSICS_ID}-N${N_RAYS}-S${SEED}-$(date +%s)-$(uuidgen | cut -c1-6)"
RUN_DIR="${JUNO_SCRATCH}/stage2b/${RUN_TAG}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

cp -r "${JUNO_WORK}"/{src,experiments,scripts,pyproject.toml,uv.lock} .
ln -s "${JUNO_SCRATCH}/sherwood" Sherwood

source "${JUNO_WORK}/.venv/bin/activate"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${RUN_DIR}/mlflow"
export GIT_PYTHON_REFRESH=quiet

echo "=== run config ==="
echo "RUN_TAG=${RUN_TAG}"
echo "physics=${PHYSICS_ID} n_rays=${N_RAYS} microbatch=${MICROBATCH} accum=${ACCUM_STEPS} max_steps=${MAX_STEPS} warmup=${WARMUP_STEPS} seed=${SEED}"
nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv | tail -1

python -u experiments/nerf/pipeline.py \
    --physics "${PHYSICS_ID}" \
    --n_rays "${N_RAYS}" \
    --microbatch "${MICROBATCH}" \
    --accum_steps "${ACCUM_STEPS}" \
    --max_steps "${MAX_STEPS}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --seed "${SEED}"

# === Producer-Consumer Verification (PCV) — copy-out + assert ===
# pipeline.py writes step_*.pt under experiments/nerf/artifacts/checkpoints/
# (its --checkpoint_dir default), NOT under ./checkpoints/. Copy by the
# correct nested path and assert the artifact exists before cleanup.
DEST="${JUNO_WORK%/CosmoGasVision}/stage2b_results/${RUN_TAG}"
mkdir -p "${DEST}"

# 1. MLflow file-store — required for downstream metric replay.
[[ -d mlflow ]] || { echo "FATAL: pipeline produced no mlflow/ store" >&2; exit 2; }
cp -r mlflow "${DEST}/"

# 2. Checkpoints — required for [D-13] evaluators downstream. Hard-fail on absence.
CKPT_SRC="experiments/nerf/artifacts/checkpoints"
[[ -d "${CKPT_SRC}" ]] || { echo "FATAL: no checkpoint dir at ${CKPT_SRC}" >&2; exit 3; }
N_CKPT=$(ls -1 "${CKPT_SRC}"/*.pt 2>/dev/null | wc -l)
[[ "${N_CKPT}" -gt 0 ]] || { echo "FATAL: ${CKPT_SRC} contains zero *.pt files" >&2; exit 4; }
mkdir -p "${DEST}/checkpoints"
cp "${CKPT_SRC}"/*.pt "${DEST}/checkpoints/"

# 3. Verify what was copied — visible in the .out log for downstream debugging.
echo "=== copied artifacts ==="
echo "--- ${DEST}/mlflow ---"
find "${DEST}/mlflow" -maxdepth 4 -type d | head -10
echo "--- ${DEST}/checkpoints ---"
ls -la "${DEST}/checkpoints/"

cd "${HOME}"
rm -rf "${RUN_DIR}"
echo "=== done, results at ${DEST} ==="
