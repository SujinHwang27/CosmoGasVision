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

# [D-39] anchor + batch discriminator. Defaults preserve pre-publication-class
# behavior (cost-survey runs did not pass --mean_flux_obs through this wrapper).
# For the publication-class T1 batch at the [D-34] corrected anchor:
#   MEAN_FLUX_OBS=0.979  JUNO_BATCH=pub-t1
: "${MEAN_FLUX_OBS:=0.877}"
: "${JUNO_BATCH:=stage2b}"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SHORTHASH=$(cd "${JUNO_WORK}" && git rev-parse --short HEAD 2>/dev/null || echo nogit)

RUN_TAG="P${PHYSICS_ID}-N${N_RAYS}-S${SEED}-$(date +%s)-$(uuidgen | cut -c1-6)"

# Stage-prefixed RUN_NAME for publication-class batches; empty default lets
# pipeline.py auto-generate via the [D-12] pattern (cost-survey/default mode).
case "${JUNO_BATCH}" in
  pub-t1) RUN_NAME="Stage2bPub-T1-P${PHYSICS_ID}-S${SEED}-${SHORTHASH}-${TIMESTAMP}" ;;
  pub-t4) RUN_NAME="Stage2bPub-T4-P${PHYSICS_ID}-S${SEED}-${SHORTHASH}-${TIMESTAMP}" ;;
  *)      RUN_NAME="" ;;
esac

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
echo "RUN_NAME=${RUN_NAME:-<pipeline.py auto-generated>}"
echo "JUNO_BATCH=${JUNO_BATCH}"
echo "MEAN_FLUX_OBS=${MEAN_FLUX_OBS}"
echo "physics=${PHYSICS_ID} n_rays=${N_RAYS} microbatch=${MICROBATCH} accum=${ACCUM_STEPS} max_steps=${MAX_STEPS} warmup=${WARMUP_STEPS} seed=${SEED}"
nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv | tail -1

python -u experiments/nerf/pipeline.py \
    --physics "${PHYSICS_ID}" \
    --n_rays "${N_RAYS}" \
    --microbatch "${MICROBATCH}" \
    --accum_steps "${ACCUM_STEPS}" \
    --max_steps "${MAX_STEPS}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --seed "${SEED}" \
    --mean_flux_obs "${MEAN_FLUX_OBS}" \
    ${RUN_NAME:+--run_name "${RUN_NAME}"}

# === Post-run MLflow tag injection (publication-class batches only) ===
# pipeline.py sets the canonical Stage 2b tags via mlflow.set_tags. For pub-*
# batches we add provenance tags so the host-side replay can discriminate
# publication-class runs from cost-survey runs and trace [D-34] anchor +
# [D-24] loss-bundle context. Hits the local file store directly.
if [[ "${JUNO_BATCH}" == pub-* ]]; then
  python -u - <<PYEOF
import os, sys
from mlflow.tracking import MlflowClient

uri = os.environ["MLFLOW_TRACKING_URI"]
client = MlflowClient(tracking_uri=uri)
exp = client.get_experiment_by_name("CosmoGasVision/NeRF")
if exp is None:
    print("FATAL: CosmoGasVision/NeRF experiment not found.", file=sys.stderr); sys.exit(10)
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="tags.\`mlflow.runName\` = '${RUN_NAME}'",
    max_results=1,
)
if not runs:
    print("FATAL: run '${RUN_NAME}' not found after pipeline exit.", file=sys.stderr); sys.exit(11)
run = runs[0]
client.set_tag(run.info.run_id, "stage", "2b-publication")
client.set_tag(run.info.run_id, "juno_batch", "${JUNO_BATCH}")
client.set_tag(run.info.run_id, "mean_flux_obs_target", "${MEAN_FLUX_OBS}")
client.set_tag(run.info.run_id, "mean_flux_obs_source", "kirkman2007")
client.set_tag(run.info.run_id, "anchor_corrected", "true")
client.set_tag(run.info.run_id, "loss_bundle", "log1p+cap+mask")
print(f"[pub-tagger] run_id={run.info.run_id} batch=${JUNO_BATCH} stage=2b-publication OK")
PYEOF
fi

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
