#!/bin/bash
#SBATCH --job-name=Stage1a-1b-SkipRich
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-9%4
#SBATCH --output=stage1a-1b-%A_%a.out
#SBATCH --error=stage1a-1b-%A_%a.err

# Stage 1a (1b) skip-rich-MLP feasibility seed sweep on UTD Juno HPC.
# n=10 seeds x 500 steps x P1 z=0.3, single-A30 per seed. Array indices 0..9
# map to SEED 0..9; %4 caps concurrency to the Juno 4-job-per-user GPU limit.
#
# Design source-of-truth: D70_stage1_architectural_reframe_scoping.md Rev 5.1
# §2.2 (amended) + PI memo §8a + Amendment A.
#
# PI Amendment A (BINDING):  The 'current' arch path is "structurally
# invariant under the Rev-5.1 refactor (init layout 63/319 asserted)" — NOT
# "bit-equivalent to pre-Rev-5.1 production". This sbatch dispatches the
# 'skip-rich-mlp' variant; the framing remark applies to any reference made
# to the 'current' branch in operator comments.
#
# See `.claude/skills/juno-hpc/SKILL.md` for the cluster contract; PCV
# (Producer-Consumer Verification) per infrastructure-manager.md.

set -euo pipefail

if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=/work/sxh240010/CosmoGasVision}"
: "${JUNO_SCRATCH:=/scratch/juno/sxh240010}"

# Array-index -> SEED. Trivial 1:1 mapping for the n=10 sweep.
SEED="${SLURM_ARRAY_TASK_ID:-0}"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SHORTHASH=$(cd "${JUNO_WORK}" && git rev-parse --short HEAD 2>/dev/null || echo nogit)
RUN_TAG="stage1a-1b-skiprich-P1-S${SEED}-${SHORTHASH}-${TIMESTAMP}-$(uuidgen | cut -c1-6)"
RUN_NAME="Stage1a-1b-SkipRichMLP-P1-N768-S${SEED}-${SHORTHASH}-${TIMESTAMP}"

RUN_DIR="${JUNO_SCRATCH}/stage1a/${RUN_TAG}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

# --- Copy in repo source + symlink Sherwood data root ---
cp -r "${JUNO_WORK}"/{src,experiments,scripts,pyproject.toml,uv.lock} .
ln -s "${JUNO_SCRATCH}/sherwood" Sherwood

# --- Environment ---
source "${JUNO_WORK}/.venv/bin/activate"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${RUN_DIR}/mlflow"
export GIT_PYTHON_REFRESH=quiet

# Rho-field cache override — the host-default may be a non-portable absolute
# path. Point at the Juno-side scratch mirror under Sherwood/.rho_field_cache/.
# The 768-grid cache for P1 z=0.3 (1.81 GB) must be pre-staged on Juno-side via:
#   rsync -avzP <host-repo-root>/Sherwood/.rho_field_cache/rho_field_p1_z0.300_n768.* \
#       juno:${JUNO_SCRATCH}/sherwood/.rho_field_cache/
# (one-time prerequisite; not done by this script).
export COSMOGAS_RHO_CACHE_DIR="${JUNO_SCRATCH}/sherwood/.rho_field_cache"

echo "=== Stage 1a (1b) skip-rich-MLP — run config ==="
echo "RUN_TAG=${RUN_TAG}"
echo "RUN_NAME=${RUN_NAME}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-N/A} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-N/A}"
echo "SEED=${SEED}"
echo "ARCH=skip-rich-mlp PHYSICS=1 REDSHIFT=0.3 N_GRID=768 MAX_STEPS=500"
echo "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"
echo "COSMOGAS_RHO_CACHE_DIR=${COSMOGAS_RHO_CACHE_DIR}"
nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv | tail -1

# Verify the cache file is present before training; pretrain loader will
# otherwise CIC-deposit from raw particles (slow, and would silently change
# the substrate). Hard-fail to keep the sweep substrate-consistent.
RHO_CACHE_FILE="${COSMOGAS_RHO_CACHE_DIR}/rho_field_p1_z0.300_n768.npy"
[[ -f "${RHO_CACHE_FILE}" ]] || {
    echo "FATAL: rho-field cache missing at ${RHO_CACHE_FILE}" >&2
    echo "  Pre-stage with: rsync from host Sherwood/.rho_field_cache/" >&2
    exit 5
}

# --- Compute ---
# Per-seed Stage 1a entry. Argparse signatures verified against
# experiments/nerf/pipeline.py:
#   --arch              (line 320, choices={current, skip-rich-mlp})
#   --pretrain-density  (line 277, dest=pretrain_density)
#   --physics           (line 55, required, choices={1,2,3,4})
#   --n_rays            (line 52, required, choices={16384,1024,256,64})
#   --microbatch        (line 62)
#   --accum_steps       (line 64)
#   --max_steps         (line 69)
#   --warmup_steps      (line 72)
#   --seed              (line 58, required)
#   --run_name          (line 82)
# NOTE: PI brief specified --stage_tag but no such argparse flag exists in
# pipeline.py. Surfaced as DEVIATION in dispatch return; stage tagging is
# handled by pipeline.py's internal mlflow.set_tags(stage="1a-density-pretrain")
# at line ~779 — no CLI flag needed.
python -u experiments/nerf/pipeline.py \
    --arch skip-rich-mlp \
    --pretrain-density \
    --physics 1 \
    --n_rays 256 \
    --microbatch 1024 \
    --accum_steps 1 \
    --max_steps 500 \
    --warmup_steps 100 \
    --seed "${SEED}" \
    --run_name "${RUN_NAME}"

# --- Post-run MLflow tag injection ---
# pipeline.py's train_pretrain sets {model_type, stage, physics_id, redshift,
# pretrain_target, design_doc, loss_variant}. Add Stage 1a (1b) discriminators
# so the host-side replay can pick out this sweep.
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
client.set_tag(run.info.run_id, "body_arch", "skip-rich-mlp")
client.set_tag(run.info.run_id, "compute", "juno")
client.set_tag(run.info.run_id, "juno_batch", "stage1a-1b-skiprich")
client.set_tag(run.info.run_id, "seed", "${SEED}")
client.set_tag(run.info.run_id, "stage_substep", "1a-(1b)")
client.set_tag(run.info.run_id, "design_doc_ref", "D70 Rev 5.1 §2.2")
client.set_tag(run.info.run_id, "framing_amendment_A", "structural-invariance-not-bit-equivalent")
print(f"[tagger] run_id={run.info.run_id} SEED=${SEED} OK")
PYEOF

# --- Copy out + Producer-Consumer Verification (PCV) ---
DEST="${JUNO_WORK%/CosmoGasVision}/stage1a_results/${RUN_TAG}"
mkdir -p "${DEST}"

# 1. MLflow file-store — required for downstream metric replay.
[[ -d mlflow ]] || { echo "FATAL: pipeline produced no mlflow/ store" >&2; exit 2; }
cp -r mlflow "${DEST}/"

# 2. Stage 1a checkpoints — NOTE: train_pretrain does NOT write step_*.pt
# checkpoints by default (it is a feasibility-loop; the run lives via M1/M2/M3
# metrics + tags in MLflow). Skip checkpoint PCV for this sweep; the
# downstream consumer is the seed-aggregator (Wilcoxon harness, post-Juno),
# which reads MLflow metrics not .pt files. If a future Stage 1a variant adds
# checkpoint emission, reinstate the [[ -d "${CKPT_SRC}" ]] PCV block from
# submit_juno_stage2b.sh lines 199-204.
echo "=== Stage 1a (1b) consumes MLflow metrics, not checkpoints ==="
echo "  pretrain loop logs L_pre/M1/M2/M3 to file-store; no .pt files emitted"

# 3. Verify what was copied — visible in the .out log for downstream debugging.
echo "=== copied artifacts ==="
echo "--- ${DEST}/mlflow ---"
find "${DEST}/mlflow" -maxdepth 4 -type d | head -10

# 4. Cleanup
cd "${HOME}"
rm -rf "${RUN_DIR}"
echo "=== done, results at ${DEST} ==="
