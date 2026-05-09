#!/bin/bash
#SBATCH --job-name=Stage2b-AblationS5S7
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=stage2b-ablation-s5s7-%j.out
#SBATCH --error=stage2b-ablation-s5s7-%j.err

# =============================================================================
# Stage 2b ablation S5/S7 dispatch — defense-of-novelty bundle
# =============================================================================
#
# Per PI ruling 2 (minimum bundle = A + B + C; D optional if quota allows):
#   Cell A (full):           --tau_max=10.0 (default), no --disable_dla_mask
#                            -> published baseline, [D-24] log1p+cap+mask
#   Cell B (no-cap):         --tau_max=1e9
#                            -> numerical no-op of the cap; isolates cap effect
#   Cell C (no-mask):        --disable_dla_mask (default --tau_max=10.0)
#                            -> pre-[D-24] supervision regime; isolates mask
#   Cell D (no-cap-no-mask): --tau_max=1e9 --disable_dla_mask
#                            -> joint ablation (optional 4th cell)
#
# All four cells share P1 / T2 / seed=1 to compare apples-to-apples with the
# existing Batch 2 P1-T2 published baseline. Sequential dispatch in one sbatch
# job for simplicity (~6 hr/cell × 4 = ~24 hr; fits the 1-day wallclock cap).
#
# IMPORTANT — anchor decision: --mean_flux_obs=0.877 is the existing pipeline
# default (the [D-34]-noted broken anchor). We deliberately KEEP 0.877 here so
# this defense-of-novelty bundle is comparable to the existing P1-T2 baseline.
# The corrected anchor (0.979) belongs in the publication-class re-train, NOT
# in this ablation. Do not change this without re-running the baseline.
#
# Patch dependency: pipeline.py must expose --disable_dla_mask (action="store_true")
# and --tau_max (float, default 10.0). Verified at HEAD on 2026-05-08:
#   - line ~93:  --tau_max
#   - line ~101: --disable_dla_mask
#   - line ~282: mask override block (mask_no_dla_profile = ones_like when flag)
#   - line ~376: log_params includes tau_max + disable_dla_mask + dynamic loss_form
#
# PCV (Producer-Consumer Verification): every cell hard-fails on missing MLflow
# store or zero checkpoints. Any cell failure aborts the whole sbatch via
# `set -euo pipefail`. The cleanup step rms RUN_DIR only after DEST has been
# verified for that cell.
# =============================================================================

set -euo pipefail

if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=/work/sxh240010/CosmoGasVision}"
: "${JUNO_SCRATCH:=/scratch/juno/sxh240010}"

# Common training params (T2 row of [D-23] tier table, P1, seed=1)
PHYSICS_ID=1
N_RAYS=256
MICROBATCH=1024
ACCUM_STEPS=1
MAX_STEPS=25000
WARMUP_STEPS=1000
SEED=1
MEAN_FLUX_OBS=0.877  # [D-34] broken-anchor baseline; KEEP for apples-to-apples

# Cells to run. Comment out Cell D ("no-cap-no-mask") to drop to the 3-cell
# minimum bundle if quota or wallclock pressure demands it.
CELLS=(full no-cap no-mask no-cap-no-mask)

# Build per-cell flag suffixes. Cell A = no extra flags.
declare -A CELL_FLAGS
CELL_FLAGS[full]=""
CELL_FLAGS[no-cap]="--tau_max 1e9"
CELL_FLAGS[no-mask]="--disable_dla_mask"
CELL_FLAGS[no-cap-no-mask]="--tau_max 1e9 --disable_dla_mask"

# Short hash of the source tree at submission time, for run-name traceability.
SHORTHASH=$(cd "${JUNO_WORK}" && git rev-parse --short HEAD 2>/dev/null || echo nogit)

source "${JUNO_WORK}/.venv/bin/activate"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export GIT_PYTHON_REFRESH=quiet

echo "=== sbatch launched ==="
echo "job_id=${SLURM_JOB_ID:-LOCAL} host=$(hostname) shorthash=${SHORTHASH}"
nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv | tail -1
echo "cells=${CELLS[*]}"

# =============================================================================
# Per-cell loop
# =============================================================================
for CELL in "${CELLS[@]}"; do
  echo ""
  echo "============================================================"
  echo "=== CELL: ${CELL}  (start: $(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
  echo "============================================================"

  TIMESTAMP=$(date +%Y%m%d-%H%M%S)
  RUN_NAME="Stage2b-AblationS5S7-P${PHYSICS_ID}-T2-S${SEED}-${CELL}-${TIMESTAMP}-${SHORTHASH}"
  RUN_TAG="ablation-s5s7-${CELL}-$(date +%s)-$(uuidgen | cut -c1-6)"
  RUN_DIR="${JUNO_SCRATCH}/stage2b/${RUN_TAG}"

  mkdir -p "${RUN_DIR}"
  cd "${RUN_DIR}"

  cp -r "${JUNO_WORK}"/{src,experiments,scripts,pyproject.toml,uv.lock} .
  ln -s "${JUNO_SCRATCH}/sherwood" Sherwood

  export MLFLOW_TRACKING_URI="file://${RUN_DIR}/mlflow"

  echo "RUN_NAME=${RUN_NAME}"
  echo "RUN_TAG=${RUN_TAG}"
  echo "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"
  echo "extra_flags=${CELL_FLAGS[${CELL}]}"

  # -------------------------------------------------------------------------
  # Compute — pipeline.py sets the canonical Stage 2b tags (model_type, stage,
  # physics_id, redshift, n_rays, seed, ablation_matrix). The post-run tagger
  # below adds {stage=2b-ablation-s5s7, cell=<cell>} for this bundle.
  # -------------------------------------------------------------------------
  python -u experiments/nerf/pipeline.py \
      --physics "${PHYSICS_ID}" \
      --n_rays "${N_RAYS}" \
      --microbatch "${MICROBATCH}" \
      --accum_steps "${ACCUM_STEPS}" \
      --max_steps "${MAX_STEPS}" \
      --warmup_steps "${WARMUP_STEPS}" \
      --seed "${SEED}" \
      --mean_flux_obs "${MEAN_FLUX_OBS}" \
      --run_name "${RUN_NAME}" \
      ${CELL_FLAGS[${CELL}]}

  # -------------------------------------------------------------------------
  # Post-run tag injection. pipeline.py's set_tags dict is fixed; the ablation
  # bundle needs an extra `cell` tag and a `stage=2b-ablation-s5s7` override
  # so the host-side replay can discriminate ablation runs from production.
  # We hit the local file store directly via MlflowClient — no network, no
  # auth, no race condition.
  # -------------------------------------------------------------------------
  python -u - <<PYEOF
import os
import sys
from mlflow.tracking import MlflowClient

uri = os.environ["MLFLOW_TRACKING_URI"]
client = MlflowClient(tracking_uri=uri)
# CosmoGasVision/NeRF is the canonical experiment name (see CLAUDE.md).
exp = client.get_experiment_by_name("CosmoGasVision/NeRF")
if exp is None:
    print("FATAL: CosmoGasVision/NeRF experiment not found in file store.", file=sys.stderr)
    sys.exit(10)
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="tags.\`mlflow.runName\` = '${RUN_NAME}'",
    max_results=1,
)
if not runs:
    print("FATAL: run '${RUN_NAME}' not found after pipeline exit.", file=sys.stderr)
    sys.exit(11)
run = runs[0]
client.set_tag(run.info.run_id, "stage", "2b-ablation-s5s7")
client.set_tag(run.info.run_id, "cell", "${CELL}")
client.set_tag(run.info.run_id, "ablation_bundle", "S5S7")
print(f"[ablation-tagger] run_id={run.info.run_id} cell=${CELL} stage=2b-ablation-s5s7 OK")
PYEOF

  # -------------------------------------------------------------------------
  # PCV copy-out — same contract as scripts/submit_juno_stage2b.sh.
  # Hard-fail on missing mlflow store or zero checkpoints; do NOT silence
  # these checks. If a cell fails verification, abort the whole sbatch
  # rather than continuing into a polluted bundle.
  # -------------------------------------------------------------------------
  DEST="${JUNO_WORK%/CosmoGasVision}/stage2b_results/${RUN_TAG}"
  mkdir -p "${DEST}"

  [[ -d mlflow ]] || { echo "FATAL: cell ${CELL} produced no mlflow/ store" >&2; exit 2; }
  cp -r mlflow "${DEST}/"

  CKPT_SRC="experiments/nerf/artifacts/checkpoints"
  [[ -d "${CKPT_SRC}" ]] || { echo "FATAL: cell ${CELL} no checkpoint dir at ${CKPT_SRC}" >&2; exit 3; }
  N_CKPT=$(ls -1 "${CKPT_SRC}"/*.pt 2>/dev/null | wc -l)
  [[ "${N_CKPT}" -gt 0 ]] || { echo "FATAL: cell ${CELL} ${CKPT_SRC} contains zero *.pt files" >&2; exit 4; }
  mkdir -p "${DEST}/checkpoints"
  cp "${CKPT_SRC}"/*.pt "${DEST}/checkpoints/"

  echo "=== cell ${CELL} copied artifacts ==="
  echo "--- ${DEST}/mlflow ---"
  find "${DEST}/mlflow" -maxdepth 4 -type d | head -10
  echo "--- ${DEST}/checkpoints (count=${N_CKPT}) ---"
  ls -la "${DEST}/checkpoints/"
  echo "RUN_NAME_DONE=${RUN_NAME}"
  echo "RUN_TAG_DONE=${RUN_TAG}"

  cd "${HOME}"
  rm -rf "${RUN_DIR}"
  echo "=== cell ${CELL} done at $(date -u +%Y-%m-%dT%H:%M:%SZ); results at ${DEST} ==="
done

echo ""
echo "============================================================"
echo "=== ALL CELLS COMPLETE ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
echo "============================================================"
echo "Per-cell DEST under: ${JUNO_WORK%/CosmoGasVision}/stage2b_results/"
echo "Cells: ${CELLS[*]}"
