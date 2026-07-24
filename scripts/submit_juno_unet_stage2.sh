#!/bin/bash
# [U-06] Stage-2 U-Net production run on Juno — spec v2 §(b2) + juno-hpc skill.
#
# DO NOT SUBMIT until A10 both-band clearance is confirmed (Juno dispatch
# requires A1-A10 green; coordinator holds the submission gate).
#
# Usage (login node, from ${JUNO_WORK}):
#   bash scripts/submit_juno_unet_stage2.sh preflight     # sync + verify only
#   sbatch --export=ALL,SEED=42 scripts/submit_juno_unet_stage2.sh
# Seed policy (spec v2 S6): primary 42; contingency repeats pinned {142, 242}.
#
#SBATCH --job-name=Stage2-UNet-Full
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# ---- .env conventions (juno-hpc skill) ------------------------------------
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=$HOME/work/CosmoGasVision}"
: "${JUNO_SCRATCH:=$HOME/scratch}"
: "${SEED:=42}"

EXPECTED_BRANCH="exp/unet-inversion"
# Truth cubes are DVC/gitignored — must be rsynced from the host before
# dispatch (see preflight); the job hard-fails if absent.
REQUIRED_ARTIFACTS=(
  "experiments/nerf/artifacts/d75_rescore/cubes/truth_real_192.npy"
  "experiments/unet-inversion/artifacts/stage1/cubes/truth_real_192_p2.npy"
  "experiments/unet-inversion/artifacts/stage2/null_band_val_n200.json"
)

# ---- preflight mode (login node; A9 found work repo STALE at exp/nerf
# c6f3aed -> fetch + checkout this track's HEAD) ------------------------------
if [[ "${1:-}" == "preflight" ]]; then
  cd "${JUNO_WORK}"
  git fetch --all --prune
  git checkout "${EXPECTED_BRANCH}"
  git pull --ff-only
  echo "HEAD: $(git rev-parse --short HEAD) on $(git branch --show-current)"
  # torch cu124 re-pin check (juno-hpc skill: uv sync clobbers the override)
  .venv/bin/python -c "import torch; print('torch', torch.__version__)" \
    || echo "WARN: venv missing/broken — run uv sync + cu124 re-pin"
  # scratch mirror freshness (45-day purge clock)
  ls -d "${JUNO_SCRATCH}/sherwood/Physics1_nofeedback" \
    && find "${JUNO_SCRATCH}/sherwood" -maxdepth 1 -exec touch -a {} + \
    || echo "FATAL: sherwood mirror absent — re-mirror per juno-hpc skill"
  for f in "${REQUIRED_ARTIFACTS[@]}"; do
    [[ -f "${JUNO_WORK}/${f}" ]] && echo "OK  ${f}" \
      || echo "MISSING ${f}  <- rsync from host before dispatch"
  done
  exit 0
fi

# ---- git-uncommitted guard (job side; tracked files only) -------------------
cd "${JUNO_WORK}"
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] \
  || { echo "FATAL: work repo on $(git branch --show-current), expected ${EXPECTED_BRANCH}" >&2; exit 10; }
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "FATAL: uncommitted tracked changes in ${JUNO_WORK} — provenance guard" >&2
  git status --porcelain --untracked-files=no >&2
  exit 11
fi
GIT_HEAD=$(git rev-parse --short HEAD)
for f in "${REQUIRED_ARTIFACTS[@]}"; do
  [[ -f "${JUNO_WORK}/${f}" ]] \
    || { echo "FATAL: required artifact missing: ${f}" >&2; exit 12; }
done

# ---- 1. Copy in (scratch; copy in -> compute -> copy out -> clean) ----------
RUN_TAG="UNetS2-S${SEED}-${GIT_HEAD}-$(date +%s)-$(uuidgen | cut -c1-6)"
RUN_DIR="${JUNO_SCRATCH}/unet_stage2/${RUN_TAG}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
cp -r "${JUNO_WORK}"/{src,experiments,scripts,pyproject.toml,uv.lock} .
ln -s "${JUNO_SCRATCH}/sherwood" Sherwood

# ---- 2. Environment ---------------------------------------------------------
source "${JUNO_WORK}/.venv/bin/activate"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${RUN_DIR}/mlflow"
export GIT_PYTHON_REFRESH=quiet
export COSMOGAS_COMPUTE=juno

# ---- 3. Compute (spec v2 (b2): batch 8, lr 3e-4 cosine+500 warmup,
#         60000 steps / 11 h ceiling, VAL eval per 1000, early stop 10x1e-4,
#         ckpt per 2500 retain best+final+last3, best-VAL of record) ----------
python -u experiments/unet-inversion/pipeline.py train-full \
    --steps 60000 --batch 8 --accum 1 \
    --lr 3e-4 --warmup 500 \
    --eval-every 1000 --ckpt-every 2500 \
    --patience 10 --min-delta 1e-4 \
    --wallclock-hours 11 \
    --seed "${SEED}"

# ---- 4. Copy out + PCV (hard-fail; producer paths cited from pipeline.py) ---
DEST="${JUNO_WORK%/CosmoGasVision}/unet_stage2_results/${RUN_TAG}"
mkdir -p "${DEST}"
STAGE2_OUT="experiments/unet-inversion/artifacts/stage2"

[[ -d mlflow ]] || { echo "FATAL: no mlflow/ file store produced" >&2; exit 2; }
cp -r mlflow "${DEST}/"

CKPT_SRC="${STAGE2_OUT}/checkpoints"
[[ -d "${CKPT_SRC}" ]] || { echo "FATAL: no checkpoint dir ${CKPT_SRC}" >&2; exit 3; }
for req in best_val.pt final.pt; do
  [[ -f "${CKPT_SRC}/${req}" ]] \
    || { echo "FATAL: required checkpoint ${req} absent" >&2; exit 4; }
done
mkdir -p "${DEST}/checkpoints"
cp "${CKPT_SRC}"/*.pt "${DEST}/checkpoints/"

[[ -f "${STAGE2_OUT}/train_full_record.json" ]] \
  || { echo "FATAL: train_full_record.json absent" >&2; exit 5; }
cp "${STAGE2_OUT}/train_full_record.json" "${DEST}/"
CSV=$(ls "${STAGE2_OUT}"/Stage2-TrainFull_metrics.csv 2>/dev/null || true)
[[ -n "${CSV}" ]] || { echo "FATAL: CSV metric mirror absent" >&2; exit 6; }
cp "${CSV}" "${DEST}/"

echo "=== copied artifacts ==="
ls -la "${DEST}" "${DEST}/checkpoints"

# ---- 5. Cleanup --------------------------------------------------------------
cd "${HOME}"
rm -rf "${RUN_DIR}"
echo "=== done: ${DEST} (RUN_TAG=${RUN_TAG}) ==="
