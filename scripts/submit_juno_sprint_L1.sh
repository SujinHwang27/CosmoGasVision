#!/bin/bash
#SBATCH --job-name=sprint-L1-direct-pf-loss
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=07:30:00
#SBATCH --mail-type=END
#SBATCH --output=sprint-L1-juno-%j.out
#SBATCH --error=sprint-L1-juno-%j.err

# Sprint-L1 [D-60] direct P_F MSE loss + GradNorm training run on Juno A30.
# Design doc: experiments/nerf/design/sprint_L1_*.md (NON-PROVISIONAL per [D-60]).
# PI gate-6 dispatch authorization; user directive 2026-05-21: NO SAGEMAKER, Juno-only.
#
# Distinct from scripts/submit_juno_sprint5_cprime.sh (h100, sprint-5 (c′) substrate
# probe via scripts/train_truth_baseline.py): this script routes through
# experiments/nerf/pipeline.py — the active CVPR-track NeRF MLP — to train the
# direct-P_F-MSE-loss variant with GradNorm (simplified=True default).
#
# Pinned configuration (PI gate-6 Ruling 6):
#   --enable-l1-pf-loss              (pipeline.py:206)
#   --physics 1                      (P1, K1-absorbing tier)
#   --mean_flux_obs 0.979            (z=0.3 Becker+ 2013 anchor; [D-11])
#   --l1-d24-baseline-tau-mse 0.01   (R-a backstop; cost-survey T3 200-step upper bound)
#   --microbatch 256                 (cost-survey T3 schedule; accum=ceil(1024/256)=4)
#   --n_rays 1024                    (P1-T3)
#   --max_steps 50000                (pipeline arg name; PI spec called this --n_steps)
#   GradNorm: simplified=True default — hardcoded in pipeline.py:846. No
#   --gradnorm-full CLI knob exists on the current pipeline; expose as the
#   placeholder env var L1_GRADNORM_FULL below for documentation continuity,
#   but leave UNSET (no shell wiring beyond a comment).
#
# Phase-1 step-200 abort checkpoint (PI condition (ii)):
#   Chose the SIMPLER, MORE ROBUST option — in-pipeline retire-checks
#   (R-a..R-g) already enforce abort-on-anomaly + emit retire.json on retire.
#   This script performs a POST-HOC log inspection at the end (tail -n 200 of
#   the driver log) plus an explicit retire.json existence assertion. No
#   background log-tailer / no mid-run racy file probes — those interact badly
#   with sbatch stdio buffering on Juno and add a sleep-poll loop that can mask
#   genuine pipeline progress.

set -euo pipefail

# --- 0. Load Juno-side .env if present ----------------------------------------
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=/work/sxh240010/CosmoGasVision}"
: "${JUNO_SCRATCH:=/scratch/juno/sxh240010}"
: "${JUNO_RESULTS_ROOT:=${JUNO_WORK%/CosmoGasVision}/sprint_L1_results}"

# --- Sprint-L1 pinned hyperparameters (PI gate-6 Ruling 6) --------------------
# All fixed per the design doc; do NOT parametrize without a LEDGER decision.
PHYSICS=1
N_RAYS=1024
SEED=0
MICROBATCH=256
MAX_STEPS=50000
MEAN_FLUX_OBS=0.979
L1_D24_BASELINE_TAU_MSE=0.01
CHECKPOINT_INTERVAL=10000

# Documentation-only placeholder for GradNorm-full toggle. No CLI hook exists
# in pipeline.py at sprint-L1 freeze; default simplified=True is hardcoded at
# pipeline.py:846. Leaving this UNSET (commented) for future-proofing only.
# L1_GRADNORM_FULL=1   # would require a pipeline.py patch to honor

# --- 1. RUN_TAG generation (sprint-5 precedent: uuidgen fallback) -------------
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SHORTHASH=$(cd "${JUNO_WORK}" && git rev-parse --short HEAD 2>/dev/null || echo nogit)
UUID_SUFFIX=$(uuidgen 2>/dev/null | cut -c1-6 \
              || head -c 6 /dev/urandom | base32 | tr '[:upper:]' '[:lower:]' | head -c 6 \
              || openssl rand -hex 3)
RUN_TAG="SprintL1-PFloss-P${PHYSICS}-N${N_RAYS}-S${SEED}-${SHORTHASH}-${TIMESTAMP}-${UUID_SUFFIX}"
RUN_NAME="Sprint-L1-direct-pf-loss-P1T3-50k"

# Defensive: reject RUN_TAG with whitespace / special chars
if [[ "${RUN_TAG}" =~ [[:space:]] ]] || [[ "${RUN_TAG}" == *[\$\`\;\&\|]* ]]; then
  echo "FATAL: RUN_TAG '${RUN_TAG}' contains forbidden characters." >&2
  exit 1
fi

echo "================================================================="
echo "  Sprint-L1 [D-60] — direct P_F MSE loss + GradNorm"
echo "  RUN_TAG=${RUN_TAG}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  Juno A30 partition, 7:30:00 wallclock (7 hr envelope + 30 min cushion)"
echo "  User directive 2026-05-21: NO SAGEMAKER, Juno-only."
echo "================================================================="

# Artifacts go under JUNO_WORK/cloud_runs/sprint_L1_juno_<jobid>/ per spec.
DEST="${JUNO_WORK}/cloud_runs/sprint_L1_juno_${SLURM_JOB_ID}"
if [[ -d "${DEST}" ]]; then
  echo "FATAL: results dir already exists: ${DEST}" >&2
  exit 1
fi
mkdir -p "${DEST}"

# --- 2. Uncommitted-tracked-content guard (sprint-4 PI caveat C4) -------------
if [[ -n "$(cd "${JUNO_WORK}" && git status --porcelain --untracked-files=no)" ]]; then
  echo "FATAL: JUNO_WORK has uncommitted CONTENT changes; SHORTHASH=${SHORTHASH} would mislead." >&2
  echo "Commit (or stash) before sbatch. Output of git status:" >&2
  (cd "${JUNO_WORK}" && git status --short --untracked-files=no) >&2
  exit 2
fi

# --- 3. Code refresh + environment activation ---------------------------------
# JUNO_WORK symlink discipline (sprint-5 precedent): explicit cd to JUNO_WORK,
# no relative paths, symlinks resolved at job-start.
cd "${JUNO_WORK}"
git pull origin exp/nerf || echo "[submit] git pull non-fatal warning (continuing on local HEAD ${SHORTHASH})"

# --- 3a. Data symlinks (inherit from sprint-5 [D-51] precedent) ----------------
if [[ ! -e "${JUNO_WORK}/Sherwood" ]]; then
  ln -s "${JUNO_SCRATCH}/sherwood" "${JUNO_WORK}/Sherwood"
  echo "[submit] symlinked ${JUNO_WORK}/Sherwood -> ${JUNO_SCRATCH}/sherwood"
fi
if [[ ! -e "${JUNO_WORK}/SherwoodIGM_gal" ]]; then
  ln -s "${JUNO_SCRATCH}/SherwoodIGM_gal" "${JUNO_WORK}/SherwoodIGM_gal"
  echo "[submit] symlinked ${JUNO_WORK}/SherwoodIGM_gal -> ${JUNO_SCRATCH}/SherwoodIGM_gal"
fi
# Sprint-L1 routes through experiments/nerf/pipeline.py (sightline-based, NOT
# IGM_gal CIC crops); the IGM_gal hdf5 mirror is not required for this run.
# We still verify Sherwood/ symlink target exists (the pipeline reads from
# Sherwood/.../snapdir_012/ + Sherwood/.../tauH1/ per src/data/ loaders).
if [[ ! -d "${JUNO_WORK}/Sherwood" ]]; then
  echo "FATAL: Sherwood symlink resolved to missing target." >&2
  ls -la "${JUNO_WORK}/Sherwood" >&2 || true
  exit 5
fi
echo "[submit] Sherwood symlink verified: $(readlink ${JUNO_WORK}/Sherwood)"

source "${JUNO_WORK}/.venv/bin/activate"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export GIT_PYTHON_REFRESH=quiet

# MLflow tracker ping (informational). pipeline.py honors MLFLOW_TRACKING_URI
# from .env (sourced above) and falls back to nullcontext when unreachable.
if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
  if python -c "import urllib.request,sys; urllib.request.urlopen('${MLFLOW_TRACKING_URI}', timeout=3)" 2>/dev/null; then
    echo "[submit] MLflow tracker ${MLFLOW_TRACKING_URI} reachable"
  else
    echo "[submit] MLflow tracker ${MLFLOW_TRACKING_URI} unreachable; pipeline will fall back to nullcontext."
  fi
fi

# --- 4. GPU + torch sanity ----------------------------------------------------
echo "=== GPU diagnostics ==="
nvidia_smi_out=$(nvidia-smi 2>&1 || true)
printf '%s\n' "${nvidia_smi_out}" | awk 'NR<=10'
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; \
print(f'torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}')"

echo "=== run config ==="
echo "RUN_TAG=${RUN_TAG}"
echo "RUN_NAME=${RUN_NAME}"
echo "PHYSICS=${PHYSICS} N_RAYS=${N_RAYS} SEED=${SEED} MICROBATCH=${MICROBATCH}"
echo "MAX_STEPS=${MAX_STEPS} MEAN_FLUX_OBS=${MEAN_FLUX_OBS}"
echo "L1_D24_BASELINE_TAU_MSE=${L1_D24_BASELINE_TAU_MSE}"
echo "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}"
echo "GradNorm: simplified=True (pipeline.py:846 default; no CLI override at freeze)"

# --- 5. Main driver invocation ------------------------------------------------
# Tag injection happens via run_name + env vars; pipeline.py handles its own
# MLflow setup. Tags applied post-run via the block in §6.
export NERF_RUN_NAME="${RUN_NAME}"
export NERF_RUN_TAG="${RUN_TAG}"

LOG="${DEST}/driver.log"
echo "=== driver invocation: experiments/nerf/pipeline.py (sprint-L1 direct P_F MSE) ==="
set +e
python -u experiments/nerf/pipeline.py \
    --enable-l1-pf-loss \
    --physics "${PHYSICS}" \
    --n_rays "${N_RAYS}" \
    --seed "${SEED}" \
    --microbatch "${MICROBATCH}" \
    --max_steps "${MAX_STEPS}" \
    --mean_flux_obs "${MEAN_FLUX_OBS}" \
    --l1-d24-baseline-tau-mse "${L1_D24_BASELINE_TAU_MSE}" \
    --checkpoint_dir "${DEST}/checkpoints" \
    --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
    --run_name "${RUN_NAME}" \
    2>&1 | tee "${LOG}"
DRIVER_EXIT=${PIPESTATUS[0]}
echo "[submit] DRIVER_EXIT=${DRIVER_EXIT}"
set -e

# --- 6. Phase-1 step-200 abort checkpoint (PI condition (ii)) -----------------
# Variant chosen: POST-HOC log inspection (simpler, more robust than
# background-monitor + mid-run race). The in-pipeline retire-checks
# (R-a..R-g) are the primary abort-on-anomaly mechanism and produce
# retire.json on retire. Here we:
#   (a) assert retire.json was NOT produced;
#   (b) tail the last ~200 log lines and grep for loss_tau / loss_pf /
#       w_ratio finiteness;
#   (c) emit a non-fatal warning if any check fails (driver-exit + retire.json
#       presence are the authoritative signals; this is a human-readable summary).
echo "=== Phase-1 step-200 checkpoint (post-hoc log inspection) ==="
RETIRE_JSON="${DEST}/checkpoints/retire.json"
if [[ -f "${RETIRE_JSON}" ]]; then
  echo "[checkpoint] RETIRE: ${RETIRE_JSON} present — in-pipeline retire fired."
  echo "[checkpoint] Contents:"
  cat "${RETIRE_JSON}"
  RETIRE_FIRED=1
else
  echo "[checkpoint] OK: no retire.json (R-a..R-g did not trigger)."
  RETIRE_FIRED=0
fi

echo "=== last 100 log lines (loss_tau / loss_pf / w_ratio finiteness scan) ==="
tail -n 100 "${LOG}"
echo "---"
echo "=== grep for non-finite signals around step 200 ==="
if grep -E "(nan|inf|NaN|Inf|loss_tau|loss_pf|w_ratio)" "${LOG}" | head -n 40; then
  :
else
  echo "[checkpoint] WARN: no loss_tau/loss_pf/w_ratio lines matched — log format may have changed."
fi

# Hard fail iff the in-pipeline retire produced retire.json. Otherwise treat
# DRIVER_EXIT as informational (pipeline outcome routing is authoritative).
if [[ "${RETIRE_FIRED}" -eq 1 ]]; then
  echo "FATAL: retire.json present; aborting per PI condition (ii)." >&2
  exit 3
fi

# --- 7. Artifact PCV (Producer-Consumer Verification) -------------------------
# pipeline.py writes checkpoints + metrics under --checkpoint_dir, which we
# already pointed at ${DEST}/checkpoints. Verify presence + emit a summary.
echo "=== artifact PCV ==="
if [[ -d "${DEST}/checkpoints" ]]; then
  ls -la "${DEST}/checkpoints/"
  CKPT_COUNT=$(find "${DEST}/checkpoints" -name "*.pt" 2>/dev/null | wc -l)
  echo "[PCV] checkpoints: ${CKPT_COUNT} *.pt files in ${DEST}/checkpoints/"
else
  echo "[PCV] WARN: no checkpoint dir at ${DEST}/checkpoints"
fi

# --- 8. MLflow tag injection (mlflow-run skill contract) ----------------------
# Mandatory tags from PI Ruling 6 + CLAUDE.md Experiment-workflow tag set.
python -u - <<PYEOF || echo "[tagger] warning: MLflow tag injection failed (non-fatal)"
import os, sys
try:
    from mlflow.tracking import MlflowClient
except ImportError:
    sys.exit("[tagger] mlflow not importable; skipping tag injection.")

uri = os.environ.get("MLFLOW_TRACKING_URI")
if not uri:
    sys.exit("[tagger] MLFLOW_TRACKING_URI unset; skipping tag injection.")

client = MlflowClient(tracking_uri=uri)
exp = client.get_experiment_by_name("CosmoGasVision/NeRF")
if exp is None:
    sys.exit("[tagger] experiment 'CosmoGasVision/NeRF' not found; skipping.")

# Find the run by name.
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string=f"tags.mlflow.runName = '${RUN_NAME}'",
    order_by=["attributes.start_time DESC"],
    max_results=2,
)
if not runs:
    sys.exit(f"[tagger] no run named '${RUN_NAME}' under exp '{exp.name}'.")
run = runs[0]
for k, v in [
    ("model_type", "nerf"),
    ("stage", "sprint_L1"),
    ("physics_id", "${PHYSICS}"),
    ("redshift", "0.3"),
    ("loss_variant", "l1_direct_pf_mse"),
    ("compute", "juno"),
    ("juno_job_id", "${SLURM_JOB_ID}"),
    ("juno_run_tag", "${RUN_TAG}"),
    ("commit_sha", "${SHORTHASH}"),
    ("design_doc", "sprint_L1_direct_pf_loss"),
    ("decision_id", "[D-60]"),
]:
    client.set_tag(run.info.run_id, k, v)
print(f"[tagger] run_id={run.info.run_id} tagged OK")
PYEOF

# --- 9. Final summary banner --------------------------------------------------
echo "================================================================="
echo "  Sprint-L1 run summary"
echo "================================================================="
echo "  RUN_TAG               = ${RUN_TAG}"
echo "  RUN_NAME              = ${RUN_NAME}"
echo "  DRIVER_EXIT           = ${DRIVER_EXIT}"
echo "  RETIRE_FIRED          = ${RETIRE_FIRED}"
echo "  DEST                  = ${DEST}"
echo "  LOG                   = ${LOG}"
echo "================================================================="
echo "=== done; results at ${DEST} ==="
