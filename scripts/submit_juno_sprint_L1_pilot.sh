#!/bin/bash
#SBATCH --job-name=sprint-L1-pilot-fullgradnorm
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END
#SBATCH --output=sprint-L1-pilot-juno-%j.out
#SBATCH --error=sprint-L1-pilot-juno-%j.err

# Sprint-L1 [D-60] gate-8 follow-on — P1-T1 5k full-GradNorm pilot on Juno A30.
# Design doc: experiments/nerf/design/sprint_L1_direct_pf_loss.md (v2, NON-PROVISIONAL).
# PI gate-8 absorption verdict authorized this pilot (NON-PROVISIONAL per R15 (c)).
# User directive 2026-05-21: NO SAGEMAKER, Juno-only.
#
# Why this pilot exists
# ---------------------
# Job 201587 (P1-T3 50k under simplified GradNorm) retired at step 2271 on
# R-g gradnorm_runaway (w_ratio=0.000867, w_pf=1.998 saturated). The simplified
# loss-magnitude proxy G_i = w_i * |L_i| was falsified at T3 scale per
# Chen+ 2018 §3. The new --gradnorm-full CLI knob (pipeline.py:230, commit
# c470e75) exposes the full second-order Chen+ 2018 path. Before another
# ~$11 P1-T3 50k commitment, PI requires this ~$0.30 P1-T1 5k pilot to
# characterize full-GradNorm step-rate and stability under the FFT ×
# volume_render compute graph.
#
# Pinned configuration (PI gate-8 scope)
# --------------------------------------
#   --gradnorm-full                  (pipeline.py:230; full Chen+ 2018 path)
#   --enable-l1-pf-loss              (pipeline.py:206)
#   --physics 1                      (P1, K1-absorbing tier)
#   --n_rays 64                      (T1 scale — pilot only, NOT T3; primary
#                                     deliverable is step-rate + ratio-history,
#                                     NOT gate-pass evidence per K1 noise floor)
#   --seed 0
#   --microbatch 256                 (cost-survey schedule; T1 fits easily)
#   --max_steps 5000                 (pilot length)
#   --mean_flux_obs 0.979            (z=0.3 Becker+ 2013 anchor; [D-11])
#   --l1-d24-baseline-tau-mse 0.01   (R-a backstop)
#   --checkpoint_interval 5000       (end-state checkpoint only — pilot does
#                                     not interrogate mid-run weights)
#
# Pilot success criteria (PI gate-8 scope — verbatim)
# ---------------------------------------------------
#   (a) No retire trigger through 5k steps.
#   (b) w_ratio (= w_tau / w_pf) trajectory bounded in [0.01, 100] throughout
#       (well inside the widened R-g 1000:1 + clamp-ceiling rule).
#   (c) Step-rate >= 0.4 steps/s averaged over steps 500-5000.
#
# PASS -> PI authorizes P1-T3 50k full-GradNorm retry.
# FAIL -> re-design (L2 flux-domain or [D-53] supervision-target per v2 §9
#         carry-forward).
#
# Abort guard (PI gate-8 condition — REQUIRED)
# --------------------------------------------
# Step-rate logging at 100-step intervals + abort guard if step-rate
# < 0.2 steps/s at step 1k. At 0.2 steps/s the 5k pilot extrapolates to ~7 hr
# (way over 2-hr cap, and tells us full-GradNorm at T3 is unviable on Juno A30
# wallclock).
#
# Implementation: BACKGROUND awk watcher on the live driver log. Rationale:
# (i) post-hoc grep-after-driver-exit can only react after wallclock burns;
# (ii) the watcher only needs to read line timestamps the pipeline already
# emits via `python -u`; (iii) awk is line-buffered + already on every Juno
# node; (iv) we kill the watcher cleanly at driver exit. The watcher writes
# a sentinel file STEPRATE_ABORT on detected slowdown which the post-driver
# block escalates to a non-zero exit. (Compare: the T3 launcher used
# post-hoc log inspection because retire-on-anomaly was the primary signal;
# here step-rate IS the deliverable, so it must be detected live.)

set -euo pipefail

# --- 0. Load Juno-side .env if present ----------------------------------------
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=/work/sxh240010/CosmoGasVision}"
: "${JUNO_SCRATCH:=/scratch/juno/sxh240010}"

# --- Sprint-L1 pilot pinned hyperparameters (PI gate-8) -----------------------
PHYSICS=1
N_RAYS=64                  # T1 scale (pilot only — NOT T3)
SEED=0
MICROBATCH=256
MAX_STEPS=5000
MEAN_FLUX_OBS=0.979
L1_D24_BASELINE_TAU_MSE=0.01
CHECKPOINT_INTERVAL=5000   # end-state checkpoint only

# Abort-guard tunables (PI gate-8 scope):
STEPRATE_CHECK_AT_STEP=1000     # at step 1k, evaluate rolling step-rate
STEPRATE_MIN_THRESH=0.2         # < 0.2 steps/s -> abort

# --- 1. RUN_TAG generation ----------------------------------------------------
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SHORTHASH=$(cd "${JUNO_WORK}" && git rev-parse --short HEAD 2>/dev/null || echo nogit)
UUID_SUFFIX=$(uuidgen 2>/dev/null | cut -c1-6 \
              || head -c 6 /dev/urandom | base32 | tr '[:upper:]' '[:lower:]' | head -c 6 \
              || openssl rand -hex 3)
RUN_TAG="SprintL1-Pilot-FullGN-P${PHYSICS}-N${N_RAYS}-S${SEED}-${SHORTHASH}-${TIMESTAMP}-${UUID_SUFFIX}"
RUN_NAME="Sprint-L1-pilot-P1T1-fullgradnorm-5k"

if [[ "${RUN_TAG}" =~ [[:space:]] ]] || [[ "${RUN_TAG}" == *[\$\`\;\&\|]* ]]; then
  echo "FATAL: RUN_TAG '${RUN_TAG}' contains forbidden characters." >&2
  exit 1
fi

echo "================================================================="
echo "  Sprint-L1 [D-60] gate-8 follow-on — P1-T1 5k full-GradNorm pilot"
echo "  RUN_TAG=${RUN_TAG}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  Juno A30 partition, 2:00:00 wallclock"
echo "  User directive 2026-05-21: NO SAGEMAKER, Juno-only."
echo "================================================================="

DEST="${JUNO_WORK}/cloud_runs/sprint_L1_pilot_fullgradnorm_${SLURM_JOB_ID}"
if [[ -d "${DEST}" ]]; then
  echo "FATAL: results dir already exists: ${DEST}" >&2
  exit 1
fi
mkdir -p "${DEST}"

# --- 2. Uncommitted-tracked-content guard -------------------------------------
if [[ -n "$(cd "${JUNO_WORK}" && git status --porcelain --untracked-files=no)" ]]; then
  echo "FATAL: JUNO_WORK has uncommitted CONTENT changes; SHORTHASH=${SHORTHASH} would mislead." >&2
  (cd "${JUNO_WORK}" && git status --short --untracked-files=no) >&2
  exit 2
fi

# --- 3. Code refresh + env activation -----------------------------------------
cd "${JUNO_WORK}"
git pull origin exp/nerf || echo "[submit] git pull non-fatal warning (continuing on local HEAD ${SHORTHASH})"

# --- 3a. Data symlinks --------------------------------------------------------
if [[ ! -e "${JUNO_WORK}/Sherwood" ]]; then
  ln -s "${JUNO_SCRATCH}/sherwood" "${JUNO_WORK}/Sherwood"
  echo "[submit] symlinked ${JUNO_WORK}/Sherwood -> ${JUNO_SCRATCH}/sherwood"
fi
if [[ ! -e "${JUNO_WORK}/SherwoodIGM_gal" ]]; then
  ln -s "${JUNO_SCRATCH}/SherwoodIGM_gal" "${JUNO_WORK}/SherwoodIGM_gal"
fi
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
echo "GradNorm: FULL (Chen+ 2018 second-order; --gradnorm-full active)"
echo "Step-rate abort guard: < ${STEPRATE_MIN_THRESH} steps/s at step ${STEPRATE_CHECK_AT_STEP}"

export NERF_RUN_NAME="${RUN_NAME}"
export NERF_RUN_TAG="${RUN_TAG}"

LOG="${DEST}/driver.log"
SENTINEL_ABORT="${DEST}/STEPRATE_ABORT"
WATCHER_LOG="${DEST}/steprate_watcher.log"
touch "${LOG}"  # make sure file exists before watcher starts tailing

# --- 5. Background step-rate watcher (PI gate-8 abort guard) ------------------
# awk watches the driver log; when it sees the first "Step N/" line, it records
# its wall-clock timestamp; when it sees "Step ${STEPRATE_CHECK_AT_STEP}/", it
# computes (1000 - first_step) / (now - first_time). If < threshold, it touches
# SENTINEL_ABORT and exits, the trap below kills the driver.
( tail -F -n +1 "${LOG}" 2>/dev/null \
    | awk -v target="${STEPRATE_CHECK_AT_STEP}" \
          -v thresh="${STEPRATE_MIN_THRESH}" \
          -v sentinel="${SENTINEL_ABORT}" \
          -v wlog="${WATCHER_LOG}" '
      BEGIN { first_step = -1; first_time = 0; done = 0 }
      done == 1 { next }
      /^Step [0-9]+\// {
        # Extract step number from "Step N/M | ..."
        split($2, a, "/")
        step = a[1] + 0
        now = systime()
        if (first_step < 0) {
          first_step = step
          first_time = now
          print "[watcher] first_step=" step " at t=" now > wlog
          fflush(wlog)
        } else if (step >= target) {
          elapsed = now - first_time
          if (elapsed <= 0) elapsed = 1
          rate = (step - first_step) / elapsed
          printf("[watcher] step=%d elapsed=%ds rate=%.3f steps/s thresh=%.3f\n",
                 step, elapsed, rate, thresh) > wlog
          fflush(wlog)
          if (rate + 0 < thresh + 0) {
            printf("[watcher] ABORT: step-rate %.3f < %.3f at step %d\n",
                   rate, thresh, step) > wlog
            fflush(wlog)
            # Touch sentinel; the bash trap below kills the driver.
            system("touch " sentinel)
          }
          done = 1
        }
      }
    ' ) &
WATCHER_PID=$!
echo "[submit] step-rate watcher started, pid=${WATCHER_PID}"

# Ensure the watcher is cleaned up on any exit path.
cleanup_watcher() {
  if kill -0 "${WATCHER_PID}" 2>/dev/null; then
    kill "${WATCHER_PID}" 2>/dev/null || true
    wait "${WATCHER_PID}" 2>/dev/null || true
  fi
}
trap cleanup_watcher EXIT

# --- 6. Sentinel-driven driver-killer (runs in background) --------------------
# Polls every 10s; if SENTINEL_ABORT appears, kills the driver process group.
DRIVER_KILLED_BY_WATCHER=0
( while true; do
    if [[ -f "${SENTINEL_ABORT}" ]]; then
      echo "[killer] SENTINEL_ABORT detected; killing driver PGID=${DRIVER_PGID:-unknown}"
      if [[ -n "${DRIVER_PGID:-}" ]]; then
        kill -TERM -"${DRIVER_PGID}" 2>/dev/null || true
        sleep 5
        kill -KILL -"${DRIVER_PGID}" 2>/dev/null || true
      fi
      exit 0
    fi
    sleep 10
  done ) &
KILLER_PID=$!

# --- 7. Main driver invocation ------------------------------------------------
echo "=== driver invocation: experiments/nerf/pipeline.py (sprint-L1 pilot --gradnorm-full) ==="
set +e
# setsid puts the driver in its own process group so the killer can signal
# the whole tree (python + any CUDA helper threads).
setsid python -u experiments/nerf/pipeline.py \
    --gradnorm-full \
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
    2>&1 | tee "${LOG}" &
DRIVER_PID=$!
DRIVER_PGID=$(ps -o pgid= -p "${DRIVER_PID}" 2>/dev/null | tr -d ' ')
export DRIVER_PGID
wait "${DRIVER_PID}"
DRIVER_EXIT=$?
set -e

# Stop killer + watcher.
kill "${KILLER_PID}" 2>/dev/null || true
wait "${KILLER_PID}" 2>/dev/null || true
cleanup_watcher
trap - EXIT

echo "[submit] DRIVER_EXIT=${DRIVER_EXIT}"
if [[ -f "${SENTINEL_ABORT}" ]]; then
  DRIVER_KILLED_BY_WATCHER=1
  echo "[submit] STEPRATE_ABORT sentinel present; pilot failed criterion (c)."
fi

if [[ -f "${WATCHER_LOG}" ]]; then
  echo "=== step-rate watcher log ==="
  cat "${WATCHER_LOG}"
fi

# --- 8. Retire-marker guard (criterion (a)) -----------------------------------
echo "=== retire-marker check (pilot criterion (a)) ==="
RETIRE_JSON="${DEST}/checkpoints/retire.json"
if [[ -f "${RETIRE_JSON}" ]]; then
  echo "[pilot] RETIRE: ${RETIRE_JSON} present — in-pipeline retire fired."
  cat "${RETIRE_JSON}"
  RETIRE_FIRED=1
else
  echo "[pilot] OK: no retire.json (criterion (a) satisfied)."
  RETIRE_FIRED=0
fi

# --- 9. w_ratio + loss-finiteness post-hoc scan (criterion (b)) ---------------
echo "=== last 100 log lines ==="
tail -n 100 "${LOG}" || true
echo "---"
echo "=== grep: w_ratio / w_tau / w_pf / loss_pf trail (criterion (b) inspection) ==="
grep -E "(w_ratio|w_tau|w_pf|loss_pf|loss_tau)" "${LOG}" | tail -n 60 || \
  echo "[pilot] WARN: no w_ratio / loss_pf lines matched — log format may have changed."

# --- 10. Artifact PCV ---------------------------------------------------------
echo "=== artifact PCV ==="
if [[ -d "${DEST}/checkpoints" ]]; then
  ls -la "${DEST}/checkpoints/"
  CKPT_COUNT=$(find "${DEST}/checkpoints" -name "*.pt" 2>/dev/null | wc -l)
  echo "[PCV] checkpoints: ${CKPT_COUNT} *.pt files in ${DEST}/checkpoints/"
else
  echo "[PCV] WARN: no checkpoint dir at ${DEST}/checkpoints"
fi

# --- 11. MLflow tag injection -------------------------------------------------
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
    ("gradnorm_variant", "full"),
    ("sprint_kind", "pilot"),
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

# --- 12. Final summary banner -------------------------------------------------
echo "================================================================="
echo "  Sprint-L1 pilot summary"
echo "================================================================="
echo "  RUN_TAG                    = ${RUN_TAG}"
echo "  RUN_NAME                   = ${RUN_NAME}"
echo "  DRIVER_EXIT                = ${DRIVER_EXIT}"
echo "  RETIRE_FIRED               = ${RETIRE_FIRED}"
echo "  DRIVER_KILLED_BY_WATCHER   = ${DRIVER_KILLED_BY_WATCHER}"
echo "  DEST                       = ${DEST}"
echo "  LOG                        = ${LOG}"
echo "================================================================="

# --- 13. Exit-code routing (pilot criteria) -----------------------------------
# Hard-fail iff:
#   - retire.json present (criterion (a) violation), OR
#   - step-rate watcher tripped (criterion (c) violation).
# Criterion (b) (w_ratio bounded) is a human-inspection deliverable from the
# tail/grep block above; the in-pipeline R-g retire-check is the live guard
# (1000:1 + clamp-ceiling rule), and a R-g retire would already trip criterion
# (a) via retire.json.
if [[ "${RETIRE_FIRED}" -eq 1 ]]; then
  echo "FATAL: pilot criterion (a) failed (retire.json present)." >&2
  exit 3
fi
if [[ "${DRIVER_KILLED_BY_WATCHER}" -eq 1 ]]; then
  echo "FATAL: pilot criterion (c) failed (step-rate < ${STEPRATE_MIN_THRESH} steps/s at step ${STEPRATE_CHECK_AT_STEP})." >&2
  exit 4
fi

echo "=== done; results at ${DEST} ==="
