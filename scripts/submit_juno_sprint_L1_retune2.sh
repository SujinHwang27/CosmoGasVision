#!/bin/bash
#SBATCH --job-name=sprint-L1-retune2-sum-to-mean
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --output=sprint-L1-retune2-juno-%j.out
#SBATCH --error=sprint-L1-retune2-juno-%j.err

# Sprint-L1 [D-60] gate-retune-1 absorption — Retune Attempt 2 (Path-gamma)
# on Juno A30. Design doc: experiments/nerf/design/sprint_L1_direct_pf_loss.md.
# Parent decision: PI R15 NON-PROVISIONAL (clause (c)), 2026-05-22, Atom A
# revised post-self-correction.
#
# Why this retune exists
# ----------------------
# Retune Attempt 1 (--lr_max 1e-4) was authored on the hypothesis that the
# gradient-magnitude shock from pilot 201712 was an LR-conditioning issue.
# The revised Attempt 2 (Path-gamma) absorbs the PI re-derivation: the lever
# is the inertial-band REDUCTION in pf_log_mse_loss, NOT the log-domain
# switch (RETRACTED by core-implementer grep audit — the code already
# operates in log10-space inside pf_log_mse_loss). Switching reduction
# 'sum' -> 'mean' divides loss_pf by n_inertial_bins (~6 under default
# log-k binning), shrinking its raw scale to commensurate with loss_tau
# so GradNorm operates in a well-conditioned regime from step 0.
#
# Single lever vs retune1
# -----------------------
#   --pf-log-reduction mean    (NEW Attempt-2 lever; default 'sum' retained
#                              elsewhere for backward-compat)
# All other knobs match retune1 verbatim per R13 / [D-37] symmetric-
# disclosure (single-axis change so any improvement is attributable to
# the reduction change alone).
#
# Pinned configuration (PI pre-committed grid)
# --------------------------------------------
#   --lr_max 1e-4                    (kept from retune1)
#   --warmup_steps 1000              (kept from retune1)
#   --pf-log-reduction mean          (RETUNE LEVER — sum -> mean)
#   --gradnorm-full                  (full Chen+ 2018 path)
#   --enable-l1-pf-loss
#   --physics 1                      (P1, K1-absorbing tier)
#   --n_rays 64                      (T1 scale, unchanged)
#   --seed 0
#   --microbatch 256                 (unchanged)
#   --max_steps 5000                 (per design)
#   --mean_flux_obs 0.979            (z=0.3 Becker+ 2013 anchor; [D-11])
#   --l1-d24-baseline-tau-mse 0.01   (R-a backstop)
#   --checkpoint_interval 5000       (end-state checkpoint only)
#
# Pre-committed falsification criterion (Attempt 2 -> Attempt 3 escalation)
# -------------------------------------------------------------------------
# Per PI 2026-05-22 spec: ``pf/tau < 0.05 at step 500`` (the Option R live
# per-task grad-norm diagnostic ratio) is the trigger for Attempt 3
# dispatch (per-task gradient clipping path). The sbatch trailer greps the
# step-500 per-task grad-norm log line and surfaces a banner so the user's
# next-step authorization is unambiguous.

set -euo pipefail

# --- 0. Load Juno-side .env if present ----------------------------------------
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=/work/sxh240010/CosmoGasVision}"
: "${JUNO_SCRATCH:=/scratch/juno/sxh240010}"

# --- Sprint-L1 retune2 pinned hyperparameters (PI pre-committed grid) ---------
PHYSICS=1
N_RAYS=64                  # T1 scale (unchanged from retune1)
SEED=0
MICROBATCH=256
MAX_STEPS=5000
MEAN_FLUX_OBS=0.979
L1_D24_BASELINE_TAU_MSE=0.01
CHECKPOINT_INTERVAL=5000

# Retune levers (single change vs retune1):
LR_MAX=1e-4                # kept from retune1
WARMUP_STEPS=1000          # kept from retune1
PF_LOG_REDUCTION=mean      # *** NEW LEVER *** sum -> mean

# Abort-guard tunables (same relaxation as retune1 per
# feedback-no-cost-gate-on-juno):
STEPRATE_CHECK_AT_STEP=1000
STEPRATE_MIN_THRESH=0.005

# --- 1. RUN_TAG generation ----------------------------------------------------
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SHORTHASH=$(cd "${JUNO_WORK}" && git rev-parse --short HEAD 2>/dev/null || echo nogit)
UUID_SUFFIX=$(uuidgen 2>/dev/null | cut -c1-6 \
              || head -c 6 /dev/urandom | base32 | tr '[:upper:]' '[:lower:]' | head -c 6 \
              || openssl rand -hex 3)
RUN_TAG="SprintL1-Retune2-MeanRed-P${PHYSICS}-N${N_RAYS}-S${SEED}-lr${LR_MAX}-${SHORTHASH}-${TIMESTAMP}-${UUID_SUFFIX}"
RUN_NAME="Sprint-L1-retune2-P1T1-lr1e-4-meanreduction-fullgradnorm-5k"

if [[ "${RUN_TAG}" =~ [[:space:]] ]] || [[ "${RUN_TAG}" == *[\$\`\;\&\|]* ]]; then
  echo "FATAL: RUN_TAG '${RUN_TAG}' contains forbidden characters." >&2
  exit 1
fi

echo "================================================================="
echo "  Sprint-L1 [D-60] gate-retune-1 absorption — Retune Attempt 2"
echo "  RUN_TAG=${RUN_TAG}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  Retune lever: --pf-log-reduction ${PF_LOG_REDUCTION} (sum -> mean)"
echo "  Juno A30 partition, 24:00:00 wallclock"
echo "================================================================="

DEST="${JUNO_WORK}/cloud_runs/sprint_L1_retune2_${SLURM_JOB_ID}"
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
echo "LR_MAX=${LR_MAX} WARMUP_STEPS=${WARMUP_STEPS}"
echo "PF_LOG_REDUCTION=${PF_LOG_REDUCTION}  (LEVER)"
echo "L1_D24_BASELINE_TAU_MSE=${L1_D24_BASELINE_TAU_MSE}"
echo "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}"
echo "GradNorm: FULL (Chen+ 2018 second-order; --gradnorm-full active)"
echo "Step-rate abort guard: < ${STEPRATE_MIN_THRESH} steps/s at step ${STEPRATE_CHECK_AT_STEP}"

export NERF_RUN_NAME="${RUN_NAME}"
export NERF_RUN_TAG="${RUN_TAG}"

LOG="${DEST}/driver.log"
SENTINEL_ABORT="${DEST}/STEPRATE_ABORT"
WATCHER_LOG="${DEST}/steprate_watcher.log"
touch "${LOG}"

# --- 5. Background step-rate watcher (carried over from retune1) --------------
( tail -F -n +1 "${LOG}" 2>/dev/null \
    | awk -v target="${STEPRATE_CHECK_AT_STEP}" \
          -v thresh="${STEPRATE_MIN_THRESH}" \
          -v sentinel="${SENTINEL_ABORT}" \
          -v wlog="${WATCHER_LOG}" '
      BEGIN { first_step = -1; first_time = 0; done = 0 }
      done == 1 { next }
      /^Step [0-9]+\// {
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
            system("touch " sentinel)
          }
          done = 1
        }
      }
    ' ) &
WATCHER_PID=$!
echo "[submit] step-rate watcher started, pid=${WATCHER_PID}"

cleanup_watcher() {
  if kill -0 "${WATCHER_PID}" 2>/dev/null; then
    kill "${WATCHER_PID}" 2>/dev/null || true
    wait "${WATCHER_PID}" 2>/dev/null || true
  fi
}
trap cleanup_watcher EXIT

# --- 6. Sentinel-driven driver-killer -----------------------------------------
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
echo "=== driver invocation: experiments/nerf/pipeline.py (sprint-L1 retune2 reduction=${PF_LOG_REDUCTION}) ==="
set +e
setsid python -u experiments/nerf/pipeline.py \
    --gradnorm-full \
    --enable-l1-pf-loss \
    --pf-log-reduction "${PF_LOG_REDUCTION}" \
    --physics "${PHYSICS}" \
    --n_rays "${N_RAYS}" \
    --seed "${SEED}" \
    --microbatch "${MICROBATCH}" \
    --max_steps "${MAX_STEPS}" \
    --lr_max "${LR_MAX}" \
    --warmup_steps "${WARMUP_STEPS}" \
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

kill "${KILLER_PID}" 2>/dev/null || true
wait "${KILLER_PID}" 2>/dev/null || true
cleanup_watcher
trap - EXIT

echo "[submit] DRIVER_EXIT=${DRIVER_EXIT}"
if [[ -f "${SENTINEL_ABORT}" ]]; then
  DRIVER_KILLED_BY_WATCHER=1
  echo "[submit] STEPRATE_ABORT sentinel present; retune2 failed step-rate guard."
fi

if [[ -f "${WATCHER_LOG}" ]]; then
  echo "=== step-rate watcher log ==="
  cat "${WATCHER_LOG}"
fi

# --- 8. Retire-marker guard (pass-condition (a)) ------------------------------
echo "=== retire-marker check (retune2 pass-condition (a)) ==="
RETIRE_JSON="${DEST}/checkpoints/retire.json"
if [[ -f "${RETIRE_JSON}" ]]; then
  echo "[retune2] RETIRE: ${RETIRE_JSON} present — in-pipeline retire fired."
  cat "${RETIRE_JSON}"
  RETIRE_FIRED=1
else
  echo "[retune2] OK: no retire.json (pass-condition (a) satisfied)."
  RETIRE_FIRED=0
fi

# --- 9. var_pf_band_ratio + Option-R diag post-hoc scan -----------------------
echo "=== last 100 log lines ==="
tail -n 100 "${LOG}" || true
echo "---"
echo "=== grep: per-task grad-norm + var_pf_band_ratio trail ==="
grep -E "(var_pf_band_ratio|w_ratio|w_tau|w_pf|loss_pf|loss_tau|per-task grad-norm)" "${LOG}" | tail -n 80 || \
  echo "[retune2] WARN: no diagnostic lines matched."

# --- 9a. Attempt-2 falsification criterion baked into trailer -----------------
# Per PI: ``pf/tau < 0.05 at step 500`` (Option R live ratio at the freeze
# boundary) triggers Attempt-3 escalation. We grep the step-500 per-task
# grad-norm line and surface a non-fatal banner.
echo "=== Attempt-2 falsification criterion (step 500 pf/tau) ==="
STEP500_LINE=$(grep -E "per-task grad-norm @ step 500:" "${LOG}" | tail -n 1 || true)
if [[ -n "${STEP500_LINE}" ]]; then
  echo "[retune2] step-500 line: ${STEP500_LINE}"
  # Extract the ratio= field with awk; tolerate trailing punctuation.
  STEP500_RATIO=$(echo "${STEP500_LINE}" | awk -F'ratio=' '{print $2}' | awk '{print $1}' | tr -d ',')
  echo "[retune2] step-500 pf/tau ratio = ${STEP500_RATIO}"
  RATIO_FREEZE_FAIL=$(awk -v r="${STEP500_RATIO}" 'BEGIN { print (r+0 < 0.05) ? 1 : 0 }')
  if [[ "${RATIO_FREEZE_FAIL}" -eq 1 ]]; then
    echo "============================================================"
    echo "  Attempt 2 falsification criterion: pf/tau < 0.05 at step 500"
    echo "  observed ratio = ${STEP500_RATIO}"
    echo "  -> Attempt 3 dispatch path activated"
    echo "  (next-step: scripts/submit_juno_sprint_L1_retune3.sh)"
    echo "============================================================"
  else
    echo "[retune2] step-500 pf/tau >= 0.05 (no Attempt-3 escalation trigger)."
  fi
else
  echo "[retune2] WARN: no 'per-task grad-norm @ step 500' line found in log."
  echo "[retune2] Cannot evaluate Attempt-2 falsification criterion; surface to PI."
fi

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
    ("sprint_kind", "retune"),
    ("retune_attempt", "2"),
    ("lr", "${LR_MAX}"),
    ("pf_log_reduction", "${PF_LOG_REDUCTION}"),
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
echo "  Sprint-L1 Retune Attempt 2 summary"
echo "================================================================="
echo "  RUN_TAG                    = ${RUN_TAG}"
echo "  RUN_NAME                   = ${RUN_NAME}"
echo "  LR_MAX                     = ${LR_MAX}"
echo "  WARMUP_STEPS               = ${WARMUP_STEPS}"
echo "  PF_LOG_REDUCTION (lever)   = ${PF_LOG_REDUCTION}"
echo "  DRIVER_EXIT                = ${DRIVER_EXIT}"
echo "  RETIRE_FIRED               = ${RETIRE_FIRED}"
echo "  DRIVER_KILLED_BY_WATCHER   = ${DRIVER_KILLED_BY_WATCHER}"
echo "  DEST                       = ${DEST}"
echo "  LOG                        = ${LOG}"
echo "================================================================="

# --- 13. Exit-code routing (retune2 pass conditions) --------------------------
if [[ "${RETIRE_FIRED}" -eq 1 ]]; then
  echo "FATAL: retune2 pass-condition (a) failed (retire.json present)." >&2
  exit 3
fi
if [[ "${DRIVER_KILLED_BY_WATCHER}" -eq 1 ]]; then
  echo "FATAL: step-rate watcher tripped (< ${STEPRATE_MIN_THRESH} steps/s at step ${STEPRATE_CHECK_AT_STEP})." >&2
  exit 4
fi

echo "=== done; results at ${DEST} ==="
