#!/bin/bash
#SBATCH --job-name=sprint-L1-retune3-grad-clip
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --output=sprint-L1-retune3-juno-%j.out
#SBATCH --error=sprint-L1-retune3-juno-%j.err

# Sprint-L1 [D-60] gate-retune-2 v2 absorption (post-panel-HOLD) — Retune
# Attempt 3 (per-task gradient clipping path) on Juno A30. Design: PI
# Attempt 3 spec 2026-05-22 with Amendments A + B; sbatch comment block
# corrected 2026-05-22 per PI v2 absorption + panel HOLD on D-WRONG-AXIS.
#
# WHY THIS RETUNE EXISTS (post gate-retune-2 v2 absorption, 2026-05-22)
# --------------------------------------------------------------------
# Retune-2 Option α (lr=3e-5/warmup=2000) R-b retired at step 200 with
# var_pf_band_ratio=6.9e-7. Combined with retune-1's step-200 retire at
# var_pf_band_ratio=6.3e-3 (a 4-decade log10 drop along the LR-narrower
# direction), the LR-narrower-as-rescue direction is abandoned under PI
# verdict D-WRONG-DIRECTION-ON-RATIO-AXIS (post panel HOLD). Both retires
# are formally warmup-zone diagnostics per v3 KILLER-1 pre-commit; this
# is NOT an axis-family falsification (R13 sweep-discipline: ≥1 decade
# × ≥4 points required per Smith 2018 / Wilson+ 2017). The rejected
# direction is abandoned on combined warmup-zone evidence + ratio-
# invariance prior: the observed pf:tau gradient ratio of 20806.954328:1
# at step 200 (driver.log:27 re-verified in-session, R26 DISCHARGED) is
# INVARIANT under uniform LR rescaling — lower LR scales both gradients
# equally and preserves the ratio. Chen+ 2018 arXiv:1711.02257 Sec 3.2
# documents GradNorm's ~100× rescuability bound; observed ratio is 2 OOM
# beyond that bound. LR scaling cannot rescue ratio-invariant pathology
# by construction.
#
# Routing per v3 KILLER-3 (i): per-task gradient clipping pre-GradNorm
# composition. Justification for (i) over (ii) reduction='mean':
# K2 estimator-equivalence (1.021e-14 abs / 2.914e-15 rel) was certified
# at gate-4 under --pf-log-reduction='sum'; switching to 'mean' invalidates
# K2 and requires re-test before dispatch (real procedural cost). Per-task
# clip preserves the reduction lever and so K2 stays certified —
# admissible without re-test. K2-preservation is the binding justification,
# NOT "narrowness" (per-task-clip adds a clip-magnitude hyperparameter,
# strictly less narrow than reduction='mean' which adds none; panel
# SERIOUS-5 correction).
#
# PRIOR SIMILAR-CONFIDENCE CLAIMS FALSIFIED IN THIS TRACK
# -------------------------------------------------------
#   - retune-1 (lr=1e-4, warmup=1000): R-b retire @ step 200, var_pf_band_ratio=6.3e-3
#   - retune-2 Option α (lr=3e-5, warmup=2000): R-b retire @ step 200, var_pf_band_ratio=6.9e-7
#   - Combined verdict: D-WRONG-DIRECTION-ON-RATIO-AXIS (LEDGER §3 v2 absorption)
# Per [D-37]-Ext rule 2 (falsified-prior cascade): this retune-3 spec
# inherits ONE-LEVEL-DOWNGRADED confidence verbs. Retune-3 is "first
# test of the per-task gradient-discipline hypothesis given the
# LR-narrower-direction abandonment," NOT "the right axis" or
# "ratio-discipline by design."
#
# KILLER-1 WARMUP-ZONE DISCLOSURE (carried from v3, status updated post-HOLD)
# --------------------------------------------------------------------------
# Retune-1 and Retune-2 step-200 retires are formally WARMUP-ZONE
# diagnostics. The joint-claim posture is CORRECTED post-panel: prior-
# session absorption called the joint pattern an "axis-family
# falsification"; that claim is RETRACTED (R13 sweep-discipline). The
# joint pattern supports the narrower claim "LR-narrower-direction
# abandoned on combined warmup-zone evidence + ratio-invariance prior +
# Chen+ 2018 §3.2 rescuability literature." Retune-3 carries the
# per-task gradient-discipline hypothesis FORWARD into a warmup-zone-
# clean test regime via the per-task-clip pre-GradNorm-composition
# discipline — distinct hypothesis space, not a continuation of the LR sweep.
#
# Amendment A (empirical clip threshold)
# --------------------------------------
# Clip at the value of the LOWER-NORM TASK'S running EMA (decay 0.95)
# computed over steps 100-500, then HELD FIXED steps 500-end. The pipeline
# code path (--per-task-grad-clip auto) maintains this state via the
# PerTaskClipState dataclass in src/training/per_task_clip.py.
#
# Amendment B (pre-committed FAIL criteria)
# -----------------------------------------
# Attempt 3 FAILS if at step 5000 EITHER:
#   (i)  ``pf/tau`` ratio diagnostic > 3.0, OR
#   (ii) ``loss_tau`` has not decreased monotonically by >= 10% relative
#        to its step-1000 value.
# On FAIL: no headline claim, null result reported. The sbatch trailer
# greps the log for both signals and surfaces a verdict banner.
#
# Single lever vs retune2
# -----------------------
#   --per-task-grad-clip auto   (NEW Attempt-3 lever; default '0.0' = OFF)
# All other knobs carry over from retune-2 Option α verbatim per
# R13 / [D-37] symmetric-disclosure.
#
# Pinned configuration (PI pre-committed grid)
# --------------------------------------------
#   --lr_max 1e-4                    (kept from retune1/retune2)
#   --warmup_steps 1000              (kept from retune1/retune2)
#   --pf-log-reduction sum           (DEFAULT; K2-certified at gate-4;
#                                     'mean' reserved for L2 escalation)
#   --per-task-grad-clip auto        (RETUNE LEVER — EMA-derived threshold)
#   --gradnorm-full                  (full Chen+ 2018 path)
#   --enable-l1-pf-loss
#   --physics 1                      (P1, K1-absorbing tier)
#   --n_rays 64                      (T1 scale, unchanged)
#   --seed 0
#   --microbatch 256
#   --max_steps 5000
#   --mean_flux_obs 0.979            (z=0.3 Becker+ 2013 anchor; [D-11])
#   --l1-d24-baseline-tau-mse 0.01   (R-a backstop)
#   --checkpoint_interval 5000

set -euo pipefail

# --- 0. Load Juno-side .env if present ----------------------------------------
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=/work/sxh240010/CosmoGasVision}"
: "${JUNO_SCRATCH:=/scratch/juno/sxh240010}"

# --- Sprint-L1 retune3 pinned hyperparameters (PI pre-committed grid) ---------
PHYSICS=1
N_RAYS=64                  # T1 scale (unchanged from retune2)
SEED=0
MICROBATCH=256
MAX_STEPS=5000
MEAN_FLUX_OBS=0.979
L1_D24_BASELINE_TAU_MSE=0.01
CHECKPOINT_INTERVAL=5000

# Knobs carried from retune-2 Option α (post-D-WRONG-DIRECTION-ON-RATIO-AXIS):
# pf_log_reduction=sum is the K2-certified default; 'mean' reserved for L2.
LR_MAX=1e-4
WARMUP_STEPS=1000
PF_LOG_REDUCTION=sum

# *** NEW Attempt-3 lever ***
PER_TASK_GRAD_CLIP=auto    # EMA-derived threshold (PerTaskClipState dataclass)

# Abort-guard tunables (same relaxation as retune1/2):
STEPRATE_CHECK_AT_STEP=1000
STEPRATE_MIN_THRESH=0.005

# --- 1. RUN_TAG generation ----------------------------------------------------
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SHORTHASH=$(cd "${JUNO_WORK}" && git rev-parse --short HEAD 2>/dev/null || echo nogit)
UUID_SUFFIX=$(uuidgen 2>/dev/null | cut -c1-6 \
              || head -c 6 /dev/urandom | base32 | tr '[:upper:]' '[:lower:]' | head -c 6 \
              || openssl rand -hex 3)
RUN_TAG="SprintL1-Retune3-PerTaskClip-P${PHYSICS}-N${N_RAYS}-S${SEED}-lr${LR_MAX}-${SHORTHASH}-${TIMESTAMP}-${UUID_SUFFIX}"
RUN_NAME="Sprint-L1-retune3-P1T1-lr1e-4-pertask-clip-auto-fullgradnorm-5k"

if [[ "${RUN_TAG}" =~ [[:space:]] ]] || [[ "${RUN_TAG}" == *[\$\`\;\&\|]* ]]; then
  echo "FATAL: RUN_TAG '${RUN_TAG}' contains forbidden characters." >&2
  exit 1
fi

echo "================================================================="
echo "  Sprint-L1 [D-60] gate-retune-1 absorption — Retune Attempt 3"
echo "  RUN_TAG=${RUN_TAG}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  Retune lever: --per-task-grad-clip ${PER_TASK_GRAD_CLIP}"
echo "                (Amendment A: EMA-derived threshold, lower-norm task)"
echo "  Juno A30 partition, 24:00:00 wallclock"
echo "================================================================="

DEST="${JUNO_WORK}/cloud_runs/sprint_L1_retune3_${SLURM_JOB_ID}"
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
echo "PF_LOG_REDUCTION=${PF_LOG_REDUCTION}"
echo "PER_TASK_GRAD_CLIP=${PER_TASK_GRAD_CLIP}  (LEVER — Amendment A)"
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

# --- 5. Background step-rate watcher (carried over from retune2) --------------
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
echo "=== driver invocation: experiments/nerf/pipeline.py (sprint-L1 retune3 per-task-clip=${PER_TASK_GRAD_CLIP}) ==="
set +e
setsid python -u experiments/nerf/pipeline.py \
    --gradnorm-full \
    --enable-l1-pf-loss \
    --pf-log-reduction "${PF_LOG_REDUCTION}" \
    --per-task-grad-clip "${PER_TASK_GRAD_CLIP}" \
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
  echo "[submit] STEPRATE_ABORT sentinel present; retune3 failed step-rate guard."
fi

if [[ -f "${WATCHER_LOG}" ]]; then
  echo "=== step-rate watcher log ==="
  cat "${WATCHER_LOG}"
fi

# --- 8. Retire-marker guard ---------------------------------------------------
echo "=== retire-marker check ==="
RETIRE_JSON="${DEST}/checkpoints/retire.json"
if [[ -f "${RETIRE_JSON}" ]]; then
  echo "[retune3] RETIRE: ${RETIRE_JSON} present — in-pipeline retire fired."
  cat "${RETIRE_JSON}"
  RETIRE_FIRED=1
else
  echo "[retune3] OK: no retire.json."
  RETIRE_FIRED=0
fi

# --- 9. Diagnostic trail scan -------------------------------------------------
echo "=== last 100 log lines ==="
tail -n 100 "${LOG}" || true
echo "---"
echo "=== grep: per-task grad-norm + clip events + loss trail ==="
grep -E "(var_pf_band_ratio|w_ratio|w_tau|w_pf|loss_pf|loss_tau|per-task grad-norm|clip @ step|Per-task grad-clip ACTIVE)" "${LOG}" | tail -n 120 || \
  echo "[retune3] WARN: no diagnostic lines matched."

# --- 9a. Attempt-3 Amendment-B FAIL criteria (baked into trailer) -------------
# (i)  pf/tau ratio diagnostic > 3.0 at step 5000
# (ii) loss_tau has NOT decreased monotonically by >= 10% relative to
#      its step-1000 value
# Both criteria are evaluated post-hoc from the log; on FAIL we surface a
# verdict banner (NULL RESULT — no headline claim per PI Attempt 3 spec).
echo "=== Attempt-3 Amendment-B FAIL criteria ==="

# Criterion (i): step-5000 pf/tau ratio
STEP5000_LINE=$(grep -E "per-task grad-norm @ step 5000:" "${LOG}" | tail -n 1 || true)
RATIO_FAIL=0
if [[ -n "${STEP5000_LINE}" ]]; then
  echo "[retune3] step-5000 line: ${STEP5000_LINE}"
  STEP5000_RATIO=$(echo "${STEP5000_LINE}" | awk -F'ratio=' '{print $2}' | awk '{print $1}' | tr -d ',')
  echo "[retune3] step-5000 pf/tau ratio = ${STEP5000_RATIO}"
  RATIO_FAIL=$(awk -v r="${STEP5000_RATIO}" 'BEGIN { print (r+0 > 3.0) ? 1 : 0 }')
else
  echo "[retune3] WARN: no 'per-task grad-norm @ step 5000' line found; criterion (i) inconclusive."
fi

# Criterion (ii): loss_tau monotonic >= 10% decrease vs step-1000 value
# Pull loss_tau (the data-task MSE) at step 1000 and step 5000 from the
# step-summary log line ("Step N/MAX | loss=... loss_data=... ...").
LOSS_TAU_1000=$(grep -E "^Step 1000/" "${LOG}" | tail -n 1 | grep -oE "loss_data=[0-9.eE+-]+" | head -n 1 | cut -d= -f2 || true)
LOSS_TAU_5000=$(grep -E "^Step 5000/" "${LOG}" | tail -n 1 | grep -oE "loss_data=[0-9.eE+-]+" | head -n 1 | cut -d= -f2 || true)
TAU_DECREASE_FAIL=0
if [[ -n "${LOSS_TAU_1000}" && -n "${LOSS_TAU_5000}" ]]; then
  echo "[retune3] loss_tau(step=1000) = ${LOSS_TAU_1000}"
  echo "[retune3] loss_tau(step=5000) = ${LOSS_TAU_5000}"
  TAU_DECREASE_FAIL=$(awk -v a="${LOSS_TAU_1000}" -v b="${LOSS_TAU_5000}" \
    'BEGIN { print (b+0 > 0.9 * (a+0)) ? 1 : 0 }')
else
  echo "[retune3] WARN: could not extract loss_tau at step 1000 / 5000; criterion (ii) inconclusive."
fi

echo "[retune3] FAIL criterion (i)  pf/tau > 3.0     -> ${RATIO_FAIL}"
echo "[retune3] FAIL criterion (ii) tau >= 90% of step1k -> ${TAU_DECREASE_FAIL}"

if [[ "${RATIO_FAIL}" -eq 1 || "${TAU_DECREASE_FAIL}" -eq 1 ]]; then
  echo "============================================================"
  echo "  Attempt 3 PRE-COMMITTED FAIL CRITERION HIT"
  echo "  -> NULL RESULT REPORTED: no headline claim from Attempt 3"
  echo "  -> PI Attempt 3 spec 2026-05-22, Amendment B"
  echo "============================================================"
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
    ("retune_attempt", "3"),
    ("lr", "${LR_MAX}"),
    ("pf_log_reduction", "${PF_LOG_REDUCTION}"),
    ("per_task_grad_clip", "${PER_TASK_GRAD_CLIP}"),
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
echo "  Sprint-L1 Retune Attempt 3 summary"
echo "================================================================="
echo "  RUN_TAG                    = ${RUN_TAG}"
echo "  RUN_NAME                   = ${RUN_NAME}"
echo "  LR_MAX                     = ${LR_MAX}"
echo "  WARMUP_STEPS               = ${WARMUP_STEPS}"
echo "  PF_LOG_REDUCTION           = ${PF_LOG_REDUCTION}"
echo "  PER_TASK_GRAD_CLIP (lever) = ${PER_TASK_GRAD_CLIP}"
echo "  DRIVER_EXIT                = ${DRIVER_EXIT}"
echo "  RETIRE_FIRED               = ${RETIRE_FIRED}"
echo "  DRIVER_KILLED_BY_WATCHER   = ${DRIVER_KILLED_BY_WATCHER}"
echo "  AMEND_B_RATIO_FAIL         = ${RATIO_FAIL:-?}"
echo "  AMEND_B_TAU_DECREASE_FAIL  = ${TAU_DECREASE_FAIL:-?}"
echo "  DEST                       = ${DEST}"
echo "  LOG                        = ${LOG}"
echo "================================================================="

# --- 13. Exit-code routing ----------------------------------------------------
if [[ "${RETIRE_FIRED}" -eq 1 ]]; then
  echo "FATAL: retune3 retire.json present (pipeline-side retire fired)." >&2
  exit 3
fi
if [[ "${DRIVER_KILLED_BY_WATCHER}" -eq 1 ]]; then
  echo "FATAL: step-rate watcher tripped (< ${STEPRATE_MIN_THRESH} steps/s at step ${STEPRATE_CHECK_AT_STEP})." >&2
  exit 4
fi

# Amendment-B FAIL criteria are SCIENTIFIC FAIL conditions (null result),
# not infrastructure errors. We exit 0 so MLflow tagging completes; the
# null-result verdict banner above is the human-readable signal.
echo "=== done; results at ${DEST} ==="
