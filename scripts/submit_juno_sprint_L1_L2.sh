#!/bin/bash
#SBATCH --job-name=sprint-L1-L2-meanReduction-lr1e-4-wu1000
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --output=sprint-L1-L2-juno-%j.out
#SBATCH --error=sprint-L1-L2-juno-%j.err
#SBATCH --comment="D-60 L2 reduction='mean' escalation; LEDGER v5 NON-PROVISIONAL 2026-05-22"

# Sprint-L1 [D-60] gate-retune-1+2+3 v5 NON-PROVISIONAL — L2 escalation
# on Juno A30. Routing source: PI v5 pre-commit at commit 6e5d63e
# (defense-panel round-3 APPROVE-with-minor-disclosures + 2nd R26 catch
# on K2-layer-confusion). Parent: LEDGER §3 [D-60] gate-retune v5.
#
# WHY THIS L2 EXISTS (v5 routing)
# --------------------------------
# L1 in-loss-function intervention class has now exhausted three ladder
# points (retune-1 baseline lr=1e-4/wu=1000; retune-2 Option-alpha
# lr=3e-5/wu=2000; retune-3 per-task clip) — all FAILed by ratio-
# invariance pf/tau imbalance ~20807x at step-500 + flat-loss signatures.
# L2 isolates the v3 KILLER-3 reduction-flip axis: --pf-log-reduction
# 'mean' replaces the default 'sum'. Single-axis vs retune-1 baseline;
# all other hyperparameters identical to retune-1.
#
# PI v5 HONEST CAVEAT (per [D-37] honest-reporting; absorption text verbatim)
# --------------------------------------------------------------------------
# L2 is procedural-exhaustion of v3 KILLER-3 ladder, NOT mechanism-driven.
# PI cannot mechanism-predict reduction='mean' success: 'mean' reduces the
# pf-loss magnitude by ~n_velocity_bins (~6x), but the observed pf/tau
# imbalance is ~20807x, still 30+x above any demonstrated regime where
# GradNorm has been shown to stabilize. Expected outcome under that read:
# FAIL -> escalate to [D-53] supervision-target redesign per LEDGER §3 v5
# [D-53] stub. This sbatch closes the L1-in-loss intervention class so the
# falsification record is complete before [D-53] design opens.
#
# PRIOR-FAILURE LEDGER ([D-37]-Ext rule 2; 3 prior FAILs in this track)
# ---------------------------------------------------------------------
#   1. Sprint-L1 retune-1 (lr=1e-4, wu=1000, reduction=sum)
#      -> FAIL: pf/tau ratio ~20807x at step-500, flat tau loss.
#   2. Sprint-L1 retune-2 Option-alpha (lr=3e-5, wu=2000, reduction=sum)
#      -> FAIL: frozen-network mechanism (effective LR ~3e-6 at step 200
#         under warmup taper); LR-axis terminal probe closed.
#   3. Sprint-L1 retune-3 (per-task clip, reduction=sum)
#      -> FAIL: ratio-invariance signature preserved; per-task clip alone
#         does not break the pf/tau magnitude trap.
# This L2 is the LAST pre-committed grid point on the L1 in-loss-function
# intervention class. On FAIL, gate closes; [D-53] supervision-target
# redesign opens per panel SERIOUS-3 + v5 [D-53] stub.
#
# Pinned configuration (L2 — single-axis lever vs retune-1 baseline)
# ------------------------------------------------------------------
#   --lr_max 1e-4                    (retune-1 baseline; NOT retune-2's 3e-5)
#   --warmup_steps 1000              (retune-1 baseline; NOT retune-2's 2000)
#   --pf-log-reduction mean          (*** L2 LEVER *** vs retune-1's 'sum')
#   --gradnorm-full                  (full Chen+ 2018 path; unchanged)
#   --enable-l1-pf-loss              (unchanged)
#   --physics 1                      (P1, K1-absorbing tier; unchanged)
#   --n_rays 64                      (T1 scale; unchanged)
#   --seed 0                         (unchanged)
#   --microbatch 256                 (unchanged)
#   --max_steps 5000                 (unchanged)
#   --mean_flux_obs 0.979            (z=0.3 Becker+ 2013 anchor; [D-11])
#   --l1-d24-baseline-tau-mse 0.01   (R-a backstop)
#   --checkpoint_interval 5000       (end-state checkpoint only)
# Per-task clip is OFF (default 0.0) — L2 tests reduction lever in
# isolation per R13.
#
# FALSIFICATION CRITERIA (post-hoc grep of driver.log; trailer reports raw)
# ------------------------------------------------------------------------
# Amendment B (carry from v3 KILLER-3):
#   - pf/tau ratio > 3.0 at step 5000  -> FAIL
#   - loss_tau(5000) > 0.9 * loss_tau(1000)  -> FAIL (tau-loss did not move)
# R-b retire:
#   - any retire.json present in checkpoints dir  -> FAIL
# L2-specific (D-REDUCTION-AXIS-ORTHOGONAL signature):
#   - if per-task w_ratio > 1000 AND var_pf_band_ratio < 1e-3 at step 200,
#     surface as "D-REDUCTION-AXIS-ORTHOGONAL — escalate immediately to
#     [D-53] per LEDGER §3 v5 [D-53] stub". This is a STRENGTH signal that
#     the reduction axis is orthogonal to the trap, justifying [D-53]
#     opening without waiting for step-5000 conclusion.
#
# Defense-panel round-3 round-up: APPROVE-with-minor-disclosures (lifted
# R15; 2nd R26 catch on K2-layer-confusion addressed in absorption text).

set -euo pipefail

# --- 0. Load Juno-side .env if present ----------------------------------------
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=/work/sxh240010/CosmoGasVision}"
: "${JUNO_SCRATCH:=/scratch/juno/sxh240010}"

# --- Sprint-L1 L2 pinned hyperparameters (PI v5 pre-committed) ----------------
PHYSICS=1
N_RAYS=64
SEED=0
MICROBATCH=256
MAX_STEPS=5000
MEAN_FLUX_OBS=0.979
L1_D24_BASELINE_TAU_MSE=0.01
CHECKPOINT_INTERVAL=5000

# L2 lever vs retune-1 baseline (single-axis):
LR_MAX=1e-4                # retune-1 baseline (NOT retune-2's 3e-5)
WARMUP_STEPS=1000          # retune-1 baseline (NOT retune-2's 2000)
PF_LOG_REDUCTION=mean      # *** L2 LEVER *** vs retune-1's default 'sum'

# Abort-guard tunables (carry from retune2):
STEPRATE_CHECK_AT_STEP=1000
STEPRATE_MIN_THRESH=0.005

# --- 1. RUN_TAG generation ----------------------------------------------------
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SHORTHASH=$(cd "${JUNO_WORK}" && git rev-parse --short HEAD 2>/dev/null || echo nogit)
UUID_SUFFIX=$(uuidgen 2>/dev/null | cut -c1-6 \
              || head -c 6 /dev/urandom | base32 | tr '[:upper:]' '[:lower:]' | head -c 6 \
              || openssl rand -hex 3)
RUN_TAG="SprintL1-L2-meanRed-P${PHYSICS}-N${N_RAYS}-S${SEED}-lr${LR_MAX}-wu${WARMUP_STEPS}-${SHORTHASH}-${TIMESTAMP}-${UUID_SUFFIX}"
RUN_NAME="Sprint-L1-L2-meanReduction-P1T1-lr1e-4-warmup1000-fullgradnorm-5k"

if [[ "${RUN_TAG}" =~ [[:space:]] ]] || [[ "${RUN_TAG}" == *[\$\`\;\&\|]* ]]; then
  echo "FATAL: RUN_TAG '${RUN_TAG}' contains forbidden characters." >&2
  exit 1
fi

echo "================================================================="
echo "  Sprint-L1 [D-60] v5 NON-PROVISIONAL — L2 escalation (reduction='mean')"
echo "  RUN_TAG=${RUN_TAG}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  L2 lever: --pf-log-reduction ${PF_LOG_REDUCTION}  (vs retune-1 'sum')"
echo "  Baseline carry: lr=${LR_MAX} warmup=${WARMUP_STEPS} (retune-1 baseline)"
echo "  Juno A30 partition, 24:00:00 wallclock"
echo "================================================================="

DEST="${JUNO_WORK}/cloud_runs/sprint_L1_L2_${SLURM_JOB_ID}"
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

# --- 4. 1-step CPU dry-run pre-flight (replaces v4's CPU-30min smoke) ---------
# Per PI v5 absorption of panel SERIOUS-6: a 1-step CPU dry-run with the EXACT
# SAME reduction-affecting args is the cheapest possible proof that the
# reduction='mean' code path does not crash and produces both task losses.
# If this fails, abort BEFORE consuming Juno GPU time.
#
# pipeline.py has NO --device flag; device is selected by torch.cuda.is_available().
# Force CPU via CUDA_VISIBLE_DEVICES='' in a subshell so the main GPU invocation
# is unaffected.
echo "=== 1-step CPU dry-run pre-flight (PI v5 SERIOUS-6 absorption) ==="
DRYRUN_LOG="${DEST}/dryrun_cpu_1step.log"
DRYRUN_DEST="${DEST}/dryrun"
mkdir -p "${DRYRUN_DEST}/checkpoints"

set +e
(
  export CUDA_VISIBLE_DEVICES=""
  export NERF_RUN_NAME="${RUN_NAME}-DRYRUN"
  export NERF_RUN_TAG="${RUN_TAG}-DRYRUN"
  # MLflow off for the dry-run so we don't pollute the tracker with 1-step runs.
  export MLFLOW_TRACKING_URI=""
  python -u experiments/nerf/pipeline.py \
      --gradnorm-full \
      --enable-l1-pf-loss \
      --pf-log-reduction "${PF_LOG_REDUCTION}" \
      --physics "${PHYSICS}" \
      --n_rays 4 \
      --seed "${SEED}" \
      --microbatch 4 \
      --max_steps 1 \
      --lr_max "${LR_MAX}" \
      --warmup_steps "${WARMUP_STEPS}" \
      --mean_flux_obs "${MEAN_FLUX_OBS}" \
      --l1-d24-baseline-tau-mse "${L1_D24_BASELINE_TAU_MSE}" \
      --checkpoint_dir "${DRYRUN_DEST}/checkpoints" \
      --checkpoint_interval 1 \
      --run_name "${RUN_NAME}-DRYRUN" \
      2>&1 | tee "${DRYRUN_LOG}"
)
DRYRUN_EXIT=${PIPESTATUS[0]}
set -e

echo "[preflight] DRYRUN_EXIT=${DRYRUN_EXIT}"

# Pass criteria: exit 0, no NaN, both task losses present in log.
if [[ "${DRYRUN_EXIT}" -ne 0 ]]; then
  echo "FATAL: 1-step CPU dry-run exited non-zero (${DRYRUN_EXIT}); aborting before Juno GPU." >&2
  exit 10
fi
if grep -Eiq "(nan|inf)" "${DRYRUN_LOG}"; then
  echo "FATAL: 1-step CPU dry-run log contains NaN/Inf; aborting before Juno GPU." >&2
  grep -Ei "(nan|inf)" "${DRYRUN_LOG}" | head -n 20 >&2 || true
  exit 11
fi
if ! grep -Eq "loss_pf" "${DRYRUN_LOG}"; then
  echo "FATAL: 1-step CPU dry-run missing loss_pf signal; aborting before Juno GPU." >&2
  exit 12
fi
if ! grep -Eq "loss_tau" "${DRYRUN_LOG}"; then
  echo "FATAL: 1-step CPU dry-run missing loss_tau signal; aborting before Juno GPU." >&2
  exit 13
fi
echo "[preflight] PASS: dry-run completed 1 step, no NaN/Inf, both task losses observed."
echo "[preflight] dry-run artifacts retained at ${DRYRUN_DEST} for forensics."

# --- 5. GPU + torch sanity ----------------------------------------------------
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
echo "PF_LOG_REDUCTION=${PF_LOG_REDUCTION}  (L2 LEVER)"
echo "L1_D24_BASELINE_TAU_MSE=${L1_D24_BASELINE_TAU_MSE}"
echo "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}"
echo "GradNorm: FULL (Chen+ 2018 second-order; --gradnorm-full active)"
echo "Per-task clip: OFF (default 0.0; isolating reduction lever per R13)"
echo "Step-rate abort guard: < ${STEPRATE_MIN_THRESH} steps/s at step ${STEPRATE_CHECK_AT_STEP}"

export NERF_RUN_NAME="${RUN_NAME}"
export NERF_RUN_TAG="${RUN_TAG}"

LOG="${DEST}/driver.log"
SENTINEL_ABORT="${DEST}/STEPRATE_ABORT"
WATCHER_LOG="${DEST}/steprate_watcher.log"
touch "${LOG}"

# --- 6. Background step-rate watcher (carried from retune2) -------------------
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

# --- 7. Sentinel-driven driver-killer -----------------------------------------
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

# --- 8. Main driver invocation ------------------------------------------------
echo "=== driver invocation: experiments/nerf/pipeline.py (sprint-L1 L2 reduction=${PF_LOG_REDUCTION}) ==="
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
  echo "[submit] STEPRATE_ABORT sentinel present; L2 failed step-rate guard."
fi

if [[ -f "${WATCHER_LOG}" ]]; then
  echo "=== step-rate watcher log ==="
  cat "${WATCHER_LOG}"
fi

# --- 9. Retire-marker guard (R-b retire FAIL criterion) -----------------------
echo "=== retire-marker check (R-b retire FAIL criterion) ==="
RETIRE_JSON="${DEST}/checkpoints/retire.json"
if [[ -f "${RETIRE_JSON}" ]]; then
  echo "[L2] RETIRE: ${RETIRE_JSON} present — in-pipeline retire fired."
  cat "${RETIRE_JSON}"
  RETIRE_FIRED=1
else
  echo "[L2] OK: no retire.json (R-b retire condition clear)."
  RETIRE_FIRED=0
fi

# --- 10. Diagnostic tails (post-hoc grep for Amendment B FAIL criteria) -------
echo "=== last 100 log lines ==="
tail -n 100 "${LOG}" || true
echo "---"
echo "=== grep: per-task grad-norm + var_pf_band_ratio + w_ratio trail ==="
grep -E "(var_pf_band_ratio|w_ratio|w_tau|w_pf|loss_pf|loss_tau|per-task grad-norm)" "${LOG}" | tail -n 200 || \
  echo "[L2] WARN: no diagnostic lines matched."

# --- 10a. Amendment B + L2-specific FAIL criteria (POST-HOC; raw signal only) -
# Amendment B (carry from v3 KILLER-3):
#   - pf/tau ratio > 3.0 at step 5000  -> FAIL
#   - loss_tau(5000) > 0.9 * loss_tau(1000)  -> FAIL
# R-b retire: any retire.json present  -> FAIL (handled in §9)
# L2-specific D-REDUCTION-AXIS-ORTHOGONAL surface:
#   - per-task w_ratio > 1000 AND var_pf_band_ratio < 1e-3 at step 200
#     -> escalate IMMEDIATELY to [D-53] per LEDGER §3 v5 [D-53] stub.
# Trailer emits raw step-200 + step-1000 + step-5000 lines so PI absorption
# can compute the ratios from a single grep tail; no pass/fail call is made
# in-sbatch (avoids step-500 mis-targeting class of errors per v3 KILLER-2).
echo "=== step-200 / step-1000 / step-5000 diagnostic snapshots (Amendment B + L2 surface) ==="
for STEP in 200 1000 5000; do
  echo "--- step ${STEP} window (+/- 5) ---"
  grep -E "^Step (${STEP}|$((STEP-1))|$((STEP+1))|$((STEP-2))|$((STEP+2))|$((STEP-3))|$((STEP+3))|$((STEP-4))|$((STEP+4))|$((STEP-5))|$((STEP+5)))/" "${LOG}" \
    | head -n 20 || echo "[L2] step ${STEP} window not observed."
done
echo "[L2] PI absorption pass computes: pf/tau ratio at step 5000;"
echo "     loss_tau(5000)/loss_tau(1000) ratio; w_ratio + var_pf_band_ratio at step 200."
echo "     D-REDUCTION-AXIS-ORTHOGONAL trigger: w_ratio>1000 AND var_pf_band_ratio<1e-3 @step200."

# --- 11. Artifact PCV ---------------------------------------------------------
echo "=== artifact PCV ==="
if [[ -d "${DEST}/checkpoints" ]]; then
  ls -la "${DEST}/checkpoints/"
  CKPT_COUNT=$(find "${DEST}/checkpoints" -name "*.pt" 2>/dev/null | wc -l)
  echo "[PCV] checkpoints: ${CKPT_COUNT} *.pt files in ${DEST}/checkpoints/"
else
  echo "[PCV] WARN: no checkpoint dir at ${DEST}/checkpoints"
fi

# --- 12. MLflow tag injection (mirrored from retune2 + L2 fields) -------------
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
    ("sprint_kind", "L2"),
    ("pf_log_reduction", "${PF_LOG_REDUCTION}"),
    ("lr", "${LR_MAX}"),
    ("warmup_steps", "${WARMUP_STEPS}"),
    ("compute", "juno"),
    ("juno_job_id", "${SLURM_JOB_ID}"),
    ("juno_run_tag", "${RUN_TAG}"),
    ("commit_sha", "${SHORTHASH}"),
    ("design_doc", "sprint_L1_direct_pf_loss"),
    ("decision_id", "[D-60]"),
    ("ledger_amendment", "v5-NON-PROVISIONAL-2026-05-22"),
]:
    client.set_tag(run.info.run_id, k, v)
print(f"[tagger] run_id={run.info.run_id} tagged OK")
PYEOF

# --- 13. Final summary banner -------------------------------------------------
echo "================================================================="
echo "  Sprint-L1 L2 escalation (reduction='mean') summary"
echo "================================================================="
echo "  RUN_TAG                    = ${RUN_TAG}"
echo "  RUN_NAME                   = ${RUN_NAME}"
echo "  PF_LOG_REDUCTION (L2 lever)= ${PF_LOG_REDUCTION}   [vs retune-1 'sum']"
echo "  LR_MAX (retune-1 baseline) = ${LR_MAX}"
echo "  WARMUP_STEPS (retune-1)    = ${WARMUP_STEPS}"
echo "  DRYRUN_EXIT (CPU 1-step)   = ${DRYRUN_EXIT}"
echo "  DRIVER_EXIT                = ${DRIVER_EXIT}"
echo "  RETIRE_FIRED               = ${RETIRE_FIRED}"
echo "  DRIVER_KILLED_BY_WATCHER   = ${DRIVER_KILLED_BY_WATCHER}"
echo "  DEST                       = ${DEST}"
echo "  LOG                        = ${LOG}"
echo "================================================================="

# --- 14. Exit-code routing ----------------------------------------------------
if [[ "${RETIRE_FIRED}" -eq 1 ]]; then
  echo "FATAL: L2 R-b retire criterion failed (retire.json present)." >&2
  exit 3
fi
if [[ "${DRIVER_KILLED_BY_WATCHER}" -eq 1 ]]; then
  echo "FATAL: step-rate watcher tripped (< ${STEPRATE_MIN_THRESH} steps/s at step ${STEPRATE_CHECK_AT_STEP})." >&2
  exit 4
fi

echo "=== done; results at ${DEST} ==="
echo "[L2] On PI-absorption FAIL verdict: gate closes; open [D-53] supervision-target redesign."
