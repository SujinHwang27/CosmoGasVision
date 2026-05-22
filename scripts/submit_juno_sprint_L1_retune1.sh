#!/bin/bash
#SBATCH --job-name=sprint-L1-retune1-lr1e-4
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --output=sprint-L1-retune1-juno-%j.out
#SBATCH --error=sprint-L1-retune1-juno-%j.err

# Sprint-L1 [D-60] gate-pilot final-absorption — Retune Attempt 1 on Juno A30.
# Design doc: experiments/nerf/design/sprint_L1_direct_pf_loss.md (v2).
# PI authorized via amended (D)-hybrid ruling on scientific grounds
# (3 qualitatively-distinct retune directions per R13; NOT cost-gated per
# binding memory `feedback-no-cost-gate-on-juno`).
#
# Why this retune exists
# ----------------------
# Pilot 201712 (post-D1+D2 fix at HEAD cca67af) RETIRED at step 200 on
# R-b:pf_pred_variance_collapse. Under default --lr_max=5e-4 with warmup
# 5e-7 -> 5e-4 by step 200 + full Chen+ 2018 GradNorm at P1-T1, Var_k P_F^pred
# collapsed to 0.6% of truth (R-b threshold 10%) within 200 steps. Pre-clip
# grad norms hit 305107 at step 1 — gradient-magnitude shock from
# poorly-conditioned P_F-MSE term.
#
# Retune Attempt 1 lever: --lr_max 1e-4 (10x lower peak). All other knobs
# held constant per R13 / [D-37] symmetric-disclosure — single-axis change
# so any improvement is attributable to the LR retune alone.
#
# Pinned configuration (PI pre-committed grid)
# --------------------------------------------
#   --lr_max 1e-4                    (RETUNE LEVER — 10x lower peak)
#   --warmup_steps 1000              (kept; LR REDUCTION is the lever)
#   --gradnorm-full                  (full Chen+ 2018 path)
#   --enable-l1-pf-loss
#   --physics 1                      (P1, K1-absorbing tier)
#   --n_rays 64                      (T1 scale, unchanged)
#   --seed 0
#   --microbatch 256                 (unchanged)
#   --max_steps 5000                 (per design; clear 5k-survival bar)
#   --mean_flux_obs 0.979            (z=0.3 Becker+ 2013 anchor; [D-11])
#   --l1-d24-baseline-tau-mse 0.01   (R-a backstop)
#   --checkpoint_interval 5000       (end-state checkpoint only)
#
# Pre-committed pass condition (PI verdict — verbatim)
# ----------------------------------------------------
#   (a) Survive 5000 steps without R-b retire (or any other R-rule retire).
#   (b) var_pf_band_ratio > 0.3 at step 5000 (training-dynamics-survival bar;
#       lower than [D-13] gate's >0.9 expectation, but adequate to
#       demonstrate non-collapse).
#
# PASS -> PI authorizes escalation to 4-physics x T3 cross-physics tally
#         (>=3/4 PASS => CLOSE; <=2/4 => FAIL) per design v2 sec 5.
# FAIL -> Retune Attempt 2 (lr=3e-5 + warmup 2000 + log-domain P_F MSE)
#         — separately authored after Attempt 1 outcome.
#
# Commit-A step-100 contract assertion + step-rate watcher abort guard
# carried over from pilot 201712 — DO NOT touch.

set -euo pipefail

# --- 0. Load Juno-side .env if present ----------------------------------------
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=/work/sxh240010/CosmoGasVision}"
: "${JUNO_SCRATCH:=/scratch/juno/sxh240010}"

# --- Sprint-L1 retune1 pinned hyperparameters (PI pre-committed grid) ---------
PHYSICS=1
N_RAYS=64                  # T1 scale (unchanged from pilot 201712)
SEED=0
MICROBATCH=256
MAX_STEPS=5000
MEAN_FLUX_OBS=0.979
L1_D24_BASELINE_TAU_MSE=0.01
CHECKPOINT_INTERVAL=5000   # end-state checkpoint only

# Retune lever:
LR_MAX=1e-4                # 10x lower than pilot 201712 default 5e-4
WARMUP_STEPS=1000          # kept (LR REDUCTION is the lever, not warmup length)

# Abort-guard tunables (relaxed for cost-lift regime per memory
# feedback-no-cost-gate-on-juno: Juno compute is not cost-gated; only true
# stall signals warrant abort, not slow-but-progressing second-order autograd):
STEPRATE_CHECK_AT_STEP=1000
STEPRATE_MIN_THRESH=0.005   # ~3.5 hr for 60 steps = definite genuine-stall signal.
                            # Pilot 201712 ran at 0.07 steps/s under heavy clip events;
                            # retune1 should reach ≥0.07 with the smaller LR. The original
                            # 0.2 threshold was cost-calibrated, now obsolete.

# --- 1. RUN_TAG generation ----------------------------------------------------
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SHORTHASH=$(cd "${JUNO_WORK}" && git rev-parse --short HEAD 2>/dev/null || echo nogit)
UUID_SUFFIX=$(uuidgen 2>/dev/null | cut -c1-6 \
              || head -c 6 /dev/urandom | base32 | tr '[:upper:]' '[:lower:]' | head -c 6 \
              || openssl rand -hex 3)
RUN_TAG="SprintL1-Retune1-FullGN-P${PHYSICS}-N${N_RAYS}-S${SEED}-lr${LR_MAX}-${SHORTHASH}-${TIMESTAMP}-${UUID_SUFFIX}"
RUN_NAME="Sprint-L1-retune1-P1T1-lr1e-4-fullgradnorm-5k"

if [[ "${RUN_TAG}" =~ [[:space:]] ]] || [[ "${RUN_TAG}" == *[\$\`\;\&\|]* ]]; then
  echo "FATAL: RUN_TAG '${RUN_TAG}' contains forbidden characters." >&2
  exit 1
fi

echo "================================================================="
echo "  Sprint-L1 [D-60] gate-pilot final-absorption — Retune Attempt 1"
echo "  RUN_TAG=${RUN_TAG}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  Retune lever: --lr_max ${LR_MAX} (10x lower than pilot 201712)"
echo "  Juno A30 partition, 4:00:00 wallclock"
echo "================================================================="

DEST="${JUNO_WORK}/cloud_runs/sprint_L1_retune1_${SLURM_JOB_ID}"
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

# --- 5. Background step-rate watcher (carried over from pilot 201712) ---------
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
echo "=== driver invocation: experiments/nerf/pipeline.py (sprint-L1 retune1 lr=${LR_MAX}) ==="
set +e
setsid python -u experiments/nerf/pipeline.py \
    --gradnorm-full \
    --enable-l1-pf-loss \
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
  echo "[submit] STEPRATE_ABORT sentinel present; retune1 failed step-rate guard."
fi

if [[ -f "${WATCHER_LOG}" ]]; then
  echo "=== step-rate watcher log ==="
  cat "${WATCHER_LOG}"
fi

# --- 8. Retire-marker guard (pass-condition (a)) ------------------------------
echo "=== retire-marker check (retune1 pass-condition (a)) ==="
RETIRE_JSON="${DEST}/checkpoints/retire.json"
if [[ -f "${RETIRE_JSON}" ]]; then
  echo "[retune1] RETIRE: ${RETIRE_JSON} present — in-pipeline retire fired."
  cat "${RETIRE_JSON}"
  RETIRE_FIRED=1
else
  echo "[retune1] OK: no retire.json (pass-condition (a) satisfied)."
  RETIRE_FIRED=0
fi

# --- 9. var_pf_band_ratio + loss-finiteness post-hoc scan (pass-condition (b))
echo "=== last 100 log lines ==="
tail -n 100 "${LOG}" || true
echo "---"
echo "=== grep: var_pf_band_ratio / loss_pf trail (pass-condition (b) inspection) ==="
grep -E "(var_pf_band_ratio|w_ratio|w_tau|w_pf|loss_pf|loss_tau)" "${LOG}" | tail -n 80 || \
  echo "[retune1] WARN: no var_pf_band_ratio / loss_pf lines matched."

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
    ("retune_attempt", "1"),
    ("lr", "${LR_MAX}"),
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
echo "  Sprint-L1 Retune Attempt 1 summary"
echo "================================================================="
echo "  RUN_TAG                    = ${RUN_TAG}"
echo "  RUN_NAME                   = ${RUN_NAME}"
echo "  LR_MAX (retune lever)      = ${LR_MAX}"
echo "  WARMUP_STEPS               = ${WARMUP_STEPS}"
echo "  DRIVER_EXIT                = ${DRIVER_EXIT}"
echo "  RETIRE_FIRED               = ${RETIRE_FIRED}"
echo "  DRIVER_KILLED_BY_WATCHER   = ${DRIVER_KILLED_BY_WATCHER}"
echo "  DEST                       = ${DEST}"
echo "  LOG                        = ${LOG}"
echo "================================================================="

# --- 13. Exit-code routing (retune1 pass conditions) --------------------------
# Hard-fail iff retire.json present (pass-condition (a) violation) OR
# step-rate watcher tripped. Pass-condition (b) (var_pf_band_ratio > 0.3 at
# step 5000) is a human-inspection deliverable from the grep block above.
if [[ "${RETIRE_FIRED}" -eq 1 ]]; then
  echo "FATAL: retune1 pass-condition (a) failed (retire.json present)." >&2
  exit 3
fi
if [[ "${DRIVER_KILLED_BY_WATCHER}" -eq 1 ]]; then
  echo "FATAL: step-rate watcher tripped (< ${STEPRATE_MIN_THRESH} steps/s at step ${STEPRATE_CHECK_AT_STEP})." >&2
  exit 4
fi

echo "=== done; results at ${DEST} ==="
