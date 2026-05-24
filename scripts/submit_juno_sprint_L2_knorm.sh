#!/bin/bash
#SBATCH --job-name=sprint-L2-knorm-bspec-redispatch-fullladder
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --output=sprint-L2-knorm-juno-%j.out
#SBATCH --error=sprint-L2-knorm-juno-%j.err
#SBATCH --comment="D-53 (b) re-dispatch post-R20-v2; full ladder max_steps=5000; ledger amend v6"

# Sprint-L2 knorm [D-53] candidate (b) — k-space-normalized P_F target,
# panel-bound first dispatch on Juno A30. Routing source: defense-panel
# verdict 2026-05-23 (commit 3074596) + design doc
# experiments/nerf/design/D53_supervision_target_redesign.md candidate (b).
#
# WHY THIS SMOKE EXISTS
# ---------------------
# [D-61] closed sprint-L1 with the in-loss-function intervention class
# empirically exhausted across 4 attempts (LR, warmup, per-task clip, sum→
# mean reduction); all 4 R-b retired at step 200 with var_pf_band_ratio
# in [6.9e-7, 2.93e-6] (0.63 log10-decade spread). Per [D-61] inference:
# the pathology lives in the supervision-target structure, NOT in any
# scalar weighting / LR / reduction-op choice over a fixed target.
#
# [D-53] candidates are the first test of the supervision-target-redesign
# class — they are NOT pre-justified as structurally addressing the
# upstream pathology. Candidate (b) was panel-selected as first-binding
# (commit 3074596): k-space-normalized P_F target with three verbatim
# binding pre-commits (selectors 1–3 below).
#
# THREE BINDING PRE-COMMITS (from defense-panel verdict, verbatim)
# ----------------------------------------------------------------
# 1. σ_k² estimator: truth-side, batch-sample, EMA-stabilized with decay
#    0.99, floor 0.01 × median_k(σ_k²_truth) (relative, NOT absolute 1e-12).
# 2. Falsification trigger (post-hoc, step 200): max_k(|∂L/∂F|) /
#    median_k(|∂L/∂F|) > 100 AND var_pf_band_ratio < 1e-3 → R-b → close
#    immediately + escalate to [D-62].
# 3. Loss form: L = Σ_k (P_pred(k) − P_truth(k))² / σ_k²_truth(k) (NOT
#    log-MSE; SEPARATE new loss function pf_knorm_loss).
#
# STOP-GATE CALIBRATION (panel-bound clarification)
# -------------------------------------------------
# - Step 100: R20-v2 contract assertion (wrapper liveness + balanced-regime
#   admission per PI v6 2026-05-24 absorption of 202259 false-positive).
# - Step 200: PASS is PROVISIONAL (necessary, not sufficient — broke from
#   L1 cluster but does not establish rescue). Per-task ratio in [0.01, 100]
#   sustained, no NaN.
# - Step 1000: PASS is BINDING rescue verdict (var_pf_band_ratio > 0.05).
#   This threshold is locked per [D-37] symmetric-disclosure; post-hoc
#   relaxation forbidden absent symmetric-disclosure entry. 0.04 = FAIL,
#   NOT "close enough."
# - Step 5000: PASS is generalization confirmation (var_pf_band_ratio > 0.05
#   sustained, no late-train collapse).
# This re-dispatch (post-R20-v2 calibration) targets the FULL ladder
# (max_steps=5000) to reach step-1000 BINDING gate. Prior 202259 dispatch
# (max_steps=200) crashed at step 100 on R20-v1 false-positive — substantive
# finding: per-task ratio = 0.98 at step 100 (4.36 log10-decade reduction
# vs L1-class O(10^4)). See LEDGER §3 [D-53] (b) 202259 absorption block.
#
# HONEST FRAMING ([D-37] rule (a), hedged verbs throughout)
# ---------------------------------------------------------
# - This is the FIRST TEST of a k-space-normalized P_F supervision target
#   in this project. No published IGM-NeRF precedent (closest: Cabayol-
#   García+2023 log-k-MSE emulator, NOT a reconstruction-loss precedent).
# - Per-mode normalization MAY rescue variance collapse — pending step-
#   200 / step-1000 / step-5000 evidence.
# - AVOIDED: "structurally addresses the L1 pathology", "canonical fix".
#
# PRIOR-FAILURE LEDGER ([D-37]-ext rule 4 symmetric-disclosure)
# -------------------------------------------------------------
#   retune-1 (201734): var_pf_band_ratio=2.51e-6 (LR axis lr=1e-4/wu=1000)
#   retune-2 (201814): var_pf_band_ratio=6.9e-7  (LR axis lr=3e-5/wu=2000)
#   retune-3 (201856): var_pf_band_ratio=1.75e-6 (per-task clip; dead lever)
#   L2       (202109): var_pf_band_ratio=2.93e-6 (sum→mean reduction)
# All 4: R-b:pf_pred_variance_collapse at step 200. L2-knorm is the first
# dispatched [D-53] candidate; on R-b at step 200 the [D-62] stop-gate
# fires (architectural pivot — model-side capacity / inductive-bias,
# NOT loss-side) and we do NOT iterate to candidate (a) without panel
# re-review.
#
# Pinned configuration (single-axis: pf_knorm_loss replaces pf_log_mse_loss)
# -------------------------------------------------------------------------
#   --pf-knorm-loss                  *** L2-KNORM LEVER *** (new path)
#   --enable-l1-pf-loss              precondition for the L1 P_F path
#   --gradnorm-full                  full Chen+ 2018 second-order path
#   --lr_max 1e-4                    retune-1 baseline
#   --warmup_steps 1000              retune-1 baseline
#   --physics 1                      P1 (K1-absorbing tier)
#   --n_rays 64                      T1 scale
#   --seed 0
#   --microbatch 256
#   --max_steps 5000                 *** FULL LADDER *** (post-R20-v2 re-dispatch)
#   --mean_flux_obs 0.979            z=0.3 Becker+ 2013 anchor [D-11]
#   --l1-d24-baseline-tau-mse 0.01   R-a backstop
#   --checkpoint_interval 200        end-state checkpoint only
# NOTE: --pf-log-reduction is intentionally NOT passed — the knorm path
# is a different loss function (sum reduction is structural to the
# verbatim panel form L = Σ_k r_k² / σ_k²).

set -euo pipefail

# --- 0. Load Juno-side .env if present ----------------------------------------
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=/work/sxh240010/CosmoGasVision}"
: "${JUNO_SCRATCH:=/scratch/juno/sxh240010}"

# --- Pinned hyperparameters (panel-bound pre-commits) -------------------------
PHYSICS=1
N_RAYS=64
SEED=0
MICROBATCH=256
MAX_STEPS=5000                    # full ladder (post-R20-v2 re-dispatch) per PI v6
MEAN_FLUX_OBS=0.979
L1_D24_BASELINE_TAU_MSE=0.01
CHECKPOINT_INTERVAL=1000          # checkpoint at step-1000 BINDING gate + step-5000
LR_MAX=1e-4
WARMUP_STEPS=1000

# Abort-guard tunables (carry from L2):
STEPRATE_CHECK_AT_STEP=1000       # standard 1000-step check for full ladder
STEPRATE_MIN_THRESH=0.005

# --- 1. RUN_TAG generation ----------------------------------------------------
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SHORTHASH=$(cd "${JUNO_WORK}" && git rev-parse --short HEAD 2>/dev/null || echo nogit)
UUID_SUFFIX=$(uuidgen 2>/dev/null | cut -c1-6 \
              || head -c 6 /dev/urandom | base32 | tr '[:upper:]' '[:lower:]' | head -c 6 \
              || openssl rand -hex 3)
RUN_TAG="SprintL2-knorm-P${PHYSICS}-N${N_RAYS}-S${SEED}-lr${LR_MAX}-wu${WARMUP_STEPS}-${SHORTHASH}-${TIMESTAMP}-${UUID_SUFFIX}"
RUN_NAME="Sprint-L2-knorm-bspec-firstdispatch-P1T1-lr1e-4-wu1000-200steps"

if [[ "${RUN_TAG}" =~ [[:space:]] ]] || [[ "${RUN_TAG}" == *[\$\`\;\&\|]* ]]; then
  echo "FATAL: RUN_TAG '${RUN_TAG}' contains forbidden characters." >&2
  exit 1
fi

echo "================================================================="
echo "  Sprint-L2 knorm [D-53] candidate (b) — panel-bound first dispatch"
echo "  RUN_TAG=${RUN_TAG}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  L2-knorm lever: --pf-knorm-loss  (verbatim panel selector 3)"
echo "  σ_k² estimator: truth-side, EMA decay 0.99, floor 0.01×median (selector 1)"
echo "  Step-200 falsification trigger: grad-inflation>100 AND var_pf<1e-3 (selector 2)"
echo "  Juno A30, 04:00:00 wallclock (smoke 200-step window)"
echo "================================================================="

DEST="${JUNO_WORK}/cloud_runs/sprint_L2_knorm_${SLURM_JOB_ID}"
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
fi
if [[ ! -e "${JUNO_WORK}/SherwoodIGM_gal" ]]; then
  ln -s "${JUNO_SCRATCH}/SherwoodIGM_gal" "${JUNO_WORK}/SherwoodIGM_gal"
fi
if [[ ! -d "${JUNO_WORK}/Sherwood" ]]; then
  echo "FATAL: Sherwood symlink resolved to missing target." >&2
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

# --- 4. 1-step CPU dry-run pre-flight (carried from L2 commit 3dcf621) --------
# Cheapest possible proof that the --pf-knorm-loss code path does not crash
# and produces both task losses. If this fails, abort BEFORE consuming Juno GPU.
echo "=== 1-step CPU dry-run pre-flight ==="
DRYRUN_LOG="${DEST}/dryrun_cpu_1step.log"
DRYRUN_DEST="${DEST}/dryrun"
mkdir -p "${DRYRUN_DEST}/checkpoints"

set +e
(
  export CUDA_VISIBLE_DEVICES=""
  export NERF_RUN_NAME="${RUN_NAME}-DRYRUN"
  export NERF_RUN_TAG="${RUN_TAG}-DRYRUN"
  export MLFLOW_TRACKING_URI=""
  python -u experiments/nerf/pipeline.py \
      --gradnorm-full \
      --enable-l1-pf-loss \
      --pf-knorm-loss \
      --physics "${PHYSICS}" \
      --n_rays 64 \
      --seed "${SEED}" \
      --microbatch 64 \
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

if [[ "${DRYRUN_EXIT}" -ne 0 ]]; then
  echo "FATAL: 1-step CPU dry-run exited non-zero (${DRYRUN_EXIT}); aborting before Juno GPU." >&2
  exit 10
fi
# Word-boundary NaN/Inf check restricted to numeric output lines
# (avoids false-positive on MLflow 'INFO' log lines; carry from L2 fix 3dcf621).
if grep -E "(Step [0-9]|loss=|grad=|sprint-L2-knorm)" "${DRYRUN_LOG}" | grep -Eiq "\b(nan|inf|-inf)\b"; then
  echo "FATAL: 1-step CPU dry-run log contains NaN/Inf in numeric field; aborting before Juno GPU." >&2
  grep -E "(Step [0-9]|loss=|grad=|sprint-L2-knorm)" "${DRYRUN_LOG}" | grep -Ei "\b(nan|inf|-inf)\b" | head -n 20 >&2 || true
  exit 11
fi
if ! grep -Eq "^Step 1/" "${DRYRUN_LOG}"; then
  echo "FATAL: 1-step CPU dry-run missing 'Step 1' line; aborting before Juno GPU." >&2
  exit 12
fi
if ! grep -Eq "Training finished\." "${DRYRUN_LOG}"; then
  echo "FATAL: 1-step CPU dry-run did not reach 'Training finished.'; aborting before Juno GPU." >&2
  exit 13
fi
echo "[preflight] PASS: dry-run completed 1 step, no NaN/Inf, Training finished."
echo "[preflight] dry-run artifacts retained at ${DRYRUN_DEST} for forensics."

# --- 5. GPU + torch sanity ----------------------------------------------------
echo "=== GPU diagnostics ==="
nvidia-smi 2>&1 | awk 'NR<=10' || true
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; \
print(f'torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}')"

echo "=== run config ==="
echo "RUN_TAG=${RUN_TAG}"
echo "RUN_NAME=${RUN_NAME}"
echo "PHYSICS=${PHYSICS} N_RAYS=${N_RAYS} SEED=${SEED} MICROBATCH=${MICROBATCH}"
echo "MAX_STEPS=${MAX_STEPS} MEAN_FLUX_OBS=${MEAN_FLUX_OBS}"
echo "LR_MAX=${LR_MAX} WARMUP_STEPS=${WARMUP_STEPS}"
echo "L1_D24_BASELINE_TAU_MSE=${L1_D24_BASELINE_TAU_MSE}"
echo "GradNorm: FULL (Chen+ 2018 second-order; --gradnorm-full active)"
echo "Step-rate abort guard: < ${STEPRATE_MIN_THRESH} steps/s at step ${STEPRATE_CHECK_AT_STEP}"

export NERF_RUN_NAME="${RUN_NAME}"
export NERF_RUN_TAG="${RUN_TAG}"

LOG="${DEST}/driver.log"
SENTINEL_ABORT="${DEST}/STEPRATE_ABORT"
WATCHER_LOG="${DEST}/steprate_watcher.log"
touch "${LOG}"

# --- 6. Background step-rate watcher ------------------------------------------
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
echo "=== driver invocation: experiments/nerf/pipeline.py (sprint-L2-knorm) ==="
set +e
setsid python -u experiments/nerf/pipeline.py \
    --gradnorm-full \
    --enable-l1-pf-loss \
    --pf-knorm-loss \
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
  echo "[submit] STEPRATE_ABORT sentinel present; L2-knorm failed step-rate guard."
fi

if [[ -f "${WATCHER_LOG}" ]]; then
  echo "=== step-rate watcher log ==="
  cat "${WATCHER_LOG}"
fi

# --- 9. Retire-marker guard (R-b retire FAIL criterion) -----------------------
echo "=== retire-marker check (R-b retire FAIL criterion) ==="
RETIRE_JSON="${DEST}/checkpoints/retire.json"
if [[ -f "${RETIRE_JSON}" ]]; then
  echo "[L2-knorm] RETIRE: ${RETIRE_JSON} present — in-pipeline retire fired."
  cat "${RETIRE_JSON}"
  RETIRE_FIRED=1
else
  echo "[L2-knorm] OK: no retire.json (R-b retire condition clear)."
  RETIRE_FIRED=0
fi

# --- 10. Diagnostic tails -----------------------------------------------------
echo "=== last 100 log lines ==="
tail -n 100 "${LOG}" || true
echo "---"
echo "=== grep: knorm + var_pf_band_ratio + w_ratio + per-task grad-norm trail ==="
grep -E "(sprint-L2-knorm|var_pf_band_ratio|w_ratio|w_tau|w_pf|loss_pf|loss_tau|per-task grad-norm)" "${LOG}" | tail -n 200 || \
  echo "[L2-knorm] WARN: no diagnostic lines matched."

# --- 10a. Panel-bound step-200 falsification trigger (selector 2, verbatim) ---
# Trigger: max_k(|∂L/∂F|) / median_k(|∂L/∂F|) > 100 AND var_pf_band_ratio < 1e-3
# at step 200. The pipeline emits the proxy grad_inflation_metric per step on
# the [sprint-L2-knorm] diag line; we parse it together with the var_pf_band_ratio
# metric (which lands in retire.json on R-b retire, or in the MLflow metric
# scrape window otherwise). Trailer reports raw values + verdict; no in-sbatch
# pass/fail call beyond the retire.json gate (which fires at the pipeline-side
# R-b check on var_pf_band_ratio < 0.1).
echo "=== step-200 panel-bound falsification snapshot (selector 2 verbatim) ==="
echo "--- step 200 window (+/- 5) ---"
grep -E "^Step (200|19[5-9]|20[0-5])/" "${LOG}" | head -n 20 || echo "[L2-knorm] step 200 window not observed."
echo "--- [sprint-L2-knorm] diag lines (sigma_k_floor / median / grad_inflation_metric) ---"
grep "sprint-L2-knorm" "${LOG}" | tail -n 40 || echo "[L2-knorm] no knorm diag lines."
echo "[L2-knorm] PI absorption pass computes: grad_inflation_metric at step 200,"
echo "          var_pf_band_ratio at step 200 (from retire.json or last metric scrape)."
echo "[L2-knorm] panel-bound trigger: grad_inflation_metric > 100 AND var_pf_band_ratio < 1e-3"
echo "          -> verdict 'D-REDUCTION-AXIS-ORTHOGONAL-analog OR normalization-amplification-without-rescue'"
echo "          -> close immediately as R-b, escalate to [D-62] architectural pivot."

# --- 11. Artifact PCV ---------------------------------------------------------
echo "=== artifact PCV ==="
if [[ -d "${DEST}/checkpoints" ]]; then
  ls -la "${DEST}/checkpoints/"
  CKPT_COUNT=$(find "${DEST}/checkpoints" -name "*.pt" 2>/dev/null | wc -l)
  echo "[PCV] checkpoints: ${CKPT_COUNT} *.pt files in ${DEST}/checkpoints/"
else
  echo "[PCV] WARN: no checkpoint dir at ${DEST}/checkpoints"
fi

# --- 12. MLflow tag injection -------------------------------------------------
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
    ("stage", "sprint_L2_knorm"),
    ("physics_id", "${PHYSICS}"),
    ("redshift", "0.3"),
    ("loss_variant", "l1_knorm_pf"),
    ("gradnorm_variant", "full"),
    ("sprint_kind", "L2_knorm"),
    ("panel_first_dispatch", "true"),
    ("lr", "${LR_MAX}"),
    ("warmup_steps", "${WARMUP_STEPS}"),
    ("compute", "juno"),
    ("juno_job_id", "${SLURM_JOB_ID}"),
    ("juno_run_tag", "${RUN_TAG}"),
    ("commit_sha", "${SHORTHASH}"),
    ("design_doc", "D53_supervision_target_redesign"),
    ("decision_id", "[D-53]"),
    ("ledger_amendment", "v6-D53-(b)-binding-2026-05-23"),
]:
    client.set_tag(run.info.run_id, k, v)
print(f"[tagger] run_id={run.info.run_id} tagged OK")
PYEOF

# --- 13. Final summary banner -------------------------------------------------
echo "================================================================="
echo "  Sprint-L2 knorm [D-53] candidate (b) first-dispatch summary"
echo "================================================================="
echo "  RUN_TAG                    = ${RUN_TAG}"
echo "  RUN_NAME                   = ${RUN_NAME}"
echo "  pf_knorm_loss (LEVER)      = ACTIVE  (verbatim panel selector 3)"
echo "  σ_k² estimator             = truth-side EMA decay 0.99, floor 0.01×med"
echo "  LR_MAX                     = ${LR_MAX}"
echo "  WARMUP_STEPS               = ${WARMUP_STEPS}"
echo "  MAX_STEPS (smoke)          = ${MAX_STEPS}"
echo "  DRYRUN_EXIT (CPU 1-step)   = ${DRYRUN_EXIT}"
echo "  DRIVER_EXIT                = ${DRIVER_EXIT}"
echo "  RETIRE_FIRED               = ${RETIRE_FIRED}"
echo "  DRIVER_KILLED_BY_WATCHER   = ${DRIVER_KILLED_BY_WATCHER}"
echo "  DEST                       = ${DEST}"
echo "  LOG                        = ${LOG}"
echo "================================================================="

# --- 14. Exit-code routing ----------------------------------------------------
if [[ "${RETIRE_FIRED}" -eq 1 ]]; then
  echo "FATAL: L2-knorm R-b retire criterion failed (retire.json present)." >&2
  exit 3
fi
if [[ "${DRIVER_KILLED_BY_WATCHER}" -eq 1 ]]; then
  echo "FATAL: step-rate watcher tripped (< ${STEPRATE_MIN_THRESH} steps/s at step ${STEPRATE_CHECK_AT_STEP})." >&2
  exit 4
fi

echo "=== done; results at ${DEST} ==="
echo "[L2-knorm] On step-200 PROVISIONAL PASS verdict: dispatch step-1000 BINDING + step-5000 generalization runs."
echo "[L2-knorm] On step-200 R-b retire OR panel-bound trigger fire: gate closes; escalate to [D-62] architectural pivot."
