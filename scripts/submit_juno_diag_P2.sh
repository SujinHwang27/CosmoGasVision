#!/bin/bash
#SBATCH --job-name=diag-P2-T1-microbatch256-knorm-200steps
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END
#SBATCH --output=diag-P2-juno-%j.out
#SBATCH --error=diag-P2-juno-%j.err
#SBATCH --comment="[D-62] addendum diagnostic (5) per panel BINDING + [D-64] routing on PASS"

# Sprint-L2 absorption-stack diagnostic (5) — physics-variant P2 single-axis
# Template: scripts/submit_juno_sprint_L2_knorm.sh (HEAD f91e398, the
# post-D-63 commit with v3 (b) absorption text). Single-axis change from
# the (b) re-dispatch baseline: --physics 2 (was 1). All other
# configuration IDENTICAL (including microbatch=256).
#
# WHY THIS DIAGNOSTIC EXISTS
# --------------------------
# Defense-panel design review on [D-62] (2026-05-24) returned NEEDS WORK
# with KILLER category-conflation finding: candidates originally tagged
# as architectural-pivot were in fact diagnostic levers on the absorption
# stack itself (P1-T1, microbatch=256, k-norm loss). PI re-tagged
# candidates (4) microbatch-ablation and (5) physics-variant-ablation as
# **absorption-stack diagnostics**, NOT [D-62] architectural-pivot
# candidates. Panel BINDING: dispatch (4) FIRST, (5) SECOND
# (parallel-eligible), before any [D-62] architectural candidate.
# LEDGER §3 [D-62] addendum + [D-64] stub (commit pending 2026-05-24).
#
# THIS DIAGNOSTIC TESTS whether the variance-collapse pathology observed
# across the 5-attempt L1+knorm sweep (all R-b retired at step 200 with
# var_pf_band_ratio in [6.9e-7, 2.93e-6]) is **P1-thermal-state-specific**.
# A PASS at P2 would mean the [D-60]/[D-61]/[D-63] absorption-stack
# inference scope narrows to the P1 thermal state; the CVPR claim
# narrows to a multi-physics generalization gap.
#
# ROUTING (panel-bound)
# ---------------------
# - PASS at step 200 (var_pf_band_ratio > 1e-3): **[D-64]
#   L1-scope-revision activates**. Inference: pathology is
#   P1-thermal-state-specific (one (gamma, T_0) combination from
#   Sherwood Tab. 1). CVPR claim narrows from "in-loss-function class
#   exhausted" to "in-loss-function class exhausted at P1; multi-physics
#   generalization is the open question."
# - FAIL at step 200 (R-b retire): pathology is **NOT P1-specific**;
#   inference is strengthened across thermal states. On (4) ALSO FAIL,
#   the [D-62] candidate ladder activates per the panel sequence:
#     (3) -> (2-pretraining) -> (1-reframed).
#
# PER-PHYSICS-VARIANT HONESTY CAVEAT
# ----------------------------------
# P2 corresponds to a specific (gamma, T_0) thermal-state combination
# per the Sherwood suite (Bolton+ 2017 Tab. 1). Generalizing
# "supervision-target structurally pathological" from a single variant
# is selection-bias-indictment territory: a single P2 PASS does not
# license claims about P3/P4/P5 thermal states. The PASS-routed [D-64]
# scope-revision must restrict the narrowed claim to {P1} explicitly
# and flag P3-P5 as untested unless additional variants are dispatched.
#
# Pre-committed FAIL criteria (per [D-37] symmetric-disclosure)
# -------------------------------------------------------------
# Same as the (b) re-dispatch:
#  - R-b:pf_pred_variance_collapse retire at step 200 (in-pipeline)
#  - (b)-specific FAIL trigger inherited from (b) sbatch trailer:
#    grad_inflation_metric > 100 AND var_pf_band_ratio < 1e-3 at step 200
#
# HONEST FRAMING ([D-37] rule (a))
# --------------------------------
# This is a DIAGNOSTIC on the absorption stack, NOT an architectural
# pivot. PASS does NOT vindicate any architectural intervention; PASS
# routes to [D-64] L1-scope-revision (the L1 absorption-claim scope
# narrows to "at P1," not "across thermal states"). AVOIDED: "rescue",
# "candidate-5 success", "architectural fix".
#
# PRIOR-FAILURE LEDGER ([D-37]-ext rule 4 symmetric-disclosure)
# -------------------------------------------------------------
#   retune-1 (201734): var_pf_band_ratio=2.51e-6 (LR axis lr=1e-4/wu=1000)
#   retune-2 (201814): var_pf_band_ratio=6.9e-7  (LR axis lr=3e-5/wu=2000)
#   retune-3 (201856): var_pf_band_ratio=1.75e-6 (per-task clip; dead lever)
#   L2       (202109): var_pf_band_ratio=2.93e-6 (sum->mean reduction)
#   L2-knorm-bspec (b re-dispatch): R-b at step 200 (5th attempt; k-norm
#                                   loss-function-class lever, still R-b)
# All 5: R-b:pf_pred_variance_collapse at step 200, microbatch=256, P1.
#
# Pinned configuration (single-axis: physics 1 -> 2)
# --------------------------------------------------
#   --pf-knorm-loss                  (carried from b re-dispatch)
#   --enable-l1-pf-loss              (carried)
#   --gradnorm-full                  (carried)
#   --lr_max 1e-4                    (carried)
#   --warmup_steps 1000              (carried)
#   --physics 2                      *** DIAGNOSTIC LEVER *** (was 1)
#   --n_rays 64                      (carried — T1)
#   --seed 0                         (carried)
#   --microbatch 256                 (carried — IDENTICAL to b re-dispatch)
#   --max_steps 200                  (panel cost estimate ~15 min Juno)
#   --mean_flux_obs 0.979            (carried — z=0.3 Becker+ 2013 [D-11])
#   --l1-d24-baseline-tau-mse 0.01   (carried — R-a backstop)

set -euo pipefail

# --- 0. Load Juno-side .env if present ----------------------------------------
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=/work/sxh240010/CosmoGasVision}"
: "${JUNO_SCRATCH:=/scratch/juno/sxh240010}"

# --- Pinned hyperparameters (single-axis diagnostic) --------------------------
PHYSICS=2                         # *** DIAGNOSTIC LEVER *** (was 1)
N_RAYS=64
SEED=0
MICROBATCH=256                    # carried IDENTICAL from b re-dispatch
MAX_STEPS=200                     # smoke: 200-step retire-or-pass window
MEAN_FLUX_OBS=0.979
L1_D24_BASELINE_TAU_MSE=0.01
CHECKPOINT_INTERVAL=200           # end-state checkpoint only (200-step window)
LR_MAX=1e-4
WARMUP_STEPS=1000

# Abort-guard tunables:
STEPRATE_CHECK_AT_STEP=100        # short window — early check
STEPRATE_MIN_THRESH=0.005

# --- 1. RUN_TAG generation ----------------------------------------------------
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SHORTHASH=$(cd "${JUNO_WORK}" && git rev-parse --short HEAD 2>/dev/null || echo nogit)
UUID_SUFFIX=$(uuidgen 2>/dev/null | cut -c1-6 \
              || head -c 6 /dev/urandom | base32 | tr '[:upper:]' '[:lower:]' | head -c 6 \
              || openssl rand -hex 3)
RUN_TAG="DiagP2-P${PHYSICS}-N${N_RAYS}-S${SEED}-mb${MICROBATCH}-${SHORTHASH}-${TIMESTAMP}-${UUID_SUFFIX}"
RUN_NAME="Diag-P2-T1-microbatch256-knorm-200steps"

if [[ "${RUN_TAG}" =~ [[:space:]] ]] || [[ "${RUN_TAG}" == *[\$\`\;\&\|]* ]]; then
  echo "FATAL: RUN_TAG '${RUN_TAG}' contains forbidden characters." >&2
  exit 1
fi

echo "================================================================="
echo "  [D-62] addendum diagnostic (5) — physics=2 single-axis"
echo "  RUN_TAG=${RUN_TAG}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  Lever: --physics 2 (was 1 in b re-dispatch); all else IDENTICAL"
echo "  Panel BINDING: dispatch (5) SECOND (parallel-eligible w/ (4))"
echo "  PASS routing: var_pf_band_ratio > 1e-3 at step 200 -> [D-64] scope-revision"
echo "                (claim narrows to P1-thermal-state-specific)"
echo "  FAIL routing: R-b at step 200 -> pathology NOT P1-specific;"
echo "                on (4) ALSO FAIL -> [D-62] candidate ladder activates"
echo "  Juno A30, 04:00:00 wallclock"
echo "================================================================="

DEST="${JUNO_WORK}/cloud_runs/diag_P2_${SLURM_JOB_ID}"
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
# Wiring-only check on the SAME diagnostic lever (--physics 2) AND the
# production microbatch=256. Dry-run uses --microbatch 256 (matches
# production; no oversized-batch concern as in diagnostic 4).
echo "=== 1-step CPU dry-run pre-flight (physics=2, microbatch=256) ==="
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
      --microbatch 256 \
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
echo "PHYSICS=${PHYSICS} (LEVER, was 1) N_RAYS=${N_RAYS} SEED=${SEED} MICROBATCH=${MICROBATCH}"
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
echo "=== driver invocation: experiments/nerf/pipeline.py (diag-P2) ==="
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
  echo "[submit] STEPRATE_ABORT sentinel present; diag-P2 failed step-rate guard."
fi

if [[ -f "${WATCHER_LOG}" ]]; then
  echo "=== step-rate watcher log ==="
  cat "${WATCHER_LOG}"
fi

# --- 9. Retire-marker guard (R-b retire FAIL criterion) -----------------------
echo "=== retire-marker check (R-b retire FAIL criterion) ==="
RETIRE_JSON="${DEST}/checkpoints/retire.json"
if [[ -f "${RETIRE_JSON}" ]]; then
  echo "[diag-P2] RETIRE: ${RETIRE_JSON} present — in-pipeline retire fired."
  cat "${RETIRE_JSON}"
  RETIRE_FIRED=1
else
  echo "[diag-P2] OK: no retire.json (R-b retire condition clear)."
  RETIRE_FIRED=0
fi

# --- 10. Diagnostic tails -----------------------------------------------------
echo "=== last 100 log lines ==="
tail -n 100 "${LOG}" || true
echo "---"
echo "=== grep: knorm + var_pf_band_ratio + w_ratio + per-task grad-norm trail ==="
grep -E "(sprint-L2-knorm|var_pf_band_ratio|w_ratio|w_tau|w_pf|loss_pf|loss_tau|per-task grad-norm)" "${LOG}" | tail -n 200 || \
  echo "[diag-P2] WARN: no diagnostic lines matched."

# --- 10a. Inherited (b)-spec step-200 falsification trigger -------------------
# Inherited verbatim from b re-dispatch trailer (selector 2):
# max_k(|dL/dF|) / median_k(|dL/dF|) > 100 AND var_pf_band_ratio < 1e-3 at step 200
echo "=== step-200 falsification snapshot (inherited from b re-dispatch trailer) ==="
echo "--- step 200 window (+/- 5) ---"
grep -E "^Step (200|19[5-9]|20[0-5])/" "${LOG}" | head -n 20 || echo "[diag-P2] step 200 window not observed."
echo "--- [sprint-L2-knorm] diag lines (sigma_k_floor / median / grad_inflation_metric) ---"
grep "sprint-L2-knorm" "${LOG}" | tail -n 40 || echo "[diag-P2] no knorm diag lines."
echo "[diag-P2] PI absorption pass computes: grad_inflation_metric at step 200,"
echo "          var_pf_band_ratio at step 200 (from retire.json or last metric scrape)."
echo "[diag-P2] inherited trigger: grad_inflation_metric > 100 AND var_pf_band_ratio < 1e-3"
echo "          -> verdict 'pathology NOT P1-specific; inference strengthened across"
echo "             thermal states; on (4) ALSO FAIL -> [D-62] candidate ladder activates'."
echo "[diag-P2] PASS routing (var_pf_band_ratio > 1e-3 at step 200):"
echo "          -> [D-64] L1-scope-revision activates; CVPR claim narrows to"
echo "             P1-thermal-state-specific. NOTE: single-variant PASS does NOT license"
echo "             generalization to P3/P4/P5 (selection-bias-indictment territory)."

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
    ("stage", "sprint_L2_knorm_diag"),
    ("physics_id", "${PHYSICS}"),
    ("redshift", "0.3"),
    ("loss_variant", "l1_knorm_pf"),
    ("gradnorm_variant", "full"),
    ("sprint_kind", "L2_knorm_diag"),
    ("diagnostic_lever", "physics_2"),
    ("panel_binding", "true"),
    ("lr", "${LR_MAX}"),
    ("warmup_steps", "${WARMUP_STEPS}"),
    ("microbatch", "${MICROBATCH}"),
    ("compute", "juno"),
    ("juno_job_id", "${SLURM_JOB_ID}"),
    ("juno_run_tag", "${RUN_TAG}"),
    ("commit_sha", "${SHORTHASH}"),
    ("design_doc", "D62_addendum_D64_stub"),
    ("decision_id", "[D-62]-addendum"),
    ("ledger_amendment", "D-62_addendum_D-64_stub_2026-05-24"),
]:
    client.set_tag(run.info.run_id, k, v)
print(f"[tagger] run_id={run.info.run_id} tagged OK")
PYEOF

# --- 13. Final summary banner -------------------------------------------------
echo "================================================================="
echo "  [D-62] addendum diagnostic (5) physics=2 summary"
echo "================================================================="
echo "  RUN_TAG                    = ${RUN_TAG}"
echo "  RUN_NAME                   = ${RUN_NAME}"
echo "  DIAGNOSTIC LEVER           = --physics ${PHYSICS} (was 1)"
echo "  pf_knorm_loss              = ACTIVE (carried from b re-dispatch)"
echo "  LR_MAX                     = ${LR_MAX}"
echo "  WARMUP_STEPS               = ${WARMUP_STEPS}"
echo "  MICROBATCH                 = ${MICROBATCH} (carried IDENTICAL)"
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
  echo "FATAL: diag-P2 R-b retire criterion failed (retire.json present)." >&2
  echo "       -> pathology NOT P1-specific; on (4) ALSO FAIL -> [D-62] candidate ladder." >&2
  exit 3
fi
if [[ "${DRIVER_KILLED_BY_WATCHER}" -eq 1 ]]; then
  echo "FATAL: step-rate watcher tripped (< ${STEPRATE_MIN_THRESH} steps/s at step ${STEPRATE_CHECK_AT_STEP})." >&2
  exit 4
fi

echo "=== done; results at ${DEST} ==="
echo "[diag-P2] On PASS (var_pf_band_ratio > 1e-3 at step 200):"
echo "          [D-64] L1-scope-revision activates — claim narrows to P1-thermal-state."
echo "          Single-variant PASS does NOT license P3/P4/P5 generalization."
echo "[diag-P2] On R-b retire: pathology NOT P1-specific; on (4) ALSO FAIL,"
echo "          [D-62] candidate ladder activates per (3) -> (2-pretraining) -> (1-reframed)."
