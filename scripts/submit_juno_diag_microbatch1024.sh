#!/bin/bash
#SBATCH --job-name=diag-microbatch1024-P1T1-knorm-200steps
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END
#SBATCH --output=diag-microbatch1024-juno-%j.out
#SBATCH --error=diag-microbatch1024-juno-%j.err
#SBATCH --comment="[D-62] addendum diagnostic (4) per panel BINDING + [D-64] routing on PASS"

# Sprint-L2 absorption-stack diagnostic (4) — microbatch=1024 single-axis
# Template: scripts/submit_juno_sprint_L2_knorm.sh (HEAD f91e398, the
# post-D-63 commit with v3 (b) absorption text). Single-axis change from
# the (b) re-dispatch baseline: --microbatch 1024 (was 256). All other
# configuration IDENTICAL.
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
# var_pf_band_ratio in [6.9e-7, 2.93e-6]) is **microbatch-coupled**. A
# PASS at microbatch=1024 would mean the entire [D-60]/[D-61]/[D-63]
# absorption stack at microbatch=256 is invalidated as an inference
# substrate for "supervision-target structurally pathological"; the
# microbatch axis (NOT the loss-function class) was the binding lever.
#
# ROUTING (panel-bound)
# ---------------------
# - PASS at step 200 (var_pf_band_ratio > 1e-3): **[D-64]
#   L1-scope-revision activates**. Inference: pathology is
#   microbatch-coupled at N=256, not loss-function-class-exhausted.
#   This does NOT vindicate any [D-62] architectural intervention —
#   PASS routes to scope-revision of the L1 absorption claim, NOT to
#   candidate-4 "success" framing.
# - FAIL at step 200 (R-b retire, var_pf_band_ratio < 1e-3): L1
#   in-loss-function exhaustion **holds at N=1024**; the microbatch
#   axis is empirically orthogonal to the pathology; the
#   supervision-target-structure inference (per [D-61]) is strengthened
#   across batch scales. Continue to diagnostic (5) — physics-variant
#   ablation — for the second falsification axis.
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
# narrows to "at microbatch=256," not "in-loss-function class
# exhausted"). AVOIDED: "rescue", "candidate-4 success", "architectural
# fix".
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
# Pinned configuration (single-axis: microbatch 256 -> 1024)
# ----------------------------------------------------------
#   --pf-knorm-loss                  (carried from b re-dispatch)
#   --enable-l1-pf-loss              (carried)
#   --gradnorm-full                  (carried)
#   --lr_max 1e-4                    (carried)
#   --warmup_steps 1000              (carried)
#   --physics 1                      (carried — P1 K1-absorbing tier)
#   --n_rays 64                      (carried — T1)
#   --seed 0                         (carried)
#   --microbatch 1024                *** DIAGNOSTIC LEVER *** (was 256)
#   --max_steps 200                  (panel cost estimate ~15 min Juno)
#   --mean_flux_obs 0.979            (carried — z=0.3 Becker+ 2013 [D-11])
#   --l1-d24-baseline-tau-mse 0.01   (carried — R-a backstop)
#
# NOTE on dry-run microbatch: the CPU 1-step dry-run below keeps
# microbatch=64 (vs production microbatch=1024). The dry-run verifies
# wiring only — argparse, loss-path traversal, both task losses present,
# Training-finished marker. Microbatch=1024 on CPU would be wasteful and
# offers no additional wiring-coverage. The PRODUCTION GPU run
# separately invokes microbatch=1024 (the diagnostic lever).

set -euo pipefail

# --- 0. Load Juno-side .env if present ----------------------------------------
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=/work/sxh240010/CosmoGasVision}"
: "${JUNO_SCRATCH:=/scratch/juno/sxh240010}"

# --- Pinned hyperparameters (single-axis diagnostic) --------------------------
PHYSICS=1
N_RAYS=64
SEED=0
MICROBATCH=1024                   # *** DIAGNOSTIC LEVER *** (was 256)
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
RUN_TAG="DiagMb1024-P${PHYSICS}-N${N_RAYS}-S${SEED}-mb${MICROBATCH}-${SHORTHASH}-${TIMESTAMP}-${UUID_SUFFIX}"
RUN_NAME="Diag-microbatch1024-P1T1-knorm-200steps"

if [[ "${RUN_TAG}" =~ [[:space:]] ]] || [[ "${RUN_TAG}" == *[\$\`\;\&\|]* ]]; then
  echo "FATAL: RUN_TAG '${RUN_TAG}' contains forbidden characters." >&2
  exit 1
fi

echo "================================================================="
echo "  [D-62] addendum diagnostic (4) — microbatch=1024 single-axis"
echo "  RUN_TAG=${RUN_TAG}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  Lever: --microbatch 1024 (was 256 in b re-dispatch); all else IDENTICAL"
echo "  Panel BINDING: dispatch (4) FIRST before any [D-62] architectural candidate"
echo "  PASS routing: var_pf_band_ratio > 1e-3 at step 200 -> [D-64] scope-revision"
echo "  FAIL routing: R-b at step 200 -> continue to diagnostic (5) physics-variant"
echo "  Juno A30, 04:00:00 wallclock"
echo "================================================================="

DEST="${JUNO_WORK}/cloud_runs/diag_microbatch1024_${SLURM_JOB_ID}"
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
# Wiring-only check: --pf-knorm-loss code path does not crash and produces
# both task losses. Dry-run uses microbatch=64 (NOT the production
# microbatch=1024) — see header NOTE. Production GPU run separately
# invokes microbatch=1024 below.
echo "=== 1-step CPU dry-run pre-flight (microbatch=64 wiring-only) ==="
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
echo "PHYSICS=${PHYSICS} N_RAYS=${N_RAYS} SEED=${SEED} MICROBATCH=${MICROBATCH} (LEVER, was 256)"
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
echo "=== driver invocation: experiments/nerf/pipeline.py (diag-microbatch1024) ==="
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
  echo "[submit] STEPRATE_ABORT sentinel present; diag-microbatch1024 failed step-rate guard."
fi

if [[ -f "${WATCHER_LOG}" ]]; then
  echo "=== step-rate watcher log ==="
  cat "${WATCHER_LOG}"
fi

# --- 9. Retire-marker guard (R-b retire FAIL criterion) -----------------------
echo "=== retire-marker check (R-b retire FAIL criterion) ==="
RETIRE_JSON="${DEST}/checkpoints/retire.json"
if [[ -f "${RETIRE_JSON}" ]]; then
  echo "[diag-mb1024] RETIRE: ${RETIRE_JSON} present — in-pipeline retire fired."
  cat "${RETIRE_JSON}"
  RETIRE_FIRED=1
else
  echo "[diag-mb1024] OK: no retire.json (R-b retire condition clear)."
  RETIRE_FIRED=0
fi

# --- 10. Diagnostic tails -----------------------------------------------------
echo "=== last 100 log lines ==="
tail -n 100 "${LOG}" || true
echo "---"
echo "=== grep: knorm + var_pf_band_ratio + w_ratio + per-task grad-norm trail ==="
grep -E "(sprint-L2-knorm|var_pf_band_ratio|w_ratio|w_tau|w_pf|loss_pf|loss_tau|per-task grad-norm)" "${LOG}" | tail -n 200 || \
  echo "[diag-mb1024] WARN: no diagnostic lines matched."

# --- 10a. Inherited (b)-spec step-200 falsification trigger -------------------
# Inherited verbatim from b re-dispatch trailer (selector 2):
# max_k(|dL/dF|) / median_k(|dL/dF|) > 100 AND var_pf_band_ratio < 1e-3 at step 200
echo "=== step-200 falsification snapshot (inherited from b re-dispatch trailer) ==="
echo "--- step 200 window (+/- 5) ---"
grep -E "^Step (200|19[5-9]|20[0-5])/" "${LOG}" | head -n 20 || echo "[diag-mb1024] step 200 window not observed."
echo "--- [sprint-L2-knorm] diag lines (sigma_k_floor / median / grad_inflation_metric) ---"
grep "sprint-L2-knorm" "${LOG}" | tail -n 40 || echo "[diag-mb1024] no knorm diag lines."
echo "[diag-mb1024] PI absorption pass computes: grad_inflation_metric at step 200,"
echo "             var_pf_band_ratio at step 200 (from retire.json or last metric scrape)."
echo "[diag-mb1024] inherited trigger: grad_inflation_metric > 100 AND var_pf_band_ratio < 1e-3"
echo "             -> verdict 'microbatch-axis orthogonal to pathology; L1 in-loss-function"
echo "                exhaustion holds at N=1024; continue to diagnostic (5) P2'."
echo "[diag-mb1024] PASS routing (var_pf_band_ratio > 1e-3 at step 200):"
echo "             -> [D-64] L1-scope-revision activates (NOT [D-62] candidate-4 success)."

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
    ("diagnostic_lever", "microbatch_1024"),
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
echo "  [D-62] addendum diagnostic (4) microbatch=1024 summary"
echo "================================================================="
echo "  RUN_TAG                    = ${RUN_TAG}"
echo "  RUN_NAME                   = ${RUN_NAME}"
echo "  DIAGNOSTIC LEVER           = --microbatch ${MICROBATCH} (was 256)"
echo "  pf_knorm_loss              = ACTIVE (carried from b re-dispatch)"
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
  echo "FATAL: diag-microbatch1024 R-b retire criterion failed (retire.json present)." >&2
  echo "       -> microbatch axis orthogonal to pathology; continue to diagnostic (5) P2." >&2
  exit 3
fi
if [[ "${DRIVER_KILLED_BY_WATCHER}" -eq 1 ]]; then
  echo "FATAL: step-rate watcher tripped (< ${STEPRATE_MIN_THRESH} steps/s at step ${STEPRATE_CHECK_AT_STEP})." >&2
  exit 4
fi

echo "=== done; results at ${DEST} ==="
echo "[diag-mb1024] On PASS (var_pf_band_ratio > 1e-3 at step 200):"
echo "             [D-64] L1-scope-revision activates — pathology microbatch-coupled at N=256."
echo "[diag-mb1024] On R-b retire: L1 in-loss-function exhaustion holds at N=1024;"
echo "             continue to diagnostic (5) physics-variant ablation (P2)."
