#!/bin/bash
#SBATCH --job-name=sprint5-cprime-48cube
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --output=sprint5cprime-juno-%j.out
#SBATCH --error=sprint5cprime-juno-%j.err

# Sprint-5 (c′)-at-48³ substrate-extension probe on Juno HPC H100.
# Design doc: experiments/nerf/design/sprint5_cprime_48cube.md v4 (NON-PROVISIONAL per [D-55]).
# Predecessor: scripts/submit_juno_sprint4_30epoch.sh; sprint-4 run 198571 (run_id sprint4_1778774878).
#
# Distinct from sprint-4: (i) H100 partition (sprint-4 was a30); (ii) routes through
# run_sprint5_cprime_substrate_extension via --n_seeds 5 --baseline mvsk --crop_size 48;
# (iii) 5-seed loop emits per-seed headlines + per-crop JSONLs under
# cloud_runs/${RUN_TAG}/eval/ (NOT experiments/nerf/artifacts/eval/sprint4/);
# (iv) S4 cuDNN-determinism pre-flight at 48³ on H100 is the hard gate-(c) prereq.
#
# 8th-gap defense (inherited verbatim from sprint-4 submit script post-2026-05-14 patch):
#   - `set +e` wraps the driver invocation. The driver (scripts/train_truth_baseline.py:903)
#     now always exits 0 on completed outcome routing per [D-52] amendment 7; non-zero
#     exit here indicates a genuine runtime exception, not branch-iv routing.
#   - `set -e` is re-enabled around PCV so PCV's own hard-fails still trip.
# Do NOT remove the wrap: it preserves the invariant "if artifacts exist on disk, PCV copies them out"
# even if a future driver change re-introduces non-zero exits on legitimate outcomes.

set -euo pipefail

# --- 0. Load Juno-side .env if present ----------------------------------------
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=/work/sxh240010/CosmoGasVision}"
: "${JUNO_SCRATCH:=/scratch/juno/sxh240010}"
: "${JUNO_RESULTS_ROOT:=${JUNO_WORK%/CosmoGasVision}/sprint5_cprime_results}"

# Optional overrides (export at sbatch time):
#   EPOCHS=30 N_CROPS_TRAIN=5000 N_CROPS_VAL=1000 N_CROPS_TEST=2000
#   BATCH_SIZE=64 LR_MAX=3e-4
# Defaults match design doc v4 §5.3 + §6.
: "${EPOCHS:=30}"
: "${N_CROPS_TRAIN:=5000}"
: "${N_CROPS_VAL:=1000}"
: "${N_CROPS_TEST:=2000}"
: "${BATCH_SIZE:=64}"
: "${LR_MAX:=3e-4}"
: "${LR_MIN:=3e-6}"
: "${WEIGHT_DECAY:=1e-4}"
: "${WARMUP_EPOCHS:=1}"
: "${EARLY_STOP_PATIENCE:=5}"
: "${REDSHIFT:=0.300}"
: "${N_BOOTSTRAP:=1000}"
: "${CROP_SIZE:=48}"
: "${N_SEEDS:=5}"
: "${N_GRID:=768}"
: "${BASELINE:=mvsk}"

# --- 1. RUN_TAG generation (PI caveat C3: uuidgen fallback) -------------------
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SHORTHASH=$(cd "${JUNO_WORK}" && git rev-parse --short HEAD 2>/dev/null || echo nogit)
UUID_SUFFIX=$(uuidgen 2>/dev/null | cut -c1-6 \
              || head -c 6 /dev/urandom | base32 | tr '[:upper:]' '[:lower:]' | head -c 6 \
              || openssl rand -hex 3)
RUN_TAG="Sprint5cprime-48cube-${N_SEEDS}seed-${SHORTHASH}-${TIMESTAMP}-${UUID_SUFFIX}"

# Defensive: reject RUN_TAG with whitespace / special chars
if [[ "${RUN_TAG}" =~ [[:space:]] ]] || [[ "${RUN_TAG}" == *[\$\`\;\&\|]* ]]; then
  echo "FATAL: RUN_TAG '${RUN_TAG}' contains forbidden characters." >&2
  exit 1
fi

echo "================================================================="
echo "  Sprint-5 (c′) at 48³ — substrate-extension probe"
echo "  RUN_TAG=${RUN_TAG}"
echo "  Design doc: experiments/nerf/design/sprint5_cprime_48cube.md v4"
echo "  [D-55] NON-PROVISIONAL"
echo "================================================================="

DEST="${JUNO_RESULTS_ROOT}/${RUN_TAG}"
if [[ -d "${DEST}" ]]; then
  echo "FATAL: results dir already exists: ${DEST}" >&2
  echo "  Suggestion: rm -rf '${DEST}' OR pick a different RUN_TAG suffix." >&2
  exit 1
fi
mkdir -p "${DEST}"

# --- 2. PI caveat C4: uncommitted-tracked-content guard -----------------------
# (Verbatim from sprint-4 precedent — --untracked-files=no ignores cloud_runs/,
# old logs, etc.; core.fileMode false on Juno-side ignores chmod-only changes;
# experiments/nerf/talk/*.pptx untracked-content carve-out implicit via
# --untracked-files=no.)
if [[ -n "$(cd "${JUNO_WORK}" && git status --porcelain --untracked-files=no)" ]]; then
  echo "FATAL: JUNO_WORK has uncommitted CONTENT changes; SHORTHASH=${SHORTHASH} would mislead." >&2
  echo "Commit (or stash) before sbatch. Output of git status:" >&2
  (cd "${JUNO_WORK}" && git status --short --untracked-files=no) >&2
  exit 2
fi

# --- 3. Code refresh + environment activation ---------------------------------
cd "${JUNO_WORK}"
git pull origin exp/nerf || echo "[submit] git pull non-fatal warning (continuing on local HEAD ${SHORTHASH})"

source "${JUNO_WORK}/.venv/bin/activate"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export GIT_PYTHON_REFRESH=quiet

# PI caveat C1 (informational for sprint-5: driver does NOT integrate with MLflow
# at current code surface, but we preserve the fresh-store invariant for any
# future tag-injection step). Ping if MLFLOW_TRACKING_URI is set; fall through
# to nullcontext semantics if unreachable.
if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
  if python -c "import urllib.request,sys; urllib.request.urlopen('${MLFLOW_TRACKING_URI}', timeout=3)" 2>/dev/null; then
    echo "[submit] MLflow tracker ${MLFLOW_TRACKING_URI} reachable (informational; (c′) driver does not log)"
  else
    echo "[submit] MLflow tracker ${MLFLOW_TRACKING_URI} unreachable; (c′) driver does not log — proceeding."
  fi
fi

# --- 4. GPU + torch sanity ----------------------------------------------------
echo "=== GPU diagnostics ==="
# `nvidia-smi | head -10` under `pipefail` triggers SIGPIPE (exit 141) because
# `head` closes the pipe early. Run `nvidia-smi` unpiped (it's already concise);
# capture full output to log + echo first 10 lines via shell-only ops.
nvidia_smi_out=$(nvidia-smi 2>&1 || true)
printf '%s\n' "${nvidia_smi_out}" | awk 'NR<=10'
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; \
print(f'torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}')"

echo "=== run config ==="
echo "RUN_TAG=${RUN_TAG}"
echo "CROP_SIZE=${CROP_SIZE} N_SEEDS=${N_SEEDS} BASELINE=${BASELINE}"
echo "EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE}"
echo "N_CROPS train/val/test = ${N_CROPS_TRAIN}/${N_CROPS_VAL}/${N_CROPS_TEST} per physics × 4"
echo "N_GRID=${N_GRID} REDSHIFT=${REDSHIFT} N_BOOTSTRAP=${N_BOOTSTRAP}"

# --- 5. S4 cuDNN-determinism pre-flight at 48³ on H100 (design v4 §6 gate-(c)) -
# Hard pre-dispatch gate. If FAIL, the relaxed |Δp| < 1e-5 threshold applies
# with footnote per v4 §6; we do NOT exit on S4 FAIL (the relaxed gate is the
# documented fallback per design doc).
echo "=== S4 pre-flight: cuDNN-determinism at 48³ on H100 ==="
mkdir -p "${DEST}"
set +e
python -u -m pytest tests/test_sprint5_pre_flight.py::TestCuDNNDeterminism::test_cudnn_determinism_at_48cube -v \
    2>&1 | tee "${DEST}/s4_preflight.log"
S4_EXIT=${PIPESTATUS[0]}
set -e

if [[ "${S4_EXIT}" -eq 0 ]]; then
  echo "[submit] S4 cuDNN-determinism PASS; gate-(c) bit-identical threshold applies."
  S4_VERDICT="PASS_bit_identical"
else
  echo "[submit] S4 cuDNN-determinism FAIL on H100 (exit=${S4_EXIT}); gate-(c) RELAXED to |Δp| < 1e-5 with footnote per design doc v4 §6 gate-(c)."
  S4_VERDICT="FAIL_relaxed_to_1e-5_with_footnote"
fi
echo "S4_VERDICT=${S4_VERDICT}" >> "${DEST}/s4_preflight.log"

# --- 6. Main driver invocation (8th-gap defense: set +e around driver) --------
# Driver always exits 0 on completed routing per train_truth_baseline.py:903.
# Non-zero exit indicates a genuine runtime exception; outcome routing lives in
# cloud_runs/${RUN_TAG}/eval/headline.json `outcome_branch` field, NOT exit code.
echo "=== driver invocation: scripts/train_truth_baseline.py --crop_size ${CROP_SIZE} --n_seeds ${N_SEEDS} --baseline ${BASELINE} ==="
mkdir -p "${DEST}"
set +e
python -u scripts/train_truth_baseline.py \
    --crop_size "${CROP_SIZE}" \
    --n_seeds "${N_SEEDS}" \
    --baseline "${BASELINE}" \
    --run_tag "${RUN_TAG}" \
    --n_grid "${N_GRID}" \
    --epochs "${EPOCHS}" \
    --n_crops_train "${N_CROPS_TRAIN}" \
    --n_crops_val "${N_CROPS_VAL}" \
    --n_crops_test "${N_CROPS_TEST}" \
    --batch_size "${BATCH_SIZE}" \
    --lr_max "${LR_MAX}" \
    --lr_min "${LR_MIN}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --early_stop_patience "${EARLY_STOP_PATIENCE}" \
    --redshift "${REDSHIFT}" \
    --device auto \
    --n_bootstrap "${N_BOOTSTRAP}" \
    2>&1 | tee "${DEST}/driver.log"
DRIVER_EXIT=${PIPESTATUS[0]}
echo "[submit] DRIVER_EXIT=${DRIVER_EXIT} (informational; outcome routing in headline.json)"
set -e

# Parse outcome_branch from headline.json (graceful if missing)
HEADLINE_PATH="${JUNO_WORK}/cloud_runs/${RUN_TAG}/eval/headline.json"
OUTCOME_BRANCH="<unknown>"
if [[ -f "${HEADLINE_PATH}" ]]; then
  OUTCOME_BRANCH=$(python -c "import json; \
print(json.load(open('${HEADLINE_PATH}')).get('outcome_branch', '<not-set>'))" 2>/dev/null || echo "<parse-failure>")
fi
echo "[submit] outcome_branch=${OUTCOME_BRANCH}"

# --- 7. PCV (Producer-Consumer Verification) — copy-out + assert --------------
# Per producer-consumer pattern: (c′) driver writes to cloud_runs/${RUN_TAG}/eval/
# and cloud_runs/${RUN_TAG}/checkpoints/. The driver currently does NOT save
# per-seed ResNet checkpoints (gap surfaced 2026-05-15; logged as PCV MISSING
# per spec, NOT a script-exit condition). Top-level confusion_matrix.json /
# r_bin_edges.json / training_log.csv likewise are not emitted by the (c′)
# path (those are sprint-4 legacy artifacts under experiments/nerf/artifacts/
# eval/sprint4/); PCV logs MISSING and continues.
SRC_ROOT="${JUNO_WORK}/cloud_runs/${RUN_TAG}"
mkdir -p "${DEST}/eval" "${DEST}/checkpoints"

# (a) Per-seed eval artifacts: headline_seed_{42,142,242,342,442}.json + per_crop_seed_*.jsonl
PCV_MISSING_COUNT=0
PCV_PRESENT_COUNT=0
SEED_SCHEDULE_FULL=(42 142 242 342 442)
SEED_SCHEDULE=("${SEED_SCHEDULE_FULL[@]:0:${N_SEEDS}}")
for s in "${SEED_SCHEDULE[@]}"; do
  for suffix in "headline_seed_${s}.json" "per_crop_seed_${s}.jsonl"; do
    SRC="${SRC_ROOT}/eval/${suffix}"
    if [[ -f "${SRC}" ]]; then
      cp "${SRC}" "${DEST}/eval/"
      PCV_PRESENT_COUNT=$((PCV_PRESENT_COUNT + 1))
    else
      echo "PCV MISSING: ${SRC}"
      PCV_MISSING_COUNT=$((PCV_MISSING_COUNT + 1))
    fi
  done
done

# (b) Top-level eval artifacts (seed-averaged headline + legacy-style aggregates)
for suffix in headline.json confusion_matrix.json r_bin_edges.json training_log.csv; do
  SRC="${SRC_ROOT}/eval/${suffix}"
  if [[ -f "${SRC}" ]]; then
    cp "${SRC}" "${DEST}/eval/"
    PCV_PRESENT_COUNT=$((PCV_PRESENT_COUNT + 1))
  else
    echo "PCV MISSING: ${SRC}"
    PCV_MISSING_COUNT=$((PCV_MISSING_COUNT + 1))
  fi
done

# (c) Per-seed best checkpoints (>10 MB each → DVC-track candidates)
for s in "${SEED_SCHEDULE[@]}"; do
  SRC="${SRC_ROOT}/checkpoints/resnet18_3d_4class_best_seed_${s}.pt"
  if [[ -f "${SRC}" ]]; then
    cp "${SRC}" "${DEST}/checkpoints/"
    PCV_PRESENT_COUNT=$((PCV_PRESENT_COUNT + 1))
  else
    echo "PCV MISSING: ${SRC}"
    PCV_MISSING_COUNT=$((PCV_MISSING_COUNT + 1))
  fi
done

# (d) R23 BANK requirement: per-crop JSONL non-empty + correct line count
# Expected: 4 physics × 2000 test crops = 8000 lines per seed (at production
# N_CROPS_TEST=2000). Compute expected lines from env to handle SMOKE/override.
EXPECTED_JSONL_LINES=$((4 * N_CROPS_TEST))
echo "=== R23 per-crop JSONL line-count verification (expect ${EXPECTED_JSONL_LINES}/seed) ==="
for s in "${SEED_SCHEDULE[@]}"; do
  JSONL="${DEST}/eval/per_crop_seed_${s}.jsonl"
  if [[ -f "${JSONL}" ]]; then
    LINES=$(wc -l < "${JSONL}")
    if [[ "${LINES}" -eq "${EXPECTED_JSONL_LINES}" ]]; then
      echo "  seed=${s}: ${LINES} lines  OK"
    else
      echo "  seed=${s}: ${LINES} lines  WARN (expected ${EXPECTED_JSONL_LINES})"
    fi
  else
    echo "  seed=${s}: <missing>  PCV MISSING (logged above)"
  fi
done

# --- 8. DVC tracking on checkpoints (>10 MB; CLAUDE.md threshold) -------------
# Best-effort; script does NOT auto-commit. DVC push deferred to manual step.
if command -v dvc >/dev/null 2>&1; then
  echo "=== DVC add (best-effort) ==="
  CKPT_GLOB="${DEST}/checkpoints/resnet18_3d_4class_best_seed_*.pt"
  if compgen -G "${CKPT_GLOB}" >/dev/null; then
    set +e
    (cd "${JUNO_WORK}" && dvc add ${CKPT_GLOB#${JUNO_WORK}/}) || \
        echo "[submit] dvc add returned non-zero; review manually."
    set -e
    echo "[submit] Reminder: manually \`git add cloud_runs/${RUN_TAG}/checkpoints/*.pt.dvc\` + commit + \`dvc push\`."
  else
    echo "[submit] No per-seed checkpoints to DVC-track (driver did not emit; logged in PCV)."
  fi
else
  echo "[submit] dvc not in PATH on Juno; skipping DVC add. Run from host after rsync."
fi

# --- 9. Final summary banner --------------------------------------------------
echo "================================================================="
echo "  Sprint-5 (c′) run summary"
echo "================================================================="
echo "  RUN_TAG               = ${RUN_TAG}"
echo "  S4_EXIT               = ${S4_EXIT}  (${S4_VERDICT})"
echo "  DRIVER_EXIT           = ${DRIVER_EXIT}"
echo "  outcome_branch        = ${OUTCOME_BRANCH}"
echo "  PCV artifacts present = ${PCV_PRESENT_COUNT}"
echo "  PCV artifacts missing = ${PCV_MISSING_COUNT}"
echo "  DEST                  = ${DEST}"
echo "-----------------------------------------------------------------"
if [[ -f "${HEADLINE_PATH}" ]]; then
  python -u - <<PYEOF
import json
hl = json.load(open("${HEADLINE_PATH}"))
def g(k, fmt=None):
    v = hl.get(k, "<missing>")
    if fmt and isinstance(v, (int, float)):
        return fmt.format(v)
    return v
print(f"  seed_averaged A_resnet            = {g('seed_averaged_resnet_accuracy', '{:.4f}')}")
print(f"  seed_averaged A_mvsk @ 48^3       = {g('seed_averaged_mvsk_at_48cube', '{:.4f}')}")
print(f"  seed_averaged A_mvsk @ 32^3 (A4)  = {g('seed_averaged_mvsk_at_32cube', '{:.4f}')}")
print(f"  seed_averaged AD-5 margin (pp)    = {g('seed_averaged_ad5_margin_pp', '{:.4f}')}")
print(f"  AD-5 gate-(e) PASS (seed-avg)     = {g('ad5_gate_e_pass_seed_averaged')}")
print(f"  mvsk_threshold_tightened (any)    = {g('mvsk_threshold_tightened_any_seed')}")
# R20/R21 empirical σ_seed + ρ_seed surfaces if the driver added them
for fld in ('seed_emp_sigma', 'seed_emp_rho', 'rho_emp'):
    if fld in hl:
        print(f"  {fld:30s}    = {hl[fld]}")
PYEOF
else
  echo "  headline.json missing — cannot summarize driver outputs."
fi
echo "================================================================="
echo "=== done; results at ${DEST} ==="
