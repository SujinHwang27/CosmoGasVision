#!/bin/bash
#SBATCH --job-name=Sprint4-TruthBaseline-30ep
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=sprint4-juno-%j.out
#SBATCH --error=sprint4-juno-%j.err

# Sprint-4 truth-baseline 3D ResNet classifier ([D-47] option-C step-1 →
# [D-52] option-(c) post-pre-review amended) on UTD Juno HPC.
#
# Distinct from scripts/submit_juno_stage2b.sh: that template trains the
# NeRF MLP via experiments/nerf/pipeline.py on sightline data; this
# script trains a 3D ResNet-18 classifier via scripts/train_truth_baseline.py
# on IGM_gal CIC ρ crops. Sprint-4 is a single job consuming all 4 physics
# jointly (not per-physics-cell like Stage 2b), so no PHYSICS_ID env var.
#
# See `.claude/skills/juno-hpc/SKILL.md` for the cluster contract; [D-52]
# post-pre-review amendments (LEDGER §3) define the success criterion.
# This is the first dispatch on the post-[D-52] driver; the script
# expects:
#   1. .venv already built per the juno-hpc skill SKILL.md "Environment
#      bring-up" section (cu124 torch override applied).
#   2. SherwoodIGM_gal/extracted/<prefix>/snapdir_012/snap_012.*.hdf5
#      mirrored to ${JUNO_SCRATCH}/SherwoodIGM_gal/extracted/ (4 physics
#      variants × 16 hdf5 files = 64 total; ~160 GiB extracted, OR ~87
#      GiB tarballs that the user extracts on Juno-side once).
#   3. Sherwood/ sightline tree also mirrored (~6 GiB, needed for [D-48]
#      cache lookup metadata; see ${JUNO_SCRATCH}/sherwood symlink).
#
# PCV (Producer-Consumer Verification) — sprint-4 writes to:
#   experiments/nerf/artifacts/eval/sprint4/sprint4_<runid>_{headline,
#     r_bin_edges, confusion_matrix, training_log, smoke}.{json,csv}
#   experiments/nerf/artifacts/sprint4/checkpoints/resnet18_3d_4class_best.pt
# Copy-out verifies all required artifacts exist before scratch cleanup.

set -euo pipefail

# --- 0. Load Juno-side .env if present ----------------------------------------
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=/work/sxh240010/CosmoGasVision}"
: "${JUNO_SCRATCH:=/scratch/juno/sxh240010}"

# Optional overrides (export at sbatch time to bump any default):
#   EPOCHS=30 N_CROPS_TRAIN=5000 N_CROPS_VAL=1000 N_CROPS_TEST=2000
#   BATCH_SIZE=64 LR_MAX=3e-4 SEED_TRAIN=42 SEED_MODEL=42 N_BOOTSTRAP=1000
# Defaults match the [D-51] / [D-52] amended success criterion.
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
: "${SEED_TRAIN:=42}"
: "${SEED_VAL:=142}"
: "${SEED_TEST:=242}"
: "${SEED_MODEL:=42}"
: "${SEED_AUG:=42}"
: "${REDSHIFT:=0.300}"
: "${N_BOOTSTRAP:=1000}"
: "${JUNO_BATCH:=sprint4-30epoch}"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SHORTHASH=$(cd "${JUNO_WORK}" && git rev-parse --short HEAD 2>/dev/null || echo nogit)

# PI caveat C4 (binding, per 2026-05-13d Juno-dispatch review): the script
# cp -r's ${JUNO_WORK} to scratch; if ${JUNO_WORK} has uncommitted CONTENT
# changes to tracked files when sbatch fires, those run but SHORTHASH
# misrepresents what executed. [D-37] honest-reporting hygiene rule for HPC
# dispatches.
#
# Refinement (2026-05-14, first-Juno-dispatch-feedback): use
# --untracked-files=no to ignore safe untracked artifacts (old .err/.out
# logs, backup files, cloud_runs/). Rely on `git config core.fileMode false`
# on Juno-side .git/config to ignore mode-only changes (Juno workflow
# chmod's some scripts executable without committing). The remaining
# tracked-content diff is the only thing that can mis-represent what
# executed.
if [[ -n "$(cd "${JUNO_WORK}" && git status --porcelain --untracked-files=no)" ]]; then
  echo "FATAL: JUNO_WORK has uncommitted CONTENT changes; SHORTHASH=${SHORTHASH} would mislead." >&2
  echo "Commit (or stash) before sbatch. Output of git status:" >&2
  (cd "${JUNO_WORK}" && git status --short --untracked-files=no) >&2
  exit 9
fi

# PI caveat C3: uuidgen is not guaranteed on minimal Juno login nodes; fall
# back to openssl for entropy. Either path produces 6 hex chars.
UUID_SUFFIX=$(uuidgen 2>/dev/null | cut -c1-6 || openssl rand -hex 3)
RUN_TAG="Sprint4-30ep-${SHORTHASH}-${TIMESTAMP}-${UUID_SUFFIX}"
RUN_NAME="Sprint4-TruthBaseline-3DResNet18-${SHORTHASH}-${TIMESTAMP}"

# --- 1. Stage code + data into scratch RUN_DIR --------------------------------
RUN_DIR="${JUNO_SCRATCH}/sprint4/${RUN_TAG}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

# Code: src/ + experiments/ + scripts/ + tests/ + pyproject + lockfile
cp -r "${JUNO_WORK}"/{src,experiments,scripts,tests,pyproject.toml,uv.lock} .

# Data symlinks — both Sherwood/ and SherwoodIGM_gal/ live on scratch.
# Sherwood/ (sightline + tauH1 + .rho_field_cache/) for [D-48] cache.
# SherwoodIGM_gal/ (extracted hdf5 snapshots) for the IGM_gal CIC field.
ln -s "${JUNO_SCRATCH}/sherwood" Sherwood
ln -s "${JUNO_SCRATCH}/SherwoodIGM_gal" SherwoodIGM_gal

# Verify the IGM_gal mirror has all 4 physics × 16 hdf5 = 64 files.
EXPECTED_HDF5=64
ACTUAL_HDF5=$(find SherwoodIGM_gal/extracted -name "snap_012.*.hdf5" 2>/dev/null | wc -l)
if [[ "${ACTUAL_HDF5}" -ne "${EXPECTED_HDF5}" ]]; then
  echo "FATAL: SherwoodIGM_gal mirror has ${ACTUAL_HDF5}/${EXPECTED_HDF5} hdf5 files." >&2
  echo "Expected 4 physics variants × 16 sub-files at:" >&2
  echo "  ${JUNO_SCRATCH}/SherwoodIGM_gal/extracted/{planck1_60_768_z0.300," >&2
  echo "    planck1_60_768_ps13_z0.300, planck1_60_768_ps13agn_z0.300," >&2
  echo "    planck1_60_768_ps13agn_strong_z0.300}/snapdir_012/snap_012.*.hdf5" >&2
  exit 5
fi

# --- 2. Environment -----------------------------------------------------------
source "${JUNO_WORK}/.venv/bin/activate"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file://${RUN_DIR}/mlflow"
export GIT_PYTHON_REFRESH=quiet

# Run-name + Stage-3 experiment for the MLflow contract (mlflow-run skill).
# train_truth_baseline.py uses its own MLflow setup; export the run-name as
# a hint variable the driver can pick up if it consults the env.
export SPRINT4_RUN_NAME="${RUN_NAME}"
export SPRINT4_EXPERIMENT="CosmoGasVision/Stage3-TruthBaseline"

# --- 3. Diagnostics -----------------------------------------------------------
echo "=== run config ==="
echo "RUN_TAG=${RUN_TAG}"
echo "RUN_NAME=${RUN_NAME}"
echo "JUNO_BATCH=${JUNO_BATCH}"
echo "epochs=${EPOCHS} batch_size=${BATCH_SIZE}"
echo "crops train/val/test=${N_CROPS_TRAIN}/${N_CROPS_VAL}/${N_CROPS_TEST} per physics × 4"
echo "lr_max=${LR_MAX} lr_min=${LR_MIN} weight_decay=${WEIGHT_DECAY}"
echo "warmup_epochs=${WARMUP_EPOCHS} early_stop_patience=${EARLY_STOP_PATIENCE}"
echo "seeds: train=${SEED_TRAIN} val=${SEED_VAL} test=${SEED_TEST} model=${SEED_MODEL} aug=${SEED_AUG}"
echo "n_bootstrap=${N_BOOTSTRAP} redshift=${REDSHIFT}"
nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv | tail -1
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')"
echo "=== data mirror tally ==="
echo "Sherwood/ → $(readlink Sherwood)"
echo "SherwoodIGM_gal/ → $(readlink SherwoodIGM_gal)"
echo "IGM_gal hdf5 files: ${ACTUAL_HDF5}/${EXPECTED_HDF5} ✓"

# --- 4. Compute ---------------------------------------------------------------
# Defense-in-depth (2026-05-14 8th-gap fix): wrap the driver in `set +e` so a
# non-zero exit (legitimate process-failure outcome under prior semantics, or
# any other code path) does NOT skip the post-run MLflow tagging + PCV
# copy-out. The driver itself now exits 0 on any successfully-completed
# outcome routing per the [D-52] 4-branch table, but we keep this defensive
# wrap so the submit-script invariant "if artifacts exist on disk, copy them
# out" holds even if a future driver change re-introduces non-zero exits on
# legitimate outcomes. PCV's own hard-fails (exit codes 2-8) still trigger
# correctly because they're inside the `set -euo pipefail` region after the
# `set -e` re-enable below.
set +e
python -u scripts/train_truth_baseline.py \
    --crop_size 32 \
    --n_grid 768 \
    --n_crops_train "${N_CROPS_TRAIN}" \
    --n_crops_val "${N_CROPS_VAL}" \
    --n_crops_test "${N_CROPS_TEST}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr_max "${LR_MAX}" \
    --lr_min "${LR_MIN}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --early_stop_patience "${EARLY_STOP_PATIENCE}" \
    --seed_train "${SEED_TRAIN}" \
    --seed_val "${SEED_VAL}" \
    --seed_test "${SEED_TEST}" \
    --seed_model "${SEED_MODEL}" \
    --seed_aug "${SEED_AUG}" \
    --redshift "${REDSHIFT}" \
    --device auto \
    --n_bootstrap "${N_BOOTSTRAP}"
DRIVER_EXIT=$?
echo "[submit] driver exit code: ${DRIVER_EXIT} (informational; PCV runs regardless)"
# Re-enable strict-fail mode now that the driver is done. PCV's own
# explicit exit codes (2-8) still trip the trap; legitimate non-zero from
# the driver (under historical semantics) no longer skips PCV.
set -e

# --- 5. Post-run MLflow tag injection (mlflow-run skill contract) -------------
# train_truth_baseline.py uses its own MLflow setup; we tag the run here so
# the host-side replay can identify it. The driver writes one run per
# invocation under experiment ${SPRINT4_EXPERIMENT}.
python -u - <<PYEOF
import os, sys
from mlflow.tracking import MlflowClient

uri = os.environ["MLFLOW_TRACKING_URI"]
client = MlflowClient(tracking_uri=uri)
exp = client.get_experiment_by_name(os.environ["SPRINT4_EXPERIMENT"])
if exp is None:
    print(f"FATAL: experiment '{os.environ['SPRINT4_EXPERIMENT']}' not found.", file=sys.stderr)
    sys.exit(10)
# Pick the most recent run in this experiment (sprint-4 should be the only
# run on the freshly-created file store). PI caveat C1: assert the
# fresh-store invariant so a future code change producing 2 runs does not
# silently mis-tag the wrong one.
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=2,
)
if not runs:
    print("FATAL: no run found in the experiment after pipeline exit.", file=sys.stderr)
    sys.exit(11)
assert len(runs) == 1, (
    f"PI caveat C1: expected exactly 1 run in the fresh file-store at "
    f"${MLFLOW_TRACKING_URI}, got {len(runs)}. Mis-tag risk; abort."
)
run = runs[0]
client.set_tag(run.info.run_id, "stage", "Stage3-truth-baseline")
client.set_tag(run.info.run_id, "model_type", "resnet18_3d")
client.set_tag(run.info.run_id, "physics_id", "all")
client.set_tag(run.info.run_id, "redshift", "${REDSHIFT}")
client.set_tag(run.info.run_id, "juno_batch", "${JUNO_BATCH}")
client.set_tag(run.info.run_id, "compute", "juno")
client.set_tag(run.info.run_id, "decision_id", "[D-51]")
client.set_tag(run.info.run_id, "successor_of", "[D-52]")
client.set_tag(run.info.run_id, "sprint4_amendments", "post-pre-review-2026-05-13b")
client.set_tag(run.info.run_id, "discipline_bar", "candidate-paper-claim-surface-upon-AD-5-clearance")
print(f"[tagger] run_id={run.info.run_id} OK")
PYEOF

# --- 6. PCV (Producer-Consumer Verification) — copy-out + assert --------------
DEST="${JUNO_WORK%/CosmoGasVision}/sprint4_results/${RUN_TAG}"
mkdir -p "${DEST}"

# (a) MLflow file-store — required for downstream metric replay.
[[ -d mlflow ]] || { echo "FATAL: pipeline produced no mlflow/ store" >&2; exit 2; }
cp -r mlflow "${DEST}/"

# (b) sprint4 eval artifacts (headline.json + supporting JSON/CSV) — hard-fail
# if headline.json or training_log.csv missing.
ART_SRC="experiments/nerf/artifacts/eval/sprint4"
[[ -d "${ART_SRC}" ]] || { echo "FATAL: no sprint4 eval dir at ${ART_SRC}" >&2; exit 3; }

# Pull the most recent run_id from the eval dir (the script tags artifacts
# with a unix-timestamp run_id like sprint4_1778677480_headline.json).
HEADLINE_FILE=$(ls -t "${ART_SRC}"/sprint4_*_headline.json 2>/dev/null | head -1)
[[ -n "${HEADLINE_FILE}" ]] || { echo "FATAL: no headline.json in ${ART_SRC}" >&2; exit 4; }
SPRINT4_RUN_ID=$(basename "${HEADLINE_FILE}" | sed 's/_headline.json//')
echo "[PCV] detected sprint4 run_id = ${SPRINT4_RUN_ID}"

mkdir -p "${DEST}/eval"
for suffix in headline.json r_bin_edges.json confusion_matrix.json training_log.csv; do
  SRC="${ART_SRC}/${SPRINT4_RUN_ID}_${suffix}"
  if [[ -f "${SRC}" ]]; then
    cp "${SRC}" "${DEST}/eval/"
  else
    echo "FATAL: required artifact missing: ${SRC}" >&2
    exit 6
  fi
done
# smoke.json is only present if the driver fell through smoke-mode fields
# (post-pre-review amendments did this for the wiring smoke). Copy if present.
if [[ -f "${ART_SRC}/${SPRINT4_RUN_ID}_smoke.json" ]]; then
  cp "${ART_SRC}/${SPRINT4_RUN_ID}_smoke.json" "${DEST}/eval/"
fi

# (c) Best checkpoint — required for [D-52] paper-claim surface + sprint-5
# follow-on. Hard-fail on absence.
CKPT_SRC="experiments/nerf/artifacts/sprint4/checkpoints"
[[ -d "${CKPT_SRC}" ]] || { echo "FATAL: no checkpoint dir at ${CKPT_SRC}" >&2; exit 7; }
BEST_CKPT="${CKPT_SRC}/resnet18_3d_4class_best.pt"
[[ -f "${BEST_CKPT}" ]] || { echo "FATAL: best checkpoint missing: ${BEST_CKPT}" >&2; exit 8; }
mkdir -p "${DEST}/checkpoints"
cp "${BEST_CKPT}" "${DEST}/checkpoints/"
# Also copy any per-epoch checkpoints if present (optional; for inspection).
find "${CKPT_SRC}" -name "*.pt" -not -name "resnet18_3d_4class_best.pt" -exec cp {} "${DEST}/checkpoints/" \;

# (d) Verify what was copied — visible in .out for downstream debugging.
echo "=== copied artifacts ==="
echo "--- ${DEST}/mlflow ---"
find "${DEST}/mlflow" -maxdepth 4 -type d | head -10
echo "--- ${DEST}/eval ---"
ls -la "${DEST}/eval/"
echo "--- ${DEST}/checkpoints ---"
ls -la "${DEST}/checkpoints/"
echo "--- headline summary ---"
python -u - <<PYEOF
import json, sys
with open("${DEST}/eval/${SPRINT4_RUN_ID}_headline.json") as f:
    hl = json.load(f)
print(f"outcome_branch = {hl.get('outcome_branch')}")
print(f"deliverable_framing = {hl.get('deliverable_framing','<missing>')[:120]}")
for k in ("test_overall_ordinary_bootstrap", "test_overall_block_bootstrap"):
    v = hl.get(k)
    if v: print(f"{k} = mean={v.get('mean'):.4f} ci=[{v.get('lower_ci'):.4f}, {v.get('upper_ci'):.4f}]")
gates = hl.get("gates", {})
ge = gates.get("gate_e_trivial_baseline", {})
print(f"gate_(e) margin_pp_required={ge.get('required_margin_pp')} escalation_band_low={ge.get('escalation_band_low_pp')} escalate={ge.get('escalate_to_capacity_matched_mlp')}")
PYEOF

# --- 7. Cleanup ---------------------------------------------------------------
cd "${HOME}"
rm -rf "${RUN_DIR}"
echo "=== done, results at ${DEST} ==="
echo "=== sprint4_run_id: ${SPRINT4_RUN_ID} ==="
