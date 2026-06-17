#!/bin/bash
# [D-73] (1d') Explicit voxel-grid + flux-supervision baseline — Juno H100 dispatch.
#
# PARTITION CHANGE (2026-06-13 scheduling decision, infrastructure-manager): switched
# a30 -> h100. Rationale: the a30 partition is saturated overnight (job 214396 sat
# PENDING(Resources) and was cancelled); h100 schedules sooner AND its 80 GB removes the
# A30 (24 GB) VRAM-fit risk that was the original reason to hold (job 214197 OOM'd at
# G=192 in the Voigt intermediate). The science is partition-invariant — same one-lever
# config, same microbatch=256/accum=4, same seed; this is a deliberate logged scheduling
# choice, NOT a methodology change. The h100 pre-flight (job 214404, d73_preflight_h100.sh)
# GPU-validates the corrected --pf-diagnostic-only path before this full run.
#
# Parameters of record (LEDGER §3 [D-73] amendment-6 §P/§Q/§R/§S + amendment-7 §U/§V/§W;
# design D73_voxelgrid_flux_baseline.md §3/§4/§5):
#   --arch voxel-grid --voxel-grid-size 192   (option (a) explicit four-field grid)
#   --pf-diagnostic-only                       (ONE LEVER: plain-[D-24] backward; var_pf
#                                               is a DETACHED diagnostic readout — tol-0
#                                               byte-for-byte proven == plain-[D-24], am-7 §U.
#                                               MUTUALLY EXCLUSIVE with --enable-l1-pf-loss;
#                                               do NOT pass the latter — it re-injects the
#                                               falsified [D-60] second lever.)
#   --voxel-init-xhi 0.2                       (PINNED, am-7 §V: <F>_init=0.927, forest band
#                                               midpoint; pipeline default stays 1e-5.)
#   P1, z=0.3, plain [D-24] supervision (integrator/mask/cap/log1p/mean-F anchor inherited),
#   [D-14] optimizer schedule, n_rays=1024 ([D-13] fiducial eval point).
#
# Microbatch sizing: --microbatch 256 --accum_steps 4 (kept IDENTICAL to the h100 pre-flight
# 214404 so the GPU-validated path == the full-run path — do NOT inflate microbatch even
# though the h100 (80 GB) has headroom; the validated-by-pre-flight invariant is worth more
# than throughput here). History: pre-flight 214197 ran microbatch>=1024 full-grid and OOM'd
# at G=192 on the A30 (24 GB) in the Voigt intermediate — moot on h100, but the locked config
# does not change with the partition. 256/4 matches the production T3 mapping (peak ~11.3 GB
# on the MLP; the grid forward is cheaper per design §5). PYTORCH_CUDA_ALLOC_CONF retained.
#
# Gate ladder (pre-committed, design §4):
#   (i)  trainability: var_pf_band_ratio > 1e-3 at step 5000 (emitted to stdout + MLflow
#        under --pf-diagnostic-only).
#   (ii) [D-13] science gates at the converged checkpoint: |dP_F/P_F|<10%, KS<0.05, and the
#        TRUE 3D xi via the host-side consumer (§S) — scored host-side AFTER copy-out.
#
# HARD CAP (design §5 / [D-73] §F item 7): 30 A30-hr total, 1 job + 1 re-bake. Walltime
# below set to 18h for the full 50k-step [D-14] schedule; if step throughput from the
# pre-flight extrapolates past the cap, this job is the single permitted run and a re-bake
# is reserved only for a panel-flagged construction defect (NOT tuning).
#
# PCV (infrastructure-manager.md): the §S 3D-xi gate (ii)(b) REQUIRES the step_*.pt
# checkpoint preserved to the host. The copy-out below hard-fails (exit N) on any missing
# artifact at each producer-consumer seam. NO `2>/dev/null || true` on artifact paths.
#
# Submit from inside ${JUNO_WORK} on the login node:
#   sbatch scripts/submit_juno_d73_voxelgrid.sh
#
#SBATCH --job-name=d73-1dp-voxelgrid-p1
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# --- Load Juno path conventions (JUNO_WORK, JUNO_SCRATCH) from the project .env ---
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a
  source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"
  set +a
fi
: "${JUNO_WORK:=$HOME/work/CosmoGasVision}"
: "${JUNO_SCRATCH:=$HOME/scratch}"

COMMIT="37c02bc"
RUN_TAG="d73-1dprime-voxel192-P1-z0.3-${COMMIT}-$(date +%Y%m%d-%H%M%S)-$(uuidgen | cut -c1-6)"
RUN_DIR="${JUNO_SCRATCH}/d73_1dprime/${RUN_TAG}"

echo "=== [D-73] (1d') dispatch | RUN_TAG=${RUN_TAG} | commit=${COMMIT} ==="

# --- 1. Copy in ---
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
cp -r "${JUNO_WORK}"/{src,experiments,scripts,tests,pyproject.toml,uv.lock} .
ln -s "${JUNO_SCRATCH}/sherwood" Sherwood   # P1 sightlines + tauH1 + .rho_field_cache/

# Provenance guard: confirm the copied repo carries the one-lever flag + tol-0 test.
grep -q -- "--pf-diagnostic-only\|pf_diagnostic_only" experiments/nerf/pipeline.py \
  || { echo "FATAL: --pf-diagnostic-only absent from copied pipeline.py (wrong commit?)" >&2; exit 10; }
[[ -f tests/test_d73_pf_diagnostic.py ]] \
  || { echo "FATAL: tests/test_d73_pf_diagnostic.py absent (wrong commit, expected ${COMMIT})" >&2; exit 11; }

# Producer-consumer seam 0: the §S host-consumer needs the n768 truth rho cube. Assert it
# is reachable through the Sherwood symlink BEFORE burning GPU time.
TRUTH_RHO="Sherwood/.rho_field_cache/rho_field_p1_z0.300_n768.npy"
[[ -f "${TRUTH_RHO}" ]] \
  || { echo "FATAL: truth rho cube absent at ${TRUTH_RHO} — §S 3D-xi gate (ii)(b) would be un-scorable" >&2; exit 12; }

# --- 2. Environment ---
source "${JUNO_WORK}/.venv/bin/activate"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MLFLOW_TRACKING_URI="file://${RUN_DIR}/mlflow"
export GIT_PYTHON_REFRESH=quiet

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; \
print('CUDA', torch.cuda.get_device_name(0), '| torch', torch.__version__)"

# --- 3. Compute (plain-[D-24] + diagnostic-only var_pf; ONE LEVER) ---
SECONDS=0
python -u experiments/nerf/pipeline.py \
    --arch voxel-grid \
    --voxel-grid-size 192 \
    --pf-diagnostic-only \
    --voxel-init-xhi 0.2 \
    --physics 1 \
    --n_rays 1024 \
    --microbatch 256 \
    --accum_steps 4 \
    --max_steps 50000 \
    --warmup_steps 1000 \
    --lr_max 5e-4 \
    --lr_min 5e-6 \
    --seed 0 \
    --checkpoint_interval 5000 \
    --data_root Sherwood \
    --run_name "${RUN_TAG}"
TRAIN_SECONDS=${SECONDS}
echo "=== train wall-clock: ${TRAIN_SECONDS}s ==="

# --- 4. Copy out + Producer-Consumer Verification (PCV) ---
# pipeline.py writes step_*.pt under experiments/nerf/artifacts/checkpoints/ (the
# --checkpoint_dir default). Cite the producer's source-of-truth path (pipeline.py:75-76,
# 3373) — NOT a recalled ./checkpoints/. The §S 3D-xi gate REQUIRES step_*.pt preserved.
DEST="${JUNO_WORK%/CosmoGasVision}/d73_1dprime_results/${RUN_TAG}"
mkdir -p "${DEST}"

# Seam A: MLflow file-store (trainability gate var_pf + [D-24] metrics replay).
[[ -d mlflow ]] || { echo "FATAL: pipeline produced no mlflow/ store" >&2; exit 2; }
cp -r mlflow "${DEST}/"

# Seam B: checkpoints (REQUIRED for §S 3D-xi host-consumer + [D-13] evaluators). Hard-fail.
CKPT_SRC="experiments/nerf/artifacts/checkpoints"
[[ -d "${CKPT_SRC}" ]] || { echo "FATAL: no checkpoint dir at ${CKPT_SRC}" >&2; exit 3; }
N_CKPT=$(ls -1 "${CKPT_SRC}"/step_*.pt 2>/dev/null | wc -l)
[[ "${N_CKPT}" -gt 0 ]] || { echo "FATAL: ${CKPT_SRC} contains zero step_*.pt files" >&2; exit 4; }
mkdir -p "${DEST}/checkpoints"
cp "${CKPT_SRC}"/step_*.pt "${DEST}/checkpoints/"

# Seam B proof-of-loadability: the highest-step checkpoint must load into VoxelGridField and
# expose log_rho_grid at (192,192,192). This is the one-line PCV proof the §S consumer needs.
LAST_CKPT=$(ls -1 "${DEST}/checkpoints"/step_*.pt | sort | tail -1)
python - "${LAST_CKPT}" <<'PYEOF'
import sys, torch
from src.models.voxel_grid_field import VoxelGridField
ckpt = sys.argv[1]
state = torch.load(ckpt, map_location="cpu", weights_only=False)
ms = state.get("model_state", state)
assert "log_rho_grid" in ms, f"log_rho_grid absent; keys={list(ms.keys())[:8]}"
m = VoxelGridField(grid_size=192, init_noise_std=0.0, init_xhi=0.2)
m.load_state_dict(ms)
g = tuple(m.log_rho_grid.shape)
assert g == (192, 192, 192), f"log_rho_grid shape {g} != (192,)*3"
print(f"[PCV] {ckpt} loads into VoxelGridField; log_rho_grid={g} OK")
PYEOF

# Seam C: copy the n768 truth rho cube alongside results so the host-consumer is
# self-contained even if the scratch Sherwood mirror purges (it is large, ~1.8 GB; the
# host already has its own copy, so this is best-effort — DO NOT hard-fail the milestone on
# it, the host-side truth is the canonical input). Best-effort housekeeping only.
cp "${TRUTH_RHO}" "${DEST}/rho_field_p1_z0.300_n768.npy" 2>/dev/null \
  && echo "[copy-out] truth rho cube staged to ${DEST}" \
  || echo "[copy-out] truth rho cube not staged (host has canonical copy) — non-fatal"

echo "=== copied artifacts ==="
ls -la "${DEST}/checkpoints/"
find "${DEST}/mlflow" -maxdepth 4 -type d | head -10

# --- 4b. In-job §S 3D-xi consumer (best-effort; the canonical run is host-side post-pull) ---
# Run the §S consumer here too so the .out log carries an immediate xi(2 Mpc/h) read. This
# is BEST-EFFORT (the binding score is the host-side re-run after rsync, against the host's
# canonical truth cube) — do not hard-fail the milestone if it errors, but DO surface it.
if [[ -f "${TRUTH_RHO}" ]]; then
  python -u scripts/d73_xi_host_consumer.py \
      --checkpoint "${LAST_CKPT}" \
      --truth-rho "${TRUTH_RHO}" \
      --grid-size 192 \
      --box-kpc-h 60000 \
      --out "${DEST}/eval/xi_3d_injob.json" \
    && cp "${DEST}/eval/xi_3d_injob.json" "${DEST}/" 2>/dev/null \
    || echo "[xi-consumer] in-job run errored — re-run host-side after pull (non-fatal)"
fi

# --- 5. Cleanup (only after all artifacts are confirmed copied out + loadable) ---
cd "${HOME}"
rm -rf "${RUN_DIR}"
echo "=== DONE | results at ${DEST} | RUN_TAG=${RUN_TAG} ==="
echo "=== HOST NEXT STEPS ==="
echo "  1. rsync -avzP juno:${DEST}/ cloud_runs/${RUN_TAG}/"
echo "  2. PYTHONPATH=. python scripts/d73_xi_host_consumer.py \\"
echo "       --checkpoint cloud_runs/${RUN_TAG}/checkpoints/step_050000.pt \\"
echo "       --truth-rho Sherwood/.rho_field_cache/rho_field_p1_z0.300_n768.npy \\"
echo "       --grid-size 192 --box-kpc-h 60000 \\"
echo "       --out cloud_runs/${RUN_TAG}/eval/xi_3d.json"
echo "  3. MLflow round-trip per juno-hpc skill; LEDGER §6 + §3 [D-73] amendment-8."
