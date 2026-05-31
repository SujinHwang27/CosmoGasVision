#!/bin/bash
#SBATCH --job-name=D71-SmoothingFloor
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=d71-smoothing-%j.out
#SBATCH --error=d71-smoothing-%j.err

# D71 Rev 1.2 §10.E K6 #3 — variance-preservation smoothing-floor compute on
# Juno HPC. PI-authorized per defense-panel S-A7 absorption + LEDGER §3 [D-71]
# Rev 1.1 Option 1 (methodology-preservation re-route from OOM'd Windows host).
#
# Compute: scipy.ndimage.gaussian_filter1d separable per-axis on Sherwood p1
# z=0.300 768^3 rho field, sigma in {1, 2, 3, 5} voxels, periodic wrap.
# Empirical wall ~5-10 min; 30 min wallclock + queue buffer.
# CPU-only (no GPU dependency in script); `normal` partition is the right home
# vs `a30` (which would burn a 24GB GPU slot for nothing).
#
# Input  cache: ${JUNO_SCRATCH}/sherwood/.rho_field_cache/rho_field_p1_z0.300_n768.npy
#               (pre-staged 2026-05-14, 1.81 GB, verified present 2026-05-30)
# Output JSON: ${RUN_DIR}/d71_var_smoothing_floor.json
#              → copied to ${DEST}/d71_var_smoothing_floor.json post-run
#              → host pulls to cloud_runs/d71_var_smoothing_floor.json
#
# Script under test:  scripts/d71_compute_var_truth_smoothing_floor.py
#   The script resolves REPO_ROOT via env var COSMOGAS_REPO_ROOT (override) or
#   the script's own parent-of-parent (default). We pass the Juno RUN_DIR via
#   that env var rather than patching the source.
#
# See `.claude/skills/juno-hpc/SKILL.md` for the cluster contract; PCV
# (Producer-Consumer Verification) per infrastructure-manager.md.

set -euo pipefail

if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env" ]]; then
  set -a; source "${JUNO_WORK:-$HOME/work/CosmoGasVision}/.env"; set +a
fi
: "${JUNO_WORK:=/work/${USER}/CosmoGasVision}"
: "${JUNO_SCRATCH:=/scratch/juno/${USER}}"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SHORTHASH=$(cd "${JUNO_WORK}" && git rev-parse --short HEAD 2>/dev/null || echo nogit)
RUN_TAG="d71-smoothing-floor-${SHORTHASH}-${TIMESTAMP}-$(uuidgen | cut -c1-6)"

RUN_DIR="${JUNO_SCRATCH}/d71/${RUN_TAG}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

# --- Copy in repo source + symlink Sherwood data root ---
# We only need the d71 script + its imports (numpy/scipy already in .venv),
# but copy the scripts/ tree wholesale for consistency with stage1a precedent.
cp -r "${JUNO_WORK}"/{src,scripts,pyproject.toml,uv.lock} .
ln -s "${JUNO_SCRATCH}/sherwood" Sherwood

# --- Environment ---
source "${JUNO_WORK}/.venv/bin/activate"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

echo "=== D71 smoothing-floor — run config ==="
echo "RUN_TAG=${RUN_TAG}"
echo "SHORTHASH=${SHORTHASH}"
echo "PARTITION=normal CPUS=4 MEM=24G WALL=00:30:00"
echo "JUNO_SCRATCH=${JUNO_SCRATCH}"
echo "Python: $(python --version 2>&1)"
echo "scipy: $(python -c 'import scipy; print(scipy.__version__)' 2>&1)"
echo "numpy: $(python -c 'import numpy; print(numpy.__version__)' 2>&1)"
free -h | sed -n '1,3p'

# --- Pre-flight: rho cache presence ---
RHO_CACHE_FILE="${JUNO_SCRATCH}/sherwood/.rho_field_cache/rho_field_p1_z0.300_n768.npy"
[[ -f "${RHO_CACHE_FILE}" ]] || {
    echo "FATAL: rho-field cache missing at ${RHO_CACHE_FILE}" >&2
    echo "  Pre-stage with: rsync from host Sherwood/.rho_field_cache/" >&2
    exit 5
}
echo "rho cache: $(stat -c '%s bytes, %y' "${RHO_CACHE_FILE}")"

# --- Compute ---
# The d71 script's module-level constants (REPO_ROOT, INPUT_REL, OUTPUT_REL)
# are Windows-paths-as-strings. Override them on the live module object before
# calling main(). This keeps the source-of-truth script unedited.
#
# Output JSON lands at ${RUN_DIR}/cloud_runs/d71_var_smoothing_floor.json
# (the script's OUTPUT_REL is "cloud_runs/d71_var_smoothing_floor.json" under REPO_ROOT).
OUTPUT_JSON="${RUN_DIR}/cloud_runs/d71_var_smoothing_floor.json"
mkdir -p "${RUN_DIR}/cloud_runs"

export COSMOGAS_REPO_ROOT="${RUN_DIR}"
export COSMOGAS_RHO_CACHE="${RHO_CACHE_FILE}"
echo "[launcher] COSMOGAS_REPO_ROOT = ${COSMOGAS_REPO_ROOT}"
echo "[launcher] COSMOGAS_RHO_CACHE = ${COSMOGAS_RHO_CACHE}"

python -u "${RUN_DIR}/scripts/d71_compute_var_truth_smoothing_floor.py"

# --- Producer-Consumer Verification (PCV) ---
# The downstream consumer is the D71 Rev 1.2 §10.E K6 anchor — it reads the
# four mu_smoothing_floor scalars and the var_truth_full scalar from the JSON.
# Hard-fail if any of those are missing.

echo "=== PCV: verify JSON ==="
[[ -f "${OUTPUT_JSON}" ]] || { echo "FATAL: no JSON at ${OUTPUT_JSON}" >&2; exit 6; }
JSON_BYTES=$(stat -c '%s' "${OUTPUT_JSON}")
[[ "${JSON_BYTES}" -gt 100 ]] || { echo "FATAL: JSON too small (${JSON_BYTES} bytes)" >&2; exit 7; }

python -u <<PYEOF
import json, sys
with open("${OUTPUT_JSON}") as fh:
    obj = json.load(fh)
required_top = {"var_truth_full", "smoothing_floor_per_scale", "meta"}
missing = required_top - set(obj.keys())
if missing:
    print(f"FATAL: top-level keys missing: {missing}", file=sys.stderr); sys.exit(8)
sigmas = [row["sigma_voxels"] for row in obj["smoothing_floor_per_scale"]]
if sorted(sigmas) != [1, 2, 3, 5]:
    print(f"FATAL: expected sigmas [1,2,3,5], got {sorted(sigmas)}", file=sys.stderr); sys.exit(9)
for row in obj["smoothing_floor_per_scale"]:
    mu = row["mu_smoothing_floor"]
    if not (0.0 < mu <= 1.0 + 1e-9):
        print(f"FATAL: mu out of band at sigma={row['sigma_voxels']}: {mu}", file=sys.stderr); sys.exit(10)
print(f"[PCV] var_truth_full = {obj['var_truth_full']:.6e}")
for row in obj["smoothing_floor_per_scale"]:
    print(f"[PCV] sigma={row['sigma_voxels']} mu={row['mu_smoothing_floor']:.6e}")
print("[PCV] OK -- all four sigmas present, mu monotonic-checked by script.")
PYEOF

# --- Copy out ---
DEST="${JUNO_WORK%/CosmoGasVision}/d71_results/${RUN_TAG}"
mkdir -p "${DEST}"
cp "${OUTPUT_JSON}" "${DEST}/d71_var_smoothing_floor.json"
echo "=== copied: ${DEST}/d71_var_smoothing_floor.json ==="
stat -c '%s bytes, %y' "${DEST}/d71_var_smoothing_floor.json"

# --- Cleanup ---
cd "${HOME}"
rm -rf "${RUN_DIR}"
echo "=== done, result at ${DEST}/d71_var_smoothing_floor.json ==="
echo "=== host-side pull: rsync -avzP juno:${DEST}/d71_var_smoothing_floor.json cloud_runs/ ==="
