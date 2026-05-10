#!/bin/bash
#SBATCH --job-name=PubT1-Babysitter
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=pubt1-babysitter-%j.out
#SBATCH --error=pubt1-babysitter-%j.err

# =============================================================================
# Babysitter for the [D-39] T1 publication-class bundle (4 cells P1..P4).
# Queued with --dependency=afterany:${J1}:${J2}:${J3}:${J4} immediately after
# the 4 training sbatch jobs. Tarballs all four cells' MLflow stores +
# checkpoints into a single tarball under cloud_runs/ for host-side scp.
#
# RUN_TAG pattern from submit_juno_stage2b.sh: P${PID}-N64-S0-${EPOCH}-${UUID6}.
# We scope by mtime (-mmin -360) to avoid catching unrelated stage2b dirs.
# =============================================================================

set -euo pipefail

: "${JUNO_WORK:=/work/sxh240010/CosmoGasVision}"
RESULT_BASE="${JUNO_WORK%/CosmoGasVision}/stage2b_results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TARBALL_DIR="${JUNO_WORK}/cloud_runs"
TARBALL="${TARBALL_DIR}/pub-t1-${TIMESTAMP}.tar.gz"

mkdir -p "${TARBALL_DIR}"
echo "=== pub-t1 babysitter start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Scanning ${RESULT_BASE} for P*-N64-S0-* dirs (mtime within last 6 hr)"

DIRS=$(find "${RESULT_BASE}" -maxdepth 1 -type d -name "P*-N64-S0-*" -mmin -360 | sort)
N_DIRS=$(printf '%s\n' "${DIRS}" | grep -c '^P\|^/' || true)
echo "Found dirs:"
printf '%s\n' "${DIRS}"
echo "N_DIRS=${N_DIRS}"

if [[ "${N_DIRS}" -lt 4 ]]; then
  echo "WARN: expected 4 cells, found ${N_DIRS}. Tarballing what's present." >&2
fi

cd "${RESULT_BASE}"
BASENAMES=$(printf '%s\n' "${DIRS}" | xargs -n1 basename)
echo "Tarballing basenames into ${TARBALL}:"
echo "${BASENAMES}"
tar czf "${TARBALL}" ${BASENAMES}
ls -la "${TARBALL}"

echo "TARBALL_DONE=${TARBALL}"
echo "=== pub-t1 babysitter done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
