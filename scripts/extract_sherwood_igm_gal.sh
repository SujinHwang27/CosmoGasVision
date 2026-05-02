#!/usr/bin/env bash
# Extract one or all SherwoodIGM_gal/*.tar.gz tarballs to SherwoodIGM_gal/extracted/.
# Each tarball contains a GADGET multi-part HDF5 snapshot (snapdir_012/snap_012.{0..15}.hdf5)
# plus FoF group catalogues. Each unpacked tree is ~30 GB.
#
# Usage:
#   bash scripts/extract_sherwood_igm_gal.sh                  # extract all four
#   bash scripts/extract_sherwood_igm_gal.sh planck1_60_768_z0.300   # extract one by basename
#
# Idempotent: skips a tarball if its target directory already exists and is non-empty.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/SherwoodIGM_gal"
DST="${SRC}/extracted"
mkdir -p "${DST}"

declare -a TARBALLS
if [[ $# -gt 0 ]]; then
    for name in "$@"; do
        TARBALLS+=("${name}.tar.gz")
    done
else
    TARBALLS=(
        "planck1_60_768_z0.300.tar.gz"
        "planck1_60_768_ps13_z0.300.tar.gz"
        "planck1_60_768_ps13agn_z0.300.tar.gz"
        "planck1_60_768_ps13agn_strong_z0.300.tar.gz"
    )
fi

for tarball in "${TARBALLS[@]}"; do
    src_path="${SRC}/${tarball}"
    base="${tarball%.tar.gz}"
    target="${DST}/${base}"

    if [[ ! -f "${src_path}" ]]; then
        echo "skip: ${tarball} not found at ${src_path}"
        continue
    fi

    if [[ -d "${target}" && "$(ls -A "${target}" 2>/dev/null)" ]]; then
        echo "skip: ${target} already populated"
        continue
    fi

    echo "extracting ${tarball} -> ${target}"
    cd "${SRC}"
    tar -xzf "${tarball}" -C "${DST}/"
    n_files=$(find "${target}" -type f | wc -l)
    size_gb=$(du -sh "${target}" 2>/dev/null | awk '{print $1}')
    echo "  done: ${n_files} files, ${size_gb}"
done
