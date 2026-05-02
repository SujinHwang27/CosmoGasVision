#!/usr/bin/env bash
# Local-GPU detection report (Stage 2b dispatch C6, [D-14]).
#
# Prints a 5-line plain-text report:
#   GPU model, total VRAM (GB), CUDA version, driver version, NVML availability.
#
# Usage:
#   bash ./scripts/check_local_gpu.sh

set -u

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU model: no NVIDIA GPU detected"
    echo "Total VRAM (GB): n/a"
    echo "CUDA version: n/a"
    echo "Driver version: n/a"
    echo "NVML availability: false"
    exit 0
fi

# Query first GPU.
query=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>/dev/null | head -n1)
if [[ -z "${query}" ]]; then
    echo "GPU model: no NVIDIA GPU detected"
    echo "Total VRAM (GB): n/a"
    echo "CUDA version: n/a"
    echo "Driver version: n/a"
    echo "NVML availability: false"
    exit 0
fi

IFS=',' read -r gpu_name mem_mib driver_version <<< "${query}"
gpu_name=$(echo "${gpu_name}" | sed 's/^ *//;s/ *$//')
mem_mib=$(echo "${mem_mib}" | sed 's/^ *//;s/ *$//')
driver_version=$(echo "${driver_version}" | sed 's/^ *//;s/ *$//')

# Round MiB -> GB to 2 decimals.
vram_gb=$(awk -v m="${mem_mib}" 'BEGIN { printf "%.2f", m/1024 }')

# CUDA version from the smi header.
cuda_version=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | head -n1 | awk '{print $3}')
[[ -z "${cuda_version}" ]] && cuda_version="unknown"

# NVML available iff nvidia-smi succeeded.
nvml_available="true"

echo "GPU model: ${gpu_name}"
echo "Total VRAM (GB): ${vram_gb}"
echo "CUDA version: ${cuda_version}"
echo "Driver version: ${driver_version}"
echo "NVML availability: ${nvml_available}"
