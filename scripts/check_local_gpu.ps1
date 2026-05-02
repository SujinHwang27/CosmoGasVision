# Local-GPU detection report (Stage 2b dispatch C6, [D-14]).
#
# Prints a 5-line plain-text report:
#   GPU model, total VRAM (GB), CUDA version, driver version, NVML availability.
#
# Usage:
#   pwsh ./scripts/check_local_gpu.ps1

$ErrorActionPreference = 'SilentlyContinue'

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    Write-Output "GPU model: no NVIDIA GPU detected"
    Write-Output "Total VRAM (GB): n/a"
    Write-Output "CUDA version: n/a"
    Write-Output "Driver version: n/a"
    Write-Output "NVML availability: false"
    exit 0
}

# Query name, total memory (MiB), driver via CSV.
$query  = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>$null
if (-not $query) {
    Write-Output "GPU model: no NVIDIA GPU detected"
    Write-Output "Total VRAM (GB): n/a"
    Write-Output "CUDA version: n/a"
    Write-Output "Driver version: n/a"
    Write-Output "NVML availability: false"
    exit 0
}

# nvidia-smi may report multiple GPUs (one row each). Use the first row.
$row = ($query -split "`r?`n")[0]
$parts = $row -split '\s*,\s*'
$gpuName = $parts[0]
$memMiB  = [double]$parts[1]
$driver  = $parts[2]
$vramGB  = [math]::Round($memMiB / 1024, 2)

# CUDA version comes from the header of `nvidia-smi` (top-right field).
$cudaVersion = "unknown"
$smiHeader = & nvidia-smi 2>$null
if ($smiHeader) {
    $match = ($smiHeader | Select-String -Pattern 'CUDA Version:\s*([\d.]+)').Matches
    if ($match.Count -gt 0) { $cudaVersion = $match[0].Groups[1].Value }
}

# NVML availability: nvidia-smi works iff the NVML library is loaded successfully.
$nvmlAvailable = "true"

Write-Output "GPU model: $gpuName"
Write-Output "Total VRAM (GB): $vramGB"
Write-Output "CUDA version: $cudaVersion"
Write-Output "Driver version: $driver"
Write-Output "NVML availability: $nvmlAvailable"
