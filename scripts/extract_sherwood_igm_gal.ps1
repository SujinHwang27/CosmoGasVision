# Extract one or all SherwoodIGM_gal/*.tar.gz tarballs to SherwoodIGM_gal/extracted/.
# Each tarball is a GADGET multi-part HDF5 snapshot (snapdir_012/snap_012.{0..15}.hdf5)
# plus FoF group catalogues. ~30 GB unpacked per variant.
#
# Usage:
#   pwsh ./scripts/extract_sherwood_igm_gal.ps1
#   pwsh ./scripts/extract_sherwood_igm_gal.ps1 -Names planck1_60_768_z0.300
#
# Idempotent.

param(
    [string[]]$Names = @(
        "planck1_60_768_z0.300",
        "planck1_60_768_ps13_z0.300",
        "planck1_60_768_ps13agn_z0.300",
        "planck1_60_768_ps13agn_strong_z0.300"
    )
)

$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..')
$Src  = Join-Path $Root 'SherwoodIGM_gal'
$Dst  = Join-Path $Src  'extracted'
New-Item -ItemType Directory -Path $Dst -Force | Out-Null

foreach ($name in $Names) {
    $tarball = Join-Path $Src "$name.tar.gz"
    $target  = Join-Path $Dst $name

    if (-not (Test-Path $tarball)) {
        Write-Host "skip: $name.tar.gz not found"
        continue
    }
    if (Test-Path $target) {
        $existing = Get-ChildItem -Path $target -ErrorAction SilentlyContinue
        if ($existing) {
            Write-Host "skip: $target already populated"
            continue
        }
    }

    Write-Host "extracting $name.tar.gz -> $target"
    Push-Location $Src
    try {
        tar -xzf "$name.tar.gz" -C $Dst
    }
    finally {
        Pop-Location
    }
    $nFiles  = (Get-ChildItem -Path $target -Recurse -File).Count
    $sizeGB  = [math]::Round((Get-ChildItem -Path $target -Recurse -File | Measure-Object Length -Sum).Sum / 1GB, 1)
    Write-Host "  done: $nFiles files, $sizeGB GB"
}
