# Launch the local MLflow tracking server.
#
# Backend store : SQLite (mlflow.db at repo root, gitignored)
# Artifact root : S3 bucket shared with the team — server uploads, clients only see URIs
# Default bind  : 127.0.0.1:5000 (loopback only). Set MLFLOW_HOST=0.0.0.0 for LAN access.
#
# Usage:
#   pwsh ./scripts/start_mlflow.ps1
# or with overrides:
#   $env:MLFLOW_HOST = '0.0.0.0'; pwsh ./scripts/start_mlflow.ps1

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

# Load .env so AWS_* are visible to the server process for S3 artifact uploads
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+?)\s*=\s*(.*)\s*$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim().Trim('"').Trim("'")
            [Environment]::SetEnvironmentVariable($name, $value, 'Process')
        }
    }
} else {
    Write-Warning "No .env file found — S3 artifact uploads will fail without AWS credentials."
}

$BackendUri   = 'sqlite:///mlflow.db'
$ArtifactRoot = 's3://cosmo-gas-vision-storage/mlflow-artifacts'
$BindHost     = if ($env:MLFLOW_HOST) { $env:MLFLOW_HOST } else { '127.0.0.1' }
$Port         = if ($env:MLFLOW_PORT) { $env:MLFLOW_PORT } else { '5000' }

Write-Host "Starting MLflow server"
Write-Host "  Backend   : $BackendUri"
Write-Host "  Artifacts : $ArtifactRoot"
Write-Host "  Listening : http://${BindHost}:${Port}"
Write-Host ""

uv run mlflow server `
    --backend-store-uri $BackendUri `
    --default-artifact-root $ArtifactRoot `
    --host $BindHost `
    --port $Port
