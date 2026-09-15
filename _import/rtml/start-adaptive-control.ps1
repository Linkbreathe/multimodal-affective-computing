[CmdletBinding()]
param(
    [string]$Bundle = "realtime_multimodal_window_v1",
    [switch]$SkipModelCheck
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"

Write-Host "Adaptive Control" -ForegroundColor Cyan
Write-Host "Model bundle: $Bundle"

if (-not $SkipModelCheck) {
    & conda run --no-capture-output -n rtml-p002-p016 python -m real_time_ml.cli adaptive-model verify --bundle $Bundle
    if ($LASTEXITCODE -ne 0) {
        throw "Model preflight failed. Adaptive control was not started."
    }
}

Write-Host "Starting local Python service. Leave this window open while Unity is running." -ForegroundColor Yellow
& conda run --no-capture-output -n rtml-p002-p016 python -m real_time_ml.cli adaptive-control --bundle $Bundle
exit $LASTEXITCODE
