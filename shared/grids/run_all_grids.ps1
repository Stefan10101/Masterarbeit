# ============================================================
# run_all_grids.ps1
# Master grids + projected station lists. Always use the Master env.
# ============================================================

$ErrorActionPreference = "Stop"

$python = $null
$candidates = @(
    "$env:USERPROFILE\miniforge3\envs\Master\python.exe",
    "$env:USERPROFILE\mambaforge\envs\Master\python.exe",
    "$env:USERPROFILE\anaconda3\envs\Master\python.exe",
    "$env:USERPROFILE\miniconda3\envs\Master\python.exe"
)
if ($env:CONDA_PREFIX -like "*\envs\Master") {
    $candidates = @("$env:CONDA_PREFIX\python.exe") + $candidates
}
foreach ($c in $candidates) {
    if ($c -and (Test-Path $c)) { $python = $c; break }
}
if (-not $python) {
    throw "Master python.exe not found at miniforge3\envs\Master."
}

Write-Host "Python: $python" -ForegroundColor Cyan
& $python -c "import yaml, sys; print('yaml ok', sys.executable)"
if ($LASTEXITCODE -ne 0) {
    throw "This python has no PyYAML. Run: conda activate Master; conda install pyyaml"
}

$methods = @("IDW", "BSS", "RFSI")
$domains = @("full", "central_alps", "inn_valley", "rhine_valley", "etsch_valley", "northern_edge", "alpine_foreland")
$resolutions = @(50, 100, 500, 1000, 2000)

$scriptPath = Join-Path $PSScriptRoot "create_grids.py"
$configBase = Join-Path $PSScriptRoot "..\..\"
$failures = @()

function Invoke-GridJob([string]$label, [string[]]$jobArgs) {
    Write-Host ""
    Write-Host $label -ForegroundColor Yellow
    & $python $scriptPath @jobArgs
    if ($LASTEXITCODE -ne 0) {
        $failures += $label
        Write-Host "FAILED $label (exit $LASTEXITCODE)" -ForegroundColor Red
    }
}

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "Creating MASTER GRIDS + STATION LISTS" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

$current = 0
$totalGrids = $methods.Count * $resolutions.Count
foreach ($method in $methods) {
    foreach ($res in $resolutions) {
        $current++
        $configPath = Join-Path $configBase "$method\config.yaml"
        Invoke-GridJob "[$current / $totalGrids] Master Grid -> $method | ${res}m" @(
            "--method", $method,
            "--master",
            "--resolutions", "$res",
            "--config", $configPath
        )
    }
}

$current = 0
$totalStations = $methods.Count * $domains.Count
foreach ($method in $methods) {
    foreach ($domain in $domains) {
        $current++
        $configPath = Join-Path $configBase "$method\config.yaml"
        Invoke-GridJob "[$current / $totalStations] Stations -> $method | $domain" @(
            "--method", $method,
            "--domain", $domain,
            "--resolutions", "100",
            "--config", $configPath
        )
    }
}

Write-Host ""
if ($failures.Count -gt 0) {
    Write-Host "Finished with $($failures.Count) failure(s):" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}
Write-Host "All master grids and station lists created." -ForegroundColor Green
