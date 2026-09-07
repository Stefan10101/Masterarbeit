# Build Source/grids + Source/stations. Run from anywhere after bootstrap_source.py.

$ErrorActionPreference = "Stop"
$python = $null
$candidates = @(
    "$env:USERPROFILE\miniforge3\envs\Master\python.exe",
    "$env:USERPROFILE\mambaforge\envs\Master\python.exe",
    "$env:CONDA_PREFIX\python.exe"
)
foreach ($c in $candidates) {
    if ($c -and (Test-Path $c)) { $python = $c; break }
}
if (-not $python) { throw "Master python.exe not found." }

$scriptPath = Join-Path $PSScriptRoot "create_grids.py"
Write-Host "Python: $python"
& $python -u $scriptPath --master --resolutions 1000
if ($LASTEXITCODE -ne 0) { throw "create_grids failed" }
Write-Host "grids done"
