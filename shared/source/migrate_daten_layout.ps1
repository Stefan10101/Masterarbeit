# Move Daten/ into CODE, Methods, Source, Plots, Network_providers, QC, Variables.
# Run from Daten/:
#   powershell -ExecutionPolicy Bypass -File CODE\shared\source\migrate_daten_layout.ps1

$ErrorActionPreference = "Stop"
$Daten = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path (Join-Path $Daten "CODE"))) {
    $Daten = (Get-Location).Path
}
if (-not (Test-Path (Join-Path $Daten "CODE"))) {
    throw "Run from Daten\ or pass the script under CODE\shared\source. Found: $Daten"
}

Write-Host "Daten root: $Daten"

function Ensure-Dir([string]$p) {
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null }
}

function Move-Into([string]$src, [string]$dstDir) {
    $name = Split-Path $src -Leaf
    $dst = Join-Path $dstDir $name
    if (-not (Test-Path $src)) {
        Write-Host "skip missing $src"
        return
    }
    if (Test-Path $dst) {
        Write-Host "already $dst"
        return
    }
    Ensure-Dir $dstDir
    Write-Host "move $src -> $dst"
    Move-Item -LiteralPath $src -Destination $dst
}

Ensure-Dir (Join-Path $Daten "Methods")
Ensure-Dir (Join-Path $Daten "Network_providers")
Ensure-Dir (Join-Path $Daten "Variables")
Ensure-Dir (Join-Path $Daten "Plots")

$methods = @("IDW","BSS","RFSI","RGI","Kriging","GAM","TPS","CNN","Frei")
foreach ($m in $methods) {
    $src = Join-Path $Daten $m
    if (Test-Path $src) { Move-Into $src (Join-Path $Daten "Methods") }
}

$providers = @("HYDRO","HYDRO_VO","LWD","MeteoSuisse","Stationen")
foreach ($n in $providers) {
    Move-Into (Join-Path $Daten $n) (Join-Path $Daten "Network_providers")
}

$vars = @("Wind","Temperatur","Schneehoehe","Luftfeuchte","Niederschlag")
foreach ($n in $vars) {
    Move-Into (Join-Path $Daten $n) (Join-Path $Daten "Variables")
}

$komb = Join-Path $Daten "Kombiniert"
$qc = Join-Path $Daten "QC"
if (Test-Path $komb) {
    if (Test-Path $qc) { Write-Host "QC already exists, not moving Kombiniert" }
    else {
        Write-Host "move $komb -> $qc"
        Move-Item -LiteralPath $komb -Destination $qc
    }
}

Write-Host "done"
Write-Host "Expect: CODE, Methods, Source, Plots, Network_providers, QC, Variables"
Get-ChildItem $Daten -Directory | ForEach-Object { $_.Name }
