# Full monthly suite: 5 variables, 10 station folds, all months 2020-2025.
# Resume-safe: skips a step when its output already exists and is large enough.
# Force a redo:  .\run_full_monthly.ps1 -Force
#
# From an already-activated Master prompt in CODE\:
#   .\run_full_monthly.ps1

param([switch]$Force)

$ErrorActionPreference = "Continue"
$Code = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Code
$Data = Split-Path -Parent $Code

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

if ($env:CONDA_PREFIX) {
    $Python = Join-Path $env:CONDA_PREFIX "python.exe"
} else {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    $Python = if ($cmd) { $cmd.Source } else { $null }
}
if (-not $Python -or -not (Test-Path $Python)) {
    Write-Host "No python. Activate conda env Master in THIS shell, then run:"
    Write-Host "  .\run_full_monthly.ps1"
    exit 1
}

$Vars = @("temp_mean", "precip_sum", "wind_mean", "rh_mean", "snow_mean")
$Folds = 10
$LogDir = Join-Path $Code "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $LogDir "full_monthly_$Stamp.log"
$FailLog = Join-Path $LogDir "full_monthly_$Stamp.fail.txt"

function Write-Log([string]$msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $Log -Value $line
}

function Test-Done([string]$path, [long]$minBytes) {
    if ($Force) { return $false }
    if (-not $path) { return $false }
    if (-not (Test-Path $path)) { return $false }
    $len = (Get-Item $path).Length
    return ($len -ge $minBytes)
}

function Out-Params([string]$methodFolder, [string]$subdir, [string]$var, [string]$suffix) {
    return Join-Path $Data "Methods\$methodFolder\Output\cluster_params\monthly\$subdir\${var}_$suffix"
}
function Out-Llocv([string]$methodFolder, [string]$var) {
    return Join-Path $Data "Methods\$methodFolder\Output\llocv\full\nested\monthly\${var}_monthly_nested_llocv.parquet"
}
function Out-Map([string]$methodFolder, [string]$var) {
    return Join-Path $Data "Methods\$methodFolder\Output\interpolated_maps\full\res_1000m\$var\${var}_1000m_monthly_20200101-20251231.nc"
}
function Out-Cluster([string]$methodFolder, [string]$canon) {
    return Join-Path $Data "Methods\$methodFolder\Output\cluster_params\monthly\gmm\${canon}_params.parquet"
}
function Out-CnnModel([string]$var) {
    $dir = Join-Path $Data "Methods\CNN\Output\models\monthly"
    if (-not (Test-Path $dir)) { return $null }
    $hit = Get-ChildItem $dir -Filter "*$var*" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($hit) { return $hit.FullName }
    return $null
}

function Invoke-Step([string]$name, [string[]]$argv, [string]$marker, [long]$minBytes) {
    if (Test-Done $marker $minBytes) {
        Write-Log "SKIP  $name  ($marker)"
        return $true
    }
    Write-Log "START $name"
    Write-Log ("CMD   $Python " + ($argv -join " "))
    if ($marker) { Write-Log "MARK  $marker" }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $stdout = Join-Path $env:TEMP ("fullrun_out_{0}.txt" -f [guid]::NewGuid())
    $stderr = Join-Path $env:TEMP ("fullrun_err_{0}.txt" -f [guid]::NewGuid())
    $argline = ($argv | ForEach-Object { if ($_ -match '[\s,]') { '"{0}"' -f $_ } else { $_ } }) -join " "
    $proc = Start-Process -FilePath $Python -ArgumentList $argline -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $code = $proc.ExitCode
    foreach ($f in @($stdout, $stderr)) {
        if (Test-Path $f) {
            Get-Content $f -ErrorAction SilentlyContinue | ForEach-Object {
                Write-Host $_
                Add-Content -Path $Log -Value $_
            }
            Remove-Item $f -Force -ErrorAction SilentlyContinue
        }
    }
    $sw.Stop()
    if ($null -eq $code) { $code = 0 }
    if ($code -ne 0) {
        $line = "FAIL  $name exit=$code after $([int]$sw.Elapsed.TotalMinutes) min"
        Write-Log $line
        Add-Content -Path $FailLog -Value $line
        return $false
    }
    Write-Log "OK    $name $([int]$sw.Elapsed.TotalMinutes) min"
    return $true
}

Write-Log "============================================================"
Write-Log "FULL MONTHLY RUN  folds=$Folds  force=$Force  vars=$($Vars -join ',')"
Write-Log "cwd=$Code"
Write-Log "python=$Python"
Write-Log "data=$Data"
Write-Log "log=$Log"
Write-Log "============================================================"

$Canon = @{
    temp_mean = "temperature"
    precip_sum = "precipitation"
    wind_mean = "wind_speed"
    rh_mean = "relative_humidity"
    snow_mean = "snow_height"
}

# maps: ~50 MB when complete. <5 MB after a crash mid-write -> redo.
$MinYaml = 40
$MinParq = 50000
$MinNc   = 5000000
$MinPt   = 100000

foreach ($v in $Vars) {
    $c = $Canon[$v]
    [void](Invoke-Step "IDW tune $v" @("IDW/train_cluster_params.py", "--method", "IDW", "--resolution", "monthly", "--variables", $c) (Out-Cluster "IDW" $c) $MinParq)
    [void](Invoke-Step "IDW produce $v" @("IDW/produce_idw_maps.py", "--variables", $v) (Out-Map "IDW" $v) $MinNc)
}
foreach ($v in $Vars) {
    $c = $Canon[$v]
    [void](Invoke-Step "BSS tune $v" @("BSS/train_cluster_params.py", "--method", "BSS", "--resolution", "monthly", "--variables", $c) (Out-Cluster "BSS" $c) $MinParq)
    [void](Invoke-Step "BSS produce $v" @("BSS/produce_bss_maps.py", "--variables", $v) (Out-Map "BSS" $v) $MinNc)
}

foreach ($v in $Vars) {
    [void](Invoke-Step "Frei tune $v"  @("FREI/tune_frei.py", "--variable", $v, "--folds", "$Folds", "--rh-t-mode", "predicted") (Out-Params "Frei" "frei" $v "frei_params.yaml") $MinYaml)
    [void](Invoke-Step "Frei llocv $v" @("FREI/llocv_frei.py", "--variable", $v, "--folds", "$Folds", "--fit", "train", "--score", "dev,test", "--rh-t-mode", "predicted") (Out-Llocv "Frei" $v) $MinParq)
    [void](Invoke-Step "Frei maps $v"  @("FREI/produce_frei_maps.py", "--variable", $v, "--rh-t-mode", "predicted") (Out-Map "Frei" $v) $MinNc)
}
foreach ($v in $Vars) {
    [void](Invoke-Step "GAM tune $v"  @("GAM/tune_gam.py", "--variable", $v, "--folds", "$Folds", "--rh-t-mode", "predicted") (Out-Params "GAM" "gam" $v "gam_params.yaml") $MinYaml)
    [void](Invoke-Step "GAM llocv $v" @("GAM/llocv_gam.py", "--variable", $v, "--folds", "$Folds", "--fit", "train", "--score", "dev,test", "--rh-t-mode", "predicted") (Out-Llocv "GAM" $v) $MinParq)
    [void](Invoke-Step "GAM maps $v"  @("GAM/produce_gam_maps.py", "--variable", $v, "--rh-t-mode", "predicted") (Out-Map "GAM" $v) $MinNc)
}
foreach ($v in $Vars) {
    [void](Invoke-Step "TPS tune $v"  @("TPS/tune_tps.py", "--variable", $v, "--folds", "$Folds", "--rh-t-mode", "predicted") (Out-Params "TPS" "tps" $v "tps_params.yaml") $MinYaml)
    [void](Invoke-Step "TPS llocv $v" @("TPS/llocv_tps.py", "--variable", $v, "--folds", "$Folds", "--fit", "train", "--score", "dev,test", "--rh-t-mode", "predicted") (Out-Llocv "TPS" $v) $MinParq)
    [void](Invoke-Step "TPS maps $v"  @("TPS/produce_tps_maps.py", "--variable", $v, "--rh-t-mode", "predicted") (Out-Map "TPS" $v) $MinNc)
}
foreach ($v in $Vars) {
    [void](Invoke-Step "Kriging tune $v"  @("Kriging/tune_kriging.py", "--variables", $v, "--folds", "$Folds", "--rh-t-mode", "predicted") (Out-Params "Kriging" "kriging" $v "kriging_params.yaml") $MinYaml)
    [void](Invoke-Step "Kriging llocv $v" @("Kriging/llocv_kriging.py", "--variables", $v, "--folds", "$Folds", "--mode", "folds", "--fit", "train", "--score", "dev,test", "--rh-t-mode", "predicted") (Out-Llocv "Kriging" $v) $MinParq)
    [void](Invoke-Step "Kriging maps $v"  @("Kriging/produce_kriging_maps.py", "--variables", $v, "--rh-t-mode", "predicted", "--skip-station-check") (Out-Map "Kriging" $v) $MinNc)
}
foreach ($v in $Vars) {
    [void](Invoke-Step "RFSI tune $v"  @("RFSI/tune_pooled.py", "--variables", $v, "--n-splits", "$Folds") (Out-Params "RFSI" "pooled" $v "pooled_params.yaml") $MinYaml)
    [void](Invoke-Step "RFSI llocv $v" @("RFSI/llocv_pooled.py", "--variables", $v, "--folds", "$Folds", "--fit", "train", "--score", "dev,test") (Out-Llocv "RFSI" $v) $MinParq)
    [void](Invoke-Step "RFSI maps $v"  @("RFSI/produce_rfsi_maps.py", "--variables", $v) (Out-Map "RFSI" $v) $MinNc)
}
foreach ($v in $Vars) {
    [void](Invoke-Step "RGI tune $v"  @("RGI/tune_rgi.py", "--variables", $v, "--phase", "all", "--folds", "$Folds") (Out-Params "RGI" "rgi" $v "rgi_params.yaml") $MinYaml)
    [void](Invoke-Step "RGI llocv $v" @("RGI/llocv_rgi.py", "--variables", $v, "--folds", "$Folds", "--mode", "folds", "--fit", "train", "--score", "dev,test") (Out-Llocv "RGI" $v) $MinParq)
    [void](Invoke-Step "RGI maps $v"  @("RGI/produce_rgi_maps.py", "--variables", $v) (Out-Map "RGI" $v) $MinNc)
}
foreach ($v in $Vars) {
    [void](Invoke-Step "CNN tune $v"  @("CNN/tune_cnn.py", "--variable", $v) (Out-Params "CNN" "cnn" $v "cnn_params.yaml") $MinYaml)
    [void](Invoke-Step "CNN train $v" @("CNN/train_cnn.py", "--variable", $v) (Out-CnnModel $v) $MinPt)
    [void](Invoke-Step "CNN llocv $v" @("CNN/llocv_cnn.py", "--variable", $v, "--nested", "--folds", "$Folds", "--score", "dev,test") (Out-Llocv "CNN" $v) $MinParq)
    [void](Invoke-Step "CNN maps $v"  @("CNN/produce_cnn_maps.py", "--variable", $v) (Out-Map "CNN" $v) $MinNc)
}

Write-Log "============================================================"
if (Test-Path $FailLog) {
    Write-Log "DONE WITH FAILURES  see $FailLog"
    Get-Content $FailLog | ForEach-Object { Write-Log $_ }
} else {
    Write-Log "DONE ALL STEPS OK"
}
Write-Log "log $Log"
