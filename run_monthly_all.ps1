# run_monthly_all.ps1
# Monthly produce maps for all methods, limited to the 12 months of 2025.
# LLOCV lines active. Tune and produce left commented.
# Stops on first non-zero exit.

$ErrorActionPreference = "Stop"
$env:WEEKLY_HOLDOUT_STAMP = Join-Path $PSScriptRoot "Output\_run_stamp\holdout_stamp.json"
python (Join-Path $PSScriptRoot "shared\splits\stamp_holdout.py")
if ($LASTEXITCODE -ne 0) { throw "holdout stamp failed" }
$vars = @("temp_mean","precip_sum","wind_mean","rh_mean","snow_mean")
$res  = "monthly"
$months2025 = "2025-01,2025-02,2025-03,2025-04,2025-05,2025-06,2025-07,2025-08,2025-09,2025-10,2025-11,2025-12"

function Run-Step($label, $cmd) {
    Write-Host "`n========== $label ==========" -ForegroundColor Cyan
    Write-Host $cmd
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $label (exit $LASTEXITCODE)" }
    $sw.Stop()
    Write-Host "OK ($([math]::Round($sw.Elapsed.TotalMinutes,1)) min)" -ForegroundColor Green
}

# IDW
#Run-Step "IDW tune"  "python IDW/tune_idw.py --time-resolution $res --max-times 0 --variables $($vars -join ' ')"
Run-Step "IDW llocv" "python IDW/llocv_idw.py --time-resolution $res --variables $($vars -join ' ')"
#Run-Step "IDW produce" "python IDW/produce_idw_maps.py --time-resolution $res --variables $($vars -join ' ') --months $months2025"

# Kriging
#Run-Step "Kriging tune"  "python Kriging/tune_kriging.py --time-resolution $res --max-times 0 --variables $($vars -join ' ')"
Run-Step "Kriging llocv" "python Kriging/llocv_kriging.py --time-resolution $res --variables $($vars -join ' ')"
#Run-Step "Kriging produce" "python Kriging/produce_kriging_maps.py --time-resolution $res --variables $($vars -join ' ') --months $months2025"

# FREI
foreach ($v in $vars) {
    #Run-Step "FREI tune $v"  "python FREI/tune_frei.py --time-resolution $res --max-times 0 --variable $v"
    Run-Step "FREI llocv $v" "python FREI/llocv_frei.py --time-resolution $res --variable $v"
 #   Run-Step "FREI produce $v" "python FREI/produce_frei_maps.py --time-resolution $res --variable $v --months $months2025"
}

# GAM
foreach ($v in $vars) {
    #Run-Step "GAM tune $v"  "python GAM/tune_gam.py --time-resolution $res --max-times 0 --variable $v"
    Run-Step "GAM llocv $v" "python GAM/llocv_gam.py --time-resolution $res --variable $v"
  #  Run-Step "GAM produce $v" "python GAM/produce_gam_maps.py --time-resolution $res --variable $v --months $months2025"
}

# RFSI
#Run-Step "RFSI tune"  "python RFSI/tune_pooled.py --time-resolution $res --max-times 0 --variables $($vars -join ' ')"
Run-Step "RFSI llocv" "python RFSI/llocv_pooled.py --time-resolution $res --variables $($vars -join ' ')"
#Run-Step "RFSI produce" "python RFSI/produce_rfsi_maps.py --time-resolution $res --variables $($vars -join ' ') --months $months2025"

# CNN
foreach ($v in $vars) {
    #Run-Step "CNN tune $v"  "python CNN/tune_cnn.py --time-resolution $res --variable $v"
    #Run-Step "CNN train $v" "python CNN/train_cnn.py --time-resolution $res --variable $v"
    Run-Step "CNN llocv $v" "python CNN/llocv_cnn.py --time-resolution $res --variable $v"
 #   Run-Step "CNN produce $v" "python CNN/produce_cnn_maps.py --time-resolution $res --variable $v --months $months2025"
}

# RGI
#Run-Step "RGI tune"  "python RGI/tune_rgi.py --time-resolution $res --max-times 0 --variables $($vars -join ' ')"
Run-Step "RGI llocv" "python RGI/llocv_rgi.py --time-resolution $res --variables $($vars -join ' ')"
#Run-Step "RGI produce" "python RGI/produce_rgi_maps.py --time-resolution $res --variables $($vars -join ' ') --months $months2025"

# SMACNP
foreach ($v in $vars) {
    #Run-Step "SMACNP tune $v"  "python SMACNP/tune_smacnp.py --time-resolution $res --max-times 0 --variable $v"
    Run-Step "SMACNP llocv $v" "python SMACNP/llocv_smacnp.py --time-resolution $res --variable $v"
    #Run-Step "SMACNP produce $v" "python SMACNP/produce_smacnp_maps.py --time-resolution $res --variable $v --months $months2025"
}

Write-Host "`nMonthly 2025 maps finished." -ForegroundColor Green
