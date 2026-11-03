# run_weekly_all.ps1
# Full weekly sequence for all methods.
# Methods that already finished are commented out so you can re-run the rest.
# Map production is left commented (expensive; enable later only for the methods you need).
# Stops on the first non-zero exit code.

$ErrorActionPreference = "Stop"
$vars = "temp_mean precip_sum wind_mean rh_mean snow_mean"

function Run-Step($label, $cmd) {
    Write-Host "`n========== $label ==========" -ForegroundColor Cyan
    Write-Host $cmd
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        throw "FAILED: $label (exit $LASTEXITCODE)"
    }
    $sw.Stop()
    Write-Host "OK ($([math]::Round($sw.Elapsed.TotalMinutes,1)) min)" -ForegroundColor Green
}

# =============================================================================
# IDW  — finished (maps + LLOCV + params for all 5 vars, 29 Oct)
# =============================================================================
# Run-Step "IDW tune"    "python IDW/tune_idw.py --time-resolution weekly --max-times 0 --variables $vars"
# Run-Step "IDW llocv"   "python IDW/llocv_idw.py --time-resolution weekly --variables $vars"
# Run-Step "IDW produce" "python IDW/produce_idw_maps.py --time-resolution weekly --variables $vars"

# =============================================================================
# Kriging  — tune + nested LLOCV finished for all 5; maps only temp_mean
# =============================================================================
# Run-Step "Kriging tune"    "python Kriging/tune_kriging.py --time-resolution weekly --max-times 0 --variables $vars"
# Run-Step "Kriging llocv"   "python Kriging/llocv_kriging.py --time-resolution weekly --variables $vars"
# Run-Step "Kriging produce" "python Kriging/produce_kriging_maps.py --time-resolution weekly --variables $vars"

# =============================================================================
# FREI  — tune + nested LLOCV finished for all 5 (30 Oct); maps missing
# =============================================================================
# Run-Step "FREI tune"    "python FREI/tune_frei.py --time-resolution weekly --max-times 0 --variables $vars"
# Run-Step "FREI llocv"   "python FREI/llocv_frei.py --time-resolution weekly --variables $vars"
# Run-Step "FREI produce" "python FREI/produce_frei_maps.py --time-resolution weekly --variables $vars"

# =============================================================================
# GAM  — not started
# =============================================================================
Run-Step "GAM tune"    "python GAM/tune_gam.py --time-resolution weekly --max-times 0 --variables $vars"
Run-Step "GAM llocv"   "python GAM/llocv_gam.py --time-resolution weekly --variables $vars"
# Run-Step "GAM produce" "python GAM/produce_gam_maps.py --time-resolution weekly --variables $vars"

# =============================================================================
# RFSI  — tune + nested LLOCV finished for all 5 (30 Oct); maps missing
# =============================================================================
# Run-Step "RFSI tune"    "python RFSI/tune_pooled.py --time-resolution weekly --max-times 0 --variables $vars"
# Run-Step "RFSI llocv"   "python RFSI/llocv_pooled.py --time-resolution weekly --variables $vars"
# Run-Step "RFSI produce" "python RFSI/produce_rfsi_maps.py --time-resolution weekly --variables $vars"

# =============================================================================
# CNN  — only temp_mean params + IDW cache exist; rest unfinished
# =============================================================================
Run-Step "CNN tune"    "python CNN/tune_cnn.py --time-resolution weekly"
Run-Step "CNN train"   "python CNN/train_cnn.py --time-resolution weekly"
Run-Step "CNN llocv"   "python CNN/llocv_cnn.py --time-resolution weekly"
# Run-Step "CNN produce" "python CNN/produce_cnn_maps.py --time-resolution weekly"

# =============================================================================
# RGI  — not started
# =============================================================================
Run-Step "RGI tune"    "python RGI/tune_rgi.py --time-resolution weekly --max-times 0 --variables $vars"
Run-Step "RGI llocv"   "python RGI/llocv_rgi.py --time-resolution weekly --variables $vars"
# Run-Step "RGI produce" "python RGI/produce_rgi_maps.py --time-resolution weekly --variables $vars"

# =============================================================================
# SMACNP  — not started (one variable at a time)
# =============================================================================
foreach ($v in $vars.Split()) {
    Run-Step "SMACNP tune $v"  "python SMACNP/tune_smacnp.py --time-resolution weekly --max-times 0 --variable $v"
    Run-Step "SMACNP llocv $v" "python SMACNP/llocv_smacnp.py --time-resolution weekly --variable $v"
    # Run-Step "SMACNP produce $v" "python SMACNP/produce_smacnp_maps.py --time-resolution weekly --variable $v"
}

Write-Host "`nWeekly tune + LLOCV finished for the active methods (maps still commented out)." -ForegroundColor Green