# ============================================================
# run_identify_regimes.ps1
# Full production run of regime clustering for all variables
# ============================================================

$methods     = @("IDW", "BSS", "RFSI")
$resolution  = "half_hourly"
$kMin        = 2
$kMax        = 12
$nMedoids    = 42

# Set to $true only if RAM is insufficient for the full horizon
$yearByYear  = $false
$years       = @(2020, 2021, 2022, 2023, 2024, 2025)

$scriptPath  = "identify_regimes.py"
$total       = if ($yearByYear) { $methods.Count * $years.Count } else { $methods.Count }
$current     = 0

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "Regime clustering - full production run" -ForegroundColor Cyan
Write-Host ("resolution={0}  k=[{1}..{2}]  n_medoids={3}" -f $resolution, $kMin, $kMax, $nMedoids) -ForegroundColor Cyan
Write-Host ("year-by-year={0}" -f $yearByYear) -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

foreach ($method in $methods) {
    if ($yearByYear) {
        foreach ($year in $years) {
            $current++
            $start = "{0}-01-01" -f $year
            $end   = "{0}-12-31" -f $year

            Write-Host ""
            Write-Host ("[{0} / {1}] {2}  {3}  {4}" -f $current, $total, $method, $resolution, $year) -ForegroundColor Yellow

            $argList = @(
                "--method",      $method,
                "--resolution",  $resolution,
                "--k-min",       "$kMin",
                "--k-max",       "$kMax",
                "--n-medoids",   "$nMedoids",
                "--start-date",  $start,
                "--end-date",    $end
            )
            & python $scriptPath @argList

            if ($LASTEXITCODE -ne 0) {
                Write-Host ("FAILED: {0} {1} (exit {2})" -f $method, $year, $LASTEXITCODE) -ForegroundColor Red
            }
        }
    }
    else {
        $current++
        Write-Host ""
        Write-Host ("[{0} / {1}] {2}  {3}  full period" -f $current, $total, $method, $resolution) -ForegroundColor Yellow

        $argList = @(
            "--method",      $method,
            "--resolution",  $resolution,
            "--k-min",       "$kMin",
            "--k-max",       "$kMax",
            "--n-medoids",   "$nMedoids"
        )
        & python $scriptPath @argList

        if ($LASTEXITCODE -ne 0) {
            Write-Host ("FAILED: {0} (exit {1})" -f $method, $LASTEXITCODE) -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Green
Write-Host "All regime clustering runs finished." -ForegroundColor Green
Write-Host "==============================================================" -ForegroundColor Green
