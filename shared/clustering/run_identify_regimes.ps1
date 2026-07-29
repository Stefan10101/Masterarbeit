# ============================================================
# run_identify_regimes.ps1
# Full production run of regime clustering (once for all methods)
# ============================================================

# Clustering is method-agnostic: run once, write into every method folder
$methods        = @("IDW", "BSS", "RFSI")
$resolution     = "seasonal"
$kMin           = 2
$kMax           = 12
$nMedoids       = 42
$clusterMethod  = "som"       # kmeans | gmm | som
$pcaVariance    = 0.95        # 0 = disable PCA

$scriptPath  = "identify_regimes.py"

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "Regime clustering - full production run" -ForegroundColor Cyan
Write-Host ("methods={0}" -f ($methods -join ", ")) -ForegroundColor Cyan
Write-Host ("resolution={0}  cluster={1}  PCA={2}" -f $resolution, $clusterMethod, $pcaVariance) -ForegroundColor Cyan
Write-Host ("k=[{0}..{1}]  n_medoids={2}" -f $kMin, $kMax, $nMedoids) -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""

$argList = @(
    "--method"
) + $methods + @(
    "--resolution",      $resolution,
    "--k-min",           "$kMin",
    "--k-max",           "$kMax",
    "--n-medoids",       "$nMedoids",
    "--cluster-method",  $clusterMethod,
    "--pca-variance",    "$pcaVariance"
)

& python $scriptPath @argList

if ($LASTEXITCODE -ne 0) {
    Write-Host ("FAILED (exit {0})" -f $LASTEXITCODE) -ForegroundColor Red
} else {
    Write-Host ""
    Write-Host "==============================================================" -ForegroundColor Green
    Write-Host "Regime clustering finished (results in all method folders)." -ForegroundColor Green
    Write-Host "==============================================================" -ForegroundColor Green
}
