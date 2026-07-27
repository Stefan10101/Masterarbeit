# ============================================================
# run_master_grids.ps1
# Creates master grids + projected station lists for all domains
# ============================================================

$methods = @("IDW", "BSS", "RFSI")
$domains = @("full", "central_alps", "inn_valley", "rhine_valley", "etsch_valley", "northern_edge", "alpine_foreland")
$resolutions = @(50, 100, 500, 1000, 2000)

$scriptPath = "create_grids.py"
$configBase = "..\..\"
$totalGrids = $methods.Count * $resolutions.Count
$totalStations = $methods.Count * $domains.Count
$current = 0

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "Creating MASTER GRIDS + STATION LISTS for all methods" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

# === 1. Create Master Grids ===
foreach ($method in $methods) {
    foreach ($res in $resolutions) {
        $current++
        $configPath = "$configBase$method\config.yaml"

        Write-Host ""
        Write-Host "[$current / $totalGrids] Master Grid → $method | ${res}m" -ForegroundColor Yellow

        $args = @(
            "--method", $method,
            "--master",
            "--resolutions", $res,
            "--config", $configPath
        )
        & python $scriptPath @args
    }
}

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Green
Write-Host "Master grids created. Now creating station lists per domain..." -ForegroundColor Green
Write-Host "==============================================================" -ForegroundColor Green

# === 2. Create Projected Stations for ALL domains ===
$current = 0
foreach ($method in $methods) {
    foreach ($domain in $domains) {
        $current++
        $configPath = "$configBase$method\config.yaml"

        Write-Host ""
        Write-Host "[$current / $totalStations] Stations → $method | $domain" -ForegroundColor Yellow

        $args = @(
            "--method", $method,
            "--domain", $domain,
            "--resolutions", 100,
            "--config", $configPath
        )
        & python $scriptPath @args
    }
}

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Green
Write-Host "All master grids and station lists created successfully!" -ForegroundColor Green
Write-Host "==============================================================" -ForegroundColor Green