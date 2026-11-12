# pack_findings.ps1
# Pack LLOCV scores, search/tune artefacts and models for one (or all) time resolutions.
# Run from the Daten directory:
#   .\CODE\pack_findings.ps1 monthly
#   .\CODE\pack_findings.ps1 weekly
#   .\CODE\pack_findings.ps1 all

param(
    [Parameter(Mandatory=$true, Position=0)]
    [ValidateSet("monthly","weekly","daily","half_hourly","all")]
    [string]$TimeResolution
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$resolutions = if ($TimeResolution -eq "all") {
    @("monthly","weekly","daily","half_hourly")
} else {
    @($TimeResolution)
}

function To-Rel($path) {
    $full = (Resolve-Path $path -ErrorAction SilentlyContinue).Path
    if (-not $full) { return $null }
    $rel = $full.Substring($root.Length).TrimStart('\','/')
    return $rel.Replace('\','/')
}

function Get-ItemsForResolution($res) {
    $items = [System.Collections.Generic.List[string]]::new()

    $methods = @("CNN","Frei","GAM","IDW","Kriging","RFSI","RGI","SMACNP")
    foreach ($m in $methods) {
        $base = "Methods/$m/Output"

        $nested = "$base/llocv/full/nested/$res"
        if (Test-Path $nested) { $items.Add((To-Rel $nested)) }

        if (Test-Path "$base/llocv") {
            Get-ChildItem "$base/llocv" -Recurse -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -eq $res -or $_.FullName -match "[\\/]$res([\\/]|$)" } |
                ForEach-Object {
                    $rel = To-Rel $_.FullName
                    if ($rel) { $items.Add($rel) }
                }
        }

        $cp = "$base/cluster_params/$res"
        if (Test-Path $cp) { $items.Add((To-Rel $cp)) }

        $tuned = "$base/tuned"
        if (Test-Path $tuned) {
            Get-ChildItem $tuned -File -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match $res } |
                ForEach-Object {
                    $rel = To-Rel $_.FullName
                    if ($rel) { $items.Add($rel) }
                }
        }

        $models = "$base/models/$res"
        if (Test-Path $models) { $items.Add((To-Rel $models)) }
    }

    $sm = "Methods/SMACNP/Output/llocv/$res"
    if (Test-Path $sm) { $items.Add((To-Rel $sm)) }

    Get-ChildItem "Methods" -Recurse -File -Include *.parquet -ErrorAction SilentlyContinue |
        Where-Object {
            ($_.Name -like '*heldout*.parquet' -or $_.Name -like '*_llocv.parquet') -and
            $_.FullName -match $res
        } |
        ForEach-Object {
            $rel = To-Rel $_.FullName
            if ($rel) { $items.Add($rel) }
        }

    if (Test-Path "CODE/Output") { $items.Add("CODE/Output") }
    $plot = "Plots/llocv_compare/$res"
    if (Test-Path $plot) { $items.Add((To-Rel $plot)) }
    if (Test-Path "Plots/completeness") { $items.Add("Plots/completeness") }

    return $items | Where-Object { $_ } | Sort-Object -Unique
}

foreach ($res in $resolutions) {
    $items = @(Get-ItemsForResolution $res)
    if (-not $items -or $items.Count -eq 0) {
        Write-Host "No artefacts found for $res - skipping." -ForegroundColor Yellow
        continue
    }

    $out = if ($res -eq "monthly") { "monthly_run_findings.tar.gz" } else { "findings_$res.tar.gz" }
    if (Test-Path $out) { Remove-Item $out }

    Write-Host "Packing $($items.Count) roots for $res -> $out" -ForegroundColor Cyan
    $items | Where-Object { $_ -like '*SMACNP*' } | ForEach-Object { Write-Host "  $_" }
    tar -czf $out --exclude='*.nc' --exclude='*.pt' --exclude='*.joblib' @items
    Write-Host "OK: $out" -ForegroundColor Green
}

Write-Host "Done."
