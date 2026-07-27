#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Thesis - Combined Multi-Variable Station Analysis (Cold Seasons v2.1 - RH)
- Dewpoint replaced by Relative Humidity
- All paths updated for Luftfeuchte
- Bar chart showing number of stations available per meteorological variable
- Donut pie showing distribution of stations by number of variables measured (1-5)
- Map of all physical stations colored by number of variables measured
- Bar chart of total valid records per variable
- Summary metrics + CSV with per-physical-station details
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import logging
from collections import defaultdict
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT


# ==================== CONFIG ====================
BASE_DIR = Path(DATA_ROOT)

VAR_DIRS = {
    "Precipitation":     BASE_DIR / "Niederschlag" / "AAAData" / "full_data_cold_seasons",
    "Snowheight":        BASE_DIR / "Schneehoehe" / "AAAData" / "full_data_cold_seasons",
    "Relative_Humidity": BASE_DIR / "Luftfeuchte" / "AAAData" / "full_data_cold_seasons",   # â† NEW
    "Temperature":       BASE_DIR / "Temperatur" / "AAAData" / "full_data_cold_seasons",
    "Wind":              BASE_DIR / "Wind"       / "AAAData" / "full_data_cold_seasons",
}

PLOTS_DIR = BASE_DIR / "Plots" / "Kombiniert"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LON_MIN, LON_MAX = 7.5, 13.5
LAT_MIN, LAT_MAX = 45.0, 49

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

# ==================== HELPER ====================
def collect_var_metadata(base_dir: Path, var_name: str):
    """Collect station metadata + valid record count for one variable."""
    if not base_dir.exists():
        logger.error(f"Directory not found for {var_name}: {base_dir}")
        return pd.DataFrame(), 0

    stations = []
    total_valid_records = 0
    json_files = list(base_dir.rglob("*.json"))
    logger.info(f"[{var_name}] Found {len(json_files)} JSON files...")

    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                meta = json.load(f)

            raw_station = meta.get("name", json_path.stem.replace("_full_cold_seasons", ""))
            station = raw_station.strip().lower().replace(" ", "_").replace("-", "_").replace("Ã¤", "ae").replace("Ã¶", "oe").replace("Ã¼", "ue")
            hoehe = float(meta.get("hoehe")) if meta.get("hoehe") is not None else None

            lat = float(meta.get("lat") or meta.get("latitude") or 0)
            lon = float(meta.get("lon") or meta.get("longitude") or 0)
            if abs(lat) > 90 or abs(lon) > 180:
                lat, lon = lon, lat
            lat = max(min(lat, 90), -90)
            lon = max(min(lon, 180), -180)

            data_list = meta.get("data", [])
            n_total = len(data_list)
            valid = sum(1 for d in data_list if d.get("value") is not None)
            total_valid_records += valid
            coverage = (valid / n_total * 100) if n_total > 0 else 0.0

            stations.append({
                "station": station,
                "hoehe": hoehe,
                "lat": lat,
                "lon": lon,
                "coverage_pct": round(coverage, 1),
                "n_records": valid
            })
        except Exception as e:
            logger.warning(f"Skipped {json_path.name} ({var_name}): {type(e).__name__}")

    df = pd.DataFrame(stations)
    if not df.empty:
        df = df.drop_duplicates(subset=["station"])
    logger.info(f"[{var_name}] {len(df)} stations, {total_valid_records:,} valid records")
    return df, total_valid_records

# ==================== MAIN ====================
def main():
    logger.info("=== Combined Multi-Variable Station Analysis (Cold Seasons - RH) ===")

    var_dfs = {}
    var_totals = {}

    for var_name, base_dir in VAR_DIRS.items():
        df, tot = collect_var_metadata(base_dir, var_name)
        if not df.empty:
            var_dfs[var_name] = df
            var_totals[var_name] = tot

    if not var_dfs:
        logger.error("No stations found in any variable directory!")
        return

    # Build physical station summary using ONLY location (lat/lon rounded to 0.001Â° â‰ˆ 100m)
    all_location_rows = []
    for var_name, df in var_dfs.items():
        for _, r in df.iterrows():
            lat_r = round(r["lat"], 3)
            lon_r = round(r["lon"], 3)
            location_key = f"{lat_r}_{lon_r}"
            all_location_rows.append({
                "location_key": location_key,
                "lat": r["lat"],
                "lon": r["lon"],
                "hoehe": r["hoehe"],
                "var": var_name
            })
    
    all_loc_df = pd.DataFrame(all_location_rows)
    
    # Group by physical location â€” each row is one unique station
    station_summary = all_loc_df.groupby("location_key").agg(
        lat=("lat", "first"),
        lon=("lon", "first"),
        hoehe=("hoehe", "mean"),
        num_vars=("var", "nunique"),
        vars=("var", lambda x: ", ".join(sorted(set(x))))
    ).reset_index()
    
    unique_by_location = len(station_summary)
    n_all5 = int((station_summary["num_vars"] == 5).sum())
    n_4plus = int((station_summary["num_vars"] >= 4).sum())
    
    counts = station_summary["num_vars"].value_counts().sort_index()

    # ==================== KEY METRICS ====================
    total_records = sum(var_totals.values())

    logger.info(f"\n{'='*60}")
    logger.info(f"TOTAL UNIQUE STATIONS (by location only): {unique_by_location:,}")
    logger.info(f"TOTAL VALID RECORDS (all variables):      {total_records:,}")
    logger.info(f"Stations measuring ALL 5 variables:       {n_all5}")
    logger.info(f"Stations measuring 4 or 5 variables:      {n_4plus}")
    logger.info(f"{'='*60}")

    for var in ["Precipitation", "Snowheight", "Relative_Humidity", "Temperature", "Wind"]:
        if var in var_totals:
            n_st = len(var_dfs[var])
            logger.info(f"  {var:17s}: {n_st:3d} stations, {var_totals[var]:>10,} records")

    # Save detailed CSV
    csv_path = PLOTS_DIR / "station_overlap_summary.csv"
    station_summary.sort_values(["num_vars", "location_key"], ascending=[False, True]).to_csv(csv_path, index=False)
    logger.info(f"\nDetailed per-station CSV saved â†’ {csv_path.name}")

    # ==================== PLOT 1: STATIONS PER VARIABLE ====================
    fig, ax = plt.subplots(figsize=(11, 7))

    var_names = ["Precipitation", "Snowheight", "Relative_Humidity", "Temperature", "Wind"]
    station_counts = [len(var_dfs.get(v, pd.DataFrame())) for v in var_names]
    var_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    bars = ax.bar(var_names, station_counts, color=var_colors, edgecolor='black', linewidth=1.3, width=0.7)
    ax.set_xlabel('Meteorological Variable', fontsize=13)
    ax.set_ylabel('Number of Stations', fontsize=13)
    ax.set_title('Number of Stations Available per Meteorological Variable\n(Cold Season Data â€“ Relative Humidity instead of Dewpoint)', fontsize=14, pad=18)
    ax.tick_params(axis='x', rotation=20, labelsize=11)
    ax.grid(True, alpha=0.35, axis='y', linestyle='--')

    max_h = max(station_counts) if station_counts else 10
    for bar, val in zip(bars, station_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_h*0.025,
                str(val), ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_stations_per_variable.png", dpi=350, bbox_inches='tight')
    logger.info("Saved 01_stations_per_variable.png")
    plt.close()

    # ==================== PLOT 2: DONUT PIE ====================
    colors = {1: '#d62728', 2: '#ff7f0e', 3: '#2ca02c', 4: '#1f77b4', 5: '#9467bd'}
    if len(counts) > 0:
        fig, ax = plt.subplots(figsize=(9, 9))
        sizes = counts.values
        labels = [f'{k} vars\n({v} stations)' for k, v in zip(counts.index, sizes)]
        pie_colors = [colors.get(k, '#555555') for k in counts.index]
        explode = [0.08 if k == 5 else 0.03 for k in counts.index]

        wedges, texts = ax.pie(sizes, explode=explode, labels=labels, colors=pie_colors,
                               startangle=90, pctdistance=0.78,
                               wedgeprops=dict(width=0.48, edgecolor='white', linewidth=2.5))

        centre_circle = plt.Circle((0, 0), 0.32, fc='white')
        ax.add_patch(centre_circle)
        ax.text(0, 0.03, f'{unique_by_location:,}', ha='center', va='center', fontsize=22, fontweight='bold', color='#222222')
        ax.text(0, -0.12, 'Unique Stations (by location)', ha='center', va='center', fontsize=11, color='#555555')

        ax.set_title('Proportion of Stations by Number of Variables Measured\n(Cold Season Data â€“ 5 variables: Precip, Snow, RH, Temp, Wind)', fontsize=14, pad=25)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "02_overlap_pie.png", dpi=350, bbox_inches='tight')
        logger.info("Saved 02_overlap_pie.png")
        plt.close()

    # ==================== PLOT 3: RECORDS PER VARIABLE ====================
    ordered = ["Precipitation", "Snowheight", "Relative_Humidity", "Temperature", "Wind"]
    var_names = [v for v in ordered if v in var_totals]
    rec_values = [var_totals[v] for v in var_names]
    var_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.bar(var_names, rec_values, color=var_colors[:len(var_names)], edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Total Valid Records (Cold Season)', fontsize=12)
    ax.set_title('Total Datapoints per Meteorological Variable\n(After quality control)', fontsize=14, pad=15)
    ax.tick_params(axis='x', rotation=18, labelsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    for bar, val in zip(bars, rec_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.015, f'{val:,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_records_per_var.png", dpi=350, bbox_inches='tight')
    logger.info("Saved 03_records_per_var.png")
    plt.close()

    # ==================== PLOT 4: MAP COLORED BY #VARIABLES ====================
    if not station_summary.empty:
        fig, ax = plt.subplots(figsize=(12.5, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#f4f4f4')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.9, edgecolor='#444444')
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='black')
        ax.add_feature(cfeature.RIVERS, linewidth=0.45, edgecolor='#1f78b4', alpha=0.65)

        sc = ax.scatter(station_summary["lon"], station_summary["lat"],
                        c=station_summary["num_vars"], cmap='viridis',
                        s=85, edgecolor='black', linewidth=0.65, alpha=0.93,
                        transform=ccrs.PlateCarree(), vmin=1, vmax=5)

        cbar = plt.colorbar(sc, ax=ax, shrink=0.62, pad=0.015, ticks=[1, 2, 3, 4, 5])
        cbar.set_label('Number of Variables Measured', fontsize=11)

        info = (f"Unique stations: {unique_by_location:,}\n"
                f"Total Records: {total_records:,}\n"
                f"All 5 vars: {n_all5} stations")
        ax.text(0.015, 0.985, info, transform=ax.transAxes, fontsize=10.5,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.96, edgecolor='black', linewidth=1.1))

        ax.set_title("Spatial Distribution of Stations\nColored by Multi-Variable Coverage (Cold Seasons â€“ Alps / Tyrol)", fontsize=14, pad=18)
        ax.gridlines(draw_labels=True, linewidth=0.35, alpha=0.55, linestyle='--', color='gray')

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "04_station_map_by_vars.png", dpi=350, bbox_inches='tight')
        logger.info("Saved 04_station_map_by_vars.png")
        plt.close()

    # ==================== DIAGNOSTIC OUTPUT ====================
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSTIC: Station counts per variable")
    for var_name, df in var_dfs.items():
        sample = list(df["station"].head(8))
        logger.info(f"  {var_name:17s}: {len(df):4d} stations   Sample: {sample}")

    logger.info(f"\n  Breakdown of unique stations by number of variables:")
    for k in sorted(counts.index):
        logger.info(f"    Exactly {k} variable(s): {counts[k]:4d} stations")
    logger.info(f"\n  >>> UNIQUE STATIONS (by location only): {unique_by_location}")
    logger.info("="*70)

    logger.info(f"\n=== All plots + CSV saved to {PLOTS_DIR} ===")
    logger.info("Open the CSV to see exactly which stations measure which combination of variables.")

if __name__ == "__main__":
    main()
