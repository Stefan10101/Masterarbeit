#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Thesis - Relative Humidity Full Analysis Plots (v2.5 - RH)
11 plots with hard QC bounds 1-100% only (no 99th percentile filtering)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import logging
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial.distance import pdist

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
BASE_DIR = Path(DATA_ROOT / "luftfeuchte" / "aaadata" / "full_data_cold_seasons")
PLOTS_DIR = Path(DATA_ROOT / "plots" / "luftfeuchte")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ELEVATION_BANDS = {
    "Low (<1000m)":          (0, 1000),
    "Mid-Low (1000-1500m)":  (1000, 1500),
    "Mid-High (1500-2000m)": (1500, 2000),
    "High (>2000m)":         (2000, 9999)
}

LON_MIN, LON_MAX = 7.5, 13.5
LAT_MIN, LAT_MAX = 45.0, 49

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

# ==================== DATA LOADING & CLEANING ====================
def load_and_clean_cold_season_data(base_dir: Path) -> pd.DataFrame:
    all_data = []
    json_files = list(base_dir.rglob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files.")

    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                meta = json.load(f)
            
            station = meta.get("name", json_path.stem.replace("_full_cold_seasons", ""))
            hoehe = float(meta.get("hoehe")) if meta.get("hoehe") is not None else None
            
            df_station = pd.DataFrame(meta.get("data", []))
            if df_station.empty: continue
                
            df_station["timestamp"] = pd.to_datetime(df_station["timestamp"], errors="coerce")
            df_station["value"] = pd.to_numeric(df_station["value"], errors="coerce")
            
            # Relative Humidity Quality Control:
            # Hard bounds only: 1% to 100% (no 99th percentile filtering)
            df_station = df_station[(df_station["value"] >= 1) & (df_station["value"] <= 100)]
            
            df_station = df_station.dropna(subset=["timestamp", "value"])
            
            if not df_station.empty and hoehe is not None:
                df_station["station"] = station
                df_station["hoehe"] = hoehe
                df_station["month"] = df_station["timestamp"].dt.month
                all_data.append(df_station[["timestamp", "value", "station", "hoehe", "month"]])
        except Exception as e:
            logger.warning(f"Skipped {json_path.name}: {type(e).__name__}")
    
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} records from {df['station'].nunique()} stations after QC (hard bounds 1â€“100% only)")
    return df

# ==================== STATION OVERVIEW HELPERS ====================
def collect_station_metadata(base_dir: Path) -> pd.DataFrame:
    stations = []
    json_files = list(base_dir.rglob("*.json"))
    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                meta = json.load(f)
            lat = float(meta.get("lat") or meta.get("latitude") or 0)
            lon = float(meta.get("lon") or meta.get("longitude") or 0)
            if abs(lat) > 90 or abs(lon) > 180:
                lat, lon = lon, lat
            lat = max(min(lat, 90), -90)
            lon = max(min(lon, 180), -180)
            
            data_list = meta.get("data", [])
            total = len(data_list)
            valid = sum(1 for d in data_list if d.get("value") is not None)
            coverage = (valid / total * 100) if total > 0 else 0.0
            
            stations.append({
                "station": meta.get("name", json_path.stem.replace("_full_cold_seasons", "")),
                "hoehe": float(meta.get("hoehe")) if meta.get("hoehe") is not None else None,
                "lat": lat,
                "lon": lon,
                "coverage_pct": coverage
            })
        except Exception:
            continue
    return pd.DataFrame(stations)

def add_nearest_neighbor_distances(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 2:
        df["nearest_km"] = np.nan
        return df
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c
    lat = df["lat"].values[:, None]
    lon = df["lon"].values[:, None]
    dist_matrix = haversine_km(lat, lon, lat.T, lon.T)
    np.fill_diagonal(dist_matrix, np.inf)
    df["nearest_km"] = np.min(dist_matrix, axis=1)
    return df

# ==================== STATION OVERVIEW PLOTS ====================
def plot_station_map(coords_df: pd.DataFrame, total_measurements: int):
    if coords_df.empty: return
    n_stations = len(coords_df)
    
    fig, ax = plt.subplots(figsize=(11, 9), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor='black')
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor='#1f78b4', alpha=0.7)
    
    sc = ax.scatter(coords_df["lon"], coords_df["lat"], c=coords_df["hoehe"], cmap='viridis', s=65,
                    edgecolor='black', linewidth=0.6, alpha=0.95, transform=ccrs.PlateCarree())
    
    plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02).set_label('Elevation [m]')
    
    info_text = f"Stations: {n_stations:,}\nTotal measurements: {total_measurements:,}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor='black'))
    
    if len(coords_df) <= 70:
        for _, row in coords_df.iterrows():
            ax.annotate(row["station"][:18], xy=(row["lon"], row["lat"]),
                        xytext=(4, 4), textcoords='offset points', fontsize=8, alpha=0.85)
    
    ax.set_title("Spatial Distribution of Relative Humidity Stations\n(Cold Seasons Data)", fontsize=14, pad=15)
    ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.6, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "station_map_cold_seasons.png", dpi=350, bbox_inches='tight')
    logger.info(f"Map saved â†’ station_map_cold_seasons.png (Stations: {n_stations}, Measurements: {total_measurements})")
    plt.close()

def plot_elevation_distribution(coords_df: pd.DataFrame):
    heights = coords_df["hoehe"].dropna()
    if heights.empty: return
    max_h = heights.max()
    bin_edges = np.arange(0, max_h + 200, 200)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(heights, bins=bin_edges, color='#1f77b4', edgecolor='black', alpha=0.85)
    ax.axvline(heights.mean(), color='red', linestyle='--', label=f'Mean: {heights.mean():.0f} m')
    ax.set_xlabel('Elevation [m]')
    ax.set_ylabel('Number of Stations')
    ax.set_title('Station Elevation Distribution (200 m bins)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "station_elevation_distribution.png", dpi=350, bbox_inches='tight')
    logger.info("Saved station_elevation_distribution.png")
    plt.close()

def plot_cold_season_coverage(coords_df: pd.DataFrame):
    coverage = coords_df["coverage_pct"].dropna()
    if coverage.empty: return
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(coverage, bins=20, density=True, color='#2ca02c', edgecolor='black', alpha=0.85)
    ax.set_ylabel('Percentage of Stations (%)')
    ax.set_xlabel('Cold-Season Coverage (%)')
    ax.set_title('Cold-Season Coverage Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "station_cold_season_coverage.png", dpi=350, bbox_inches='tight')
    logger.info("Saved station_cold_season_coverage.png")
    plt.close()

def plot_coverage_vs_height(coords_df: pd.DataFrame):
    if len(coords_df) < 2: return
    valid = coords_df[["hoehe", "coverage_pct"]].dropna()
    if len(valid) < 2:
        logger.warning("Not enough valid points for coverage vs height regression")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(valid["hoehe"], valid["coverage_pct"], c=valid["coverage_pct"], cmap='plasma', s=70, edgecolor='black')
    
    try:
        z = np.polyfit(valid["hoehe"], valid["coverage_pct"], 1)
        p = np.poly1d(z)
        x = np.linspace(valid["hoehe"].min(), valid["hoehe"].max(), 100)
        ax.plot(x, p(x), "r--", label=f'Trend: {z[0]:.4f} %/m')
    except Exception:
        logger.warning("polyfit failed, skipping trend line")
    
    ax.set_xlabel('Elevation [m]')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Coverage vs Station Height')
    plt.colorbar(sc, ax=ax, shrink=0.75).set_label('Coverage (%)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "coverage_vs_height.png", dpi=350, bbox_inches='tight')
    logger.info("Saved coverage_vs_height.png")
    plt.close()

def plot_nearest_neighbor_distances(coords_df: pd.DataFrame):
    if "nearest_km" not in coords_df.columns: return
    distances = coords_df["nearest_km"].dropna()
    if distances.empty: return
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(distances, bins=25, color='#d62728', edgecolor='black', alpha=0.85)
    ax.axvline(distances.mean(), color='red', linestyle='--', label=f'Mean: {distances.mean():.2f} km')
    ax.set_xlabel('Nearest Neighbor Distance (km)')
    ax.set_ylabel('Number of Stations')
    ax.set_title('Nearest-Neighbor Distance Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "station_nearest_neighbor_distances.png", dpi=350, bbox_inches='tight')
    logger.info("Saved station_nearest_neighbor_distances.png")
    plt.close()

# ==================== DETAILED ANALYSIS PLOTS ====================
def plot_A_overall_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(df["value"], bins=80, color='#1f77b4', edgecolor='black', alpha=0.85)
    mean_rh = df["value"].mean()
    ax.axvline(mean_rh, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rh:.1f} %')
    ax.set_xlabel('Relative Humidity (%)')
    ax.set_ylabel('Number of Observations')
    ax.set_title('A. Overall Relative Humidity Distribution\n(All Cold-Season Data â€“ QC: hard bounds 1â€“100% only)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "A_relative_humidity_distribution.png", dpi=350, bbox_inches='tight')
    logger.info("Saved A_relative_humidity_distribution.png")
    plt.close()

def plot_B_by_elevation_bands(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for (label, (low, high)), color in zip(ELEVATION_BANDS.items(), colors):
        mask = (df["hoehe"] >= low) & (df["hoehe"] < high)
        subset = df[mask]["value"]
        if len(subset) > 50:
            ax.hist(subset, bins=60, alpha=0.65, label=label, color=color, edgecolor='black')
    ax.set_xlabel('Relative Humidity (%)')
    ax.set_ylabel('Number of Observations')
    ax.set_title('B. Relative Humidity Distribution by Elevation Bands')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title="Elevation Band")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "B_relative_humidity_by_elevation_bands.png", dpi=350, bbox_inches='tight')
    logger.info("Saved B_relative_humidity_by_elevation_bands.png")
    plt.close()

def plot_C_mean_vs_elevation(df: pd.DataFrame):
    station_stats = df.groupby("station").agg(mean_rh=("value", "mean"), hoehe=("hoehe", "first")).reset_index()
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(station_stats["hoehe"], station_stats["mean_rh"], c=station_stats["mean_rh"], cmap='coolwarm', s=60, edgecolor='black', alpha=0.9)
    try:
        z = np.polyfit(station_stats["hoehe"], station_stats["mean_rh"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(station_stats["hoehe"].min(), station_stats["hoehe"].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2.5, label=f'Trend: {z[0]:.4f} %/m')
    except Exception:
        pass
    ax.set_xlabel('Elevation (hoehe) [m]')
    ax.set_ylabel('Mean Cold-Season Relative Humidity (%)')
    ax.set_title('C. Mean Relative Humidity vs Station Elevation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    cbar = plt.colorbar(sc, ax=ax, shrink=0.75)
    cbar.set_label('Mean RH (%)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "C_mean_relative_humidity_vs_elevation.png", dpi=350, bbox_inches='tight')
    logger.info("Saved C_mean_relative_humidity_vs_elevation.png")
    plt.close()

def plot_E_example_time_series(df: pd.DataFrame):
    stations = []
    for label, (low, high) in ELEVATION_BANDS.items():
        candidates = df[(df["hoehe"] >= low) & (df["hoehe"] < high)]["station"].unique()
        if len(candidates) > 0:
            stations.append(candidates[0])
        if len(stations) >= 3: break
    if len(stations) < 3: return
    fig, ax = plt.subplots(figsize=(12, 7))
    for station in stations[:3]:
        data = df[df["station"] == station].sort_values("timestamp")
        ax.plot(data["timestamp"], data["value"], label=f"{station} ({data['hoehe'].iloc[0]:.0f}m)", linewidth=1.0)
    ax.set_xlabel('Date')
    ax.set_ylabel('Relative Humidity (%)')
    ax.set_title('E. Example Cold-Season Time Series\n(One station per elevation band)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "E_example_time_series.png", dpi=350, bbox_inches='tight')
    logger.info("Saved E_example_time_series.png")
    plt.close()

def plot_F_monthly_mean_by_elevation(df: pd.DataFrame):
    df["month_name"] = pd.to_datetime(df["month"], format='%m').dt.strftime('%b')
    monthly = df.groupby(["month", "month_name", "hoehe"]).agg(mean_rh=("value", "mean")).reset_index()
    monthly["band"] = pd.cut(monthly["hoehe"], bins=[0,1000,1500,2000,9999], labels=list(ELEVATION_BANDS.keys()))
    fig, ax = plt.subplots(figsize=(11, 7))
    for band in ELEVATION_BANDS.keys():
        subset = monthly[monthly["band"] == band]
        if not subset.empty:
            ax.plot(subset["month_name"], subset["mean_rh"], marker='o', linewidth=2.5, label=band)
    ax.set_xlabel('Month')
    ax.set_ylabel('Mean Relative Humidity (%)')
    ax.set_title('F. Monthly Mean Relative Humidity by Elevation Band')
    ax.grid(True, alpha=0.3)
    ax.legend(title="Elevation Band")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "F_monthly_mean_by_elevation.png", dpi=350, bbox_inches='tight')
    logger.info("Saved F_monthly_mean_by_elevation.png")
    plt.close()

def plot_I_variability_vs_elevation(df: pd.DataFrame):
    station_stats = df.groupby("station").agg(std_rh=("value", "std"), hoehe=("hoehe", "first")).reset_index()
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(station_stats["hoehe"], station_stats["std_rh"], c=station_stats["std_rh"], cmap='viridis', s=65, edgecolor='black', alpha=0.9)
    ax.set_xlabel('Elevation (hoehe) [m]')
    ax.set_ylabel('Relative Humidity Standard Deviation (%)')
    ax.set_title('I. Relative Humidity Variability vs Station Elevation')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.75)
    cbar.set_label('Std Dev (%)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "I_variability_vs_elevation.png", dpi=350, bbox_inches='tight')
    logger.info("Saved I_variability_vs_elevation.png")
    plt.close()

# ==================== MAIN ====================
def main():
    logger.info("=== Relative Humidity Full Analysis (11 plots) Start ===")
    
    df = load_and_clean_cold_season_data(BASE_DIR)
    coords_df = collect_station_metadata(BASE_DIR)
    coords_df = add_nearest_neighbor_distances(coords_df)
    
    total_measurements = len(df)
    
    # 6 detailed analysis plots
    plot_A_overall_distribution(df)
    plot_B_by_elevation_bands(df)
    plot_C_mean_vs_elevation(df)
    plot_E_example_time_series(df)
    plot_F_monthly_mean_by_elevation(df)
    plot_I_variability_vs_elevation(df)
    
    # 5 station overview plots
    plot_station_map(coords_df, total_measurements=total_measurements)
    plot_elevation_distribution(coords_df)
    plot_cold_season_coverage(coords_df)
    plot_coverage_vs_height(coords_df)
    plot_nearest_neighbor_distances(coords_df)
    
    logger.info(f"=== All 11 plots saved to {PLOTS_DIR} ===")

if __name__ == "__main__":
    main()
