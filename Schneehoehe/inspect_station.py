#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick inspection script for suspicious measurements
- Plot any station for any time period
- Easy to change station, year, months, or specific days
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging
import json

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT


# ====================== CONFIG - CHANGE THESE ======================
STATION_NAME = "S. Bernardino"          # Change this to any station name
YEAR = 2023                              # Change year
START_MONTH = 4                          # Start month (1-12)
END_MONTH = 4                            # End month (1-12)
START_DAY = 12
END_DAY = 15

DATA_FOLDER = Path(DATA_ROOT / "schneehoehe" / "aaadata" / "cold_season_combined")
PLOTS_ROOT = Path(DATA_ROOT / "plots" / "schneehoehe" / "inspection")
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

log_file = PLOTS_ROOT / f"inspection_{STATION_NAME.replace(' ', '_')}_{YEAR}_{START_MONTH:02d}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# ====================== LOAD STATION DATA ======================
def load_station_data(station_name: str) -> pd.DataFrame:
    logger.info(f"Searching for station: {station_name}")
    
    # Try to find the parquet file
    parquet_files = list(DATA_FOLDER.glob("*.parquet"))
    
    for pq_file in parquet_files:
        # Check JSON metadata first (faster)
        json_file = pq_file.with_suffix(".json")
        if json_file.exists():
            with open(json_file, encoding="utf-8") as f:
                meta = json.load(f)
                if meta.get("name") == station_name:
                    logger.info(f"Found station in: {pq_file.name}")
                    df = pd.read_parquet(pq_file)
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                    df["station"] = meta.get("name", station_name)
                    df["hoehe"] = meta.get("hoehe")
                    return df
    
    # Fallback: search in parquet data
    logger.info("Searching in parquet data (slower)...")
    for pq_file in parquet_files:
        try:
            df = pd.read_parquet(pq_file)
            if "name" in df.columns:
                if station_name in df["name"].values:
                    logger.info(f"Found station in: {pq_file.name}")
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                    return df
        except:
            continue
    
    logger.error(f"Station '{station_name}' not found!")
    return pd.DataFrame()

# ====================== PLOT FUNCTION ======================
def plot_inspection(df: pd.DataFrame, station_name: str, year: int, 
                    start_month: int, end_month: int, save_path: Path):
    if df.empty:
        return
    
    # Filter by year and months
    mask = (df["timestamp"].dt.year == year) & \
           (df["timestamp"].dt.month >= start_month) & \
           (df["timestamp"].dt.month <= end_month)
    
    df_period = df[mask].copy()
    
    if df_period.empty:
        logger.warning(f"No data found for {station_name} in {year}-{start_month:02d} to {end_month:02d}")
        return
    
    df_period = df_period.sort_values("timestamp")
    
    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")
    
    # Main line
    plt.plot(df_period["timestamp"], df_period["value"], 
             color="#1f77b4", linewidth=1.5, label="Snow height")
    
    # Highlight suspicious spikes (values > 3x median in this period)
    median_val = df_period["value"].median()
    spike_threshold = median_val * 3
    spikes = df_period[df_period["value"] > spike_threshold]
    
    if not spikes.empty:
        plt.scatter(spikes["timestamp"], spikes["value"], 
                    color="red", s=40, zorder=5, label=f"Potential spikes (> {spike_threshold:.0f} cm)")
    
    # Title and labels
    hoehe = df_period["hoehe"].iloc[0] if "hoehe" in df_period.columns else "N/A"
    plt.title(f"SchneehÃ¶he {station_name} ({hoehe} m) â€“ {year}-{start_month:02d} to {end_month:02d}", 
              fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("SchneehÃ¶he [cm]", fontsize=12)
    plt.xlabel("Zeit", fontsize=12)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d.%b'))
    plt.xticks(rotation=45)
    
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"â†’ Saved inspection plot: {save_path.name}")
    logger.info(f"   Period: {df_period['timestamp'].min()} to {df_period['timestamp'].max()}")
    logger.info(f"   Max value: {df_period['value'].max():.1f} cm | Median: {median_val:.1f} cm")

# ====================== MAIN ======================
def main():
    logger.info("=== STATION INSPECTION SCRIPT ===")
    
    df = load_station_data(STATION_NAME)
    if df.empty:
        return
    
    # Create filename-safe name
    safe_name = STATION_NAME.replace(" ", "_").replace("/", "_")
    save_path = PLOTS_ROOT / f"inspect_{safe_name}_{YEAR}_{START_MONTH:02d}-{END_MONTH:02d}.png"
    
    plot_inspection(df, STATION_NAME, YEAR, START_MONTH, END_MONTH, save_path)
    
    logger.info("=== DONE ===")
    logger.info(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    main()
