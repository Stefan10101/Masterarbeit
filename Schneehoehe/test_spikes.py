#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAFE Spike Detection & Removal
- NEVER modifies original data
- Only creates NEW cleaned copies in a separate folder
- Test on S. Bernardino first, then apply to all stations
"""

import pandas as pd
import numpy as np
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


# ====================== CONFIG ======================
# === CHANGE THESE ===
STATION_NAME = "S. Bernardino"
YEAR = 2023
START_MONTH = 4
END_MONTH = 4

# Input (original data - NEVER modified)
DATA_FOLDER = Path(DATA_ROOT / "schneehoehe" / "aaadata" / "cold_season_combined")

# Output (new cleaned files only)
OUTPUT_FOLDER = Path(DATA_ROOT / "schneehoehe" / "cleaned")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

PLOTS_ROOT = OUTPUT_FOLDER / "spike_detection"
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

log_file = PLOTS_ROOT / f"spike_detection_{STATION_NAME.replace(' ', '_')}_{YEAR}_{START_MONTH:02d}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# ====================== SPIKE REMOVAL FUNCTION ======================
def remove_spikes(df: pd.DataFrame, 
                  window_hours: int = 6, 
                  mad_multiplier: float = 4.0,
                  max_rate_cm_per_hour: float = 25.0) -> pd.DataFrame:
    """
    Robust spike detection:
    - Rolling median (center)
    - Median Absolute Deviation (MAD)
    - Physical rate limit (max 25 cm/h change)
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    
    # Rolling median
    df["rolling_median"] = df["value"].rolling(
        window=window_hours*2, center=True, min_periods=3
    ).median()
    
    # MAD (robust scale)
    df["mad"] = (df["value"] - df["rolling_median"]).abs().rolling(
        window=window_hours*2, center=True, min_periods=3
    ).median()
    
    # Spike if deviation > mad_multiplier * MAD
    df["deviation"] = (df["value"] - df["rolling_median"]).abs()
    df["is_spike_mad"] = df["deviation"] > (mad_multiplier * df["mad"])
    
    # Rate of change
    df["time_diff_h"] = df["timestamp"].diff().dt.total_seconds() / 3600
    df["value_diff"] = df["value"].diff()
    df["rate_cm_h"] = df["value_diff"] / df["time_diff_h"]
    df["is_spike_rate"] = df["rate_cm_h"].abs() > max_rate_cm_per_hour
    
    # Combine both methods
    df["is_spike"] = df["is_spike_mad"] | df["is_spike_rate"]
    
    # Cleaned value: replace spikes with median
    df["value_clean"] = df["value"].where(~df["is_spike"], df["rolling_median"])
    
    # Count spikes
    n_spikes = int(df["is_spike"].sum())
    logger.info(f"Detected {n_spikes} spikes ({n_spikes/len(df)*100:.1f}% of data)")
    
    return df

# ====================== LOAD STATION ======================
def load_station(station_name: str) -> pd.DataFrame:
    logger.info(f"Loading station: {station_name}")
    parquet_files = list(DATA_FOLDER.glob("*.parquet"))
    
    for pq_file in parquet_files:
        json_file = pq_file.with_suffix(".json")
        if json_file.exists():
            with open(json_file, encoding="utf-8") as f:
                meta = json.load(f)
                if meta.get("name") == station_name:
                    logger.info(f"Found: {pq_file.name}")
                    df = pd.read_parquet(pq_file)
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                    df["station"] = meta.get("name", station_name)
                    df["hoehe"] = meta.get("hoehe")
                    return df
    
    logger.error(f"Station '{station_name}' not found in {DATA_FOLDER}")
    return pd.DataFrame()

# ====================== PLOT BEFORE/AFTER ======================
def plot_before_after(df: pd.DataFrame, station_name: str, year: int, 
                      start_month: int, end_month: int, save_path: Path):
    mask = (df["timestamp"].dt.year == year) & \
           (df["timestamp"].dt.month >= start_month) & \
           (df["timestamp"].dt.month <= end_month)
    df_plot = df[mask].copy()
    
    if df_plot.empty:
        logger.warning("No data in selected period")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    
    # Original
    ax1 = axes[0]
    ax1.plot(df_plot["timestamp"], df_plot["value"], color="#1f77b4", linewidth=1.2, label="Original")
    spikes = df_plot[df_plot["is_spike"]]
    if not spikes.empty:
        ax1.scatter(spikes["timestamp"], spikes["value"], color="red", s=35, zorder=5, label="Detected spikes")
    ax1.set_ylabel("SchneehÃ¶he [cm]", fontsize=12)
    ax1.set_title(f"Original data with detected spikes â€“ {station_name} ({year}-{start_month:02d})", 
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # Cleaned
    ax2 = axes[1]
    ax2.plot(df_plot["timestamp"], df_plot["value_clean"], color="#2ca02c", linewidth=1.5, label="Cleaned")
    ax2.set_ylabel("SchneehÃ¶he [cm]", fontsize=12)
    ax2.set_xlabel("Zeit", fontsize=12)
    ax2.set_title("After spike removal (replaced with rolling median)", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    for ax in axes:
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d.%b'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"â†’ Saved before/after plot: {save_path.name}")

# ====================== MAIN ======================
def main():
    logger.info("=== SAFE SPIKE DETECTION (NO ORIGINAL DATA MODIFIED) ===")
    
    df = load_station(STATION_NAME)
    if df.empty:
        return
    
    # Detect & remove spikes
    df_clean = remove_spikes(df, window_hours=6, mad_multiplier=4.0, max_rate_cm_per_hour=25.0)
    
    # Plot April 2023
    save_path = PLOTS_ROOT / f"spike_test_{STATION_NAME.replace(' ', '_')}_{YEAR}_{START_MONTH:02d}.png"
    plot_before_after(df_clean, STATION_NAME, YEAR, START_MONTH, END_MONTH, save_path)
    
    # Save NEW cleaned copy (never overwrite original)
    safe_name = STATION_NAME.replace(" ", "_").replace("/", "_")
    cleaned_file = OUTPUT_FOLDER / f"{safe_name}_cleaned.parquet"
    df_clean.to_parquet(cleaned_file, compression="snappy", index=False)
    logger.info(f"â†’ Saved cleaned copy: {cleaned_file.name}")
    
    # Summary
    n_spikes = int(df_clean["is_spike"].sum())
    logger.info(f"\n=== SUMMARY ===")
    logger.info(f"Spikes detected in April 2023: {n_spikes}")
    logger.info(f"Data affected: {n_spikes/len(df_clean)*100:.2f}%")
    logger.info(f"Cleaned file saved to: {OUTPUT_FOLDER}")
    
    logger.info("=== DONE (original data untouched) ===")

if __name__ == "__main__":
    main()
