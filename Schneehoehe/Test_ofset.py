#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Literature-Inspired Snow Height Cleaning Method
Combines best practices from:
- Blandini et al. (2023) Random Forest features (behavioral)
- Rolling median + MAD + rate limits (robust statistics)
- Long plateau + variation filters (operational QC)
- Conservative approach respecting intra-snowpack dynamics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging
import json
from scipy.stats import linregress

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
STATION_NAME = "Breiter Grieskogel Schneestation"
YEAR = 2023
START_MONTH = 10
END_MONTH = 10

DATA_FOLDER = Path(DATA_ROOT / "schneehoehe" / "aaadata" / "cold_season_combined")
PLOTS_ROOT = Path(DATA_ROOT / "plots" / "schneehoehe" / "hybrid_literature")
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

log_file = PLOTS_ROOT / f"hybrid_{STATION_NAME.replace(' ', '_')}_{YEAR}_{START_MONTH:02d}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# ====================== LOAD DATA ======================
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
    
    logger.error(f"Station '{station_name}' not found!")
    return pd.DataFrame()

# ====================== HYBRID LITERATURE METHOD ======================
def hybrid_literature_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hybrid method combining best practices from literature:
    - Behavioral features (variation, plateau duration, rate of change)
    - Robust statistics (rolling median + MAD)
    - Conservative: only remove clearly faulty periods
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["value_clean"] = df["value"].copy()
    
    logger.info("Step 1: Spike removal (rolling median + MAD + rate limit)...")
    
    # Rolling median + MAD
    df["rolling_median"] = df["value"].rolling(12, center=True).median()
    df["mad"] = (df["value"] - df["rolling_median"]).abs().rolling(12, center=True).median()
    df["deviation"] = (df["value"] - df["rolling_median"]).abs()
    df["is_spike_mad"] = df["deviation"] > (4.0 * df["mad"])
    
    # Rate of change
    df["time_diff_h"] = df["timestamp"].diff().dt.total_seconds() / 3600
    df["value_diff"] = df["value"].diff()
    df["rate"] = df["value_diff"] / df["time_diff_h"]
    df["is_spike_rate"] = df["rate"].abs() > 25.0
    
    df["is_spike"] = df["is_spike_mad"] | df["is_spike_rate"]
    df.loc[df["is_spike"], "value_clean"] = df.loc[df["is_spike"], "rolling_median"]
    
    n_spikes = df["is_spike"].sum()
    logger.info(f"   Removed {n_spikes} spikes")
    
    logger.info("Step 2: Long plateau + high variation noise removal (behavioral)...")
    
    # Long plateaus (18+ hours with very low variation)
    df["short_var"] = df["value_clean"].rolling(36).std()
    df["is_long_plateau"] = (df["short_var"] < 1.2) & (df["value_clean"] > 2) & (df["value_clean"] < 25)
    
    df["plateau_group"] = (df["is_long_plateau"] != df["is_long_plateau"].shift()).cumsum()
    plateau_durations = df.groupby("plateau_group")["timestamp"].agg(
        duration_hours=lambda x: (x.max() - x.min()).total_seconds() / 3600
    )
    long_plateaus = plateau_durations[plateau_durations["duration_hours"] >= 18].index
    df.loc[df["plateau_group"].isin(long_plateaus) & df["is_long_plateau"], "value_clean"] = 0.0
    
    # High short-term variation (noisy periods > 8 hours)
    df["rolling_var"] = df["value_clean"].rolling(24).std()
    df["is_noisy"] = (df["rolling_var"] > 4.0) & (df["value_clean"] < 15)
    
    df["noisy_group"] = (df["is_noisy"] != df["is_noisy"].shift()).cumsum()
    noisy_durations = df.groupby("noisy_group")["timestamp"].agg(
        duration_hours=lambda x: (x.max() - x.min()).total_seconds() / 3600
    )
    long_noisy = noisy_durations[noisy_durations["duration_hours"] >= 8].index
    df.loc[df["noisy_group"].isin(long_noisy) & df["is_noisy"], "value_clean"] = 0.0
    
    n_behavioral = (df["value"] != df["value_clean"]).sum() - n_spikes
    logger.info(f"   Removed {n_behavioral} behavioral anomalies (plateaus + noise)")
    
    logger.info("Step 3: Final conservative cleanup...")
    
    # Remove very short isolated spikes that survived previous steps
    df["change"] = df["value_clean"].diff().abs()
    df["is_isolated_spike"] = (df["change"] > 8) & (df["value_clean"] < 15)
    df.loc[df["is_isolated_spike"], "value_clean"] = 0.0
    
    # Clip negative values
    df["value_clean"] = df["value_clean"].clip(lower=0)
    
    total_removed = (df["value"] != df["value_clean"]).sum()
    logger.info(f"Total values corrected: {total_removed} ({total_removed/len(df)*100:.1f}%)")
    
    return df

# ====================== PLOT ======================
def plot_hybrid(df: pd.DataFrame, station_name: str, year: int, 
                start_month: int, end_month: int, save_path: Path):
    mask = (df["timestamp"].dt.year == year) & \
           (df["timestamp"].dt.month >= start_month) & \
           (df["timestamp"].dt.month <= end_month)
    df_plot = df[mask].copy()
    
    if df_plot.empty:
        logger.warning("No data in period")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    
    # Original
    ax1 = axes[0]
    ax1.plot(df_plot["timestamp"], df_plot["value"], color="#1f77b4", linewidth=1.3, label="Original")
    ax1.set_ylabel("SchneehÃ¶he [cm]", fontsize=12)
    ax1.set_title(f"Original â€“ {station_name} ({year}-{start_month:02d})", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # Cleaned
    ax2 = axes[1]
    ax2.plot(df_plot["timestamp"], df_plot["value_clean"], color="#2ca02c", linewidth=1.5, label="Cleaned (Hybrid Literature Method)")
    ax2.set_ylabel("SchneehÃ¶he [cm]", fontsize=12)
    ax2.set_xlabel("Zeit", fontsize=12)
    ax2.set_title("After Hybrid Literature Method (Behavioral + Robust Statistics)", 
                  fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    for ax in axes:
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d.%b'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"â†’ Saved: {save_path.name}")

# ====================== MAIN ======================
def main():
    logger.info("=== HYBRID LITERATURE-INSPIRED METHOD TEST ===")
    
    df = load_station(STATION_NAME)
    if df.empty:
        return
    
    # Apply hybrid cleaning
    df_clean = hybrid_literature_cleaning(df)
    
    # Plot October 2023
    save_path = PLOTS_ROOT / f"hybrid_{STATION_NAME.replace(' ', '_')}_{YEAR}_{START_MONTH:02d}.png"
    plot_hybrid(df_clean, STATION_NAME, YEAR, START_MONTH, END_MONTH, save_path)
    
    logger.info("=== DONE ===")

if __name__ == "__main__":
    main()
