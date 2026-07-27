#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined Time Series â€“ 10 Complete Stations v2.0
================================================
Selects stations that have all 5 variables and plots one full month
as multi-panel time series.

Adapted for new master dataset structure (output directly in Kombiniert/).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import logging
from datetime import datetime

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT


# ==================== CONFIG (UPDATED) ====================
MASTER_DIR = Path(DATA_ROOT / "kombiniert")
PLOT_DIR   = Path(DATA_ROOT / "plots" / "kombiniert")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

VARIABLES = ["temperature", "relative_humidity", "precipitation", "snow_height", "wind_speed"]

MONTH_TO_PLOT = "2022-11"          # â† Change this to any month you want (YYYY-MM)
N_STATIONS = 10
RANDOM_SEED = 11

LOG_FILE = PLOT_DIR / f"timeseries_10stations_v2_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


def find_complete_stations():
    """Find all stations that have all 5 variables."""
    complete = []
    for f in MASTER_DIR.glob("*_full_2020_2025_master.parquet"):
        try:
            df = pd.read_parquet(f, columns=["timestamp"] + VARIABLES)
            if all(v in df.columns for v in VARIABLES):
                complete.append(f)
        except:
            pass
    logger.info(f"Found {len(complete)} stations with all 5 variables")
    return complete


def plot_station_timeseries(pdf, file_path, month):
    """Create a 5-panel plot for one station and one month."""
    df = pd.read_parquet(file_path)
    df = df.set_index("timestamp")
    
    # Filter to the chosen month
    try:
        month_df = df.loc[month]
    except KeyError:
        logger.warning(f"No data in {month} for {file_path.name}")
        return

    if month_df.empty:
        logger.warning(f"No data in {month} for {file_path.name}")
        return

    station_name = file_path.stem.replace("_full_2020_2025_master", "")

    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{station_name} â€“ {month}", fontsize=14, fontweight="bold", y=0.98)

    # Temperature
    axes[0].plot(month_df.index, month_df["temperature"], color="#d62728", linewidth=0.8)
    axes[0].set_ylabel("Temperature (Â°C)")
    axes[0].grid(True, alpha=0.3)

    # Relative Humidity
    axes[1].plot(month_df.index, month_df["relative_humidity"], color="#1f77b4", linewidth=0.8)
    axes[1].set_ylabel("Relative Humidity (%)")
    axes[1].grid(True, alpha=0.3)

    # Precipitation (30-min accumulation)
    axes[2].bar(month_df.index, month_df["precipitation"], width=0.02, color="#2ca02c", alpha=0.7)
    axes[2].set_ylabel("Precipitation (mm/30min)")
    axes[2].grid(True, alpha=0.3)

    # Snow Height
    axes[3].plot(month_df.index, month_df["snow_height"], color="#9467bd", linewidth=0.8)
    axes[3].set_ylabel("Snow Height (cm)")
    axes[3].grid(True, alpha=0.3)

    # Wind Speed
    axes[4].plot(month_df.index, month_df["wind_speed"], color="#ff7f0e", linewidth=0.8)
    axes[4].set_ylabel("Wind Speed (m/s)")
    axes[4].grid(True, alpha=0.3)

    # Format x-axis
    axes[4].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d"))
    axes[4].set_xlabel("Day of month")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()
    logger.info(f"Plotted: {station_name}")


def main():
    logger.info("=== COMBINED TIME SERIES v2.0 START ===")
    logger.info(f"Month to plot: {MONTH_TO_PLOT}")

    complete_stations = find_complete_stations()
    if len(complete_stations) < N_STATIONS:
        logger.warning(f"Only {len(complete_stations)} complete stations found. Using all of them.")
        selected = complete_stations
    else:
        random.seed(RANDOM_SEED)
        selected = random.sample(complete_stations, N_STATIONS)

    logger.info(f"Selected {len(selected)} stations for plotting")

    output_pdf = PLOT_DIR / f"timeseries_10stations_{MONTH_TO_PLOT}.pdf"
    with PdfPages(output_pdf) as pdf:
        for f in selected:
            plot_station_timeseries(pdf, f, MONTH_TO_PLOT)

    logger.info(f"PDF saved: {output_pdf}")
    logger.info("=== FINISHED v2.0 ===")


if __name__ == "__main__":
    main()

