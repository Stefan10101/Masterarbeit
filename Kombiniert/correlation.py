#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Aggregation + Correlation Analysis v1.0
=============================================
Aggregates 30-min master data to daily resolution:
- Temperature: mean
- Precipitation: sum (daily total)
- All other variables: mean

Then computes global correlation across all stations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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


MASTER_DIR = Path(DATA_ROOT / "kombiniert")
PLOT_DIR   = Path(DATA_ROOT / "plots" / "kombiniert")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

VARIABLES = ["temperature", "relative_humidity", "precipitation", "snow_height", "wind_speed"]

LABELS = {
    "temperature": "Temperature (Â°C)",
    "relative_humidity": "Relative Humidity (%)",
    "precipitation": "Precipitation (mm/day)",
    "snow_height": "Snow Height (cm)",
    "wind_speed": "Wind Speed (m/s)"
}

LOG_FILE = PLOT_DIR / f"daily_correlation_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


def aggregate_to_daily(df):
    """Resample one station to daily resolution."""
    df = df.set_index("timestamp")
    
    daily = pd.DataFrame({
        "temperature":       df["temperature"].resample("D").mean(),
        "relative_humidity": df["relative_humidity"].resample("D").mean(),
        "precipitation":     df["precipitation"].resample("D").sum(),
        "snow_height":       df["snow_height"].resample("D").mean(),
        "wind_speed":        df["wind_speed"].resample("D").mean()
    })
    return daily.dropna(how="all")   # remove completely empty days


def load_and_aggregate_all():
    files = list(MASTER_DIR.glob("*_full_2020_2025_master.parquet"))
    logger.info(f"Found {len(files)} master files")

    all_daily = []
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["timestamp"] + VARIABLES)
            daily = aggregate_to_daily(df)
            if not daily.empty:
                all_daily.append(daily)
        except Exception as e:
            logger.warning(f"Skipped {f.name}: {e}")

    combined = pd.concat(all_daily, ignore_index=True)
    logger.info(f"Total daily observations after aggregation: {len(combined):,}")
    return combined


def compute_correlation(df):
    corr = df.corr(method="pearson")
    corr.index = [LABELS.get(x, x) for x in corr.index]
    corr.columns = [LABELS.get(x, x) for x in corr.columns]

    csv_path = PLOT_DIR / "daily_correlation_matrix.csv"
    corr.to_csv(csv_path)
    logger.info(f"Daily correlation matrix saved: {csv_path}")
    return corr


def create_pairs_plot(df):
    plt.figure(figsize=(14, 14))
    sns.set_theme(style="whitegrid", font_scale=1.15)

    plot_df = df.rename(columns=LABELS)

    g = sns.pairplot(
        plot_df,
        diag_kind="hist",
        plot_kws={"alpha": 0.3, "s": 12, "edgecolor": "none"},
        diag_kws={"bins": 50, "color": "#1f77b4"},
        corner=True
    )

    g.fig.suptitle(
        "Daily Aggregated Data â€“ Correlation Pairs Plot\n"
        f"Total daily observations: {len(df):,}",
        y=1.02, fontsize=14
    )
    plt.tight_layout()

    out_path = PLOT_DIR / "daily_pairs_plot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Daily pairs plot saved: {out_path}")


def main():
    logger.info("=== DAILY AGGREGATION + CORRELATION v1.0 START ===")

    daily_df = load_and_aggregate_all()
    compute_correlation(daily_df)
    create_pairs_plot(daily_df)

    logger.info("=== FINISHED ===")
    logger.info(f"Results saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()
