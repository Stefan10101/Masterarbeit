#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RH QC Diagnostic Plotter - v3 (final adapted version)
=====================================================
All fixes from temperature version applied.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import ConciseDateFormatter, YearLocator, MonthLocator, DayLocator
from datetime import datetime, timedelta
import logging
import random
from collections import defaultdict

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

ORIGINAL_ROOT = Path(DATA_ROOT / "QC" / "Schneehoehe" / "full_2020_2025")
FINAL_ROOT    = Path(DATA_ROOT / "QC" / "Schneehoehe" / "full_2020_2025_qc")
PLOT_DIR      = Path(DATA_ROOT / "Plots" / "Schneehoehe")

PLOT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "full"
MAX_STATIONS_PER_NETWORK = 5
MIN_FLAG_PCT = 1.0
ZOOM_DAYS = 30
MAX_PLOTS_PER_STATION = 2

LOG_FILE = PLOT_DIR / f"rh_qc_diagnostic_v3_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


def find_flagged_periods(original_df: pd.DataFrame, final_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        original_df[["timestamp", "value"]].rename(columns={"value": "original"}),
        final_df[["timestamp", "value", "is_missing"]].rename(columns={"value": "final"}),
        on="timestamp", how="inner"
    )
    flagged = merged[merged["original"].notna() & (merged["final"].isna() | merged["is_missing"])].copy()
    if flagged.empty:
        return pd.DataFrame()
    flagged = flagged.sort_values("timestamp")
    flagged["gap"] = flagged["timestamp"].diff() > pd.Timedelta(hours=1)
    flagged["period_id"] = flagged["gap"].cumsum()
    periods = flagged.groupby("period_id").agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        duration_hours=("timestamp", lambda x: (x.max() - x.min()).total_seconds() / 3600),
        n_points=("timestamp", "count")
    ).reset_index(drop=True)
    return periods


def plot_diagnostic(original_df, final_df, station, periods, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]})
    ax1 = axes[0]
    ax1.plot(original_df["timestamp"], original_df["value"], color="#1f77b4", linewidth=0.8, label="Original (pipeline)", alpha=0.85)
    ax1.plot(final_df["timestamp"], final_df["value"], color="#d62728", linewidth=0.9, label="Final (after QC)", alpha=0.9)
    for _, row in periods.iterrows():
        ax1.axvspan(row["start"], row["end"], alpha=0.25, color="red", label="Flagged by QC" if _ == 0 else "")

    removed_points = periods['n_points'].sum() if not periods.empty and 'n_points' in periods.columns else 0
    ax1.set_title(f"QC Diagnostic â€“ {station}\n"
                  f"Original valid: {original_df['value'].notna().sum():,} | "
                  f"Final valid: {final_df['value'].notna().sum():,} | "
                  f"Removed: {len(periods)} periods ({removed_points} points)",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    ax1.xaxis.set_major_locator(YearLocator(1))
    ax1.xaxis.set_minor_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
    ax1.xaxis.set_major_formatter(ConciseDateFormatter(ax1.xaxis.get_major_locator()))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha="center")

    ax2 = axes[1]
    if not periods.empty:
        ax2.hist(periods["duration_hours"], bins=40, color="#d62728", alpha=0.75, edgecolor="white", linewidth=0.5)
        ax2.set_xlabel("Duration of flagged period (hours)")
        ax2.set_ylabel("Number of flagged periods")
        ax2.set_title("Distribution of flagged period lengths (stuck / saturation / spike)", fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_xlim(left=0)
    else:
        ax2.text(0.5, 0.5, "No flagged periods", ha="center", va="center", transform=ax2.transAxes, fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved diagnostic plot: {out_path.name}")


def plot_zoom(original_df, final_df, station, event_start, event_end, out_path):
    start = event_start - timedelta(days=ZOOM_DAYS // 2)
    end = event_end + timedelta(days=ZOOM_DAYS // 2)
    orig_zoom = original_df[(original_df["timestamp"] >= start) & (original_df["timestamp"] <= end)]
    final_zoom = final_df[(final_df["timestamp"] >= start) & (final_df["timestamp"] <= end)]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(orig_zoom["timestamp"], orig_zoom["value"], color="#1f77b4", linewidth=1.2, label="Original", marker=".", markersize=2, alpha=0.8)
    ax.plot(final_zoom["timestamp"], final_zoom["value"], color="#d62728", linewidth=1.5, label="After QC", marker=".", markersize=2.5, alpha=0.9)
    ax.axvspan(event_start, event_end, alpha=0.3, color="red", label="Removed by QC")
    ax.set_ylabel("Relative Humidity (%)")
    ax.set_title(f"Zoom: {station} â€“ Flagged period {event_start.strftime('%Y-%m-%d %H:%M')} to {event_end.strftime('%Y-%m-%d %H:%M')}\n"
                 f"(Window: Â±{ZOOM_DAYS//2} days | Duration: {(event_end - event_start).total_seconds()/3600:.1f} hours)", fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_locator(DayLocator(interval=5))
    ax.xaxis.set_minor_locator(DayLocator(interval=1))
    ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved zoom plot: {out_path.name}")


def main():
    logger.info("=== RH QC DIAGNOSTIC v3 START ===")
    logger.info(f"Original data: {ORIGINAL_ROOT}")
    logger.info(f"Final data:    {FINAL_ROOT}")

    if DATASET == "full":
        orig_dir = ORIGINAL_ROOT
        final_dir = FINAL_ROOT
    else:
        logger.error("Only 'full' mode supported.")
        return

    if not orig_dir.exists() or not final_dir.exists():
        logger.error(f"One of the folders does not exist!")
        logger.error(f"  Original expected: {orig_dir}")
        logger.error(f"  Final expected:    {final_dir}")
        logger.error("Please check the paths at the top of the script.")
        return

    # Original is nested
    orig_files = {}
    for f in orig_dir.rglob("*_rh.parquet"):
        station = f.parent.name
        orig_files[station] = f

    # QC output is flat
    final_files = {}
    for f in final_dir.glob("*_final_qc.parquet"):
        station = f.stem.replace("_final_qc", "")
        final_files[station] = f

    common_stations = sorted(set(orig_files.keys()) & set(final_files.keys()))
    logger.info(f"Found {len(common_stations)} stations with both original and final data")

    # Group by network and randomly select 5 per network
    station_network = []
    for station in common_stations:
        try:
            df = pd.read_parquet(final_files[station])
            src = str(df["source_file"].iloc[0]) if "source_file" in df.columns else ""
            network = "Unknown"
            src_lower = src.lower()
            if "meteosuisse" in src_lower or "meteosuisse" in station.lower():
                network = "Meteosuisse"
            elif "dwd" in src_lower:
                network = "DWD"
            elif "hydrot" in src_lower or "hydrovo" in src_lower:
                network = "Hydro"
            elif "suedtirol" in src_lower:
                network = "Suedtirol"
            elif "zamg" in src_lower:
                network = "ZAMG"
            elif "lwd" in src_lower:
                network = "LWD"
            station_network.append((station, network))
        except:
            station_network.append((station, "Unknown"))

    network_groups = defaultdict(list)
    for station, net in station_network:
        network_groups[net].append(station)

    selected_stations = []
    for net, stations in sorted(network_groups.items()):
        random.shuffle(stations)
        selected = stations[:MAX_STATIONS_PER_NETWORK]
        selected_stations.extend(selected)
        logger.info(f"Network '{net}': selected {len(selected)} stations")

    logger.info(f"Total stations selected: {len(selected_stations)}")

    processed = 0
    plotted = 0

    for station in selected_stations:
        try:
            orig_path = orig_files[station]
            final_path = final_files[station]

            orig_df = pd.read_parquet(orig_path)
            final_df = pd.read_parquet(final_path)

            orig_valid = orig_df["value"].notna().sum()
            final_valid = final_df["value"].notna().sum()
            removed_pct = (orig_valid - final_valid) / orig_valid * 100 if orig_valid > 0 else 0

            if removed_pct < MIN_FLAG_PCT:
                processed += 1
                continue

            logger.info(f"Processing {station} | removed {removed_pct:.1f}%")

            periods = find_flagged_periods(orig_df, final_df)

            plot_name = f"{station}_QC_diagnostic.png"
            plot_diagnostic(orig_df, final_df, station, periods, PLOT_DIR / plot_name)

            if not periods.empty:
                top_periods = periods.nlargest(min(MAX_PLOTS_PER_STATION, len(periods)), "duration_hours")
                for i, (_, row) in enumerate(top_periods.iterrows()):
                    zoom_name = f"{station}_zoom_{i+1}_{row['start'].strftime('%Y%m%d')}.png"
                    plot_zoom(orig_df, final_df, station, row["start"], row["end"], PLOT_DIR / zoom_name)

            plotted += 1
            processed += 1

        except Exception as e:
            logger.error(f"Failed on {station}: {e}")

    logger.info(f"=== FINISHED v3 ===")
    logger.info(f"Stations processed: {processed}")
    logger.info(f"Plots generated: {plotted}")
    logger.info(f"All plots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()

