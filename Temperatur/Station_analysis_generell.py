#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Thesis - Temperature Data Analysis Plots (v2.0 - adapted to pipeline v11)
Straightforward, efficient, sound logic.
Uses new full_data_cold_seasons + 30-min coverage logic.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import random
import logging
import ast
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
ROOT = Path(DATA_ROOT / "temperatur")
AAAData = ROOT / "AAAData"
PLOT_DIR = ROOT / "AAPlots"
PLOT_DIR.mkdir(exist_ok=True)

FULL_COLD_DIR = AAAData / "full_data_cold_seasons"   # â† new v11 folder
STATS_FILE = AAAData / "station_statistics.csv"

FULL_PERIOD_START = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
FULL_PERIOD_END   = pd.Timestamp("2025-12-31 23:50:00", tz="UTC")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ==================== HELPERS ====================
def load_accepted_stations():
    df = pd.read_csv(STATS_FILE)
    accepted = df[df["accepted_cold_combined"] == True]["station"].tolist()
    logger.info(f"Loaded {len(accepted)} accepted stations")
    return accepted

def get_json_path(station: str) -> Path:
    return AAAData / "cold_season_combined" / f"{station}_cold_season_combined.json"

def get_full_cold_parquet(station: str) -> Path:
    """New v11 unfiltered cold-season data"""
    return FULL_COLD_DIR / f"{station}_full_cold_seasons.parquet"

def get_station_metadata(station: str) -> dict | None:
    """Correct lat/lon from v11 pipeline"""
    jpath = get_json_path(station)
    if not jpath.exists():
        return None
    try:
        with open(jpath, encoding="utf-8") as f:
            data = json.load(f)
        return {
            "hoehe": float(data.get("hoehe")) if data.get("hoehe") is not None else None,
            "lat": float(data.get("lat")) if data.get("lat") is not None else None,
            "lon": float(data.get("lon")) if data.get("lon") is not None else None,
        }
    except Exception:
        return None

def compute_true_cold_coverage(station: str) -> float:
    """Coverage over 2020-2025 using new full_cold_seasons parquet + 30-min logic"""
    p = get_full_cold_parquet(station)
    if not p.exists():
        return 0.0
    try:
        df = pd.read_parquet(p, columns=["timestamp", "value"])
        # Regrid to 30-min as in v11
        target_idx = pd.date_range(FULL_PERIOD_START, FULL_PERIOD_END, freq="30min", tz="UTC")
        regridded = pd.DataFrame(index=target_idx, columns=["value"], dtype=float)
        df = df.set_index("timestamp").sort_index()
        common = df.index.intersection(target_idx)
        regridded.loc[common, "value"] = df.loc[common, "value"].values
        return regridded["value"].notna().mean() * 100
    except Exception as e:
        logger.warning(f"Coverage calc failed for {station}: {e}")
        return 0.0

# ==================== PLOT 1: True cold-season coverage (2020-2025) ====================
def plot_coverage():
    df = pd.read_csv(STATS_FILE)
    accepted_df = df[df["accepted_cold_combined"] == True].copy()
    covs = [compute_true_cold_coverage(row["station"]) for _, row in accepted_df.iterrows()]
    accepted_df["true_cold_coverage"] = covs
    accepted_df = accepted_df.sort_values("true_cold_coverage", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots()
    ax.bar(range(len(accepted_df)), accepted_df["true_cold_coverage"], color="steelblue", width=1)
    ax.set_xlabel("Stations (sorted by coverage descending)")
    ax.set_ylabel("Coverage [%] (2020-2025, 30-min grid)")
    ax.set_title("True Coverage â€“ Full Cold-Season Data 2020-2025")
    ax.set_ylim(0, 105)
    ax.set_xticks([])
    plt.savefig(PLOT_DIR / "01_coverage_sorted.png")
    plt.close()
    logger.info("Saved 01_coverage_sorted.png")

# ==================== PLOT 2: Height distribution (200 m) ====================
def plot_height_distribution():
    accepted = load_accepted_stations()
    heights = [meta["hoehe"] for st in accepted if (meta := get_station_metadata(st)) and meta["hoehe"] is not None]
    if not heights:
        logger.warning("No height data")
        return
    heights = np.array(heights)
    max_h = heights.max()
    bins = np.arange(0, max_h + 250, 200)
    hist, bin_edges = np.histogram(heights, bins=bins)
    percentages = hist / len(heights) * 100

    fig, ax = plt.subplots()
    ax.bar(bin_edges[:-1] + 100, hist, width=180, color="seagreen", edgecolor="black", alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(bin_edges[:-1] + 100, percentages, color="darkred", marker="o", linewidth=2.5)
    ax.set_xlabel("Height [m] (200 m bins)")
    ax.set_ylabel("Number of stations")
    ax2.set_ylabel("Percentage [%]")
    ax.set_title("Station Distribution by Altitude")
    plt.savefig(PLOT_DIR / "02_height_distribution.png")
    plt.close()
    logger.info("Saved 02_height_distribution.png")

# ==================== PLOT 3: Pairwise distance distribution ====================
def plot_spatial_distances():
    accepted = load_accepted_stations()
    coords = []
    for st in accepted:
        meta = get_station_metadata(st)
        if meta and meta["lat"] is not None and meta["lon"] is not None:
            coords.append((meta["lat"], meta["lon"]))
    coords = np.array(coords)
    if len(coords) < 2:
        logger.warning("Not enough stations for distance calculation")
        return

    lat_km = coords[:, 0] * 111.0
    lon_km = coords[:, 1] * (111.0 * np.cos(np.deg2rad(47.0)))
    xy = np.column_stack((lat_km, lon_km))
    distances_km = pdist(xy)

    mean_dist = distances_km.mean()
    median_dist = np.median(distances_km)

    fig, ax = plt.subplots()
    ax.hist(distances_km, bins=50, color="steelblue", edgecolor="black", alpha=0.85)
    ax.axvline(mean_dist, color="red", linestyle="--", label=f"Mean: {mean_dist:.1f} km")
    ax.axvline(median_dist, color="orange", linestyle="--", label=f"Median: {median_dist:.1f} km")
    ax.set_xlabel("Distance between stations [km] (Euclidean)")
    ax.set_ylabel("Number of station pairs")
    ax.set_title("Distribution of Pairwise Distances Between Stations")
    ax.legend()
    plt.savefig(PLOT_DIR / "03_spatial_distances.png")
    plt.close()
    logger.info(f"Saved 03_spatial_distances.png (mean â‰ˆ {mean_dist:.1f} km)")

# ==================== PLOT 4: Coverage vs height ====================
def plot_coverage_vs_height():
    df = pd.read_csv(STATS_FILE)
    accepted_df = df[df["accepted_cold_combined"] == True].copy()
    covs = [compute_true_cold_coverage(row["station"]) for _, row in accepted_df.iterrows()]
    accepted_df["true_cold_coverage"] = covs

    data = []
    for _, row in accepted_df.iterrows():
        meta = get_station_metadata(row["station"])
        if meta and meta["hoehe"] is not None:
            data.append({"height": meta["hoehe"], "coverage": row["true_cold_coverage"]})

    df_plot = pd.DataFrame(data)
    fig, ax = plt.subplots()
    sns.regplot(data=df_plot, x="height", y="coverage",
                scatter_kws={"alpha": 0.65, "s": 20},
                line_kws={"color": "red"}, ax=ax)
    ax.set_xlabel("Height [m]")
    ax.set_ylabel("Coverage [%] (2020-2025)")
    ax.set_title("Coverage vs Station Height (2020-2025)")
    plt.savefig(PLOT_DIR / "04_coverage_vs_height.png")
    plt.close()
    logger.info("Saved 04_coverage_vs_height.png")

# ==================== PLOT 5: Mean temperature vs height ====================
def plot_mean_temp_vs_height():
    accepted = load_accepted_stations()
    data = []
    for st in accepted:
        p = get_full_cold_parquet(st)
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["value"])
            meta = get_station_metadata(st)
            if not df["value"].dropna().empty and meta and meta["hoehe"] is not None:
                mean_t = df["value"].mean()
                data.append({"height": meta["hoehe"], "mean_temp": mean_t})
        except Exception:
            continue

    df_plot = pd.DataFrame(data)
    fig, ax = plt.subplots()
    sns.regplot(data=df_plot, x="height", y="mean_temp",
                scatter_kws={"alpha": 0.7}, line_kws={"color": "red"}, ax=ax)
    ax.set_xlabel("Height [m]")
    ax.set_ylabel("Mean temperature cold seasons [Â°C]")
    ax.set_title("Mean Cold-Season Temperature vs Height")
    plt.savefig(PLOT_DIR / "05_mean_temp_vs_height.png")
    plt.close()
    logger.info("Saved 05_mean_temp_vs_height.png")

# ==================== PLOT 6: Diurnal cycle (50-day average) ====================
def plot_diurnal_cycles():
    accepted = load_accepted_stations()
    bins = [(0, 1000), (1001, 2000), (2001, 4000)]
    bin_data = {b: [] for b in bins}

    for st in accepted:
        p = get_full_cold_parquet(st)
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["timestamp", "value"])
        meta = get_station_metadata(st)
        if df.empty or meta is None or meta["hoehe"] is None:
            continue
        h = meta["hoehe"]
        for (low, high) in bins:
            if low <= h <= high:
                bin_data[(low, high)].append((st, df))
                break

    fig, axes = plt.subplots(3, 1, figsize=(13, 16), sharex=True)
    fig.suptitle("Average Diurnal Cycle (up to 50 days per station, 07 UTC start)", fontsize=16)

    for i, (low, high) in enumerate(bins):
        ax = axes[i]
        ax.set_title(f"{low}â€“{high} m   (n = {len(bin_data[(low,high)])} stations)")
        all_station_means = []
        for _, df_full in bin_data[(low, high)]:
            df = df_full.set_index("timestamp").sort_index()
            starts = df.index[df.index.hour == 7]
            if len(starts) == 0:
                continue
            random_starts = random.sample(list(starts), min(50, len(starts)))
            station_curves = []
            for start_ts in random_starts:
                end_ts = start_ts + pd.Timedelta(hours=24) - pd.Timedelta(minutes=10)
                window = df.loc[start_ts:end_ts]
                if len(window) != 144:
                    continue
                base = window["value"].iloc[0]
                rel = window["value"] - base
                station_curves.append(rel.values)
            if station_curves:
                station_mean = np.mean(station_curves, axis=0)
                all_station_means.append(station_mean)
                hours = np.arange(0, 24, 10/60)
                ax.plot(hours, station_mean, color="gray", alpha=0.25, linewidth=0.9)

        if all_station_means:
            grand_mean = np.mean(all_station_means, axis=0)
            ax.plot(hours, grand_mean, color="red", linewidth=3.5, label="Overall mean")
            ax.legend(loc="upper right")
        ax.set_ylabel("Î”T [Â°C]")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))

    axes[-1].set_xlabel("Hours since 07:00 UTC")
    plt.savefig(PLOT_DIR / "06_diurnal_cycles.png")
    plt.close()
    logger.info("Saved 06_diurnal_cycles.png")

# ==================== MAIN ====================
def main():
    logger.info("=== STARTING ANALYSIS PLOTS (v2.0 - v11 pipeline) ===")
    plot_coverage()
    plot_height_distribution()
    plot_spatial_distances()
    plot_coverage_vs_height()
    plot_mean_temp_vs_height()
    plot_diurnal_cycles()
    logger.info(f"=== ALL PLOTS SAVED TO {PLOT_DIR} ===")

if __name__ == "__main__":
    main()
