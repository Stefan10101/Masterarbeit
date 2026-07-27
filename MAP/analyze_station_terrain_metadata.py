#!/usr/bin/env python3
"""
analyze_station_terrain_metadata.py
Master's thesis utility â€” quick, clean analysis of the augmented station metadata.

Reads stations_metadata_with_terrain.csv and produces:
- Distribution plots (hist + KDE) for: elev_dem, slope, aspect, profile_curvature, plan_curvature, elev_diff
- Special handling for aspect (polar histogram + flat stations note)
- One spatial map coloured by number of measured variables per station
- Small summary statistics printed + saved

All plots saved as high-resolution PNG in plots/ subfolder (thesis-ready).

Minimal deps:
    pip install pandas geopandas matplotlib seaborn

Run after you have the _with_terrain.csv from the previous script.
"""

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

try:
    import contextily as ctx

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT

    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

# =============================================================================
# CONFIG â€” change only the CSV path if needed
# =============================================================================
CSV_PATH = Path(DATA_ROOT / "map_station" / "stations_metadata_with_terrain.csv")
PLOTS_DIR = Path(DATA_ROOT / "plots" / "map")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

# =============================================================================
# LOAD & PREPARE
# =============================================================================
def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Number of variables measured at each station
    df["num_variables"] = df["variables"].str.count(",") + 1

    # Clean aspect: keep -1 as "Flat", others 0-360
    df["aspect_clean"] = df["aspect"].where(df["aspect"] >= 0, np.nan)

    # Flag big elevation differences (possible station vs DEM mismatch)
    df["large_elev_diff"] = df["elev_diff"].abs() > 100

    print(f"Loaded {len(df)} stations")
    print(f"Variable count distribution:\n{df['num_variables'].value_counts().sort_index()}")
    print(f"Stations with |elev_diff| > 100 m: {df['large_elev_diff'].sum()}")

    return df


# =============================================================================
# DISTRIBUTION PLOTS
# =============================================================================
def plot_distributions(df: pd.DataFrame):
    # 1. Elevation (DEM)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["elev_dem"], bins=60, kde=True, color="#2E86AB", ax=ax)
    ax.set_xlabel("Elevation from DEM (m)")
    ax.set_title("Distribution of station elevations (DEM 30 m)")
    plt.savefig(PLOTS_DIR / "01_elev_dem_distribution.png")
    plt.close()

    # 2. Slope
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["slope"], bins=50, kde=True, color="#A23B72", ax=ax)
    ax.set_xlabel("Slope (degrees)")
    ax.set_title("Distribution of station slopes")
    plt.savefig(PLOTS_DIR / "02_slope_distribution.png")
    plt.close()

    # 3. Aspect â€” polar histogram (nice for directional data)
    aspect_data = df["aspect_clean"].dropna()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    # 16 bins = 22.5Â° each
    counts, bins = np.histogram(aspect_data, bins=16, range=(0, 360))
    # Convert to radians for polar
    theta = np.deg2rad(bins[:-1])
    width = np.deg2rad(22.5)
    bars = ax.bar(theta, counts, width=width, bottom=0.0, color="#F18F01", alpha=0.85, edgecolor="white")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise
    ax.set_title("Aspect distribution (0Â° = North, clockwise)\n(flat stations excluded)", pad=20)
    plt.savefig(PLOTS_DIR / "03_aspect_polar_distribution.png")
    plt.close()

    # Also a simple note about flat stations
    n_flat = (df["aspect"] == -1).sum()
    print(f"Flat stations (aspect = -1): {n_flat} ({n_flat/len(df)*100:.1f}%)")

    # 4. Profile curvature
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["profile_curvature"], bins=60, kde=True, color="#C73E1D", ax=ax)
    ax.set_xlabel("Profile curvature (m^-1)")
    ax.set_title("Distribution of profile curvature")
    plt.savefig(PLOTS_DIR / "04_profile_curvature_distribution.png")
    plt.close()

    # 5. Plan curvature
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["plan_curvature"], bins=60, kde=True, color="#3B1F2B", ax=ax)
    ax.set_xlabel("Plan curvature (m^-1)")
    ax.set_title("Distribution of plan curvature")
    plt.savefig(PLOTS_DIR / "05_plan_curvature_distribution.png")
    plt.close()

    # 6. Elevation difference (QC)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["elev_diff"], bins=60, kde=True, color="#6B4226", ax=ax)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero difference")
    ax.set_xlabel("elev_dem â€“ hoehe (m)")
    ax.set_title("Elevation difference: DEM vs station metadata (QC)")
    ax.legend()
    plt.savefig(PLOTS_DIR / "06_elev_diff_distribution.png")
    plt.close()

    print("Distribution plots saved to", PLOTS_DIR)


# =============================================================================
# SPATIAL MAP â€” colour by number of variables
# =============================================================================
def plot_spatial_map(df: pd.DataFrame):
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:32632")   # projected for nicer map

    fig, ax = plt.subplots(figsize=(12, 9))

    # Discrete colour map for num_variables (usually 2â€“6)
    n_vars = sorted(df["num_variables"].unique())
    cmap = plt.colormaps.get_cmap("viridis")
    colors = [cmap(i / max(len(n_vars)-1, 1)) for i in range(len(n_vars))]
    cat_cmap = ListedColormap(colors)

    scatter = gdf.plot(
        column="num_variables",
        cmap=cat_cmap,
        markersize=18,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.3,
        legend=True,
        ax=ax,
        categorical=True
    )

    # Custom legend with counts
    handles = []
    for i, nv in enumerate(n_vars):
        count = (df["num_variables"] == nv).sum()
        handles.append(
            mpatches.Patch(color=colors[i], label=f"{nv} variables (n={count})")
        )
    ax.legend(handles=handles, title="Variables measured", loc="lower left", fontsize=9)

    # Add background for orientation (highly recommended)
    if HAS_CONTEXTILY:
        try:
            ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain, crs=gdf.crs.to_string(), alpha=0.55)
        except Exception:
            pass  # fallback if tile download fails
    else:
        # Fallback: country borders only (no internet needed)
        try:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            world = world.to_crs(gdf.crs)
            world.boundary.plot(ax=ax, color="#555555", linewidth=0.6, alpha=0.7)
        except Exception:
            pass

    ax.set_title("Spatial distribution of stations\n(colour = number of measured variables)", fontsize=14, pad=15)
    ax.set_xlabel("Easting (m, UTM 32N)")
    ax.set_ylabel("Northing (m, UTM 32N)")
    ax.grid(True, alpha=0.3)

    # Optional: highlight stations with large elev_diff
    large_diff = gdf[gdf["large_elev_diff"]]
    if len(large_diff) > 0:
        large_diff.plot(
            ax=ax, marker="x", color="red", markersize=40, label="|elev_diff| > 100 m"
        )

    out_path = PLOTS_DIR / "07_spatial_map_num_variables.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Spatial map saved â†’ {out_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("Station terrain metadata analysis")
    print(f"Input: {CSV_PATH}")
    print(f"Plots will be written to: {PLOTS_DIR}")
    print("=" * 70)

    df = load_and_prepare(CSV_PATH)

    # Quick numeric summary
    summary_cols = ["elev_dem", "slope", "aspect_clean", "profile_curvature", "plan_curvature", "elev_diff"]
    summary = df[summary_cols].describe().T
    summary["median"] = df[summary_cols].median()
    summary_path = PLOTS_DIR / "00_summary_statistics.csv"
    summary.to_csv(summary_path)
    print("\nSummary statistics saved to", summary_path)
    print(summary.round(3))

    plot_distributions(df)
    plot_spatial_map(df)

    print("\nAll plots generated successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

