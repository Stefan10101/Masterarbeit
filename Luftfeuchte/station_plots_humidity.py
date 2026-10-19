#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relative humidity station and QC diagnostic plots.
One script for all relative humidity overview and QC figures.

Data sources (via paths.py):
- Station list, coordinates, elevation, network: QC/stations_overview.csv
- Time series: QC/{safe_name}_full_2020_2025_master.parquet  (masterset creator)
- Pre-QC series: QC/Temperatur/full_2020_2025/  (matched by station name / coords)
- DEM for elevation difference: Source/DEM/COP30_mosaic_EPSG31287.tif
- Map background and perimeter: same style as overview_domain_map.py
"""

from __future__ import annotations

import json
import logging
import random
import sys
import tempfile
import urllib.request
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from paths import (
    DATA_ROOT,
    get_dem_path,
    get_metadata_path,
    get_raw_data_dir,
)
from shared.ingest_paths import pipeline_output_dirs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VAR = "relative_humidity"
VAR_FOLDER = "Luftfeuchte"
ELEV_BINS = [
    ("Valley", 0, 800),
    ("Mid-mountain", 800, 2000),
    ("Mountain", 2000, 4500),
]
MAP_CLASS_COLORS = {
    "Valley": "#0346ff",
    "Mid-mountain": "#028d09",
    "Mountain": "#ffa600",
}
ELEV_MARKERS = {
    "Valley": "o",
    "Mid-mountain": "^",
    "Mountain": "p",  # pentagon
}
ELEV_HIST_BIN = 200  # m
N_QC_EXAMPLES = 20
QC_ZOOM_DAYS = 7
RANDOM_SEED = 42

PLOT_ROOT = DATA_ROOT / "Plots" / VAR_FOLDER
PLOT_ROOT.mkdir(parents=True, exist_ok=True)

OVERVIEW_CSV = get_metadata_path()
DEM_PATH = get_dem_path()
DEM_BG_PATH = DEM_PATH.with_name("COP30_mosaic_EPSG31287_unclipped.tif")
if not DEM_BG_PATH.exists():
    DEM_BG_PATH = DEM_PATH
QC_DIRS = pipeline_output_dirs(VAR_FOLDER)
ORIGINAL_DIR = QC_DIRS["full"]
QC_DIR = QC_DIRS["qc"]
MASTER_DIR = get_raw_data_dir()

PERIMETER_URL = (
    "https://services5.arcgis.com/qjjtr5pMpt3PDiiR/ArcGIS/rest/services/"
    "Alpine_Convention_Perimeter/FeatureServer/1/query?"
    "where=1%3D1&outFields=*&f=geojson"
)

# Three palettes for the first run. Each writes into its own subfolder.
PALETTES = {
    "black": {
        "primary": "#1a1a1a",
        "secondary": "#4a4a4a",
        "accent": "#c0392b",
        "line": "#000000",
        "fill": "#2c2c2c",
        "edge": "#000000",
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Inputs in degrees."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def elev_class(h):
    if pd.isna(h):
        return None
    for name, lo, hi in ELEV_BINS:
        if lo <= h < hi:
            return name
    return None


BUFFER_KM = 100


def filter_to_alpine_buffer(stations: pd.DataFrame, peri: gpd.GeoDataFrame) -> pd.DataFrame:
    """Keep stations inside the Alpine Convention polygon buffered by BUFFER_KM.
    Uses the actual perimeter geometry, not its bounding box.
    """
    if peri.crs is None:
        peri = peri.set_crs("EPSG:4326")
    peri = peri.to_crs("EPSG:4326")
    # Repair invalid rings before buffering
    peri = peri.copy()
    peri["geometry"] = peri.geometry.buffer(0)

    valid = stations.dropna(subset=["lat", "lon"]).copy()
    n_bad = len(stations) - len(valid)

    peri_m = peri.to_crs("EPSG:31287")
    buffered_m = peri_m.buffer(BUFFER_KM * 1000).union_all()
    if buffered_m is None or buffered_m.is_empty:
        logger.error("Alpine buffer geometry is empty — no stations kept")
        return valid.iloc[0:0].copy()

    gdf = gpd.GeoDataFrame(
        valid,
        geometry=gpd.points_from_xy(valid["lon"], valid["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:31287")
    inside = gdf.intersects(buffered_m)
    out = valid.loc[inside.to_numpy()].copy()

    minx, miny, maxx, maxy = peri.total_bounds
    logger.info(
        f"Alpine perimeter bounds: lon {minx:.2f}–{maxx:.2f}, lat {miny:.2f}–{maxy:.2f}"
    )
    logger.info(
        f"Domain filter (polygon + {BUFFER_KM} km): kept {len(out)} / {len(stations)} "
        f"(dropped {len(valid) - len(out)} outside, {n_bad} missing coords)"
    )
    if len(out) < len(valid):
        dropped = valid.loc[~inside.to_numpy(), "station_name"].head(8).tolist()
        logger.info(f"  example dropped: {dropped}")
    if len(out) == len(valid) and len(valid) > 50:
        logger.warning("Filter kept every station — check that the perimeter loaded correctly")
    return out


MAP_PAD = 0.2  # degrees beyond the furthest station
FIXED_MAP_EXTENT = None  # set once after the domain filter


def map_extent_from_perimeter(peri: gpd.GeoDataFrame, pad=2.0):
    """Alpine Convention bounding box plus a small fixed pad. Same for every map."""
    minx, miny, maxx, maxy = peri.total_bounds
    extent = (minx - pad, maxx + pad, miny - pad, maxy + pad)
    logger.info(
        f"Fixed map extent: lon {extent[0]:.2f}–{extent[1]:.2f}, "
        f"lat {extent[2]:.2f}–{extent[3]:.2f} (pad {pad}°)"
    )
    return extent


def map_extent_from_stations(stations: pd.DataFrame, pad=MAP_PAD):
    """Fixed rectangle: furthest station in each direction plus MAP_PAD degrees."""
    lon = stations["lon"].dropna()
    lat = stations["lat"].dropna()
    extent = (float(lon.min()) - pad, float(lon.max()) + pad,
              float(lat.min()) - pad, float(lat.max()) + pad)
    logger.info(
        f"Fixed map extent: lon {extent[0]:.2f}–{extent[1]:.2f}, "
        f"lat {extent[2]:.2f}–{extent[3]:.2f} (pad {pad}°)"
    )
    return extent


def load_variable_stations() -> pd.DataFrame:
    """Load stations_overview.csv and keep those with a relative humidity series."""
    df = pd.read_csv(OVERVIEW_CSV)
    # variables column is a comma-separated string of variable names
    mask = df["variables"].fillna("").str.contains("relative_humidity", case=False)
    out = df[mask].copy()
    out["elev_class"] = out["hoehe"].apply(elev_class)
    logger.info(f"Relative humidity stations: {len(out)} / {len(df)}")
    return out


def load_master(station_row) -> pd.DataFrame | None:
    """Load the master parquet for one station."""
    pq = MASTER_DIR / station_row["parquet"]
    if not pq.exists():
        logger.warning(f"Master parquet missing: {pq.name}")
        return None
    try:
        return pd.read_parquet(pq)
    except Exception as e:
        logger.warning(f"Failed to read {pq.name}: {e}")
        return None


_ORIGINAL_INDEX = None  # built once


def build_original_index() -> dict:
    """
    One-time index of pre-QC files: station name and rounded coords -> path.
    Avoids rescanning the folder for every station.
    """
    global _ORIGINAL_INDEX
    if _ORIGINAL_INDEX is not None:
        return _ORIGINAL_INDEX
    index = {"by_name": {}, "by_coord": {}}
    files = list(ORIGINAL_DIR.rglob("*.parquet"))
    logger.info(f"Indexing {len(files)} pre-QC files...")
    for f in files:
        try:
            peek = pd.read_parquet(f, columns=["station", "lat", "lon"])
            if peek.empty:
                continue
            row = peek.iloc[0]
            name = str(row.get("station", ""))
            if name:
                index["by_name"][name] = f
            if pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
                key = (round(float(row["lat"]), 3), round(float(row["lon"]), 3))
                index["by_coord"][key] = f
        except Exception:
            continue
    _ORIGINAL_INDEX = index
    logger.info(f"Indexed {len(index['by_name'])} stations by name")
    return index


def find_original_file(station_row) -> Path | None:
    """Locate the pre-QC parquet using the one-time index."""
    index = build_original_index()
    name = str(station_row["station_name"])
    if name in index["by_name"]:
        return index["by_name"][name]
    if pd.notna(station_row.get("lat")) and pd.notna(station_row.get("lon")):
        key = (round(float(station_row["lat"]), 3), round(float(station_row["lon"]), 3))
        if key in index["by_coord"]:
            return index["by_coord"][key]
    return None


def sample_dem_at_points(lats, lons) -> np.ndarray:
    """Sample COP30 DEM at given lat/lon. Returns array of elevations (NaN where no data)."""
    if not DEM_PATH.exists():
        logger.warning(f"DEM not found: {DEM_PATH}")
        return np.full(len(lats), np.nan)
    with rasterio.open(DEM_BG_PATH) as src:
        # Transform points to the DEM CRS if needed
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)
        coords = list(zip(xs, ys))
        samples = list(src.sample(coords))
        vals = np.array([s[0] if s is not None else np.nan for s in samples], dtype=float)
        # Treat nodata and extreme negatives as missing
        nodata = src.nodata if src.nodata is not None else -9999
        vals = np.where((vals == nodata) | (vals < -200), np.nan, vals)
    return vals


def load_alpine_perimeter() -> gpd.GeoDataFrame:
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        urllib.request.urlretrieve(PERIMETER_URL, tmp.name)
        gdf = gpd.read_file(tmp.name)
    return gdf.to_crs("EPSG:4326")


def load_dem_background(extent, target_res=0.0018):
    """Reproject DEM to lon/lat for the map background. Returns (array, extent)."""
    lon_min, lat_min, lon_max, lat_max = extent
    pad = 0.12
    bounds = (lon_min - pad, lat_min - pad, lon_max + pad, lat_max + pad)
    with rasterio.open(DEM_BG_PATH) as src:
        width = max(900, min(int((bounds[2] - bounds[0]) / target_res), 4800))
        height = max(700, min(int((bounds[3] - bounds[1]) / target_res), 3800))
        transform = rasterio.transform.from_bounds(
            bounds[0], bounds[1], bounds[2], bounds[3], width, height
        )
        arr = np.full((height, width), np.nan, dtype=np.float32)
        src_nodata = src.nodata if src.nodata is not None else 0
        reproject(
            source=rasterio.band(src, 1),
            destination=arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )
    arr = np.where(np.isnan(arr) | (arr < -200), np.nan, arr)
    return arr, (bounds[0], bounds[2], bounds[1], bounds[3])


def style_axes(ax, pal):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.grid(True, alpha=0.25, linewidth=0.4, color="0.7")
    ax.tick_params(labelsize=8)


def fine_counts(values, bin_width, align="left"):
    """Fixed-width histogram. Points sit on a bin edge, not the centre.
    align='left':  bin [n, n+w) is plotted at n   (0 % bin on the 0 line)
    align='right': bin (n-w, n] is plotted at n   (400–500 m bin on the 500 line)
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return None, None
    lo = np.floor(values.min() / bin_width) * bin_width
    hi = np.ceil(values.max() / bin_width) * bin_width
    if hi <= lo:
        hi = lo + bin_width
    edges = np.arange(lo, hi + bin_width * 0.5, bin_width)
    counts, edges = np.histogram(values, bins=edges)
    if align == "right":
        x = edges[1:]
    else:
        x = edges[:-1]
    return x, counts.astype(float)


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {path.name}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_nearest_distance(stations: pd.DataFrame, out_dir: Path, pal: dict):
    coords = stations[["lat", "lon"]].dropna().values
    if len(coords) < 2:
        return
    # Approximate nearest with cKDTree in degrees, then exact haversine to the neighbour
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=2)
    nearest = []
    for i, j in enumerate(idx[:, 1]):
        d = haversine_km(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
        nearest.append(d)
    nearest = np.array(nearest)

    fig, ax = plt.subplots(figsize=(8, 5))
    x, y = fine_counts(nearest, bin_width=1, align='left')
    if x is not None:
        ax.plot(x, y, color=pal["primary"], linewidth=1.1, marker="o", markersize=3.0, markerfacecolor=pal["primary"], markeredgecolor="white", markeredgewidth=0.3)
    ax.axvline(np.median(nearest), color=pal["accent"], linestyle="--", linewidth=1.2,
               label=f"Median {np.median(nearest):.1f} km")
    ax.set_xlabel("Distance to nearest relative humidity station (km)")
    ax.set_ylabel("Number of stations")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axes(ax, pal)
    save(fig, out_dir / "01_nearest_distance.png")


def plot_elevation_hist(stations: pd.DataFrame, out_dir: Path, pal: dict):
    h = stations["hoehe"].dropna().values
    if len(h) == 0:
        return
    bins = np.arange(0, np.nanmax(h) + ELEV_HIST_BIN, ELEV_HIST_BIN)
    counts, edges = np.histogram(h, bins=bins)
    pct = counts / counts.sum() * 100
    centers = edges[:-1] + ELEV_HIST_BIN / 2

    fig, ax = plt.subplots(figsize=(8, 5))
    x, y = fine_counts(h, bin_width=100, align='right')
    if x is not None:
        ax.plot(x, y, color=pal["primary"], linewidth=1.1, marker="o", markersize=3.0, markerfacecolor=pal["primary"], markeredgecolor="white", markeredgewidth=0.3, label="Stations")
    ax.set_xlabel("Elevation (m)")
    ax.set_ylabel("Number of stations")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axes(ax, pal)
    save(fig, out_dir / "02_elevation_distribution.png")


def plot_elev_diff(stations: pd.DataFrame, out_dir: Path, pal: dict):
    valid = stations.dropna(subset=["lat", "lon", "hoehe"])
    if valid.empty:
        return
    dem_elev = sample_dem_at_points(valid["lat"].values, valid["lon"].values)
    diff = valid["hoehe"].values - dem_elev
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        logger.warning("No valid DEM samples for elevation difference")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x, y = fine_counts(diff, bin_width=5, align='right')
    if x is not None:
        ax.plot(x, y, color=pal["primary"], linewidth=1.1, marker="o", markersize=3.0, markerfacecolor=pal["primary"], markeredgecolor="white", markeredgewidth=0.3)
    ax.axvline(0, color=pal["line"], linestyle="-", linewidth=0.8, label="Zero")
    ax.axvline(np.median(diff), color=pal["accent"], linestyle="--", linewidth=1.2,
               label=f"Median {np.median(diff):.0f} m")
    ax.set_xlabel("Metadata elevation − DEM elevation (m)")
    ax.set_ylabel("Number of stations")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axes(ax, pal)
    save(fig, out_dir / "03_elevation_difference.png")


def plot_station_map(stations: pd.DataFrame, out_dir: Path, pal: dict):
    peri = load_alpine_perimeter()
    main_extent = FIXED_MAP_EXTENT

    fig = plt.figure(figsize=(12.0, 8.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=ccrs.PlateCarree())
    ax.set_extent(main_extent, crs=ccrs.PlateCarree())
    mean_lat = (main_extent[2] + main_extent[3]) / 2
    ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)))
    ax.set_facecolor("#f5f5f5")

    if DEM_BG_PATH.exists():
        dem, dem_extent = load_dem_background(main_extent)
        ax.imshow(
            dem, origin="upper", extent=dem_extent,
            cmap="gray_r", vmin=0, vmax=3200,
            transform=ccrs.PlateCarree(), zorder=1, interpolation="nearest",
        )

    # Perimeter
    peri.plot(ax=ax, facecolor="none", edgecolor="#c0392b", linewidth=0.9,
              transform=ccrs.PlateCarree(), zorder=5)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.3, edgecolor="0.4", zorder=3)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#e6f3ff", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.RIVERS.with_scale("10m"), linewidth=0.28, edgecolor="#4a90d9", zorder=2)
    ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="#a8d5e5", edgecolor="#4a90d9",
                   linewidth=0.22, zorder=2)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.5", alpha=0.45, linestyle="-",
                      xlocs=np.arange(4, 18, 2), ylocs=np.arange(43, 50, 1))
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    # Stations
    handles = []
    for name, lo, hi in ELEV_BINS:
        sub = stations[stations["elev_class"] == name]
        if sub.empty:
            continue
        ax.scatter(
            sub["lon"], sub["lat"],
            marker=ELEV_MARKERS[name],
            s=18, c=MAP_CLASS_COLORS[name], edgecolors="white", linewidths=0.3,
            transform=ccrs.PlateCarree(), zorder=6, label=f"{name} ({lo}–{hi} m)",
        )
        handles.append(plt.Line2D(
            [0], [0], marker=ELEV_MARKERS[name], color="w",
            markerfacecolor=MAP_CLASS_COLORS[name], markeredgecolor="white",
            markersize=7, label=f"{name} ({lo}–{hi} m)",
        ))

    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.9,
              fontsize=8, title="Station elevation", title_fontsize=8)
    # North arrow
    ax.annotate("N", xy=(0.965, 0.94), xytext=(0.965, 0.88),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.15), zorder=8)
    # Scale bar
    scale_lon = main_extent[0] + 0.5
    scale_lat = main_extent[2] + 0.28
    ax.plot([scale_lon, scale_lon + 1.2], [scale_lat, scale_lat],
            color="black", linewidth=1.4, transform=ccrs.PlateCarree(), zorder=7)
    ax.text(scale_lon + 0.6, scale_lat + 0.1, "100 km", ha="center", fontsize=7,
            transform=ccrs.PlateCarree(), zorder=7)

    save(fig, out_dir / "04_station_map.png")


def plot_temp_boxplot(stations: pd.DataFrame, out_dir: Path, pal: dict):
    """One mean relative humidity per station (full series), violin plot by elevation class."""
    means = []
    classes = []
    for _, row in stations.iterrows():
        df = load_master(row)
        if df is None or VAR not in df.columns:
            continue
        vals = df[VAR].dropna()
        if len(vals) < 100:
            continue
        means.append(vals.mean())
        classes.append(row["elev_class"])
    if not means:
        return
    data = pd.DataFrame({"mean_temp": means, "elev_class": classes})
    order = [b[0] for b in ELEV_BINS]
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_data = []
    positions = []
    labels = []
    for i, name in enumerate(order):
        vals = data.loc[data["elev_class"] == name, "mean_temp"].dropna().values
        if len(vals) == 0:
            continue
        plot_data.append(vals)
        positions.append(i)
        labels.append(name)
    if not plot_data:
        return
    parts = ax.violinplot(plot_data, positions=positions, widths=0.7, showmeans=False, showmedians=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor(pal["fill"])
        body.set_edgecolor(pal["edge"])
        body.set_alpha(0.85)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color(pal["accent"] if key == "cmedians" else pal["edge"])
            parts[key].set_linewidth(1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{ELEV_BINS[i][0]}\n({ELEV_BINS[i][1]}–{ELEV_BINS[i][2]} m)" for i in positions], fontsize=8)
    ax.set_ylabel("Mean relative humidity (%)")
    ax.legend([plt.Line2D([0], [0], color=pal["accent"], linewidth=1.4)],
              ["Median"], frameon=False, fontsize=8, loc="upper right")
    style_axes(ax, pal)
    save(fig, out_dir / "05_violin_elevation.png")


def plot_temp_distribution(stations: pd.DataFrame, out_dir: Path, pal: dict):
    """Histogram of all 30-min relative humidity values."""
    all_vals = []
    for _, row in stations.iterrows():
        df = load_master(row)
        if df is None or VAR not in df.columns:
            continue
        all_vals.append(df[VAR].dropna().values)
    if not all_vals:
        return
    vals = np.concatenate(all_vals)
    # Relative humidity distribution
    bins = np.arange(-40, 45, 1)
    counts, edges = np.histogram(vals, bins=bins)
    pct = counts / counts.sum() * 100
    centers = edges[:-1] + 0.5

    fig, ax = plt.subplots(figsize=(9, 5))
    x, y = fine_counts(vals, bin_width=1, align='left')
    mean_t = float(np.mean(vals))
    if x is not None:
        ax.plot(x, y, color=pal["primary"], linewidth=1.1, marker="o", markersize=3.0, markerfacecolor=pal["primary"], markeredgecolor="white", markeredgewidth=0.3, label="Observations")
    ax.axvline(mean_t, color=pal["accent"], linestyle="--", linewidth=1.2, label=f"Mean {mean_t:.1f} %")
    ax.set_xlabel("Relative humidity (%)")
    ax.set_ylabel("Number of observations")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axes(ax, pal)
    save(fig, out_dir / "06_distribution.png")


def plot_variability_vs_elevation(stations: pd.DataFrame, out_dir: Path, pal: dict):
    """Standard deviation of each station's series vs elevation."""
    elevs, stds = [], []
    for _, row in stations.iterrows():
        if pd.isna(row["hoehe"]):
            continue
        df = load_master(row)
        if df is None or VAR not in df.columns:
            continue
        vals = df[VAR].dropna()
        if len(vals) < 100:
            continue
        elevs.append(row["hoehe"])
        stds.append(vals.std())
    if len(elevs) < 5:
        return
    elevs = np.array(elevs)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(elevs, stds, s=12, c=pal["primary"], alpha=0.7, edgecolors="none", label="Station")
    # Simple linear trend
    coef = np.polyfit(elevs, stds, 1)
    xfit = np.linspace(elevs.min(), elevs.max(), 100)
    ax.plot(xfit, np.polyval(coef, xfit), color=pal["accent"], linewidth=1.2,
            label=f"Trend ({coef[0]*1000:.2f} % / 1000 m)")
    ax.set_xlabel("Elevation (m)")
    ax.set_ylabel("Standard deviation of relative humidity (%)")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax, pal)
    save(fig, out_dir / "07_variability_vs_elevation.png")


def compute_all_removals(stations: pd.DataFrame) -> pd.DataFrame:
    """
    Count QC removals for every relative humidity station.
    Uses the one-time original-file index, so this is one pass, not a repeated scan.
    """
    records = []
    n = len(stations)
    for i, (_, row) in enumerate(stations.iterrows(), start=1):
        if i % 100 == 0 or i == n:
            logger.info(f"  removal count {i}/{n}")
        orig_path = find_original_file(row)
        if orig_path is None:
            continue
        master = load_master(row)
        if master is None or VAR not in master.columns:
            continue
        try:
            orig = pd.read_parquet(orig_path, columns=["timestamp", "value"])
        except Exception:
            continue
        orig["timestamp"] = pd.to_datetime(orig["timestamp"], utc=True)
        master = master.copy()
        master["timestamp"] = pd.to_datetime(master["timestamp"], utc=True)
        merged = pd.merge(
            orig.rename(columns={"value": "original"}),
            master[["timestamp", VAR]].rename(columns={VAR: "qc"}),
            on="timestamp", how="inner",
        )
        removed = (merged["original"].notna()) & (merged["qc"].isna())
        records.append({
            "station_name": row["station_name"],
            "n_removed": int(removed.sum()),
            "n_original": int(merged["original"].notna().sum()),
            "row": row,
            "merged": merged,
        })
    logger.info(f"Removal counts computed for {len(records)} stations")
    return pd.DataFrame(records)


def sample_qc_examples(removal_df: pd.DataFrame, n_target: int = N_QC_EXAMPLES) -> list[dict]:
    """Pick n_target stations that had removals and build their 7-day windows."""
    candidates = removal_df[removal_df["n_removed"] > 0]
    if candidates.empty:
        logger.warning("No stations with QC removals")
        return []
    n = min(n_target, len(candidates))
    chosen = candidates.sample(n=n, random_state=RANDOM_SEED)
    examples = []
    for _, rec in chosen.iterrows():
        merged = rec["merged"]
        removed_mask = (merged["original"].notna()) & (merged["qc"].isna())
        if not removed_mask.any():
            continue
        event = merged.loc[removed_mask, "timestamp"].iloc[0]
        start = event - pd.Timedelta(days=QC_ZOOM_DAYS / 2)
        end = event + pd.Timedelta(days=QC_ZOOM_DAYS / 2)
        window = merged[(merged["timestamp"] >= start) & (merged["timestamp"] <= end)]
        examples.append({
            "station_name": rec["station_name"],
            "n_removed": rec["n_removed"],
            "window": window,
        })
        logger.info(f"  QC example {len(examples)}/{n}: {rec['station_name']} ({rec['n_removed']} removed)")
    return examples


def plot_qc_examples(examples: list[dict], out_dir: Path, pal: dict):
    """Split plot: top = original, bottom = after QC, same 7-day window."""
    if not examples:
        logger.warning("No QC examples to plot")
        return
    for rec in examples:
        window = rec["window"]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax1.plot(window["timestamp"], window["original"], color=pal["secondary"],
                 linewidth=0.9, label="Original")
        ax1.set_ylabel("Original (%)")
        ax1.legend(frameon=False, fontsize=8, loc="lower right")
        style_axes(ax1, pal)

        ax2.plot(window["timestamp"], window["qc"], color=pal["primary"],
                 linewidth=0.9, label="After QC")
        # Mark removed points (original present, QC missing)
        removed = window["original"].notna() & window["qc"].isna()
        if removed.any():
            ax2.scatter(window.loc[removed, "timestamp"], window.loc[removed, "original"],
                        s=12, c=pal["accent"], marker="x", linewidths=0.8, label="Removed", zorder=5)
        ax2.set_ylabel("After QC (%)")
        ax2.set_xlabel("Time")
        ax2.legend(frameon=False, fontsize=8, loc="lower right")
        style_axes(ax2, pal)
        fig.autofmt_xdate()
        safe = str(rec["station_name"]).replace("/", "_").replace(" ", "_")[:40]
        save(fig, out_dir / f"08_qc_example_{safe}.png")


def plot_removals(removal_df: pd.DataFrame, out_dir: Path, pal: dict):
    """Sorted bar and smooth density of points removed. Values above the 99th percentile are dropped."""
    if removal_df.empty:
        return
    cap = removal_df["n_removed"].quantile(0.99)
    df = removal_df[removal_df["n_removed"] <= cap].copy()
    df = df.sort_values("n_removed", ascending=False).reset_index(drop=True)
    logger.info(f"Removal plots: kept {len(df)} / {len(removal_df)} stations (<= 99th pct = {cap:.0f})")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(df)), df["n_removed"], color=pal["fill"], edgecolor=pal["edge"],
           linewidth=0.15, label="Points removed")
    ax.set_xlabel("Stations (sorted by number of removals)")
    ax.set_ylabel("Number of points removed")
    ax.set_xticks([])
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axes(ax, pal)
    save(fig, out_dir / "09a_points_removed_sorted.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    x, y = fine_counts(df['n_removed'].values, bin_width=1, align='left')
    if x is not None:
        ax.plot(x, y, color=pal["primary"], linewidth=1.1, marker="o", markersize=3.0, markerfacecolor=pal["primary"], markeredgecolor="white", markeredgewidth=0.3)
    ax.set_xlabel("Number of points removed")
    ax.set_ylabel("Number of stations")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axes(ax, pal)
    save(fig, out_dir / "09b_points_removed_density.png")


def main():
    peri = load_alpine_perimeter()
    all_stations = pd.read_csv(OVERVIEW_CSV)
    all_inside = filter_to_alpine_buffer(all_stations, peri)
    if all_inside.empty:
        logger.error("No stations inside the Alpine buffer")
        return
    global FIXED_MAP_EXTENT
    FIXED_MAP_EXTENT = (4.0, 17.0, 43.0, 49.0)
    logger.info(f"Fixed map extent: lon 4–17, lat 43–49")

    stations = load_variable_stations()
    stations = stations[stations["station_name"].isin(all_inside["station_name"])].copy()
    if stations.empty:
        logger.error("No relative humidity stations inside the Alpine buffer")
        return
    logger.info(f"Relative humidity stations inside buffer: {len(stations)}")

    logger.info("Computing QC removal counts for all stations (indexed)...")
    removal_df = compute_all_removals(stations)
    qc_examples = sample_qc_examples(removal_df)

    for name, pal in PALETTES.items():
        out_dir = PLOT_ROOT / f"palette_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"=== Palette: {name} ===")
        plot_nearest_distance(stations, out_dir, pal)
        plot_elevation_hist(stations, out_dir, pal)
        plot_elev_diff(stations, out_dir, pal)
        plot_station_map(stations, out_dir, pal)
        plot_temp_boxplot(stations, out_dir, pal)
        plot_temp_distribution(stations, out_dir, pal)
        plot_variability_vs_elevation(stations, out_dir, pal)
        plot_qc_examples(qc_examples, out_dir, pal)
        plot_removals(removal_df, out_dir, pal)

    logger.info(f"Done. Figures under {PLOT_ROOT}")


if __name__ == "__main__":
    main()
