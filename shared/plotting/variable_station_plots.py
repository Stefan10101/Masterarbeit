#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variable-wise station plots, driven by the daily aggregate.

One entry point: run(cfg). Each variable script is a thin wrapper that passes
its config. Value-based plots (violin, distribution, variability, hold-out
time series) read Source/aggregated/daily_station_data.parquet. Station
metadata comes from stations_overview.csv. Hold-out membership comes from
the per-column YAML. QC comparison plots are optional and still use the
per-station master / pre-QC files.
"""

from __future__ import annotations

import logging
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
import yaml
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.spatial import cKDTree

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from paths import (
    DATA_ROOT,
    get_aggregated_data_path,
    get_dem_path,
    get_metadata_path,
    get_raw_data_dir,
)
from shared.ingest_paths import pipeline_output_dirs
from shared.splits.evaluation_sets import holdout_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ELEV_CUTOFF = 1500
ELEV_BINS = [
    ("Valley", 0, ELEV_CUTOFF),
    ("Mountain", ELEV_CUTOFF, 4500),
]
MAP_CLASS_COLORS = {"Valley": "#0346ff", "Mountain": "#ffa600"}
ELEV_MARKERS = {"Valley": "o", "Mountain": "^"}
HOLDOUT_COLOR = "#e67e22"
POOLED_COLOR = "#4a4a4a"
ELEV_HIST_BIN = 200
BUFFER_KM = 100
N_QC_EXAMPLES = 20
QC_ZOOM_DAYS = 7
RANDOM_SEED = 42
TS_MONTHS = (1, 4, 7, 10)  # one per season in 2025
TS_YEAR = 2025
TS_N_STATIONS = 10
FIXED_MAP_EXTENT = (4.0, 17.0, 43.0, 49.0)

PERIMETER_URL = (
    "https://services5.arcgis.com/qjjtr5pMpt3PDiiR/ArcGIS/rest/services/"
    "Alpine_Convention_Perimeter/FeatureServer/1/query?"
    "where=1%3D1&outFields=*&f=geojson"
)

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

# Set by run() so helpers can reach the active config without threading it
# through every signature.
_CFG: dict = {}
_ORIGINAL_INDEX = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def elev_class(h):
    if pd.isna(h):
        return None
    return "Valley" if h < ELEV_CUTOFF else "Mountain"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def style_axes(ax, pal):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.4, color="0.7")


def fine_counts(values, bin_width, align="left"):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None, None
    lo = np.floor(values.min() / bin_width) * bin_width
    hi = np.ceil(values.max() / bin_width) * bin_width
    edges = np.arange(lo, hi + bin_width, bin_width)
    counts, _ = np.histogram(values, bins=edges)
    x = edges[:-1] if align == "left" else edges[1:]
    return x, counts


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {path.name}")


def load_alpine_perimeter() -> gpd.GeoDataFrame:
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        urllib.request.urlretrieve(PERIMETER_URL, tmp.name)
        gdf = gpd.read_file(tmp.name)
    return gdf.to_crs("EPSG:4326")


def filter_to_alpine_buffer(stations: pd.DataFrame, peri: gpd.GeoDataFrame) -> pd.DataFrame:
    if peri.crs is None:
        peri = peri.set_crs("EPSG:4326")
    peri = peri.to_crs("EPSG:4326").copy()
    peri["geometry"] = peri.geometry.buffer(0)
    valid = stations.dropna(subset=["lat", "lon"]).copy()
    peri_m = peri.to_crs("EPSG:31287")
    buffered_m = peri_m.buffer(BUFFER_KM * 1000).union_all()
    if buffered_m is None or buffered_m.is_empty:
        return valid.iloc[0:0].copy()
    gdf = gpd.GeoDataFrame(
        valid,
        geometry=gpd.points_from_xy(valid["lon"], valid["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:31287")
    inside = gdf.intersects(buffered_m)
    out = valid.loc[inside.to_numpy()].copy()
    logger.info(
        f"Domain filter (polygon + {BUFFER_KM} km): kept {len(out)} / {len(stations)}"
    )
    return out


def load_dem_background(extent, target_res=0.0018):
    dem_path = get_dem_path()
    bg = dem_path.with_name("COP30_mosaic_EPSG31287_unclipped.tif")
    if not bg.exists():
        bg = dem_path
    lon_min, lat_min, lon_max, lat_max = extent
    pad = 0.12
    bounds = (lon_min - pad, lat_min - pad, lon_max + pad, lat_max + pad)
    with rasterio.open(bg) as src:
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
            resampling=Resampling.average,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )
    arr = np.where(np.isnan(arr) | (arr < -200), np.nan, arr)
    # imshow extent is (left, right, bottom, top)
    return arr, (bounds[0], bounds[2], bounds[1], bounds[3])


def sample_dem_at_points(lats, lons) -> np.ndarray:
    dem_path = get_dem_path()
    bg = dem_path.with_name("COP30_mosaic_EPSG31287_unclipped.tif")
    if not bg.exists():
        bg = dem_path
    if not bg.exists():
        return np.full(len(lats), np.nan)
    from pyproj import Transformer
    with rasterio.open(bg) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)
        samples = list(src.sample(list(zip(xs, ys))))
        vals = np.array([s[0] if s is not None else np.nan for s in samples], dtype=float)
        nodata = src.nodata if src.nodata is not None else -9999
        vals = np.where((vals == nodata) | (vals < -200), np.nan, vals)
    return vals


def load_holdout(code: str) -> set:
    path = holdout_path(code)
    if not path.exists():
        logger.warning(f"No holdout file for {code}: {path}")
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return set(raw.get("held_out", []))
    except Exception as exc:
        logger.warning(f"Could not load holdout for {code}: {exc}")
        return set()


def load_stations_and_daily(cfg: dict, peri: gpd.GeoDataFrame):
    """Return (stations DataFrame, daily DataFrame restricted to those stations).

    Station set = stations with at least one non-null daily value of the
    column, inside the Alpine buffer, with metadata from stations_overview.csv.
    """
    code = cfg["code"]
    agg_path = get_aggregated_data_path("Source", "daily")
    if not agg_path.exists():
        raise FileNotFoundError(f"Daily aggregate not found: {agg_path}")
    daily = pd.read_parquet(agg_path, columns=["station_name", "date", code])
    have = set(daily.loc[daily[code].notna(), "station_name"].astype(str).str.strip())
    logger.info(f"{code}: {len(have)} stations with data in daily aggregate")

    meta = pd.read_csv(get_metadata_path())
    meta["station_name"] = meta["station_name"].astype(str).str.strip()
    stations = meta[meta["station_name"].isin(have)].copy()
    stations = filter_to_alpine_buffer(stations, peri)
    stations["elev_class"] = stations["hoehe"].apply(elev_class)
    hold = load_holdout(code)
    stations["is_holdout"] = stations["station_name"].isin(hold)
    logger.info(
        f"{cfg['label']}: {len(stations)} stations inside buffer, "
        f"{int(stations['is_holdout'].sum())} held out"
    )

    keep = set(stations["station_name"])
    daily = daily[daily["station_name"].astype(str).str.strip().isin(keep)].copy()
    daily["timestamp"] = pd.to_datetime(daily["date"], utc=True)
    daily = daily.drop(columns=["date"])
    return stations, daily


# ---------------------------------------------------------------------------
# Plots that use only station metadata
# ---------------------------------------------------------------------------
def plot_nearest_distance(stations: pd.DataFrame, out_dir: Path, pal: dict, label: str):
    coords = stations.dropna(subset=["lat", "lon"])[["lat", "lon"]].to_numpy()
    if len(coords) < 2:
        return
    tree = cKDTree(np.radians(coords))
    dist, _ = tree.query(np.radians(coords), k=2)
    nearest = dist[:, 1] * 6371.0
    fig, ax = plt.subplots(figsize=(8, 5))
    x, y = fine_counts(nearest, bin_width=1, align="left")
    if x is not None:
        ax.plot(x, y, color=pal["primary"], linewidth=1.1, marker="o", markersize=3.0,
                markerfacecolor=pal["primary"], markeredgecolor="white", markeredgewidth=0.3)
    ax.axvline(np.median(nearest), color=pal["accent"], linestyle="--", linewidth=1.2,
               label=f"Median {np.median(nearest):.1f} km")
    ax.set_xlabel(f"Distance to nearest {label.lower()} station (km)")
    ax.set_ylabel("Number of stations")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axes(ax, pal)
    save(fig, out_dir / "01_nearest_distance.png")


def plot_elevation_hist(stations: pd.DataFrame, out_dir: Path, pal: dict):
    h = stations["hoehe"].dropna().values
    if len(h) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x, y = fine_counts(h, bin_width=ELEV_HIST_BIN, align="right")
    if x is not None:
        ax.plot(x, y, color=pal["primary"], linewidth=1.1, marker="o", markersize=3.0,
                markerfacecolor=pal["primary"], markeredgecolor="white", markeredgewidth=0.3)
    ax.axvline(ELEV_CUTOFF, color=pal["accent"], linestyle="--", linewidth=1.0,
               label=f"{ELEV_CUTOFF} m")
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
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x, y = fine_counts(diff, bin_width=5, align="right")
    if x is not None:
        ax.plot(x, y, color=pal["primary"], linewidth=1.1, marker="o", markersize=3.0,
                markerfacecolor=pal["primary"], markeredgecolor="white", markeredgewidth=0.3)
    ax.axvline(0, color=pal["line"], linestyle="-", linewidth=0.8)
    ax.axvline(np.median(diff), color=pal["accent"], linestyle="--", linewidth=1.2,
               label=f"Median {np.median(diff):.0f} m")
    ax.set_xlabel("Metadata elevation − DEM elevation (m)")
    ax.set_ylabel("Number of stations")
    ax.set_xlim(-100, 100)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axes(ax, pal)
    save(fig, out_dir / "03_elevation_difference.png")


def plot_station_map(stations: pd.DataFrame, out_dir: Path, pal: dict):
    peri = load_alpine_perimeter()
    extent = FIXED_MAP_EXTENT
    fig = plt.figure(figsize=(12.0, 8.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    mean_lat = (extent[2] + extent[3]) / 2
    ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)))
    ax.set_facecolor("#f5f5f5")
    dem, dem_extent = load_dem_background(extent)
    ax.imshow(dem, origin="upper", extent=dem_extent, cmap="gray_r", vmin=0, vmax=3200,
              transform=ccrs.PlateCarree(), zorder=1, interpolation="nearest")
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

    handles = []
    for name, lo, hi in ELEV_BINS:
        sub = stations[stations["elev_class"] == name]
        if sub.empty:
            continue
        ax.scatter(
            sub["lon"], sub["lat"],
            marker=ELEV_MARKERS[name], s=18, c=MAP_CLASS_COLORS[name],
            edgecolors="white", linewidths=0.3,
            transform=ccrs.PlateCarree(), zorder=6,
        )
        handles.append(plt.Line2D(
            [0], [0], marker=ELEV_MARKERS[name], color="w",
            markerfacecolor=MAP_CLASS_COLORS[name], markeredgecolor="white",
            markersize=7, label=f"{name} ({lo}–{hi} m, n={len(sub)})",
        ))
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.9,
              fontsize=8, title="Station elevation", title_fontsize=8)
    ax.annotate("N", xy=(0.965, 0.94), xytext=(0.965, 0.88),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.15), zorder=8)
    scale_lon = extent[0] + 0.5
    scale_lat = extent[2] + 0.28
    ax.plot([scale_lon, scale_lon + 1.2], [scale_lat, scale_lat],
            color="black", linewidth=1.4, transform=ccrs.PlateCarree(), zorder=7)
    ax.text(scale_lon + 0.6, scale_lat + 0.1, "100 km", ha="center", fontsize=7,
            transform=ccrs.PlateCarree(), zorder=7)
    save(fig, out_dir / "04_station_map.png")


# ---------------------------------------------------------------------------
# Plots that use the daily aggregate
# ---------------------------------------------------------------------------
def _station_means(stations: pd.DataFrame, daily: pd.DataFrame, code: str) -> pd.DataFrame:
    """Mean daily value per station, joined to elevation class and hold-out flag."""
    g = daily.groupby("station_name")[code].mean()
    out = stations[["station_name", "elev_class", "is_holdout", "hoehe"]].copy()
    out["mean"] = out["station_name"].map(g)
    return out.dropna(subset=["mean", "elev_class"])


def plot_violin(stations: pd.DataFrame, daily: pd.DataFrame, out_dir: Path, pal: dict, cfg: dict):
    """Four violins: Valley/Mountain × pooled/hold-out."""
    stats = _station_means(stations, daily, cfg["code"])
    groups = [
        ("Valley pooled", "Valley", False, POOLED_COLOR),
        ("Valley hold-out", "Valley", True, HOLDOUT_COLOR),
        ("Mountain pooled", "Mountain", False, POOLED_COLOR),
        ("Mountain hold-out", "Mountain", True, HOLDOUT_COLOR),
    ]
    plot_data, positions, labels, colors = [], [], [], []
    for i, (lab, elev, is_h, col) in enumerate(groups):
        vals = stats.loc[
            (stats["elev_class"] == elev) & (stats["is_holdout"] == is_h), "mean"
        ].dropna().values
        if len(vals) == 0:
            continue
        plot_data.append(vals)
        positions.append(i)
        labels.append(f"{lab}\n(n={len(vals)})")
        colors.append(col)
    if not plot_data:
        logger.info("No data for violin — skipped")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(plot_data, positions=positions, widths=0.7,
                          showmeans=False, showmedians=True, showextrema=True)
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_edgecolor(pal["edge"])
        body.set_alpha(0.8)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color(pal["accent"] if key == "cmedians" else pal["edge"])
            parts[key].set_linewidth(1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f"Mean daily {cfg['label'].lower()} ({cfg['unit']})")
    style_axes(ax, pal)
    save(fig, out_dir / "05_violin_elevation.png")


def plot_distribution(stations: pd.DataFrame, daily: pd.DataFrame, out_dir: Path, pal: dict, cfg: dict):
    vals = daily[cfg["code"]].dropna().values
    if len(vals) == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    # Bin width scaled to the data range so the curve is readable
    span = float(np.nanpercentile(vals, 99) - np.nanpercentile(vals, 1))
    bw = max(span / 60.0, 1e-3)
    x, y = fine_counts(vals, bin_width=bw, align="left")
    if x is not None:
        ax.plot(x, y, color=pal["primary"], linewidth=1.1, marker="o", markersize=2.5,
                markerfacecolor=pal["primary"], markeredgecolor="white", markeredgewidth=0.3)
    ax.axvline(np.mean(vals), color=pal["accent"], linestyle="--", linewidth=1.2,
               label=f"Mean {np.mean(vals):.2f} {cfg['unit']}")
    ax.set_xlabel(f"Daily {cfg['label'].lower()} ({cfg['unit']})")
    ax.set_ylabel("Number of station-days")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axes(ax, pal)
    save(fig, out_dir / "06_distribution.png")


def plot_variability(stations: pd.DataFrame, daily: pd.DataFrame, out_dir: Path, pal: dict, cfg: dict):
    g = daily.groupby("station_name")[cfg["code"]].std()
    sub = stations.dropna(subset=["hoehe"]).copy()
    sub["std"] = sub["station_name"].map(g)
    sub = sub.dropna(subset=["std"])
    if len(sub) < 5:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sub["hoehe"], sub["std"], s=12, c=pal["primary"], alpha=0.7, edgecolors="none")
    coef = np.polyfit(sub["hoehe"].values, sub["std"].values, 1)
    xfit = np.linspace(sub["hoehe"].min(), sub["hoehe"].max(), 100)
    ax.plot(xfit, np.polyval(coef, xfit), color=pal["accent"], linewidth=1.2,
            label=f"Trend ({coef[0]*1000:.2f} {cfg['unit']} / 1000 m)")
    ax.set_xlabel("Elevation (m)")
    ax.set_ylabel(f"Std. dev. of daily {cfg['label'].lower()} ({cfg['unit']})")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax, pal)
    save(fig, out_dir / "07_variability_vs_elevation.png")


def plot_holdout_timeseries(stations: pd.DataFrame, daily: pd.DataFrame, out_dir: Path, pal: dict, cfg: dict):
    """Four months in 2025, ten hold-out stations, one panel per month."""
    hold = stations[stations["is_holdout"]].copy()
    if hold.empty:
        logger.warning("No hold-out stations — skipping time series")
        return
    rng = np.random.default_rng(RANDOM_SEED)
    picked = []
    for elev, n in (("Valley", 5), ("Mountain", 5)):
        pool = hold[hold["elev_class"] == elev]
        if pool.empty:
            continue
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        picked.append(pool.iloc[idx])
    sel = pd.concat(picked) if picked else hold.sample(min(TS_N_STATIONS, len(hold)), random_state=RANDOM_SEED)
    if len(sel) < TS_N_STATIONS:
        rest = hold[~hold["station_name"].isin(sel["station_name"])]
        need = TS_N_STATIONS - len(sel)
        if len(rest):
            sel = pd.concat([sel, rest.sample(min(need, len(rest)), random_state=RANDOM_SEED)])
    names = sel["station_name"].tolist()
    logger.info(f"Time series stations: {names}")

    sub = daily[daily["station_name"].isin(names)].copy()
    sub = sub[sub["timestamp"].dt.year == TS_YEAR]
    sub = sub[sub["timestamp"].dt.month.isin(TS_MONTHS)]
    if sub.empty:
        logger.warning("No 2025 data for selected hold-out stations")
        return

    month_names = {1: "January", 4: "April", 7: "July", 10: "October"}
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, month in zip(axes.ravel(), TS_MONTHS):
        part = sub[sub["timestamp"].dt.month == month]
        for name, col in zip(names, colors):
            s = part[part["station_name"] == name].sort_values("timestamp")
            if s.empty:
                continue
            ax.plot(s["timestamp"], s[cfg["code"]], color=col, linewidth=0.9, label=name)
        ax.set_title(f"{month_names[month]} {TS_YEAR}", fontsize=10)
        ax.tick_params(labelsize=7)
        style_axes(ax, pal)
    axes[0, 0].set_ylabel(f"{cfg['label']} ({cfg['unit']})")
    axes[1, 0].set_ylabel(f"{cfg['label']} ({cfg['unit']})")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7, frameon=False)
    fig.suptitle(f"Hold-out station time series — {cfg['label']}", fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 0.85, 0.95))
    save(fig, out_dir / "10_holdout_timeseries.png")


# ---------------------------------------------------------------------------
# Optional QC plots (still use master / pre-QC files)
# ---------------------------------------------------------------------------
def _build_original_index(original_dir: Path) -> dict:
    global _ORIGINAL_INDEX
    if _ORIGINAL_INDEX is not None:
        return _ORIGINAL_INDEX
    index = {"by_name": {}, "by_coord": {}}
    files = list(original_dir.rglob("*.parquet"))
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
    return index


def _find_original(station_row, index: dict) -> Path | None:
    name = str(station_row["station_name"])
    if name in index["by_name"]:
        return index["by_name"][name]
    if pd.notna(station_row.get("lat")) and pd.notna(station_row.get("lon")):
        key = (round(float(station_row["lat"]), 3), round(float(station_row["lon"]), 3))
        if key in index["by_coord"]:
            return index["by_coord"][key]
    return None


def compute_all_removals(stations: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    master_col = cfg.get("master_col")
    if not master_col:
        return pd.DataFrame()
    qc_dirs = pipeline_output_dirs(cfg["folder"])
    original_dir = qc_dirs["full"]
    master_dir = get_raw_data_dir()
    index = _build_original_index(original_dir)
    records = []
    for _, row in stations.iterrows():
        orig_path = _find_original(row, index)
        if orig_path is None or "parquet" not in row or pd.isna(row.get("parquet")):
            continue
        master_path = master_dir / row["parquet"]
        if not master_path.exists():
            continue
        try:
            master = pd.read_parquet(master_path)
            orig = pd.read_parquet(orig_path, columns=["timestamp", "value"])
        except Exception:
            continue
        if master_col not in master.columns:
            continue
        orig["timestamp"] = pd.to_datetime(orig["timestamp"], utc=True)
        master = master.copy()
        master["timestamp"] = pd.to_datetime(master["timestamp"], utc=True)
        merged = pd.merge(
            orig.rename(columns={"value": "original"}),
            master[["timestamp", master_col]].rename(columns={master_col: "qc"}),
            on="timestamp", how="inner",
        )
        removed = (merged["original"].notna()) & (merged["qc"].isna())
        records.append({
            "station_name": row["station_name"],
            "n_removed": int(removed.sum()),
            "row": row,
            "merged": merged,
        })
    logger.info(f"Removal counts computed for {len(records)} stations")
    return pd.DataFrame(records)


def plot_qc_examples(removal_df: pd.DataFrame, out_dir: Path, pal: dict, cfg: dict):
    if removal_df.empty:
        return
    candidates = removal_df[removal_df["n_removed"] > 0]
    if candidates.empty:
        return
    sample = candidates.sample(min(N_QC_EXAMPLES, len(candidates)), random_state=RANDOM_SEED)
    for _, rec in sample.iterrows():
        merged = rec["merged"]
        if merged.empty:
            continue
        # Pick a 7-day window with removals
        removed = merged["original"].notna() & merged["qc"].isna()
        if not removed.any():
            continue
        t0 = merged.loc[removed, "timestamp"].iloc[0]
        window = merged[(merged["timestamp"] >= t0) & (merged["timestamp"] < t0 + pd.Timedelta(days=QC_ZOOM_DAYS))]
        if window.empty:
            continue
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.plot(window["timestamp"], window["original"], color=pal["secondary"], linewidth=0.9, label="Original")
        ax1.set_ylabel(f"Original ({cfg['unit']})")
        ax1.legend(frameon=False, fontsize=8, loc="lower right")
        style_axes(ax1, pal)
        ax2.plot(window["timestamp"], window["qc"], color=pal["primary"], linewidth=0.9, label="After QC")
        rem = window["original"].notna() & window["qc"].isna()
        if rem.any():
            ax2.scatter(window.loc[rem, "timestamp"], window.loc[rem, "original"],
                        s=12, c=pal["accent"], marker="x", linewidths=0.8, label="Removed", zorder=5)
        ax2.set_ylabel(f"After QC ({cfg['unit']})")
        ax2.set_xlabel("Time")
        ax2.legend(frameon=False, fontsize=8, loc="lower right")
        style_axes(ax2, pal)
        fig.autofmt_xdate()
        safe = str(rec["station_name"]).replace("/", "_").replace(" ", "_")[:40]
        save(fig, out_dir / f"08_qc_example_{safe}.png")


def plot_removals(removal_df: pd.DataFrame, out_dir: Path, pal: dict):
    if removal_df.empty:
        return
    cap = removal_df["n_removed"].quantile(0.99)
    df = removal_df[removal_df["n_removed"] <= cap].copy()
    df = df.sort_values("n_removed", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(df)), df["n_removed"], color=pal["fill"], edgecolor=pal["edge"],
           linewidth=0.15)
    ax.set_xlabel("Stations (sorted by number of removals)")
    ax.set_ylabel("Number of points removed")
    ax.set_xticks([])
    style_axes(ax, pal)
    save(fig, out_dir / "09a_points_removed_sorted.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    x, y = fine_counts(df["n_removed"].values, bin_width=1, align="left")
    if x is not None:
        ax.plot(x, y, color=pal["primary"], linewidth=1.1, marker="o", markersize=3.0,
                markerfacecolor=pal["primary"], markeredgecolor="white", markeredgewidth=0.3)
    ax.set_xlabel("Number of points removed")
    ax.set_ylabel("Number of stations")
    style_axes(ax, pal)
    save(fig, out_dir / "09b_points_removed_density.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(cfg: dict):
    """Produce the full plot set for one variable.

    cfg keys: code, folder, label, unit, overview (optional), master_col
    (optional), has_qc (optional, default False).
    """
    global _CFG, _ORIGINAL_INDEX
    _CFG = cfg
    _ORIGINAL_INDEX = None
    label = cfg["label"]
    plot_root = DATA_ROOT / "Plots" / cfg["folder"]
    plot_root.mkdir(parents=True, exist_ok=True)

    peri = load_alpine_perimeter()
    stations, daily = load_stations_and_daily(cfg, peri)
    if stations.empty:
        logger.error(f"No {label} stations inside the buffer")
        return

    removal_df = pd.DataFrame()
    if cfg.get("has_qc"):
        logger.info("Computing QC removal counts...")
        removal_df = compute_all_removals(stations, cfg)

    for name, pal in PALETTES.items():
        out_dir = plot_root / f"palette_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"=== Palette: {name} ===")
        plot_nearest_distance(stations, out_dir, pal, label)
        plot_elevation_hist(stations, out_dir, pal)
        plot_elev_diff(stations, out_dir, pal)
        plot_station_map(stations, out_dir, pal)
        plot_violin(stations, daily, out_dir, pal, cfg)
        plot_distribution(stations, daily, out_dir, pal, cfg)
        plot_variability(stations, daily, out_dir, pal, cfg)
        plot_holdout_timeseries(stations, daily, out_dir, pal, cfg)
        if cfg.get("has_qc") and not removal_df.empty:
            plot_qc_examples(removal_df, out_dir, pal, cfg)
            plot_removals(removal_df, out_dir, pal)

    logger.info(f"Done. Figures under {plot_root}")
