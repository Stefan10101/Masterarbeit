#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General station overview plots (all unique stations, not variable-specific).

Uses QC/stations_overview.csv as the source of truth and the COP30 DEM
for elevation differences. Map style matches overview_domain_map.py
(blackscale DEM, Alpine Convention perimeter, no inset).
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
from rasterio.enums import Resampling
from rasterio.warp import reproject

CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR))

from paths import DATA_ROOT, get_dem_path, get_metadata_path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
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
    "Mountain": "p",
}
ELEV_HIST_BIN = 200

PLOT_DIR = DATA_ROOT / "Plots" / "Overview"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

OVERVIEW_CSV = get_metadata_path()
DEM_PATH = get_dem_path()
DEM_BG_PATH = DEM_PATH.with_name("COP30_mosaic_EPSG31287_unclipped.tif")
if not DEM_BG_PATH.exists():
    DEM_BG_PATH = DEM_PATH

PERIMETER_URL = (
    "https://services5.arcgis.com/qjjtr5pMpt3PDiiR/ArcGIS/rest/services/"
    "Alpine_Convention_Perimeter/FeatureServer/1/query?"
    "where=1%3D1&outFields=*&f=geojson"
)

# Single palette for now; swap once the temperature run has been reviewed.
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
PAL = PALETTES["black"]  # default; main() loops over all

# Distinct but muted colours for the variable-count map.
VAR_COLORS = {
    1: "#8da0b5",
    2: "#5a7a9a",
    3: "#3d5a73",
    4: "#2c4558",
    5: "#1a2f3d",
}


PROVIDER_LABELS = {
    "DWD": "German Weather Service",
    "MeteoFrance": "MeteoFrance",
    "ZAMG": "Geosphere",
    "zamg": "Geosphere",
    "LWD": "Avalanche Service Tirol",
    "Meteosuisse": "MeteoSuisse",
    "MeteoSuisse": "MeteoSuisse",
    "ARSO": "Slovenian Environment Agency",
    "HydroT": "Hydrological Service Tirol",
    "HYDRO_TIROL": "Hydrological Service Tirol",
    "HydroVO": "Hydrological Service Vorarlberg",
    "HYDRO_VORARLBERG": "Hydrological Service Vorarlberg",
    "Suedtirol": "Autonomous Province South Tyrol",
    "Suedtriol": "Autonomous Province South Tyrol",
}


def rename_provider(name: str) -> str | None:
    if not isinstance(name, str) or name.strip() == "" or name == "Unknown":
        return None
    return PROVIDER_LABELS.get(name, name)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
    global _BUFFER_GEOM
    _BUFFER_GEOM = buffered_m

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
_BUFFER_GEOM = None


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


def load_all_stations() -> pd.DataFrame:
    df = pd.read_csv(OVERVIEW_CSV)
    df["n_variables"] = df["variables"].fillna("").apply(
        lambda s: len([v for v in s.split(",") if v.strip()])
    )
    df["elev_class"] = df["hoehe"].apply(elev_class)
    logger.info(f"Loaded {len(df)} stations from {OVERVIEW_CSV.name}")
    return df


def sample_dem_at_points(lats, lons) -> np.ndarray:
    if not DEM_PATH.exists():
        logger.warning(f"DEM not found: {DEM_PATH}")
        return np.full(len(lats), np.nan)
    from pyproj import Transformer
    with rasterio.open(DEM_BG_PATH) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)
        samples = list(src.sample(list(zip(xs, ys))))
        vals = np.array([s[0] if s is not None else np.nan for s in samples], dtype=float)
        nodata = src.nodata if src.nodata is not None else -9999
        vals = np.where((vals == nodata) | (vals < -200), np.nan, vals)
    return vals


def load_alpine_perimeter() -> gpd.GeoDataFrame:
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        urllib.request.urlretrieve(PERIMETER_URL, tmp.name)
        gdf = gpd.read_file(tmp.name)
    return gdf.to_crs("EPSG:4326")


def load_dem_background(extent, target_res=0.0018):
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


def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.grid(True, alpha=0.25, linewidth=0.4, color="0.7")
    ax.tick_params(labelsize=8)


def fine_counts(values, bin_width, align="left"):
    """Fixed-width histogram. Points sit on a bin edge, not the centre.
    align='left':  bin [n, n+w) is plotted at n   (0 °C bin on the 0 line)
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


def _base_map(ax, peri, main_extent):
    """Shared blackscale background used by both maps."""
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
    ax.annotate("N", xy=(0.965, 0.94), xytext=(0.965, 0.88),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.15), zorder=8)
    scale_lon = main_extent[0] + 0.5
    scale_lat = main_extent[2] + 0.28
    ax.plot([scale_lon, scale_lon + 1.2], [scale_lat, scale_lat],
            color="black", linewidth=1.4, transform=ccrs.PlateCarree(), zorder=7)
    ax.text(scale_lon + 0.6, scale_lat + 0.1, "100 km", ha="center", fontsize=7,
            transform=ccrs.PlateCarree(), zorder=7)



def keep_inside_buffer(stations: pd.DataFrame) -> pd.DataFrame:
    """Drop any station that is not inside the stored Alpine buffer."""
    if _BUFFER_GEOM is None or stations.empty:
        return stations
    gdf = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:31287")
    inside = gdf.intersects(_BUFFER_GEOM)
    return stations.loc[inside.to_numpy()].copy()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_station_map(stations: pd.DataFrame):
    stations = keep_inside_buffer(stations)
    peri = load_alpine_perimeter()
    main_extent = FIXED_MAP_EXTENT

    fig = plt.figure(figsize=(12.0, 8.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=ccrs.PlateCarree())
    _base_map(ax, peri, main_extent)

    logger.info(f"Station map plotting {len(stations)} filtered stations")
    handles = []
    for name, lo, hi in ELEV_BINS:
        sub = stations[stations["elev_class"] == name]
        if sub.empty:
            continue
        ax.scatter(
            sub["lon"], sub["lat"],
            marker=ELEV_MARKERS[name], s=14, c=MAP_CLASS_COLORS[name],
            edgecolors="white", linewidths=0.25,
            transform=ccrs.PlateCarree(), zorder=6,
        )
        handles.append(plt.Line2D(
            [0], [0], marker=ELEV_MARKERS[name], color="w",
            markerfacecolor=MAP_CLASS_COLORS[name], markeredgecolor="white",
            markersize=7, label=f"{name} ({lo}–{hi} m)",
        ))
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.9,
              fontsize=8, title="Station elevation", title_fontsize=8)
    save(fig, PLOT_DIR / "01_all_stations_map.png")


def plot_network_counts(stations: pd.DataFrame):
    labels = stations["network_provider"].map(rename_provider)
    counts = labels.dropna().value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(counts))))
    ax.barh(counts.index, counts.values, color=PAL["fill"], edgecolor=PAL["edge"], linewidth=0.4)
    ax.set_xlabel("Number of stations")
    for i, v in enumerate(counts.values):
        ax.text(v + max(counts.values) * 0.01, i, str(v), va="center", fontsize=7, color=PAL["line"])
    ax.legend([plt.Rectangle((0, 0), 1, 1, facecolor=PAL["fill"], edgecolor=PAL["edge"])],
              ["Stations"], frameon=False, fontsize=8)
    style_axes(ax)
    save(fig, PLOT_DIR / "02_stations_per_network.png")


def plot_variables_per_station(stations: pd.DataFrame):
    """Stacked bar: stations by number of variables, coloured by network provider."""
    df = stations.copy()
    df["provider_label"] = df["network_provider"].map(rename_provider)
    df = df.dropna(subset=["provider_label"])
    ct = pd.crosstab(df["n_variables"], df["provider_label"])
    ct = ct.reindex(sorted(ct.index))
    # Stable colour cycle for providers
    providers = list(ct.columns)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(providers), 1)))
    fig, ax = plt.subplots(figsize=(7, 5))
    bottom = np.zeros(len(ct))
    x = np.arange(len(ct))
    for i, prov in enumerate(providers):
        vals = ct[prov].values.astype(float)
        ax.bar(x, vals, bottom=bottom, color=cmap[i], edgecolor="white", linewidth=0.3, label=prov)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in ct.index])
    ax.set_xlabel("Number of variables measured")
    ax.set_ylabel("Number of stations")
    ax.legend(frameon=True, framealpha=0.92, fontsize=6, loc="upper right", title="Network", title_fontsize=6.5, borderpad=0.3, labelspacing=0.25, handlelength=1.0, handletextpad=0.4)
    style_axes(ax)
    save(fig, PLOT_DIR / "03a_variables_per_station_stacked.png")

    # Map coloured by number of variables
    peri = load_alpine_perimeter()
    main_extent = FIXED_MAP_EXTENT

    stations = keep_inside_buffer(stations)
    fig = plt.figure(figsize=(12.0, 8.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=ccrs.PlateCarree())
    _base_map(ax, peri, main_extent)

    handles = []
    for n in sorted(stations["n_variables"].dropna().unique()):
        sub = stations[stations["n_variables"] == n]
        color = VAR_COLORS.get(int(n), PAL["primary"])
        ax.scatter(
            sub["lon"], sub["lat"],
            marker="o", s=12, c=color, edgecolors="white", linewidths=0.2,
            transform=ccrs.PlateCarree(), zorder=6,
        )
        handles.append(plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=color,
            markeredgecolor="white", markersize=6, label=f"{int(n)} variable{'s' if n != 1 else ''}",
        ))
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.9,
              fontsize=8, title="Variables measured", title_fontsize=8)
    save(fig, PLOT_DIR / "03b_variables_per_station_map.png")


def plot_elevation_hist(stations: pd.DataFrame):
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
        ax.plot(x, y, color=PAL["primary"], linewidth=1.1, marker="o", markersize=3.0, markerfacecolor=PAL["primary"], markeredgecolor="white", markeredgewidth=0.3, label="Stations")
    ax.set_xlabel("Elevation (m)")
    ax.set_ylabel("Number of stations")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axes(ax)
    save(fig, PLOT_DIR / "04_elevation_distribution.png")


def plot_elev_diff(stations: pd.DataFrame):
    valid = stations.dropna(subset=["lat", "lon", "hoehe"])
    if valid.empty:
        return
    dem_elev = sample_dem_at_points(valid["lat"].values, valid["lon"].values)
    diff = valid["hoehe"].values - dem_elev
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        logger.warning("No valid DEM samples")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x, y = fine_counts(diff, bin_width=5, align='right')
    if x is not None:
        ax.plot(x, y, color=PAL["primary"], linewidth=1.1, marker="o", markersize=3.0, markerfacecolor=PAL["primary"], markeredgecolor="white", markeredgewidth=0.3)
    ax.axvline(0, color=PAL["line"], linestyle="-", linewidth=0.8, label="Zero")
    ax.axvline(np.median(diff), color=PAL["accent"], linestyle="--", linewidth=1.2,
               label=f"Median {np.median(diff):.0f} m")
    ax.set_xlabel("Metadata elevation − DEM elevation (m)")
    ax.set_ylabel("Number of stations")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axes(ax)
    save(fig, PLOT_DIR / "05_elevation_difference.png")


def main():
    global PAL, PLOT_DIR
    stations = load_all_stations()
    if stations.empty:
        logger.error("No stations in stations_overview.csv")
        return
    peri = load_alpine_perimeter()
    stations = filter_to_alpine_buffer(stations, peri)
    logger.info(f"Overview analysis uses {len(stations)} stations inside the buffer")
    if stations.empty:
        logger.error("No stations inside the Alpine buffer")
        return
    global FIXED_MAP_EXTENT
    FIXED_MAP_EXTENT = (4.0, 17.0, 43.0, 49.0)
    logger.info(f"Fixed map extent: lon 4–17, lat 43–49")

    for name, pal in PALETTES.items():
        PAL = pal
        out = PLOT_DIR / f"palette_{name}"
        out.mkdir(parents=True, exist_ok=True)
        orig = PLOT_DIR
        PLOT_DIR = out
        logger.info(f"=== Palette: {name} ===")
        plot_station_map(stations)
        plot_network_counts(stations)
        plot_variables_per_station(stations)
        plot_elevation_hist(stations)
        plot_elev_diff(stations)
        PLOT_DIR = orig
    logger.info(f"Done. Figures under {PLOT_DIR}")


if __name__ == "__main__":
    main()
