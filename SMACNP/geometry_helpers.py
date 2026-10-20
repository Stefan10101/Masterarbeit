#!/usr/bin/env python3
"""Geometry helpers for SMACNP features.

- Alpine Convention perimeter (same ArcGIS URL as station_overview_plots.py)
- Natural Earth 10 m coastline for Mediterranean and Atlantic distances
- DEM-derived main Alpine ridge (high-elevation watershed divide)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import urllib.request
import tempfile

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, box
from shapely.ops import nearest_points, unary_union

PERIMETER_URL = (
    "https://services5.arcgis.com/qjjtr5pMpt3PDiiR/ArcGIS/rest/services/"
    "Alpine_Convention_Perimeter/FeatureServer/1/query?"
    "where=1%3D1&outFields=*&f=geojson"
)

CACHE_DIR = Path(__file__).resolve().parent / "geometry_cache"


def load_alpine_perimeter(cache: bool = True) -> gpd.GeoDataFrame:
    """Download (or load cached) Alpine Convention perimeter as EPSG:4326 GeoDataFrame."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "alpine_convention_perimeter.geojson"
    if cache and cache_path.exists():
        return gpd.read_file(cache_path)
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        urllib.request.urlretrieve(PERIMETER_URL, tmp.name)
        gdf = gpd.read_file(tmp.name)
    gdf = gdf.to_crs("EPSG:4326")
    if cache:
        gdf.to_file(cache_path, driver="GeoJSON")
    return gdf


def load_natural_earth_coastline() -> gpd.GeoDataFrame:
    """Load Natural Earth 10 m coastline (same source cartopy uses)."""
    import cartopy.io.shapereader as shpreader
    path = shpreader.natural_earth(resolution="10m", category="physical", name="coastline")
    return gpd.read_file(path)


def classify_coast_segments(coast: gpd.GeoDataFrame, peri: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split Natural Earth coastline into Mediterranean and Atlantic.

    Mediterranean: coast east of Gibraltar (lon > -5.5) and south of ~46.5°N
    (Mediterranean basin, including Corsica and Sardinia).
    Atlantic: everything else (west of Gibraltar or open Atlantic-facing coasts).
    """
    # Use representative point for fast vectorized classification
    reps = coast.geometry.representative_point()
    lons = reps.x.to_numpy()
    lats = reps.y.to_numpy()
    gibraltar_lon = -5.5
    med_mask = (lons > gibraltar_lon) & (lats < 46.5)
    med = coast[med_mask].copy()
    atl = coast[~med_mask].copy()
    return med, atl


def distance_to_lines(points: gpd.GeoSeries, lines: gpd.GeoDataFrame) -> np.ndarray:
    """Minimum distance (degrees, approximate) from each point to any line.
    For metre distances reproject both to an equal-area CRS first.
    """
    if lines.empty:
        return np.full(len(points), np.nan)
    union = unary_union(lines.geometry)
    dists = []
    for pt in points:
        dists.append(pt.distance(union))
    return np.asarray(dists, dtype=np.float64)


def signed_distance_to_ridge(points: gpd.GeoSeries, ridge: LineString) -> np.ndarray:
    """Signed distance to ridge polyline.
    Positive = north of local ridge tangent, negative = south.
    Distance is in the CRS units of the geometries (reproject for metres).
    """
    if ridge is None or ridge.is_empty:
        return np.zeros(len(points))
    dists = []
    signs = []
    for pt in points:
        nearest = nearest_points(pt, ridge)[1]
        # Approximate local tangent
        # Project a small offset north and see which side
        d = pt.distance(ridge)
        # Sign by latitude relative to nearest point (simple, works for E-W trending ridge)
        sign = 1.0 if pt.y >= nearest.y else -1.0
        dists.append(d * sign)
    return np.asarray(dists, dtype=np.float64)


def derive_main_alpine_ridge() -> LineString:
    """Approximate main Alpine ridge (Alpenhauptkamm) as a polyline.

    Points chosen to follow the yellow crest in the reference image and
    standard Alpine geography: Maritime/Cottian Alps → Mont Blanc →
    Pennine Alps (Matterhorn/Monte Rosa) → Gotthard → Ötztal/Stubai →
    Hohe Tauern → eastern Alps. Not a surveyed shapefile; a reproducible
    geometric approximation for distance calculations.
    """
    # (lon, lat) turning points
    pts = [
        (6.85, 44.15),
        (6.90, 44.50),
        (6.85, 44.90),
        (6.80, 45.30),
        (6.85, 45.70),
        (6.90, 45.83),   # Mont Blanc sector
        (7.20, 45.90),
        (7.50, 45.95),
        (7.70, 45.98),   # Matterhorn / Monte Rosa
        (8.00, 46.10),
        (8.30, 46.30),
        (8.60, 46.50),   # Gotthard
        (9.00, 46.55),
        (9.50, 46.60),
        (10.00, 46.65),
        (10.50, 46.70),
        (11.00, 46.75),
        (11.50, 46.85),
        (12.00, 46.95),
        (12.50, 47.05),
        (13.00, 47.15),
        (13.50, 47.25),
        (14.00, 47.40),
        (14.50, 47.55),
        (15.00, 47.70),
    ]
    return LineString(pts)


def plot_ridge_figure(
    peri: gpd.GeoDataFrame,
    ridge: LineString,
    out_path: Path,
    med_coast: Optional[gpd.GeoDataFrame] = None,
    atl_coast: Optional[gpd.GeoDataFrame] = None,
) -> None:
    """Save a figure showing the Alpine perimeter, schematic ridge, and coast segments."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    peri.plot(ax=ax, facecolor="none", edgecolor="#c0392b", linewidth=1.2, label="Alpine Convention")
    if ridge is not None:
        gpd.GeoSeries([ridge]).plot(ax=ax, color="#1a5276", linewidth=2.0, label="Main Alpine ridge (schematic)")
    if med_coast is not None and not med_coast.empty:
        med_coast.plot(ax=ax, color="#148f77", linewidth=0.6, label="Mediterranean coast")
    if atl_coast is not None and not atl_coast.empty:
        atl_coast.plot(ax=ax, color="#1f618d", linewidth=0.6, label="Atlantic coast")
    ax.set_title("Alpine Convention perimeter, schematic main ridge, and coast segments")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
