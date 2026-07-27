#!/usr/bin/env python3
"""
grid_core.py
Core functions for station projection and grid creation.
"""

import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
from scipy.spatial import cKDTree
import xarray as xr
from pathlib import Path
from typing import Optional, Tuple


def project_stations(
    metadata_path: Path,
    projected_crs: str,
    add_network_provider: bool = True
) -> pd.DataFrame:
    """Load stations, project them, and add network_provider if available."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    stations = pd.read_csv(metadata_path, encoding="utf-8")

    transformer = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    x, y = transformer.transform(stations["lon"].values, stations["lat"].values)

    stations["x"] = x
    stations["y"] = y
    stations["elev"] = stations.get("hoehe", stations.get("elev", np.nan))

    if add_network_provider:
        overview_path = metadata_path.parent / "stations_overview.csv"
        if overview_path.exists():
            try:
                overview = pd.read_csv(overview_path, usecols=["station_name", "network_provider"])
                stations = stations.merge(overview, on="station_name", how="left")
            except Exception:
                pass

    final_cols = ["station_name", "x", "y", "elev", "lat", "lon"]
    if "network_provider" in stations.columns:
        final_cols.append("network_provider")

    return stations[[c for c in final_cols if c in stations.columns]]


def create_grid_from_dem(
    dem_path: Path,
    resolution_m: int,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    projected_crs: str = "EPSG:31287",
    domain: str = "master"
) -> xr.Dataset:
    """
    Create regular target grid from DEM (robust version for master grids).
    """
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    with rasterio.open(dem_path) as src:
        nodata = float(src.nodata) if src.nodata is not None else -9999.0

        if bbox is None:
            # Master grid: let rasterio compute the optimal transform + shape
            # in the target projected CRS at the desired resolution
            dst_transform, width, height = rasterio.warp.calculate_default_transform(
                src.crs,
                projected_crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=resolution_m
            )
            xmin, ymax = dst_transform.c, dst_transform.f          # top-left
            # Recompute bounds from the new transform for clarity
            xmax = xmin + width * resolution_m
            ymin = ymax - height * resolution_m
        else:
            xmin, ymin, xmax, ymax = bbox
            width = int(np.ceil((xmax - xmin) / resolution_m))
            height = int(np.ceil((ymax - ymin) / resolution_m))
            dst_transform = rasterio.transform.from_origin(
                xmin, ymax, resolution_m, -resolution_m
            )

        dem_data = np.full((height, width), nodata, dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=dem_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=projected_crs,
            resampling=Resampling.bilinear,
            dst_nodata=nodata
        )

        # Robust mask — we already know this works
# Replace the current mask line with this more precise version
        mask = np.isfinite(dem_data) & (np.abs(dem_data - nodata) > 1e-6)

        x_coords = np.arange(width) * dst_transform.a + dst_transform.c + dst_transform.a / 2.0
        y_coords = np.arange(height) * dst_transform.e + dst_transform.f + dst_transform.e / 2.0

        ds = xr.Dataset(
            {
                "mask": (("y", "x"), mask.astype(bool)),
                "elev": (("y", "x"), dem_data),
            },
            coords={"x": ("x", x_coords), "y": ("y", y_coords)},
            attrs={
                "resolution_m": resolution_m,
                "crs": projected_crs,
                "source_dem": str(dem_path),
                "domain": domain,
                "created": pd.Timestamp.now().isoformat()
            }
        )
        return ds
def save_grid(ds: xr.Dataset, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path, engine="netcdf4")
    print(f"  Saved grid: {output_path} ({ds.sizes['x']} x {ds.sizes['y']} cells)")


def build_kdtree(stations: pd.DataFrame) -> cKDTree:
    coords = np.vstack([stations["x"].values, stations["y"].values]).T
    return cKDTree(coords)


def add_landcover_to_stations(stations: pd.DataFrame, landcover_path: Path) -> pd.DataFrame:
    """Sample landcover values for station locations."""
    if not landcover_path.exists():
        print("  [Warning] Landcover file not found. Setting clc_code = -9999")
        stations["clc_code"] = -9999
        return stations

    print(f"  Sampling landcover from: {landcover_path.name}")
    with rasterio.open(landcover_path) as src:
        coords = list(zip(stations["x"], stations["y"]))
        values = list(src.sample(coords))
        stations["clc_code"] = [val[0] if val[0] is not None else -9999 for val in values]

    print("  Landcover sampling done.")
    return stations