#!/usr/bin/env python3
"""
plot_terrain_covariates.py

Produces one full-scale figure per core topographic covariate used by the
interpolation methods. Loads the shared grid terrain cache
(Source/terrain/grid_<res>m.nc) and writes individual high-resolution maps
suitable for thesis inclusion.

Style matches the overview domain maps: blackscale DEM background, Alpine
Convention perimeter + 100 km buffer, coastline/rivers. The covariate is
drawn only where it is defined (inside the domain); outside, the DEM
background remains visible.

Covariates (matching the DEM section of the thesis):
  elev, smooth_elev, slope, northness, tpi,
  signed_dist_to_ridge, dist_to_coast, basin_continuity, elev_above_valley

Output: Plots/terrain/<covariate>_<res>m.png  (and optional PDF)
"""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from paths import get_grid_terrain_path, get_plots_root, get_dem_path

try:
    import xarray as xr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject
except ImportError as e:
    raise SystemExit(f"missing dependency: {e}")


COVARIATES = [
    ("elev", "Elevation", "m", "terrain", False, ""),
    ("smooth_elev", "Smoothed Elevation", "m", "terrain", False, ""),
    ("slope", "Slope", "°", "viridis", False, ""),
    ("northness", "Northness", "", "RdBu_r", True, ""),
    ("tpi", "Topographic Position Index", "m", "RdBu_r", True, ""),
    ("signed_dist_to_ridge", "Signed Distance to Ridge", "m", "RdBu_r", True, ""),
    ("dist_to_coast", "Distance to Coast", "m", "viridis", False, ""),
    ("basin_continuity", "Basin Continuity", "m", "viridis", False, ""),
    ("elev_above_valley", "Elevation above Valley Floor", "m", "terrain", False, ""),
]

BUFFER_KM = 100.0
DEM_VMIN, DEM_VMAX = 0.0, 3200.0


def _load_perimeter() -> gpd.GeoDataFrame | None:
    """Alpine Convention perimeter in EPSG:4326."""
    candidates = [
        CODE_DIR / "shared" / "plotting" / "alpine_perimeter.geojson",
        CODE_DIR / "shared" / "splits" / "geometry_cache" / "alpine_convention_perimeter.geojson",
    ]
    for p in candidates:
        if p.exists():
            return gpd.read_file(p).to_crs("EPSG:4326")
    return None


def _domain_extent(peri: gpd.GeoDataFrame, pad_deg: float = 0.4) -> tuple:
    """Lon/lat extent of perimeter + buffer, with a small pad."""
    peri_m = peri.to_crs("EPSG:31287")
    buf = peri_m.buffer(BUFFER_KM * 1000).union_all()
    buf_wgs = gpd.GeoDataFrame(geometry=[buf], crs="EPSG:31287").to_crs("EPSG:4326")
    minx, miny, maxx, maxy = buf_wgs.total_bounds
    return (minx - pad_deg, maxx + pad_deg, miny - pad_deg, maxy + pad_deg)


def _load_dem_background(extent: tuple, target_res: float = 0.002) -> tuple[np.ndarray, tuple]:
    """Blackscale DEM on a lon/lat grid covering the extent. Returns (array, extent)."""
    lon_min, lat_min, lon_max, lat_max = extent
    bounds = (lon_min, lat_min, lon_max, lat_max)
    dem_path = get_dem_path()
    if not dem_path.exists():
        return None, None

    with rasterio.open(dem_path) as src:
        width = max(800, min(int((bounds[2] - bounds[0]) / target_res), 4000))
        height = max(600, min(int((bounds[3] - bounds[1]) / target_res), 3200))
        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
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
    return arr, (bounds[0], bounds[2], bounds[1], bounds[3])


def _robust_limits(arr: np.ndarray, symmetric: bool, pct: float = 2.0) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [pct, 100.0 - pct])
    if symmetric:
        lim = max(abs(lo), abs(hi))
        return -lim, lim
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def plot_one(
    da: xr.DataArray,
    dem_arr: np.ndarray | None,
    dem_extent: tuple | None,
    map_extent: tuple,
    title: str,
    unit: str,
    cmap: str,
    symmetric: bool,
    perimeter: gpd.GeoDataFrame | None,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """One figure: blackscale DEM background + covariate overlay inside domain.

    The covariate grid is in EPSG:31287. Cartopy handles the transform to
    PlateCarree so the data land on the correct lon/lat location.
    """
    values = da.values.astype(np.float64)
    xs = da.x.values
    ys = da.y.values
    vmin, vmax = _robust_limits(values, symmetric)

    src_crs = ccrs.epsg(31287)

    fig = plt.figure(figsize=(10.5, 7.2))
    ax = fig.add_axes([0.02, 0.02, 0.78, 0.96], projection=ccrs.PlateCarree())
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    ax.set_facecolor("#f5f5f5")

    if dem_arr is not None and dem_extent is not None:
        ax.imshow(
            dem_arr, origin="upper", extent=dem_extent,
            cmap="gray_r", vmin=DEM_VMIN, vmax=DEM_VMAX,
            transform=ccrs.PlateCarree(), zorder=1, interpolation="bilinear",
        )

    im = ax.pcolormesh(
        xs, ys, values,
        cmap=cmap, vmin=vmin, vmax=vmax,
        transform=src_crs, zorder=2, shading="auto",
    )

    if perimeter is not None:
        peri_m = perimeter.to_crs("EPSG:31287")
        buf = peri_m.buffer(BUFFER_KM * 1000).union_all()
        buf_gdf = gpd.GeoDataFrame(geometry=[buf], crs="EPSG:31287").to_crs("EPSG:4326")
        buf_gdf.plot(
            ax=ax, facecolor="none", edgecolor="#7aa2d4", linewidth=0.7,
            linestyle="--", transform=ccrs.PlateCarree(), zorder=4,
        )
        perimeter.plot(
            ax=ax, facecolor="none", edgecolor="#1a365d", linewidth=1.0,
            transform=ccrs.PlateCarree(), zorder=5,
        )

    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.3, edgecolor="0.35", zorder=3)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#e6f3ff", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.RIVERS.with_scale("10m"), linewidth=0.25, edgecolor="#4a90d9", zorder=3)
    ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="#a8d5e5", edgecolor="#4a90d9",
                   linewidth=0.2, zorder=3)

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="0.5", alpha=0.4, linestyle="-",
        xlocs=np.arange(4, 18, 2), ylocs=np.arange(43, 50, 1),
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    cax = fig.add_axes([0.84, 0.18, 0.018, 0.62])
    cbar = fig.colorbar(im, cax=cax)
    if unit:
        cbar.set_label(unit, fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(title, fontsize=11, pad=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-scale terrain covariate maps")
    parser.add_argument("--resolution", type=int, default=1000, help="Grid resolution in metres")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    parser.add_argument("--pdf", action="store_true", help="Also write PDF versions")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override output directory")
    args = parser.parse_args()

    path = get_grid_terrain_path(args.resolution)
    if not path.exists():
        raise SystemExit(f"terrain grid not found: {path}\nRun shared/build_terrain_features.py first.")

    ds = xr.open_dataset(path)
    perimeter = _load_perimeter()
    if perimeter is None:
        raise SystemExit("Alpine perimeter geojson not found")

    extent = _domain_extent(perimeter)
    dem_arr, dem_extent = _load_dem_background(extent)
    if dem_arr is None:
        print("warning: DEM background not found; maps will have no blackscale fill")

    out_dir = args.out_dir or (get_plots_root() / "terrain")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, title, unit, cmap, symmetric, _note in COVARIATES:
        if name not in ds:
            print(f"skip {name}: not in {path}")
            continue
        da = ds[name]
        stem = f"{name}_{args.resolution}m"
        png = out_dir / f"{stem}.png"
        plot_one(
            da, dem_arr, dem_extent, extent,
            title, unit, cmap, symmetric, perimeter, png, dpi=args.dpi,
        )
        if args.pdf:
            pdf = out_dir / f"{stem}.pdf"
            plot_one(
                da, dem_arr, dem_extent, extent,
                title, unit, cmap, symmetric, perimeter, pdf, dpi=args.dpi,
            )

    ds.close()
    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()