#!/usr/bin/env python3
"""Build shared static terrain features for stations and grids.

Computes once and writes:
  Source/terrain/station_features.parquet
  Source/terrain/grid_<res>m.nc

Features:
  elev, smooth_elev (1000 m radius), tpi, slope, aspect, sinasp, cosasp,
  northness, dist_to_coast, signed_dist_to_ridge, clc_level2 (15 classes)

Other methods load these via paths.get_station_terrain_path() and
paths.get_grid_terrain_path(resolution).
"""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from paths import (
    get_dem_path,
    get_landcover_path,
    get_domain_stations_path,
    get_master_grid_path,
    get_station_terrain_path,
    get_grid_terrain_path,
)

try:
    import xarray as xr
    import rasterio
    from rasterio.windows import Window
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError as e:
    raise SystemExit(f"missing dependency: {e}")

# CLC level-2 grouping (first two digits of the 3-digit CLC code)
# 15 classes: 11,12,13,14,21,22,23,24,31,32,33,41,42,51,52
CLC_LEVEL2 = {
    111: 11, 112: 11,
    121: 12, 122: 12, 123: 12, 124: 12,
    131: 13, 132: 13, 133: 13,
    141: 14, 142: 14,
    211: 21, 212: 21, 213: 21,
    221: 22, 222: 22, 223: 22,
    231: 23,
    241: 24, 242: 24, 243: 24, 244: 24,
    311: 31, 312: 31, 313: 31,
    321: 32, 322: 32, 323: 32, 324: 32,
    331: 33, 332: 33, 333: 33, 334: 33, 335: 33,
    411: 41, 412: 41,
    421: 42, 422: 42, 423: 42,
    511: 51, 512: 51,
    521: 52, 522: 52, 523: 52,
}


def clc_to_level2(code: int) -> int:
    return CLC_LEVEL2.get(int(code), 0)


def sample_dem_window(src, x: float, y: float, radius_m: float):
    """Return (mean_elev, centre_elev, slope_deg, aspect_deg) for one point."""
    try:
        row, col = src.index(x, y)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan
    # Skip points outside the raster
    if not (0 <= row < src.height and 0 <= col < src.width):
        return np.nan, np.nan, np.nan, np.nan
    res = abs(src.res[0])
    half = max(1, int(round(radius_m / res)))
    r0 = max(0, row - half)
    r1 = min(src.height, row + half + 1)
    c0 = max(0, col - half)
    c1 = min(src.width, col + half + 1)
    if r1 <= r0 or c1 <= c0:
        return np.nan, np.nan, np.nan, np.nan
    window = Window(c0, r0, c1 - c0, r1 - r0)
    data = src.read(1, window=window).astype(np.float64)
    nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    data[data < -200] = np.nan
    valid = data[~np.isnan(data)]
    if len(valid) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean_elev = float(np.mean(valid))
    # centre
    cr = min(half, data.shape[0] // 2)
    cc = min(half, data.shape[1] // 2)
    centre = data[cr, cc] if not np.isnan(data[cr, cc]) else mean_elev
    # simple gradient for slope/aspect
    if data.shape[0] > 2 and data.shape[1] > 2:
        gy, gx = np.gradient(np.nan_to_num(data, nan=mean_elev), res, res)
        slope = float(np.degrees(np.arctan(np.hypot(gx[cr, cc], gy[cr, cc]))))
        aspect = float((np.degrees(np.arctan2(-gx[cr, cc], gy[cr, cc])) + 360.0) % 360.0)
    else:
        slope, aspect = np.nan, np.nan
    return mean_elev, float(centre), slope, aspect


def _smooth_ridge():
    """Densified, coordinate-smoothed Alpine ridge. Shared by stations and grid."""
    from shapely.geometry import LineString
    from scipy.ndimage import uniform_filter1d
    ridge_pts = [
        (6.85, 44.15), (6.90, 44.50), (6.85, 44.90), (6.80, 45.30),
        (6.85, 45.70), (6.90, 45.83), (7.20, 45.90), (7.50, 45.95),
        (7.70, 45.98), (8.00, 46.10), (8.30, 46.30), (8.60, 46.50),
        (9.00, 46.55), (9.50, 46.60), (10.00, 46.65), (10.50, 46.70),
        (11.00, 46.75), (11.50, 46.85), (12.00, 46.95), (12.50, 47.05),
        (13.00, 47.15), (13.50, 47.25), (14.00, 47.40), (14.50, 47.55),
        (15.00, 47.70),
    ]
    raw = LineString(ridge_pts)
    fr = np.linspace(0, 1, 500)
    coords = np.array([raw.interpolate(f, normalized=True).coords[0] for f in fr])
    coords[:, 0] = uniform_filter1d(coords[:, 0], size=41, mode="nearest")
    coords[:, 1] = uniform_filter1d(coords[:, 1], size=41, mode="nearest")
    return LineString(coords)


def coast_and_ridge_distances(xs, ys):
    """Return (dist_to_coast_m, signed_dist_to_ridge_m) for points in EPSG:31287.

    Ridge distance uses the smoothed polyline so Voronoi edges do not
    appear as lines in the interpolated field.
    """
    import cartopy.io.shapereader as shpreader
    from shapely.ops import nearest_points, unary_union

    coast_path = shpreader.natural_earth(resolution="10m", category="physical", name="coastline")
    coast = gpd.read_file(coast_path)
    coast_union = unary_union(coast.geometry)
    ridge = _smooth_ridge()

    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(xs, ys)], crs="EPSG:31287").to_crs("EPSG:4326")
    coast_m = np.array([pt.distance(coast_union) for pt in pts]) * 111000.0

    ridge_m = []
    for pt in pts:
        nearest = nearest_points(pt, ridge)[1]
        d = pt.distance(ridge)
        sign = 1.0 if pt.y >= nearest.y else -1.0
        ridge_m.append(d * sign * 111000.0)
    return coast_m, np.array(ridge_m)


def build_station_features(stations_method: str = "RFSI") -> pd.DataFrame:
    stations = pd.read_parquet(get_domain_stations_path(stations_method, "full"))
    if "elev" not in stations.columns:
        if "elev_dem" in stations.columns:
            stations = stations.rename(columns={"elev_dem": "elev"})
        elif "hoehe" in stations.columns:
            stations = stations.rename(columns={"hoehe": "elev"})
    dem_path = get_dem_path()
    clc_path = get_landcover_path()
    radius = 1000.0

    xs = stations["x"].to_numpy(dtype=np.float64)
    ys = stations["y"].to_numpy(dtype=np.float64)
    n = len(stations)

    smooth = np.full(n, np.nan)
    elev = stations["elev"].to_numpy(dtype=np.float64) if "elev" in stations.columns else np.full(n, np.nan)
    slope = np.full(n, np.nan)
    aspect = np.full(n, np.nan)
    clc2 = np.zeros(n, dtype=np.int16)

    with rasterio.open(dem_path) as src:
        for i, (x, y) in enumerate(zip(xs, ys)):
            sm, ce, sl, asp = sample_dem_window(src, x, y, radius)
            smooth[i] = sm
            if np.isnan(elev[i]):
                elev[i] = ce
            slope[i] = sl
            aspect[i] = asp

    if clc_path.exists():
        with rasterio.open(clc_path) as src:
            for i, (x, y) in enumerate(zip(xs, ys)):
                try:
                    row, col = src.index(x, y)
                    val = src.read(1, window=Window(col, row, 1, 1))[0, 0]
                    clc2[i] = clc_to_level2(int(val))
                except Exception:
                    clc2[i] = 0

    coast_m, ridge_m = coast_and_ridge_distances(xs, ys)
    tpi = elev - smooth
    sinasp = np.sin(np.deg2rad(aspect))
    cosasp = np.cos(np.deg2rad(aspect))
    northness = cosasp  # positive = north-facing

    out = pd.DataFrame({
        "station_name": stations["station_name"].values,
        "x": xs,
        "y": ys,
        "elev": elev,
        "smooth_elev": smooth,
        "tpi": tpi,
        "slope": slope,
        "aspect": aspect,
        "sinasp": sinasp,
        "cosasp": cosasp,
        "northness": northness,
        "dist_to_coast": coast_m,
        "signed_dist_to_ridge": ridge_m,
        "clc_level2": clc2,
    })
    return out


def build_grid_features(resolution: int = 1000, grid_method: str = "RFSI") -> xr.Dataset:
    """Build terrain features on the master grid.

    elev comes from the master grid. smooth_elev, tpi, slope and aspect are
    derived from that elev field (no per-cell DEM windowing). Coast and ridge
    distances are computed on a subsample and interpolated to every cell.
    Output coordinates match the master grid exactly.
    """
    from scipy.ndimage import uniform_filter

    grid = xr.open_dataset(get_master_grid_path(grid_method, resolution))
    ny, nx = int(grid.y.size), int(grid.x.size)
    elev = grid["elev"].values.astype(np.float64) if "elev" in grid else np.full((ny, nx), np.nan)
    elev = np.where(elev < -200, np.nan, elev)

    # 1000 m radius at this resolution → window in cells
    dx = float(np.median(np.diff(grid.x.values))) if nx > 1 else float(resolution)
    win = max(1, int(round(1000.0 / abs(dx))))
    filled = np.where(np.isnan(elev), 0.0, elev)
    smooth = uniform_filter(filled, size=win * 2 + 1, mode="nearest")
    tpi = elev - smooth

    gy, gx = np.gradient(np.nan_to_num(elev, nan=np.nanmean(elev)), abs(dx))
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = (np.degrees(np.arctan2(-gx, gy)) + 360.0) % 360.0

    # Densify and smooth the ridge so Voronoi edges between segments
    # do not appear as lines in the interpolated field.
    from shapely.geometry import LineString
    from shapely.ops import nearest_points
    ridge_pts = [
        (6.85, 44.15), (6.90, 44.50), (6.85, 44.90), (6.80, 45.30),
        (6.85, 45.70), (6.90, 45.83), (7.20, 45.90), (7.50, 45.95),
        (7.70, 45.98), (8.00, 46.10), (8.30, 46.30), (8.60, 46.50),
        (9.00, 46.55), (9.50, 46.60), (10.00, 46.65), (10.50, 46.70),
        (11.00, 46.75), (11.50, 46.85), (12.00, 46.95), (12.50, 47.05),
        (13.00, 47.15), (13.50, 47.25), (14.00, 47.40), (14.50, 47.55),
        (15.00, 47.70),
    ]
    raw = LineString(ridge_pts)
    # Resample to 500 points and smooth coordinates aggressively
    n_dense = 500
    fr = np.linspace(0, 1, n_dense)
    coords = np.array([raw.interpolate(f, normalized=True).coords[0] for f in fr])
    from scipy.ndimage import uniform_filter1d
    coords[:, 0] = uniform_filter1d(coords[:, 0], size=41, mode="nearest")
    coords[:, 1] = uniform_filter1d(coords[:, 1], size=41, mode="nearest")
    ridge_smooth = LineString(coords)

    def _signed_to_smooth(xs, ys):
        pts = gpd.GeoSeries([Point(x, y) for x, y in zip(xs, ys)], crs="EPSG:31287").to_crs("EPSG:4326")
        out = []
        for pt in pts:
            nearest = nearest_points(pt, ridge_smooth)[1]
            d = pt.distance(ridge_smooth)
            sign = 1.0 if pt.y >= nearest.y else -1.0
            out.append(d * sign * 111000.0)
        return np.array(out)

    step = 1
    ys_idx = np.arange(0, ny, step)
    xs_idx = np.arange(0, nx, step)
    sy, sx = np.meshgrid(grid.y.values[ys_idx], grid.x.values[xs_idx], indexing="ij")
    # Coast unchanged; ridge uses the smoothed polyline
    coast_s, _ = coast_and_ridge_distances(sx.ravel(), sy.ravel())
    ridge_s = _signed_to_smooth(sx.ravel(), sy.ravel())
    coast_s = coast_s.reshape(sy.shape)
    ridge_s = ridge_s.reshape(sy.shape)

    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter
    yy, xx = np.meshgrid(grid.y.values, grid.x.values, indexing="ij")
    pts = np.column_stack([yy.ravel(), xx.ravel()])
    coast_full = RegularGridInterpolator(
        (grid.y.values[ys_idx], grid.x.values[xs_idx]), coast_s,
        bounds_error=False, fill_value=None,
    )(pts).reshape(ny, nx)
    ridge_full = RegularGridInterpolator(
        (grid.y.values[ys_idx], grid.x.values[xs_idx]), ridge_s,
        bounds_error=False, fill_value=None,
    )(pts).reshape(ny, nx)
    # Light smooth removes residual Voronoi kinks (~6 km)
    coast_full = gaussian_filter(np.nan_to_num(coast_full), sigma=3)
    ridge_full = gaussian_filter(np.nan_to_num(ridge_full), sigma=3)

    # CLC on the same subsample
    clc2 = np.zeros((ny, nx), dtype=np.int16)
    clc_path = get_landcover_path()
    if clc_path.exists():
        import rasterio
        from rasterio.windows import Window
        with rasterio.open(clc_path) as src:
            for iy in ys_idx:
                for ix in xs_idx:
                    try:
                        row, col = src.index(float(grid.x.values[ix]), float(grid.y.values[iy]))
                        if 0 <= row < src.height and 0 <= col < src.width:
                            val = src.read(1, window=Window(col, row, 1, 1))[0, 0]
                            clc2[iy, ix] = clc_to_level2(int(val))
                    except Exception:
                        pass

    ds = xr.Dataset(
        {
            "elev": (("y", "x"), elev.astype(np.float32)),
            "smooth_elev": (("y", "x"), smooth.astype(np.float32)),
            "tpi": (("y", "x"), tpi.astype(np.float32)),
            "slope": (("y", "x"), slope.astype(np.float32)),
            "aspect": (("y", "x"), aspect.astype(np.float32)),
            "sinasp": (("y", "x"), np.sin(np.deg2rad(aspect)).astype(np.float32)),
            "cosasp": (("y", "x"), np.cos(np.deg2rad(aspect)).astype(np.float32)),
            "northness": (("y", "x"), np.cos(np.deg2rad(aspect)).astype(np.float32)),
            "dist_to_coast": (("y", "x"), coast_full.astype(np.float32)),
            "signed_dist_to_ridge": (("y", "x"), ridge_full.astype(np.float32)),
            "clc_level2": (("y", "x"), clc2),
        },
        coords={"y": grid.y.values, "x": grid.x.values},
    )
    grid.close()
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations-only", action="store_true")
    parser.add_argument("--grid-only", action="store_true")
    parser.add_argument("--resolution", type=int, default=1000)
    args = parser.parse_args()

    if not args.grid_only:
        print("Building station features...")
        sta = build_station_features()
        out = get_station_terrain_path()
        sta.to_parquet(out, index=False)
        print(f"wrote {out}  ({len(sta)} stations)")

    if not args.stations_only:
        print("Building grid features...")
        ds = build_grid_features(args.resolution)
        out = get_grid_terrain_path(args.resolution)
        ds.to_netcdf(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
