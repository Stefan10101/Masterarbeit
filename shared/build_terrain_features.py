#!/usr/bin/env python3
"""Build shared static terrain features for stations and grids.

Computes once and writes:
  Source/terrain/station_features.parquet
  Source/terrain/grid_<res>m.nc

Features:
  elev, smooth_elev (1000 m radius), tpi, slope, aspect, sinasp, cosasp,
  northness, dist_to_coast, signed_dist_to_ridge, clc_level2 (15 classes),
  basin_id, elev_above_valley, dist_to_valley
  (stations: 100 m from source DEM; grids: cell-mean at map resolution)

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


def _alpine_masks(xs, ys, buffer_m: float = 100000.0):
    """Return (domain_mask, perimeter_mask) on the given 1-D x/y coordinates.

    domain_mask: Alpine Convention perimeter + buffer_m
    perimeter_mask: strict perimeter (for basin seeding)
    Both are boolean arrays shaped (len(ys), len(xs)).
    """
    import geopandas as gpd
    from shapely.ops import unary_union
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    cache = CODE_DIR / "shared" / "plotting" / "alpine_perimeter.geojson"
    if not cache.exists():
        cache = CODE_DIR / "shared" / "splits" / "geometry_cache" / "alpine_convention_perimeter.geojson"
    if not cache.exists():
        raise FileNotFoundError("Alpine perimeter geojson not found")
    peri = gpd.read_file(cache).to_crs("EPSG:31287")
    geom = unary_union(peri.geometry)
    domain = geom.buffer(buffer_m)

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    dx = float(np.median(np.diff(xs))) if len(xs) > 1 else 1000.0
    dy = float(np.median(np.diff(ys))) if len(ys) > 1 else 1000.0
    ny, nx = len(ys), len(xs)
    y_top = float(ys.max()) + abs(dy) / 2
    y_bot = float(ys.min()) - abs(dy) / 2
    x_left = float(xs.min()) - abs(dx) / 2
    x_right = float(xs.max()) + abs(dx) / 2
    transform = from_bounds(x_left, y_bot, x_right, y_top, nx, ny)

    def _rast(g):
        m = rasterize([(g, 1)], out_shape=(ny, nx), transform=transform, fill=0, dtype=np.uint8)
        if ys[0] < ys[-1]:
            m = m[::-1]
        return m.astype(bool)

    return _rast(domain), _rast(geom)


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




def _label_from_outlets(direc: np.ndarray, outlets: np.ndarray) -> np.ndarray:
    """Label every cell by the outlet it drains to. Edges are not forced."""
    from shared.watersheds import D8
    ny, nx = direc.shape
    labels = np.zeros((ny, nx), dtype=np.int32)
    oy, ox = np.where(outlets)
    if len(oy) == 0:
        return labels
    labels[oy, ox] = np.arange(1, len(oy) + 1, dtype=np.int32)

    yy, xx = np.indices((ny, nx))
    di = np.array([d[0] for d in D8], dtype=np.int32)
    dj = np.array([d[1] for d in D8], dtype=np.int32)
    d = np.clip(direc, 0, 7)
    flows = direc >= 0
    down_y = np.clip(yy + di[d], 0, ny - 1)
    down_x = np.clip(xx + dj[d], 0, nx - 1)
    down = (down_y * nx + down_x).ravel()
    self_idx = (yy * nx + xx).ravel()
    down = np.where(flows.ravel(), down, self_idx)

    flat = labels.ravel()
    pos = np.arange(flat.size, dtype=np.int64)
    nxt = np.where(flat != 0, pos, down.astype(np.int64))
    jumps = int(np.ceil(np.log2(max(flat.size, 2)))) + 1
    for _ in range(jumps):
        nxt = nxt[nxt]
    return flat[nxt].reshape(ny, nx)



def _fill_sinks(dem: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Priority-flood sink fill. Every cell inside mask drains to the mask edge."""
    from heapq import heappush, heappop
    dem = np.asarray(dem, dtype=np.float64)
    ny, nx = dem.shape
    filled = dem.copy()
    visited = np.zeros((ny, nx), dtype=bool)
    h = []
    # Seed from mask boundary
    import scipy.ndimage as ndi
    boundary = mask & ~ndi.binary_erosion(mask)
    by, bx = np.where(boundary)
    for y, x in zip(by, bx):
        visited[y, x] = True
        heappush(h, (filled[y, x], int(y), int(x)))
    neigh = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    while h:
        z, y, x = heappop(h)
        for dy, dx in neigh:
            yy, xx = y + dy, x + dx
            if not (0 <= yy < ny and 0 <= xx < nx) or visited[yy, xx] or not mask[yy, xx]:
                continue
            visited[yy, xx] = True
            filled[yy, xx] = max(dem[yy, xx], z)
            heappush(h, (filled[yy, xx], yy, xx))
    return filled


def _pour_point_basins(direc: np.ndarray, acc: np.ndarray, elev: np.ndarray,
                       mask: np.ndarray, n_basins: int,
                       elev_cut: float = 400.0) -> np.ndarray:
    """Standard pour-point catchment delineation.

    Pour points = highest-accumulation cells on the edge of the
    elev >= elev_cut zone (where major rivers leave the mountains).
    Every cell is labelled by the pour point it drains to.
    This is the method used by TauDEM / GRASS r.water.outlet.
    """
    from scipy.ndimage import binary_dilation
    high = mask & (elev >= elev_cut)
    if not high.any():
        high = mask
    edge = high & binary_dilation(~high)
    if not edge.any():
        edge = high
    ey, ex = np.where(edge)
    if len(ey) == 0:
        return np.zeros(direc.shape, dtype=np.int32)
    order = np.argsort(acc[ey, ex])[::-1][:int(n_basins)]
    outlets = np.zeros(direc.shape, dtype=bool)
    outlets[ey[order], ex[order]] = True
    print(f"      pour points: {int(outlets.sum())} on elev>={elev_cut:.0f} m edge", flush=True)
    labels = _label_from_outlets(direc, outlets)
    # Most cells drain to an edge point that is not one of the N pour points.
    # Assign those to the nearest labelled catchment so the field covers the domain.
    unlabelled = mask & (labels == 0)
    if unlabelled.any() and (labels > 0).any():
        from scipy.ndimage import distance_transform_edt
        _, (iy, ix) = distance_transform_edt(labels == 0, return_indices=True)
        labels = np.where(unlabelled, labels[iy, ix], labels)
    n = len(np.unique(labels[labels > 0]))
    print(f"      labelled cells: {(labels > 0).sum():,} in {n} basins", flush=True)
    return labels


def _basin_continuity(labels: np.ndarray, dx: float) -> np.ndarray:
    """Continuous field: distance (m) to the nearest cell of a different basin.

    0 on the divide, large in the interior. No hard covariate jump.
    """
    from scipy.ndimage import distance_transform_edt
    labels = np.asarray(labels)
    # Boundary = cell whose 4-neighbour has a different positive label
    boundary = np.zeros(labels.shape, dtype=bool)
    for dy, dx_ in ((0, 1), (1, 0)):
        a = labels
        b = np.roll(labels, -dy, axis=0)
        b = np.roll(b, -dx_, axis=1)
        boundary |= (a > 0) & (b > 0) & (a != b)
    if not boundary.any():
        return np.zeros(labels.shape, dtype=np.float32)
    dist = distance_transform_edt(~boundary).astype(np.float32) * float(dx)
    dist = np.where(labels > 0, dist, np.nan)
    return dist


def compute_valley_fields(dem: np.ndarray, dx: float,
                          accum_pct: float = 96.0,
                          valley_pct: float = 90.0,
                          min_cells: int = 10,
                          domain_mask: np.ndarray | None = None,
                          seed_mask: np.ndarray | None = None,
                          elev_min: float = 150.0,
                          n_basins: int | None = None,
                          tpi_radius_m: float = 5000.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Basin id, elev above valley floor, distance to valley axis.

    domain_mask: compute only inside Alpine perimeter + buffer (others set to 0 / nan).
    seed_mask: basin pour points only inside the strict Alpine perimeter.
    elev_min: valley-axis and distance only consider cells above this elevation.
    n_basins: if set, merge fine basins down to this count (15 = major rivers, 50-80 = finer).
    Valley axis = high accumulation AND low TPI, on land above elev_min.
    """
    from shared.watersheds import d8_flow, flow_accum, label_basins, tpi, merge_basins
    from scipy.ndimage import distance_transform_edt, uniform_filter as ndi_uniform
    import scipy.ndimage as ndi

    dem = np.asarray(dem, dtype=np.float64)
    ny, nx = dem.shape
    if domain_mask is None:
        domain_mask = np.ones((ny, nx), dtype=bool)
    else:
        domain_mask = np.asarray(domain_mask, dtype=bool)
    if seed_mask is None:
        seed_mask = domain_mask
    else:
        seed_mask = np.asarray(seed_mask, dtype=bool) & domain_mask

    finite = np.isfinite(dem) & domain_mask
    if not finite.any():
        z = np.zeros(dem.shape, dtype=np.float32)
        return (np.zeros(dem.shape, dtype=np.int32), z, z)

    # Fill nans / outside-domain for flow (nearest finite inside domain)
    filled = dem.copy()
    if not finite.all():
        _, (iy, ix) = distance_transform_edt(~finite, return_indices=True)
        filled = dem[iy, ix]

    # Sink-fill so every cell drains, then D8 + accumulation
    filled = _fill_sinks(filled, domain_mask)
    direc = d8_flow(filled)
    acc = flow_accum(direc)
    acc = np.where(domain_mask, acc, 0.0)

    # Confluence basins: split where streams join (the "go up the river" model)
    if n_basins is not None and n_basins > 0:
        basin = _pour_point_basins(direc, acc, filled, domain_mask, int(n_basins), elev_cut=400.0)
        basin = np.where(domain_mask, basin, 0).astype(np.int32)
    else:
        basin = label_basins(direc, acc, accum_pct, min_cells=min_cells, force_edges=False)
        basin = np.where(domain_mask, basin, 0).astype(np.int32)

    # Valley floor = minimum elev inside each basin (inside domain)
    floor = np.zeros_like(filled)
    for b in np.unique(basin):
        if b <= 0:
            continue
        m = (basin == b) & domain_mask
        if m.any():
            floor[m] = np.nanmin(filled[m])
    rel = np.where(domain_mask, filled - floor, np.nan).astype(np.float32)

    # Valley axis: high accumulation AND low TPI AND low elev_above_valley.
    # Restricted to elev >= elev_min so the Po plain cannot set the threshold.
    radius = max(int(tpi_radius_m / max(dx, 1.0)), 1)
    tpi_map = tpi(filled, radius)
    cand = domain_mask & (filled >= max(elev_min, 800.0)) & np.isfinite(tpi_map)
    if cand.any():
        acc_thr = np.percentile(acc[cand], max(valley_pct, 97.0))
        tpi_thr = np.percentile(tpi_map[cand], 15.0)
        # elev_above is not yet final; use a local proxy (elev - smooth)
        elev_above_proxy = filled - ndi.uniform_filter(filled, size=radius * 2 + 1)
        ea_thr = np.percentile(elev_above_proxy[cand], 20.0)
        valley = cand & (acc >= acc_thr) & (tpi_map <= tpi_thr) & (elev_above_proxy <= ea_thr)
    else:
        valley = np.zeros_like(domain_mask)
    if not valley.any() and cand.any():
        valley = cand & (acc >= np.percentile(acc[cand], 98.0))
    dist = distance_transform_edt(~valley).astype(np.float32) * float(dx)
    dist = np.where(domain_mask, dist, np.nan)

    # Size diagnostics
    ids, counts = np.unique(basin[basin > 0], return_counts=True)
    if len(counts):
        areas = counts.astype(np.float64) * (dx / 1000.0) ** 2
        print(
            f"    basins: n={len(areas):,}  "
            f"area_km2 min/p50/p90/max = "
            f"{areas.min():.1f}/{np.median(areas):.1f}/{np.percentile(areas, 90):.1f}/{areas.max():.1f}",
            flush=True,
        )
    return basin, rel, dist



def station_valley_from_dem(xs, ys, dem_path, res_m: float = 100.0,
                            accum_pct: float = 96.0, valley_pct: float = 90.0,
                            min_cells: int = 10,
                            buffer_m: float = 5000.0,
                            n_basins: int | None = None):
    """Compute valley fields on a fine DEM resample and sample at stations.

    res_m is the working resolution (default 100 m). The source DEM is
    averaged into that grid over the station bounding box (+ buffer).
    Returns (basin_id, elev_above_valley, dist_to_valley) arrays of len(xs).
    """
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n = len(xs)
    basin = np.full(n, -1, dtype=np.int32)
    rel = np.full(n, np.nan, dtype=np.float32)
    dist = np.full(n, np.nan, dtype=np.float32)

    if not Path(dem_path).exists():
        print(f"    DEM missing ({dem_path}); valley features left empty", flush=True)
        return basin, rel, dist

    with rasterio.open(dem_path) as src:
        # Station bbox in the DEM CRS (assumed EPSG:31287, same as stations)
        x0 = float(xs.min()) - buffer_m
        x1 = float(xs.max()) + buffer_m
        y0 = float(ys.min()) - buffer_m
        y1 = float(ys.max()) + buffer_m
        # Snap to res_m
        x0 = np.floor(x0 / res_m) * res_m
        y1 = np.ceil(y1 / res_m) * res_m
        nx = int(np.ceil((x1 - x0) / res_m))
        ny = int(np.ceil((y1 - y0) / res_m))
        if nx < 2 or ny < 2:
            return basin, rel, dist
        transform = from_origin(x0, y1, res_m, res_m)
        dest = np.full((ny, nx), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=src.crs,
            resampling=Resampling.average,
            dst_nodata=np.nan,
        )

    print(f"    station valley grid {ny}x{nx} @ {res_m:.0f} m", flush=True)
    # Coordinates of the fine grid (y descending from y1)
    gx = x0 + np.arange(nx) * res_m
    gy = y1 - np.arange(ny) * res_m
    try:
        domain_mask, peri_mask = _alpine_masks(gx, gy, buffer_m=100000.0)
    except Exception as exc:
        print(f"    alpine mask failed ({exc}); using full station grid", flush=True)
        domain_mask = peri_mask = None
    b, r, d = compute_valley_fields(
        dest, res_m, accum_pct, valley_pct, min_cells,
        domain_mask=domain_mask, seed_mask=peri_mask,
        elev_min=150.0, n_basins=n_basins,
    )

    # Nearest sample
    ix = np.clip(np.round((xs - x0) / res_m).astype(int), 0, nx - 1)
    iy = np.clip(np.round((y1 - ys) / res_m).astype(int), 0, ny - 1)
    basin = b[iy, ix]
    rel = r[iy, ix]
    dist = d[iy, ix]
    return basin, rel, dist


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
        # extended, always eastward so the positive side is consistently north
        (5.80, 43.60),
        (6.60, 44.00),
        (7.10, 44.30),   # Maritime Alps
        (6.95, 44.80),
        (7.05, 45.30),
        (7.20, 45.70),
        (7.60, 45.95),   # turn onto the main E-W crest
        (8.10, 46.15),
        (8.60, 46.45),
        (9.10, 46.55),
        (9.60, 46.60),
        (10.10, 46.65),
        (10.60, 46.70),
        (11.10, 46.78),
        (11.60, 46.88),
        (12.10, 46.98),
        (12.60, 47.08),
        (13.10, 47.18),
        (13.60, 47.30),
        (14.10, 47.45),
        (14.60, 47.58),
        (15.10, 47.72),
        (15.70, 47.95),
        (16.40, 48.20),
    ]
    raw = LineString(ridge_pts)
    fr = np.linspace(0, 1, 500)
    coords = np.array([raw.interpolate(f, normalized=True).coords[0] for f in fr])
    coords[:, 0] = uniform_filter1d(coords[:, 0], size=15, mode="nearest")
    coords[:, 1] = uniform_filter1d(coords[:, 1], size=15, mode="nearest")
    return LineString(coords)


def coast_and_ridge_distances(xs, ys):
    """Return (dist_to_coast_m, signed_dist_to_ridge_m) for points in EPSG:31287.

    Ridge distance uses the smoothed polyline so Voronoi edges do not
    appear as lines in the interpolated field.
    """
    import cartopy.io.shapereader as shpreader
    from shapely.ops import nearest_points, unary_union

    coast_path = shpreader.natural_earth(resolution="10m", category="physical", name="coastline")
    coast = gpd.read_file(coast_path).to_crs("EPSG:31287")
    # Full coastline: Atlantic, North Sea, Mediterranean, Adriatic, Black Sea.
    coast_union = unary_union(coast.geometry)
    ridge = _smooth_ridge()

    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(xs, ys)], crs="EPSG:31287")
    coast_m = np.array([pt.distance(coast_union) for pt in pts])

    ridge_m = []
    for pt in pts:
        nearest = nearest_points(pt, ridge)[1]
        d = pt.distance(ridge)
        sign = 1.0 if pt.y >= nearest.y else -1.0
        ridge_m.append(d * sign * 111000.0)
    return coast_m, np.array(ridge_m)


def build_station_features(stations_method: str = "RFSI",
                           station_res: float = 100.0,
                           accum_pct: float = 96.0,
                           valley_pct: float = 90.0,
                           min_cells: int = 10,
                           n_basins: int | None = None) -> pd.DataFrame:
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
    elev = stations["elev"].to_numpy(dtype=np.float64, copy=True) if "elev" in stations.columns else np.full(n, np.nan)
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


    # Valley features at fine resolution for stations (LLOCV).
    # Grid maps keep the cell-mean version at the map resolution.
    basin_id, elev_above, dist_valley = station_valley_from_dem(
        xs, ys, dem_path,
        res_m=station_res,
        accum_pct=accum_pct,
        valley_pct=valley_pct,
        min_cells=min_cells,
        n_basins=n_basins,
    )

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
        "basin_id": basin_id,
        "elev_above_valley": elev_above,
        "dist_to_valley": dist_valley,
    })
    # Basin continuity is smooth; sample the 1 km grid field when it exists.
    out["basin_continuity"] = np.nan
    try:
        gpath = get_grid_terrain_path(1000)
        if gpath.exists():
            g = xr.open_dataset(gpath)
            if "basin_continuity" in g:
                dxg = float(np.median(np.diff(g.x.values))) if g.x.size > 1 else 1000.0
                ix = np.clip(np.round((xs - float(g.x.values[0])) / dxg).astype(int), 0, g.x.size - 1)
                iy = np.clip(np.round((ys - float(g.y.values[0])) / dxg).astype(int), 0, g.y.size - 1)
                out["basin_continuity"] = g["basin_continuity"].values[iy, ix].astype(np.float32)
            g.close()
    except Exception:
        pass
    return out


def _majority_clc(grid: xr.Dataset, resolution: int) -> np.ndarray:
    """Majority CLC level-2 class inside each grid cell (GDAL mode)."""
    from shared.clc import to_level2
    clc_path = get_landcover_path()
    ny, nx = int(grid.y.size), int(grid.x.size)
    out = np.zeros((ny, nx), dtype=np.int16)
    if not clc_path.exists():
        return out
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        x = grid.x.values
        y = grid.y.values
        dx = float(np.median(np.diff(x))) if nx > 1 else float(resolution)
        dy = float(np.median(np.diff(y))) if ny > 1 else float(resolution)
        transform = rasterio.transform.from_origin(
            float(x[0]) - abs(dx) / 2.0,
            float(y[0]) + abs(dy) / 2.0,
            abs(dx), abs(dy),
        )
        with rasterio.open(clc_path) as src:
            dest = np.zeros((ny, nx), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dest,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:31287",
                resampling=Resampling.mode,
                dst_nodata=0,
            )
        flat = dest.ravel().astype(np.int32)
        out = to_level2(flat).reshape(ny, nx).astype(np.int16)
    except Exception as exc:
        print(f"majority CLC failed ({exc}); leaving zeros", flush=True)
    return out


def build_grid_features(resolution: int = 1000, grid_method: str = "RFSI",
                        accum_pct: float = 96.0, valley_pct: float = 90.0,
                        min_cells: int = 10, n_basins: int | None = None) -> xr.Dataset:
    """Build terrain features on the master grid.

    elev comes from the master grid (area mean of the source DEM).
    smooth_elev, tpi, slope and aspect are derived from that mean elev field.
    CLC is the majority level-2 class inside each cell.
    Coast and ridge distances are computed on a subsample and interpolated.
    Output coordinates match the master grid exactly.
    """
    from scipy.ndimage import uniform_filter

    grid = xr.open_dataset(get_master_grid_path(grid_method, resolution))
    ny, nx = int(grid.y.size), int(grid.x.size)
    elev = grid["elev"].values.astype(np.float64) if "elev" in grid else np.full((ny, nx), np.nan)
    elev = np.where(elev < -200, np.nan, elev)

    # Valley / basin features on the cell-mean elev
    dx = float(np.median(np.diff(grid.x.values))) if grid.x.size > 1 else float(resolution)
    # Domain = Alpine perimeter + 100 km; seeds only inside the strict perimeter
    try:
        domain_mask, peri_mask = _alpine_masks(grid.x.values, grid.y.values, buffer_m=100000.0)
    except Exception as exc:
        print(f"    alpine mask failed ({exc}); using full grid", flush=True)
        domain_mask = peri_mask = None
    n_basins = n_basins
    basin_id, elev_above, dist_valley = compute_valley_fields(
        elev, dx, accum_pct, valley_pct, min_cells,
        domain_mask=domain_mask, seed_mask=peri_mask,
        elev_min=150.0, n_basins=n_basins,
    )
    basin_cont = _basin_continuity(basin_id, dx)

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
        # extended, always eastward so the positive side is consistently north
        (5.80, 43.60),
        (6.60, 44.00),
        (7.10, 44.30),   # Maritime Alps
        (6.95, 44.80),
        (7.05, 45.30),
        (7.20, 45.70),
        (7.60, 45.95),   # turn onto the main E-W crest
        (8.10, 46.15),
        (8.60, 46.45),
        (9.10, 46.55),
        (9.60, 46.60),
        (10.10, 46.65),
        (10.60, 46.70),
        (11.10, 46.78),
        (11.60, 46.88),
        (12.10, 46.98),
        (12.60, 47.08),
        (13.10, 47.18),
        (13.60, 47.30),
        (14.10, 47.45),
        (14.60, 47.58),
        (15.10, 47.72),
        (15.70, 47.95),
        (16.40, 48.20),
    ]
    raw = LineString(ridge_pts)
    # Resample to 500 points and smooth coordinates aggressively
    n_dense = 500
    fr = np.linspace(0, 1, n_dense)
    coords = np.array([raw.interpolate(f, normalized=True).coords[0] for f in fr])
    from scipy.ndimage import uniform_filter1d
    coords[:, 0] = uniform_filter1d(coords[:, 0], size=15, mode="nearest")
    coords[:, 1] = uniform_filter1d(coords[:, 1], size=15, mode="nearest")
    ridge_smooth = LineString(coords)

    def _signed_to_smooth(xs, ys):
        pts = gpd.GeoSeries([Point(x, y) for x, y in zip(xs, ys)], crs="EPSG:31287").to_crs("EPSG:4326")
        out = []
        for pt in pts:
            nearest = nearest_points(pt, ridge_smooth)[1]
            d = pt.distance(ridge_smooth)
            proj = ridge_smooth.project(nearest)
            eps = 0.01
            p0 = ridge_smooth.interpolate(max(proj - eps, 0.0))
            p1 = ridge_smooth.interpolate(min(proj + eps, ridge_smooth.length))
            tx, ty = p1.x - p0.x, p1.y - p0.y
            cross = tx * (pt.y - nearest.y) - ty * (pt.x - nearest.x)
            sign = 1.0 if cross >= 0 else -1.0
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

    # CLC: majority class inside each cell
    clc2 = _majority_clc(grid, resolution)

    # Ocean / flat masks
    ocean = ~np.isfinite(elev) | (elev <= 0)
    coast_full = np.where(ocean, 0.0, coast_full)
    flat = ocean | (slope < 0.5)
    # northness / aspect undefined on flat water
    # (applied below when writing sin/cos/northness)

    # Clip every core covariate to the Alpine domain (perimeter + 100 km)
    if domain_mask is not None:
        elev = np.where(domain_mask, elev, np.nan)
        smooth = np.where(domain_mask, smooth, np.nan)
        tpi = np.where(domain_mask, tpi, np.nan)
        slope = np.where(domain_mask, slope, np.nan)
        aspect = np.where(domain_mask, aspect, np.nan)
        coast_full = np.where(domain_mask, coast_full, np.nan)
        ridge_full = np.where(domain_mask, ridge_full, np.nan)
        clc2 = np.where(domain_mask, clc2, 0)

    ds = xr.Dataset(
        {
            "elev": (("y", "x"), elev.astype(np.float32)),
            "smooth_elev": (("y", "x"), smooth.astype(np.float32)),
            "tpi": (("y", "x"), tpi.astype(np.float32)),
            "slope": (("y", "x"), slope.astype(np.float32)),
            "aspect": (("y", "x"), aspect.astype(np.float32)),
            "sinasp": (("y", "x"), np.where((domain_mask if domain_mask is not None else True) & ~flat, np.sin(np.deg2rad(aspect)), np.nan).astype(np.float32)),
            "cosasp": (("y", "x"), np.where((domain_mask if domain_mask is not None else True) & ~flat, np.cos(np.deg2rad(aspect)), np.nan).astype(np.float32)),
            "northness": (("y", "x"), np.where((domain_mask if domain_mask is not None else True) & ~flat, np.cos(np.deg2rad(aspect)), np.nan).astype(np.float32)),
            "dist_to_coast": (("y", "x"), coast_full.astype(np.float32)),
            "signed_dist_to_ridge": (("y", "x"), ridge_full.astype(np.float32)),
            "clc_level2": (("y", "x"), clc2),
            "basin_id": (("y", "x"), basin_id.astype(np.int32)),
            "basin_continuity": (("y", "x"), basin_cont.astype(np.float32)),
            "elev_above_valley": (("y", "x"), elev_above.astype(np.float32)),
        },
        coords={"y": grid.y.values, "x": grid.x.values},
    )
    grid.close()
    return ds



def inspect_valley_maps(resolution: int = 1000, out_dir: Path | None = None) -> Path | None:
    """Inspection maps for all core covariates written to the grid cache.

    Writes Plots/terrain/covariates_<res>m.png (multi-panel) and keeps the
    three-panel valley figure for quick checks.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = get_grid_terrain_path(resolution)
    if not path.exists():
        print(f"no grid terrain at {path}")
        return None
    ds = xr.open_dataset(path)
    if out_dir is None:
        out_dir = CODE_DIR.parent / "Plots" / "terrain"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full covariate overview
    fields = [
        ("elev", "Elevation (m)", "terrain"),
        ("smooth_elev", "Smooth elev (m)", "terrain"),
        ("slope", "Slope (deg)", "viridis"),
        ("northness", "Northness", "RdBu_r"),
        ("tpi", "TPI (m)", "RdBu_r"),
        ("clc_level2", "CLC level-2", "tab20"),
        ("dist_to_coast", "Dist to coast (m)", "viridis"),
        ("signed_dist_to_ridge", "Signed dist to ridge (m)", "RdBu_r"),
        ("basin_id", "Basin ID (inspect)", "tab20"),
        ("basin_continuity", "Basin continuity (m)", "viridis"),
        ("elev_above_valley", "Elev above valley (m)", "terrain"),
    ]
    n = len(fields)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, title, cmap) in zip(axes, fields):
        if name not in ds:
            ax.set_title(f"{title}\n(missing)")
            ax.axis("off")
            continue
        da = ds[name]
        im = ax.pcolormesh(da.x, da.y, da.values, cmap=cmap, shading="auto")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, shrink=0.6)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Core covariates @ {resolution} m (Alpine domain)")
    overview = out_dir / f"covariates_{resolution}m.png"
    fig.savefig(overview, dpi=110)
    plt.close(fig)
    print(f"wrote {overview}")

    # Keep the focused valley figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, (name, title, cmap) in zip(axes, fields[-3:]):
        if name not in ds:
            continue
        da = ds[name]
        im = ax.pcolormesh(da.x, da.y, da.values, cmap=cmap, shading="auto")
        ax.set_title(title)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"Valley features @ {resolution} m")
    valley = out_dir / f"valley_features_{resolution}m.png"
    fig.savefig(valley, dpi=120)
    plt.close(fig)
    ds.close()
    print(f"wrote {valley}")
    return overview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations-only", action="store_true")
    parser.add_argument("--grid-only", action="store_true")
    parser.add_argument("--resolution", type=int, default=1000)
    parser.add_argument("--inspect", action="store_true",
                        help="Write a three-panel map of basin_id / elev_above_valley / dist_to_valley")
    parser.add_argument("--accum-pct", type=float, default=96.0,
                        help="Pour-point percentile for basin seeding (higher = fewer, larger basins)")
    parser.add_argument("--valley-pct", type=float, default=90.0,
                        help="Accumulation percentile that defines the valley axis")
    parser.add_argument("--min-cells", type=int, default=10,
                        help="Merge basins smaller than this many cells")
    parser.add_argument("--station-res", type=float, default=100.0,
                        help="Resolution (m) for station valley features (default 100)")
    parser.add_argument("--n-basins", type=int, default=60,
                        help="Target basin count after merge (15=major rivers, 50-80=finer)")
    args = parser.parse_args()

    if not args.stations_only:
        print("Building grid features...")
        ds = build_grid_features(args.resolution, accum_pct=args.accum_pct, valley_pct=args.valley_pct, min_cells=args.min_cells, n_basins=args.n_basins)
        out = get_grid_terrain_path(args.resolution)
        ds.to_netcdf(out)
        print(f"wrote {out}")


    if not args.grid_only:
        print("Building station features...")
        sta = build_station_features(station_res=args.station_res,
                                 accum_pct=args.accum_pct,
                                 valley_pct=args.valley_pct,
                                 min_cells=args.min_cells,
                                 n_basins=args.n_basins)
        out = get_station_terrain_path()
        sta.to_parquet(out, index=False)
        print(f"wrote {out}  ({len(sta)} stations)")

    if args.inspect:
        inspect_valley_maps(args.resolution)


if __name__ == "__main__":
    main()
