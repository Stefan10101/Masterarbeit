#!/usr/bin/env python3
"""
Aggregate HydroBASINS Europe to fewer basins and plot each level
in the same style as the overview domain maps:
  blackscale DEM background, Alpine Convention perimeter + 100 km buffer,
  coastline / rivers / lakes, PlateCarree.

Expects the data extracted under Source/Basins
(run download_hydrobasins_europe.py first).

Usage (from CODE/):
  python shared/plot_hydrobasins_aggregates.py
  python shared/plot_hydrobasins_aggregates.py --level 6 --targets 60,40,20,10,8,6,4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from paths import get_source_root, get_project_root, get_dem_path  # noqa: E402


BUFFER_KM = 100.0
DEM_VMIN, DEM_VMAX = 0.0, 3200.0


def _load_perimeter():
    import geopandas as gpd
    candidates = [
        CODE_DIR / "shared" / "plotting" / "alpine_perimeter.geojson",
        CODE_DIR / "shared" / "splits" / "geometry_cache" / "alpine_convention_perimeter.geojson",
    ]
    for p in candidates:
        if p.exists():
            return gpd.read_file(p).to_crs("EPSG:4326")
    raise SystemExit("Alpine perimeter geojson not found")


def _domain_extent(peri, pad_deg: float = 0.4):
    import geopandas as gpd
    peri_m = peri.to_crs("EPSG:31287")
    buf = peri_m.buffer(BUFFER_KM * 1000).union_all()
    buf_wgs = gpd.GeoDataFrame(geometry=[buf], crs="EPSG:31287").to_crs("EPSG:4326")
    minx, miny, maxx, maxy = buf_wgs.total_bounds
    return (minx - pad_deg, maxx + pad_deg, miny - pad_deg, maxy + pad_deg)


def _load_dem_background(extent, target_res: float = 0.002):
    """Blackscale DEM on a lon/lat grid covering the extent. Returns (array, extent)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject

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


def _load_basins(level: int):
    import geopandas as gpd
    root = get_source_root() / "Basins"
    candidates = list(root.rglob(f"*lev{level:02d}*.shp")) + list(root.rglob(f"*lev{level}*.shp"))
    if not candidates:
        raise SystemExit(f"no shapefile for level {level} under {root}")
    shp = sorted(candidates, key=lambda p: len(str(p)))[0]
    print(f"loading {shp}")
    return gpd.read_file(shp)


def _aggregate(gdf, n_target: int):
    """Hierarchical clustering of centroids -> dissolve to n_target regions."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    if len(gdf) <= n_target:
        gdf = gdf.copy()
        gdf["grp"] = np.arange(1, len(gdf) + 1)
        return gdf.dissolve(by="grp").reset_index()

    cents = gdf.geometry.centroid
    xy = np.column_stack([cents.x, cents.y])
    z = linkage(pdist(xy), method="ward")
    grp = fcluster(z, t=n_target, criterion="maxclust")
    gdf = gdf.copy()
    gdf["grp"] = grp
    return gdf.dissolve(by="grp").reset_index()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, default=6, help="HydroBASINS Pfafstetter level to start from")
    p.add_argument("--targets", type=str, default="20,19,18,17,16,15,14,13,12,11,10")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--save-n", type=int, default=None,
                   help="Also write the aggregated polygons for this n as GeoJSON under Source/Basins")
    args = p.parse_args()

    targets = [int(x) for x in args.targets.split(",") if x.strip()]

    perimeter = _load_perimeter()
    extent = _domain_extent(perimeter)
    dem_arr, dem_extent = _load_dem_background(extent)
    if dem_arr is None:
        print("warning: DEM background not found; maps will have no blackscale fill")

    gdf = _load_basins(args.level)
    print(f"raw features: {len(gdf)}")
    # Keep only basins that intersect the actual domain (perimeter + 100 km buffer).
    # Bounding-box clip is not enough: most of the bbox is outside the Alps.
    import geopandas as gpd
    peri_m = perimeter.to_crs("EPSG:31287")
    domain = peri_m.buffer(BUFFER_KM * 1000).union_all()
    domain_gdf = gpd.GeoDataFrame(geometry=[domain], crs="EPSG:31287").to_crs(gdf.crs)
    # spatial index filter then precise intersection
    gdf = gdf[gdf.intersects(domain_gdf.geometry.iloc[0])].copy()
    print(f"after domain intersect: {len(gdf)}")

    out_dir = Path(args.out_dir) if args.out_dir else get_project_root() / "Plots" / "basins"
    out_dir.mkdir(parents=True, exist_ok=True)

    # domain polygon in the basins CRS for clipping the result
    domain_geom = domain_gdf.geometry.iloc[0]

    for n in targets:
        print(f"aggregating to {n} …")
        agg = _aggregate(gdf, n)
        # clip dissolved regions to the domain so the figure only shows the Alps
        agg = agg.copy()
        agg.geometry = agg.geometry.intersection(domain_geom)
        agg = agg[~agg.geometry.is_empty].reset_index(drop=True)

        fig = plt.figure(figsize=(10.5, 7.2))
        ax = fig.add_axes([0.02, 0.02, 0.78, 0.96], projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_facecolor("#f5f5f5")

        if dem_arr is not None and dem_extent is not None:
            ax.imshow(
                dem_arr, origin="upper", extent=dem_extent,
                cmap="gray_r", vmin=DEM_VMIN, vmax=DEM_VMAX,
                transform=ccrs.PlateCarree(), zorder=1, interpolation="bilinear",
            )

        # basins (semi-transparent categorical)
        agg.plot(
            ax=ax, column="grp", cmap="tab20",
            edgecolor="k", linewidth=0.4, alpha=0.55,
            transform=ccrs.PlateCarree(), zorder=2, legend=False,
        )

        # buffer (dashed) + perimeter (solid)
        import geopandas as gpd
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

        ax.set_title(f"HydroBASINS aggregated to n={n}  (from level {args.level})", fontsize=11)
        fig.tight_layout()
        out = out_dir / f"basins_n{n}_lev{args.level}.png"
        fig.savefig(out, dpi=150)
        plt.close()
        print(f"  wrote {out}")
        if args.save_n is not None and n == args.save_n:
            basins_dir = get_source_root() / "Basins"
            basins_dir.mkdir(parents=True, exist_ok=True)
            gpath = basins_dir / f"basins_n{n}.geojson"
            agg.to_crs("EPSG:4326").to_file(gpath, driver="GeoJSON")
            print(f"  saved polygons {gpath}")


if __name__ == "__main__":
    main()
