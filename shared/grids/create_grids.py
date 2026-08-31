#!/usr/bin/env python3
"""
create_grids.py
Creates master grids (from full DEM extent) and domain-specific grids.
Cleaned version - uses full landcover for RFSI.
"""

import argparse
import yaml
from pathlib import Path
import numpy as np
import joblib

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from paths import (
    get_domain_stations_path,
    get_grids_dir,
    get_metadata_path,
    get_dem_path,
    get_landcover_path,
    PROJECT_ROOT
)
from grid_core import (
    project_stations,
    create_grid_from_dem,
    save_grid,
    build_kdtree,
    add_landcover_to_stations,
    add_landcover_to_grid,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["BSS", "IDW", "RFSI"])
    parser.add_argument("--master", action="store_true",
                        help="Create master grids from full original DEM extent")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--resolutions", nargs="+", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--attach-landcover",
        action="store_true",
        help="Warp CLC onto existing master grid.nc files (no DEM rebuild)",
    )
    return parser.parse_args()


def get_full_dem_bbox(dem_path: Path) -> tuple:
    try:
        import rasterio
        with rasterio.open(dem_path) as src:
            return tuple(src.bounds)
    except Exception:
        from osgeo import gdal
        ds = gdal.Open(str(dem_path))
        if ds is None:
            raise FileNotFoundError(dem_path)
        gt = ds.GetGeoTransform()
        xmin, ymax = gt[0], gt[3]
        xmax = xmin + ds.RasterXSize * gt[1]
        ymin = ymax + ds.RasterYSize * gt[5]
        return xmin, ymin, xmax, ymax


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    method = args.method.upper()
    resolutions = args.resolutions or cfg.get("resolutions_to_process", [100])
    projected_crs = cfg.get("projected_crs", "EPSG:31287")
    dem_path = get_dem_path()

    print("=" * 80)
    if args.attach_landcover:
        mode = "ATTACH LANDCOVER"
    elif args.master:
        mode = "MASTER"
    else:
        mode = "DOMAIN"
    print(f"Creating {mode} grids | Method: {method}")
    print(f"Resolutions: {resolutions}")
    print("=" * 80)

    if args.attach_landcover:
        import xarray as xr
        lc_path = get_landcover_path()
        grids_dir = get_grids_dir(method)
        projected_crs = cfg.get("projected_crs", "EPSG:31287")
        for res in resolutions:
            out_path = grids_dir / "master" / f"res_{res}m" / "grid.nc"
            if not out_path.exists():
                raise FileNotFoundError(f"Master grid missing: {out_path}")
            print(f"\nAttaching CLC → {out_path}")
            ds = xr.open_dataset(out_path).load()
            ds.close()
            ds = add_landcover_to_grid(ds, lc_path, projected_crs=projected_crs)
            save_grid(ds, out_path)
        print("\nLandcover attach finished.")
        return

    # === Project Stations ===
    stations = project_stations(get_metadata_path(), projected_crs, add_network_provider=True)

    # === Add landcover (RFSI only - always use FULL landcover) ===
    if method == "RFSI":
        lc_path = get_landcover_path()
        if lc_path.exists():
            print(f"Adding landcover from full file: {lc_path.name}")
            stations = add_landcover_to_stations(stations, lc_path)
        else:
            print(f"[Warning] Full landcover not found at {lc_path}")

    # Save stations
    domain_name = "full" if args.master else (args.domain or "full")
    stations_path = get_domain_stations_path(method, domain_name)
    stations_path.parent.mkdir(parents=True, exist_ok=True)
    stations.to_parquet(stations_path, index=False)
    print(f"Saved stations: {stations_path}")

    if method in ["IDW", "RFSI"]:
        tree = build_kdtree(stations)
        kdtree_path = stations_path.parent / "station_kdtree.joblib"
        joblib.dump(tree, kdtree_path)
        print(f"Saved kdtree: {kdtree_path}")

    # === Grid Creation ===
    grids_dir = get_grids_dir(method)

    if args.master:
        full_bbox = get_full_dem_bbox(dem_path)
        print(f"\nFull DEM extent: {full_bbox}")

        for res in resolutions:
            print(f"\nCreating master grid @ {res} m ...")
            ds = create_grid_from_dem(
                dem_path=dem_path,
                resolution_m=res,
                bbox=None,
                projected_crs=projected_crs,
                domain="master"
            )
            lc_path = get_landcover_path()
            if lc_path.exists():
                ds = add_landcover_to_grid(ds, lc_path, projected_crs=projected_crs)
            out_path = grids_dir / "master" / f"res_{res}m" / "grid.nc"
            save_grid(ds, out_path)
    else:
        print("\n[Legacy domain mode - consider using --master]")

    print("\n" + "=" * 80)
    print("Grid creation finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()