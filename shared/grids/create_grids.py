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
import rasterio

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
    add_landcover_to_stations
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["BSS", "IDW", "RFSI"])
    parser.add_argument("--master", action="store_true",
                        help="Create master grids from full original DEM extent")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--resolutions", nargs="+", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def get_full_dem_bbox(dem_path: Path) -> tuple:
    with rasterio.open(dem_path) as src:
        return src.bounds


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    method = args.method.upper()
    resolutions = args.resolutions or cfg.get("resolutions_to_process", [100])
    projected_crs = cfg.get("projected_crs", "EPSG:31287")
    dem_path = get_dem_path()

    print("=" * 80)
    mode = "MASTER" if args.master else "DOMAIN"
    print(f"Creating {mode} grids | Method: {method}")
    print(f"Resolutions: {resolutions}")
    print("=" * 80)

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
                projected_crs=projected_crs,   # ← add this
                domain="master"
            )
            out_path = grids_dir / "master" / f"res_{res}m" / "grid.nc"
            save_grid(ds, out_path)
    else:
        print("\n[Legacy domain mode - consider using --master]")

    print("\n" + "=" * 80)
    print("Grid creation finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()