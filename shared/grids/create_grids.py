#!/usr/bin/env python3
"""Build Source/grids and Source/stations from Source DEM + CLC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import yaml

CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR))

from paths import (
    get_dem_path,
    get_domain_grid_path,
    get_domain_stations_path,
    get_grids_dir,
    get_landcover_path,
    get_master_grid_path,
    get_metadata_path,
    ensure_dir,
)
from grid_core import (
    add_landcover_to_grid,
    add_landcover_to_stations,
    build_kdtree,
    create_grid_from_dem,
    project_stations,
    save_grid,
)

DEFAULT_CFG = CODE_DIR / "shared" / "source" / "config.yaml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--master", action="store_true", help="Rebuild master grids from the DEM")
    p.add_argument("--domains", nargs="*", default=None, help="Domain names; default = all predefined")
    p.add_argument("--resolutions", nargs="+", type=int, default=None)
    p.add_argument("--config", default=str(DEFAULT_CFG))
    p.add_argument("--attach-landcover", action="store_true",
                   help="Warp CLC onto existing master grids only")
    return p.parse_args()


def domain_bbox(cfg, name):
    box = cfg["predefined_bboxes"][name]
    buf = float(cfg.get("domain", {}).get("buffer_m", 0))
    xmin, ymin, xmax, ymax = box
    return xmin - buf, ymin - buf, xmax + buf, ymax + buf


def filter_stations(stations, bbox):
    xmin, ymin, xmax, ymax = bbox
    m = (
        (stations["x"] >= xmin) & (stations["x"] <= xmax)
        & (stations["y"] >= ymin) & (stations["y"] <= ymax)
    )
    return stations.loc[m].copy()


def write_stations(stations, domain):
    path = get_domain_stations_path(None, domain)
    ensure_dir(path.parent)
    stations.to_parquet(path, index=False)
    tree = build_kdtree(stations)
    joblib.dump(tree, path.parent / "station_kdtree.joblib")
    print(f"stations {domain} n={len(stations)} -> {path}", flush=True)
    return path


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    resolutions = args.resolutions or cfg.get("resolutions_to_process", [1000])
    crs = cfg.get("projected_crs", "EPSG:31287")
    dem_path = get_dem_path()
    lc_path = get_landcover_path()
    domains = args.domains or list(cfg.get("predefined_bboxes", {}).keys())
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM missing: {dem_path}  (run shared/source/bootstrap_source.py)")
    print(f"DEM {dem_path}", flush=True)
    print(f"CLC {lc_path} exists={lc_path.exists()}", flush=True)
    print(f"res={resolutions} domains={domains}", flush=True)

    if args.attach_landcover:
        import xarray as xr
        if not lc_path.exists():
            raise FileNotFoundError(lc_path)
        for res in resolutions:
            out_path = get_master_grid_path(None, res)
            if not out_path.exists():
                raise FileNotFoundError(out_path)
            ds = xr.open_dataset(out_path).load()
            ds.close()
            ds = add_landcover_to_grid(ds, lc_path, projected_crs=crs)
            save_grid(ds, out_path)
        print("landcover attach finished", flush=True)
        return

    stations = project_stations(get_metadata_path(), crs, add_network_provider=True)
    if lc_path.exists():
        stations = add_landcover_to_stations(stations, lc_path)
    else:
        print(f"warning: no CLC at {lc_path}", flush=True)

    write_stations(stations, "full")
    for name in domains:
        if name == "full":
            continue
        write_stations(filter_stations(stations, domain_bbox(cfg, name)), name)

    if args.master:
        for res in resolutions:
            print(f"master grid {res} m", flush=True)
            ds = create_grid_from_dem(
                dem_path=dem_path,
                resolution_m=res,
                bbox=None,
                projected_crs=crs,
                domain="master",
            )
            if lc_path.exists():
                ds = add_landcover_to_grid(ds, lc_path, projected_crs=crs)
            save_grid(ds, get_master_grid_path(None, res))
            for name in domains:
                if name == "full":
                    continue
                print(f"domain grid {name} {res} m", flush=True)
                dds = create_grid_from_dem(
                    dem_path=dem_path,
                    resolution_m=res,
                    bbox=domain_bbox(cfg, name),
                    projected_crs=crs,
                    domain=name,
                )
                if lc_path.exists():
                    dds = add_landcover_to_grid(dds, lc_path, projected_crs=crs)
                save_grid(dds, get_domain_grid_path(None, name, res))

    print("grid creation finished", flush=True)


if __name__ == "__main__":
    main()
