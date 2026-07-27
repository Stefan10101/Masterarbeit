#!/usr/bin/env python3
"""
clip_landcover_to_domain.py
Clips the CORINE Land Cover raster to analysis domains.
Follows the same structure as clip_dem_to_domain.py.
"""

import argparse
import yaml
from pathlib import Path
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from paths import get_method_output_dir, ensure_dir, get_landcover_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["BSS", "IDW", "RFSI"])
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def get_bbox_for_domain(cfg: dict, domain_name: str) -> tuple:
    domain_cfg = cfg.get("domain", {})
    buffer_m = domain_cfg.get("buffer_m", 15000)
    custom_bbox = domain_cfg.get("custom_bbox", [100000, 275000, 395000, 400000])
    predefined = cfg.get("predefined_bboxes", {})

    if domain_name in predefined:
        bbox = predefined[domain_name]
    else:
        bbox = custom_bbox

    xmin, ymin, xmax, ymax = bbox
    xmin -= buffer_m
    ymin -= buffer_m
    xmax += buffer_m
    ymax += buffer_m
    return xmin, ymin, xmax, ymax

def clip_landcover(landcover_path: Path, bbox: tuple, output_path: Path, domain_name: str):
    xmin, ymin, xmax, ymax = bbox

    if not landcover_path.exists():
        raise FileNotFoundError(f"Landcover not found: {landcover_path}")

    domain_geom = box(xmin, ymin, xmax, ymax)
    domain_area = domain_geom.area

    with rasterio.open(landcover_path) as src:
        raster_geom = box(*src.bounds)
        intersection = domain_geom.intersection(raster_geom)
        overlap_area = intersection.area if not intersection.is_empty else 0.0

        coverage_percent = (overlap_area / domain_area) * 100 if domain_area > 0 else 0

        print(f"  Domain coverage by landcover: {coverage_percent:.1f}%")

        if coverage_percent == 0:
            print(f"  [SKIP] No overlap with landcover raster for domain '{domain_name}'.")
            return

        gdf = gpd.GeoDataFrame({"geometry": [domain_geom]}, crs=src.crs)

        try:
            out_image, out_transform = mask(
                src,
                gdf.geometry,
                crop=True,
                nodata=src.nodata,
                all_touched=True
            )
        except ValueError as e:
            if "do not overlap" in str(e):
                print(f"  [SKIP] Domain '{domain_name}' has no overlap with landcover raster.")
                return
            else:
                raise

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": src.nodata
        })

        ensure_dir(output_path.parent)

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

        print(f"  Saved: {output_path.name}")

def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    method = args.method.upper()
    output_dir = get_method_output_dir(method) / "clipped_landcover"
    ensure_dir(output_dir)

    landcover_path = get_landcover_path()

    domain_cfg = cfg.get("domain", {})
    predefined_bboxes = cfg.get("predefined_bboxes", {})

    domains_to_process = args.domains if args.domains else list(predefined_bboxes.keys())

    print("=" * 75)
    print(f"Clipping Landcover for method: {method}")
    print(f"Source: {landcover_path}")
    print(f"Output folder: {output_dir}")
    print(f"Domains: {domains_to_process}")
    print("=" * 75)

    for domain_name in domains_to_process:
        print(f"\n>>> {domain_name}")
        bbox = get_bbox_for_domain(cfg, domain_name)
        output_file = output_dir / f"CLC2018_clipped_{domain_name}_EPSG31287.tif"
        clip_landcover(landcover_path, bbox, output_file, domain_name)

    print("\nLandcover clipping finished.")


if __name__ == "__main__":
    main()