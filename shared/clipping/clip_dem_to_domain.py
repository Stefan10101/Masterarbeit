#!/usr/bin/env python3
"""
clip_dem_to_domain.py
Clips the master DEM to one or more analysis domains.
Uses paths.py as the single source of truth.
"""

import argparse
import yaml
from pathlib import Path
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from paths import get_method_output_dir, ensure_dir, get_dem_path, PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["BSS", "IDW", "RFSI"])
    parser.add_argument("--domains", nargs="+", default=None,
                        help="List of domains to process (default = all from config)")
    parser.add_argument("--dem", default=None,
                        help="Optional: override DEM path (absolute or relative to project root)")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def get_bbox_for_domain(domain_name, predefined_bboxes, custom_bbox, buffer_m):
    if domain_name in predefined_bboxes:
        bbox = predefined_bboxes[domain_name]
    else:
        print(f"[WARNING] Domain '{domain_name}' not found. Using custom_bbox.")
        bbox = custom_bbox

    xmin, ymin, xmax, ymax = bbox
    xmin -= buffer_m
    ymin -= buffer_m
    xmax += buffer_m
    ymax += buffer_m
    return xmin, ymin, xmax, ymax


def clip_dem(dem_path, xmin, ymin, xmax, ymax, output_path):
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    with rasterio.open(dem_path) as src:
        geom = box(xmin, ymin, xmax, ymax)
        gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs=src.crs)

        out_image, out_transform = mask(src, gdf.geometry, crop=True, nodata=src.nodata)

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

    # Output directory
    output_dir = get_method_output_dir(method) / "clipped_dems"
    ensure_dir(output_dir)

    # === DEM Path (priority: --dem > paths.py > config) ===
    if args.dem:
        dem_path = Path(args.dem)
        if not dem_path.is_absolute():
            dem_path = PROJECT_ROOT / dem_path
    else:
        # Use paths.py as default
        dem_path = get_dem_path()

    # Domain settings from config
    domain_cfg = cfg.get("domain", {})
    buffer_m = domain_cfg.get("buffer_m", 15000)
    custom_bbox = domain_cfg.get("custom_bbox", [])
    predefined_bboxes = cfg.get("predefined_bboxes", {})

    domains_to_process = args.domains if args.domains else list(predefined_bboxes.keys())

    print("=" * 75)
    print(f"Clipping DEM for method: {method}")
    print(f"DEM path     : {dem_path}")
    print(f"Output folder: {output_dir}")
    print(f"Domains      : {domains_to_process}")
    print("=" * 75)

    for domain_name in domains_to_process:
        print(f"\n>>> {domain_name}")
        xmin, ymin, xmax, ymax = get_bbox_for_domain(
            domain_name, predefined_bboxes, custom_bbox, buffer_m
        )
        output_file = output_dir / f"COP30_clipped_{domain_name}_EPSG31287.tif"
        clip_dem(dem_path, xmin, ymin, xmax, ymax, output_file)

    print("\nClipping finished successfully.")


if __name__ == "__main__":
    main()