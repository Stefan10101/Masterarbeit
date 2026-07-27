#!/usr/bin/env python3
"""
reproject_landcover.py
Reprojects the CORINE Land Cover from EPSG:3035 to EPSG:31287 (nearest neighbor).
Run this once.
"""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path

# ====================== CONFIG ======================
input_path = Path(r"C:\Users\stefa\Documents\UNI\Master\Masterarbeit\Daten\Landcover_Source\Results\u2018_clc2018_v2020_20u1_raster100m\DATA\U2018_CLC2018_V2020_20u1.tif")

output_dir = Path(r"C:\Users\stefa\Documents\UNI\Master\Masterarbeit\Daten\Landcover_Source\Reprojected")
output_path = output_dir / "CLC2018_EPSG31287_100m.tif"
# ====================================================

output_dir.mkdir(parents=True, exist_ok=True)

dst_crs = "EPSG:31287"

with rasterio.open(input_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )

    kwargs = src.meta.copy()
    kwargs.update({
        "crs": dst_crs,
        "transform": transform,
        "width": width,
        "height": height,
        "compress": "lzw",
        "predictor": 2,
        "nodata": src.nodata
    })

    with rasterio.open(output_path, "w", **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest   # Important for categorical data
            )

    print(f"Reprojected landcover saved to:\n{output_path}")
    print(f"New CRS: {dst_crs}")
    print(f"Resolution: ~100m")