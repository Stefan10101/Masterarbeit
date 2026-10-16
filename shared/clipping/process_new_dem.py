#!/usr/bin/env python3
"""
process_alpine_dem.py
Reproject the raw COP30 Alpine Convention mosaic to EPSG:31287 and write:
  - unclipped full mosaic
  - clipped to Alpine Convention perimeter (exact name expected by paths.get_dem_path)
"""
import tempfile
import urllib.request
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

# ===== CONFIG =====
RAW_PATH = Path(r"C:\Users\stefa\Documents\UNI\Master\Masterarbeit\Daten\Source\DEM\COP30_alpine_convention_raw.tif")
OUT_DIR  = Path(r"C:\Users\stefa\Documents\UNI\Master\Masterarbeit\Daten\Source\DEM")
CLIPPED_NAME   = "COP30_mosaic_EPSG31287.tif"
UNCLIPPED_NAME = "COP30_mosaic_EPSG31287_unclipped.tif"

PERIMETER_URL = (
    "https://services5.arcgis.com/qjjtr5pMpt3PDiiR/ArcGIS/rest/services/"
    "Alpine_Convention_Perimeter/FeatureServer/1/query?"
    "where=1%3D1&outFields=*&f=geojson"
)
BUFFER_M = 1000
DST_CRS  = "EPSG:31287"
RESAMPLING = Resampling.bilinear
# ==================

OUT_DIR.mkdir(parents=True, exist_ok=True)
CLIPPED_PATH   = OUT_DIR / CLIPPED_NAME
UNCLIPPED_PATH = OUT_DIR / UNCLIPPED_NAME


def main():
    # 1. Reproject full raw mosaic → unclipped
    print("Reprojecting full mosaic to EPSG:31287...")
    with rasterio.open(RAW_PATH) as src:
        dst_transform, width, height = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": DST_CRS,
            "transform": dst_transform,
            "width": width,
            "height": height,
            "compress": "lzw",
            "predictor": 2,
            "nodata": src.nodata,
        })
        with rasterio.open(UNCLIPPED_PATH, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=DST_CRS,
                    resampling=RESAMPLING,
                )
    print(f"Unclipped saved: {UNCLIPPED_PATH.name}")

    # 2. Load Alpine Convention perimeter
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        urllib.request.urlretrieve(PERIMETER_URL, tmp.name)
        gdf = gpd.read_file(tmp.name).to_crs(DST_CRS)
    if BUFFER_M > 0:
        gdf["geometry"] = gdf.geometry.buffer(BUFFER_M)

    # 3. Clip
    print("Clipping to Alpine Convention perimeter...")
    with rasterio.open(UNCLIPPED_PATH) as src:
        out_image, out_transform = mask(src, gdf.geometry, crop=True, nodata=src.nodata)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": src.nodata,
        })
        with rasterio.open(CLIPPED_PATH, "w", **out_meta) as dest:
            dest.write(out_image)

        bounds = rasterio.transform.array_bounds(
            out_image.shape[1], out_image.shape[2], out_transform
        )
        print(f"Clipped bounds (EPSG:31287): "
              f"[{bounds[0]:.0f}, {bounds[1]:.0f}, {bounds[2]:.0f}, {bounds[3]:.0f}]")

    print(f"Clipped saved: {CLIPPED_PATH.name}")
    print("Done.")


if __name__ == "__main__":
    main()