#!/usr/bin/env python3
"""
process_new_dem.py
Extract rasters_COP30.tar.gz, reproject to EPSG:31287, write:
  - unclipped full mosaic
  - clipped to study domain (exact name expected by paths.get_dem_path)
Run once.
"""
import tarfile
from pathlib import Path
import tempfile
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd

# ===== CONFIG =====
TAR_PATH = Path(r"C:\Users\stefa\Downloads\rasters_COP30.tar.gz")
OUT_DIR  = Path(r"C:\Users\stefa\Documents\UNI\Master\Masterarbeit\Daten\Source\DEM")
CLIPPED_NAME   = "COP30_mosaic_EPSG31287.tif"          # exact name used by get_dem_path
UNCLIPPED_NAME = "COP30_mosaic_EPSG31287_unclipped.tif"

LONLAT_BBOX = (7.5, 45.8, 13.6, 49.1)   # minlon, minlat, maxlon, maxlat
BUFFER_M    = 15000
DST_CRS     = "EPSG:31287"
RESAMPLING  = Resampling.bilinear
# ==================

OUT_DIR.mkdir(parents=True, exist_ok=True)
CLIPPED_PATH   = OUT_DIR / CLIPPED_NAME
UNCLIPPED_PATH = OUT_DIR / UNCLIPPED_NAME


def lonlat_to_proj(lonlat_bbox, buffer_m=BUFFER_M):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", DST_CRS, always_xy=True)
    corners = [
        (lonlat_bbox[0], lonlat_bbox[1]),
        (lonlat_bbox[0], lonlat_bbox[3]),
        (lonlat_bbox[2], lonlat_bbox[1]),
        (lonlat_bbox[2], lonlat_bbox[3]),
    ]
    xs, ys = zip(*(t.transform(lon, lat) for lon, lat in corners))
    return (min(xs) - buffer_m, min(ys) - buffer_m,
            max(xs) + buffer_m, max(ys) + buffer_m)


def main():
    xmin, ymin, xmax, ymax = lonlat_to_proj(LONLAT_BBOX)
    print(f"Clip bbox (EPSG:31287 + {BUFFER_M} m): "
          f"[{xmin:.0f}, {ymin:.0f}, {xmax:.0f}, {ymax:.0f}]")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        print("Extracting...")
        with tarfile.open(TAR_PATH) as tar:
            tar.extractall(tmp)

        tifs = list(tmp.rglob("*.tif")) + list(tmp.rglob("*.tiff"))
        if not tifs:
            raise FileNotFoundError("No GeoTIFFs found in archive")
        print(f"Found {len(tifs)} raster(s)")

        # Mosaic if needed
        if len(tifs) > 1:
            srcs = [rasterio.open(p) for p in tifs]
            mosaic, out_trans = merge(srcs)
            meta = srcs[0].meta.copy()
            meta.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
            })
            for s in srcs:
                s.close()
            mosaic_path = tmp / "mosaic.tif"
            with rasterio.open(mosaic_path, "w", **meta) as dst:
                dst.write(mosaic)
            src_path = mosaic_path
        else:
            src_path = tifs[0]

        # Reproject full extent → unclipped
        print("Reprojecting full mosaic...")
        with rasterio.open(src_path) as src:
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

        # Clip the reprojected raster
        print("Clipping to domain...")
        with rasterio.open(UNCLIPPED_PATH) as src:
            geom = box(xmin, ymin, xmax, ymax)
            gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs=src.crs)
            out_image, out_transform = mask(
                src, gdf.geometry, crop=True, nodata=src.nodata
            )
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": src.nodata,
            })
            with rasterio.open(CLIPPED_PATH, "w", **out_meta) as dest:
                dest.write(out_image)
        print(f"Clipped saved: {CLIPPED_PATH.name}")

    print("Done.")


if __name__ == "__main__":
    main()