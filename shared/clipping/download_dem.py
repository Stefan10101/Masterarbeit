#!/usr/bin/env python3
"""
download_cop30_alpine.py
Download Copernicus GLO-30 (COP30) for an expanded Alpine overview domain
via OpenTopography Global DEM API.
Bounds: 4–18 °E, 43–49 °N.
Area still exceeds the 450 000 km² limit, so the request is split and mosaicked.
"""
import urllib.request
from pathlib import Path
import rasterio
from rasterio.merge import merge

# ===== CONFIG =====
API_KEY = "74b15aaf7da99e7da38476632d08343c"          # replace
OUT_DIR = Path(r"C:\Users\stefa\Documents\UNI\Master\Masterarbeit\Daten\Source\DEM")
OUT_FILE = OUT_DIR / "COP30_alpine_convention_raw.tif"

# Expanded bounds for full overview coverage
WEST, SOUTH, EAST, NORTH = 4.0, 43.0, 18.0, 49.0
SPLIT_LON = 11.0                       # east/west split
# ==================

OUT_DIR.mkdir(parents=True, exist_ok=True)


def download_tile(west, south, east, north, dest: Path):
    url = (
        "https://portal.opentopography.org/API/globaldem"
        f"?demtype=COP30"
        f"&south={south}&north={north}"
        f"&west={west}&east={east}"
        f"&outputFormat=GTiff"
        f"&API_Key={API_KEY}"
    )
    print(f"Downloading {west:.2f}–{east:.2f} E ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  saved {dest.name}")


def main():
    tile1 = OUT_DIR / "cop30_west.tif"
    tile2 = OUT_DIR / "cop30_east.tif"

    download_tile(WEST, SOUTH, SPLIT_LON, NORTH, tile1)
    download_tile(SPLIT_LON, SOUTH, EAST, NORTH, tile2)

    print("Mosaicking...")
    srcs = [rasterio.open(tile1), rasterio.open(tile2)]
    mosaic, out_trans = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": "lzw",
    })
    with rasterio.open(OUT_FILE, "w", **meta) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()

    print(f"Done: {OUT_FILE}")
    # optional cleanup
    # tile1.unlink()
    # tile2.unlink()


if __name__ == "__main__":
    main()