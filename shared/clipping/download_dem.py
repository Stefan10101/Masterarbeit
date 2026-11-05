#!/usr/bin/env python3
"""
download_cop30_alpine.py
Download Copernicus GLO-30 (COP30) for the Alpine Convention domain
(perimeter + 100 km buffer) plus map pad, via OpenTopography Global DEM API.
Bounds: 3.0–18.5 °E, 42.0–50.0 °N  (covers map extent 3.28–18.15 / 42.18–49.62).
COP30 requests are limited to 450 000 km², so the area is split into a 2x2 grid
and mosaicked.
"""
import urllib.request
import urllib.error
from pathlib import Path
import rasterio
from rasterio.merge import merge

# ===== CONFIG =====
API_KEY = "74b15aaf7da99e7da38476632d08343c"          # replace
OUT_DIR = Path(r"C:\Users\stefa\Documents\UNI\Master\Masterarbeit\Daten\Source\DEM")
OUT_FILE = OUT_DIR / "COP30_alpine_convention_raw.tif"

# Bounds covering the domain (Alpine Convention + 100 km) and the map pad
# Map extent ≈ 3.28–18.15 °E, 42.18–49.62 °N; extra margin for reproject/edge.
WEST, SOUTH, EAST, NORTH = 3.0, 42.0, 18.5, 50.0
# 2x2 split keeps each tile well under the 450 000 km² COP30 limit
SPLIT_LON = 10.75
SPLIT_LAT = 46.0
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
    print(f"Downloading {west:.2f}–{east:.2f} E, {south:.2f}–{north:.2f} N ...")
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"OpenTopography HTTP {e.code} for tile {dest.name}: {body}") from e
    print(f"  saved {dest.name}")


def main():
    tiles = [
        (WEST, SOUTH, SPLIT_LON, SPLIT_LAT, OUT_DIR / "cop30_sw.tif"),
        (SPLIT_LON, SOUTH, EAST, SPLIT_LAT, OUT_DIR / "cop30_se.tif"),
        (WEST, SPLIT_LAT, SPLIT_LON, NORTH, OUT_DIR / "cop30_nw.tif"),
        (SPLIT_LON, SPLIT_LAT, EAST, NORTH, OUT_DIR / "cop30_ne.tif"),
    ]

    for west, south, east, north, dest in tiles:
        download_tile(west, south, east, north, dest)

    print("Mosaicking...")
    srcs = [rasterio.open(dest) for *_, dest in tiles]
    mosaic, out_trans = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": "lzw",
        "BIGTIFF": "YES",
    })
    with rasterio.open(OUT_FILE, "w", **meta) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()

    print(f"Done: {OUT_FILE}")
    # optional cleanup
    # for *_, dest in tiles:
    #     dest.unlink()


if __name__ == "__main__":
    main()