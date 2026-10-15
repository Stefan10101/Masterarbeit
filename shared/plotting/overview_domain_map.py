#!/usr/bin/env python3
"""
overview_domain_map.py
Sharp blackscale elevation map of the Eastern Alps study domain.
White = lowest, pure black = highest. No hillshade, no borders, no contours.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Rectangle
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def find_code_dir() -> Path:
    here = Path(__file__).resolve()
    candidate = here.parents[2]
    if (candidate / "paths.py").exists():
        return candidate
    current = here.parent
    for _ in range(8):
        if (current / "paths.py").exists():
            return current
        if (current / "CODE" / "paths.py").exists():
            return current / "CODE"
        current = current.parent
    raise RuntimeError("Could not locate CODE/paths.py")


CODE_DIR = find_code_dir()
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from paths import get_project_root  # noqa: E402

PROJECT_ROOT = get_project_root()
OUT_DIR = PROJECT_ROOT / "Plots" / "overview"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEM_PATH = Path(r"C:\Users\stefa\Documents\UNI\Master\Masterarbeit\Daten\Source\DEM\COP30_mosaic_EPSG31287_unclipped.tif")


def load_dem_reprojected(dem_path: Path, lonlat_extent: tuple, target_res: float = 0.0015) -> tuple[np.ndarray, tuple]:
    """High-resolution reproject to EPSG:4326. target_res ~0.0015° ≈ 150 m."""
    lon_min, lat_min, lon_max, lat_max = lonlat_extent
    pad = 0.1
    bounds = (lon_min - pad, lat_min - pad, lon_max + pad, lat_max + pad)

    with rasterio.open(dem_path) as src:
        t_src = Transformer.from_crs(4326, src.crs, always_xy=True)
        left, bottom = t_src.transform(bounds[0], bounds[1])
        right, top = t_src.transform(bounds[2], bounds[3])

        width = max(800, min(int((bounds[2] - bounds[0]) / target_res), 4500))
        height = max(700, min(int((bounds[3] - bounds[1]) / target_res), 3600))

        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
        dem = np.full((height, width), np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=dem,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,  # sharpest
            src_nodata=src.nodata if src.nodata is not None else -9999,
            dst_nodata=np.nan,
        )
        extent = (bounds[0], bounds[2], bounds[1], bounds[3])

    dem = np.where(np.isnan(dem) | (dem < -200), np.nan, dem)
    return dem, extent


def add_river_labels(ax):
    rivers = [
        ("Rhine", 9.4, 47.6),
        ("Inn", 12.1, 47.4),
        ("Etsch", 11.15, 46.35),
        ("Danube", 13.2, 48.55),
    ]
    for name, lon, lat in rivers:
        ax.text(
            lon, lat, name,
            fontsize=7.5, fontstyle="italic", color="#1a5f8a",
            transform=ccrs.PlateCarree(), zorder=6,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.8),
        )


def draw_bbox_rect(ax, lonlat_bbox, color="#c0392b", linewidth=1.4):
    lon_min, lat_min, lon_max, lat_max = lonlat_bbox
    rect = Rectangle(
        (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
        fill=False, edgecolor=color, linewidth=linewidth,
        transform=ccrs.PlateCarree(), zorder=4,
    )
    ax.add_patch(rect)


def make_inset(ax_main, domain_lonlat_bbox):
    ax_in = ax_main.inset_axes([0.80, 0.04, 0.16, 0.16], projection=ccrs.PlateCarree())
    ax_in.set_extent([-15, 35, 35, 60], crs=ccrs.PlateCarree())
    ax_in.add_feature(cfeature.LAND.with_scale("50m"), facecolor="0.94", edgecolor="none")
    ax_in.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="0.87", edgecolor="none")
    ax_in.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="0.45")
    ax_in.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.3)

    lon_min, lat_min, lon_max, lat_max = domain_lonlat_bbox
    draw_bbox_rect(ax_in, (lon_min, lat_min, lon_max, lat_max), color="#c0392b", linewidth=0.9)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    for spine in ax_in.spines.values():
        spine.set_linewidth(0.5)
    return ax_in


def main(dpi: int = 400):
    lonlat_bbox = (7.5, 45.8, 13.6, 49.1)
    print(f"Domain bbox (WGS84): {lonlat_bbox}")

    pad = 0.55
    main_extent = (
        lonlat_bbox[0] - pad,
        lonlat_bbox[2] + pad,
        lonlat_bbox[1] - pad,
        lonlat_bbox[3] + pad,
    )

    fig = plt.figure(figsize=(9.0, 9.2))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=ccrs.PlateCarree())
    ax.set_extent(main_extent, crs=ccrs.PlateCarree())
    mean_lat = (main_extent[2] + main_extent[3]) / 2
    ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)))
    ax.set_facecolor("white")

    if DEM_PATH.exists():
        dem, dem_extent = load_dem_reprojected(DEM_PATH, main_extent, target_res=0.0015)
        # pure blackscale: white = lowest, pure black = highest
        ax.imshow(
            dem, origin="upper", extent=dem_extent,
            cmap="gray_r", vmin=0, vmax=3200,
            transform=ccrs.PlateCarree(), zorder=1,
            interpolation="nearest",  # sharpest
        )
    else:
        print(f"DEM not found: {DEM_PATH}")

    # no political borders
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.4, edgecolor="0.3", zorder=3)
    ax.add_feature(cfeature.RIVERS.with_scale("10m"), linewidth=0.35, edgecolor="#4a90d9", zorder=2)
    ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="#a8d5e5", edgecolor="#4a90d9", linewidth=0.3, zorder=2)
    add_river_labels(ax)
    draw_bbox_rect(ax, lonlat_bbox)

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.4, color="0.4", alpha=0.5, linestyle="-",
        xlocs=np.arange(7, 15, 1), ylocs=np.arange(45, 50, 1),
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    make_inset(ax, lonlat_bbox)

    ax.annotate(
        "N", xy=(0.96, 0.92), xytext=(0.96, 0.86),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center", fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
        zorder=8,
    )

    scale_lon = main_extent[0] + 0.35
    scale_lat = main_extent[2] + 0.3
    ax.plot(
        [scale_lon, scale_lon + 1.35], [scale_lat, scale_lat],
        color="black", linewidth=1.6, transform=ccrs.PlateCarree(), zorder=7,
    )
    ax.text(
        scale_lon + 0.67, scale_lat + 0.08, "100 km",
        ha="center", fontsize=7, transform=ccrs.PlateCarree(), zorder=7,
    )

    out_png = OUT_DIR / "overview_domain_blackscale.png"
    print(f"Saving to: {out_png}")
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    print(f"Saved: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
