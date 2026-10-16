#!/usr/bin/env python3
"""
overview_domain_map.py
Two overview maps of the Alpine Convention domain:
  1. Blackscale elevation (COP30)
  2. CORINE Land Cover (Level-2 groups)
"""

from __future__ import annotations

import sys
import tempfile
import urllib.request
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from matplotlib.patches import Patch
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

from paths import get_project_root, get_landcover_path  # noqa: E402

PROJECT_ROOT = get_project_root()
OUT_DIR = PROJECT_ROOT / "Plots" / "overview"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEM_PATH = Path(
    r"C:\Users\stefa\Documents\UNI\Master\Masterarbeit\Daten\Source\DEM\COP30_mosaic_EPSG31287_unclipped.tif"
)
CLC_PATH = get_landcover_path()

PERIMETER_URL = (
    "https://services5.arcgis.com/qjjtr5pMpt3PDiiR/ArcGIS/rest/services/"
    "Alpine_Convention_Perimeter/FeatureServer/1/query?"
    "where=1%3D1&outFields=*&f=geojson"
)

PERIMETER_COLOR = "#6a3d9a"  # purple – distinct from landcover colours

# Level-2 labels + colours (more Alpine detail)
CLC_LEVEL2 = {
    11: ("Urban fabric",                    "#e31a1c"),
    12: ("Industrial / commercial / transport", "#fb9a99"),
    13: ("Mine, dump, construction",        "#fdbf6f"),
    14: ("Artificial non-agri. vegetation", "#cab2d6"),
    21: ("Arable land",                     "#ff7f00"),
    22: ("Permanent crops",                 "#b15928"),
    23: ("Pastures",                        "#ffff33"),
    24: ("Heterogeneous agriculture",       "#b2df8a"),
    31: ("Forests",                         "#33a02c"),
    32: ("Shrub / herbaceous",              "#7fc97f"),
    33: ("Open spaces (bare rock, glacier…)", "#c2a5cf"),
    41: ("Inland wetlands",                 "#1f78b4"),
    42: ("Coastal wetlands",                "#a6cee3"),
    51: ("Inland waters",                   "#6baed6"),
    52: ("Marine waters",                   "#08519c"),
}


def load_alpine_perimeter() -> gpd.GeoDataFrame:
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        urllib.request.urlretrieve(PERIMETER_URL, tmp.name)
        gdf = gpd.read_file(tmp.name)
    return gdf.to_crs("EPSG:4326")


def load_raster_reprojected(
    raster_path: Path,
    lonlat_extent: tuple,
    target_res: float = 0.002,
    categorical: bool = False,
) -> tuple[np.ndarray, tuple]:
    lon_min, lat_min, lon_max, lat_max = lonlat_extent
    pad = 0.12
    bounds = (lon_min - pad, lat_min - pad, lon_max + pad, lat_max + pad)

    with rasterio.open(raster_path) as src:
        width = max(900, min(int((bounds[2] - bounds[0]) / target_res), 4800))
        height = max(700, min(int((bounds[3] - bounds[1]) / target_res), 3800))
        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

        arr = np.full((height, width), np.nan, dtype=np.float32)
        src_nodata = src.nodata if src.nodata is not None else 0

        reproject(
            source=rasterio.band(src, 1),
            destination=arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )
        extent = (bounds[0], bounds[2], bounds[1], bounds[3])

    if not categorical:
        arr = np.where(np.isnan(arr) | (arr < -200), np.nan, arr)
    return arr, extent


def clc_to_level2(arr: np.ndarray) -> np.ndarray:
    """Map CLC codes to Level-2 groups. Handles 3-digit and 1–44 encodings."""
    out = np.full_like(arr, np.nan)
    valid = ~np.isnan(arr)
    if not np.any(valid):
        return out

    codes = arr[valid].astype(int)
    print(f"CLC unique (sample): {np.unique(codes)[:20]} max={codes.max()}")

    if codes.max() > 100:
        level2 = (codes // 10)
    else:
        level2 = np.zeros_like(codes, dtype=int)
        level2[(codes >= 1) & (codes <= 2)] = 11
        level2[(codes >= 3) & (codes <= 6)] = 12
        level2[(codes >= 7) & (codes <= 9)] = 13
        level2[(codes >= 10) & (codes <= 11)] = 14
        level2[(codes >= 12) & (codes <= 14)] = 21
        level2[(codes >= 15) & (codes <= 17)] = 22
        level2[codes == 18] = 23
        level2[(codes >= 19) & (codes <= 22)] = 24
        level2[(codes >= 23) & (codes <= 25)] = 31
        level2[(codes >= 26) & (codes <= 29)] = 32
        level2[(codes >= 30) & (codes <= 34)] = 33
        level2[(codes >= 35) & (codes <= 37)] = 41
        level2[(codes >= 38) & (codes <= 39)] = 42
        level2[(codes >= 40) & (codes <= 41)] = 51
        level2[(codes >= 42) & (codes <= 44)] = 52

    out[valid] = level2
    return out


def add_river_labels_dem(ax):
    """River labels for the elevation figure only."""
    rivers = [
        ("Rhine", 8.3, 47.6),
        ("Inn", 11.5, 47.3),
        ("Etsch", 11.15, 46.35),      # renamed from Adige
        ("Rhône", 6.8, 46.2),         # France
        ("Drau", 13.4, 46.7),         # eastern Alps
        ("Po", 10.5, 45.1),
        ("Danube", 13.5, 48.4),
    ]
    for name, lon, lat in rivers:
        ax.text(
            lon, lat, name,
            fontsize=7, fontstyle="italic", color="#1a5f8a",
            transform=ccrs.PlateCarree(), zorder=6,
            bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=0.6),
        )

def mark_highest_point(ax):
    """Mont Blanc – highest point in the Alps."""
    lon, lat = 6.865, 45.833
    ax.plot(
        lon, lat, marker="^", color="black", markersize=7,
        transform=ccrs.PlateCarree(), zorder=7,
    )
    ax.text(
        lon + 0.15, lat + 0.08, "Mont Blanc\n4808 m",
        fontsize=6.5, color="black", transform=ccrs.PlateCarree(), zorder=7,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.4),
    )


def make_inset(ax_main, perimeter_gdf):
    ax_in = ax_main.inset_axes([0.78, 0.03, 0.18, 0.18], projection=ccrs.PlateCarree())
    ax_in.set_extent([-12, 30, 35, 58], crs=ccrs.PlateCarree())
    ax_in.add_feature(cfeature.LAND.with_scale("50m"), facecolor="0.94", edgecolor="none")
    ax_in.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="0.87", edgecolor="none")
    ax_in.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, edgecolor="0.45")
    ax_in.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.25)
    perimeter_gdf.plot(
        ax=ax_in, facecolor="none", edgecolor=PERIMETER_COLOR, linewidth=0.6,
        transform=ccrs.PlateCarree(),
    )
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    for spine in ax_in.spines.values():
        spine.set_linewidth(0.4)
    return ax_in


def add_common_overlays(ax, peri, main_extent, perimeter_color=PERIMETER_COLOR,
                        show_rivers=True, thick_perimeter=False):
    lw = 2.2 if thick_perimeter else 1.4
    peri.plot(
        ax=ax, facecolor="none", edgecolor=perimeter_color, linewidth=lw,
        transform=ccrs.PlateCarree(), zorder=5,
    )
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.3, edgecolor="0.4", zorder=3)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#e6f3ff", edgecolor="none", zorder=0)
    if show_rivers:
        ax.add_feature(cfeature.RIVERS.with_scale("10m"), linewidth=0.28, edgecolor="#4a90d9", zorder=2)
        ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="#a8d5e5", edgecolor="#4a90d9", linewidth=0.22, zorder=2)
        add_river_labels_dem(ax)

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="0.5", alpha=0.45, linestyle="-",
        xlocs=np.arange(4, 18, 2), ylocs=np.arange(43, 50, 1),
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    make_inset(ax, peri)

    ax.annotate(
        "N", xy=(0.965, 0.94), xytext=(0.965, 0.88),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center", fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.15),
        zorder=8,
    )

    scale_lon = main_extent[0] + 0.5
    scale_lat = main_extent[2] + 0.28
    ax.plot(
        [scale_lon, scale_lon + 1.2], [scale_lat, scale_lat],
        color="black", linewidth=1.4, transform=ccrs.PlateCarree(), zorder=7,
    )
    ax.text(
        scale_lon + 0.6, scale_lat + 0.1, "100 km",
        ha="center", fontsize=7, transform=ccrs.PlateCarree(), zorder=7,
    )


def make_dem_figure(peri, main_extent, dpi):
    fig = plt.figure(figsize=(12.0, 8.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=ccrs.PlateCarree())
    ax.set_extent(main_extent, crs=ccrs.PlateCarree())
    mean_lat = (main_extent[2] + main_extent[3]) / 2
    ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)))
    ax.set_facecolor("#e6f3ff")

    if DEM_PATH.exists():
        dem, dem_extent = load_raster_reprojected(DEM_PATH, main_extent, target_res=0.0018)
        ax.imshow(
            dem, origin="upper", extent=dem_extent,
            cmap="gray_r", vmin=0, vmax=3200,
            transform=ccrs.PlateCarree(), zorder=1,
            interpolation="nearest",
        )
    else:
        print(f"DEM not found: {DEM_PATH}")

    add_common_overlays(ax, peri, main_extent, perimeter_color="#c0392b", show_rivers=True)
    # Mont Blanc marker removed

    out_png = OUT_DIR / "overview_domain_blackscale.png"
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight", pad_inches=0.015, facecolor="white")
    print(f"Saved: {out_png}")
    plt.close(fig)

    
def make_clc_figure(peri, main_extent, dpi):
    fig = plt.figure(figsize=(12.0, 8.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=ccrs.PlateCarree())
    ax.set_extent(main_extent, crs=ccrs.PlateCarree())
    mean_lat = (main_extent[2] + main_extent[3]) / 2
    ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)))
    ax.set_facecolor("white")

    if CLC_PATH.exists():
        clc_raw, clc_extent = load_raster_reprojected(
            CLC_PATH, main_extent, target_res=0.0025, categorical=True
        )
        clc_l2 = clc_to_level2(clc_raw)

        present = sorted({int(v) for v in np.unique(clc_l2) if not np.isnan(v) and v in CLC_LEVEL2})
        colors = [CLC_LEVEL2[k][1] for k in present]
        cmap = mcolors.ListedColormap(colors)
        bounds = np.array(present + [present[-1] + 1]) - 0.5
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        ax.imshow(
            clc_l2, origin="upper", extent=clc_extent,
            cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(), zorder=1,
            interpolation="nearest",
        )

        legend_elements = [
            Patch(facecolor=CLC_LEVEL2[k][1], edgecolor="none", label=CLC_LEVEL2[k][0])
            for k in present
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            ncol=2,
            frameon=True,
            framealpha=0.92,
            fontsize=6.5,
            title="CORINE Level 2",
            title_fontsize=7,
            borderpad=0.4,
            columnspacing=0.8,
        )

        # standalone legend
        fig_leg = plt.figure(figsize=(6.5, 4.2))
        ax_leg = fig_leg.add_axes([0, 0, 1, 1])
        ax_leg.axis("off")
        ax_leg.legend(
            handles=legend_elements,
            loc="center",
            ncol=2,
            frameon=False,
            fontsize=8,
            title="CORINE Level 2",
            title_fontsize=9,
        )
        leg_png = OUT_DIR / "overview_landcover_legend.png"
        fig_leg.savefig(str(leg_png), dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {leg_png}")
        plt.close(fig_leg)
    else:
        print(f"CLC not found: {CLC_PATH}")

    # thick black perimeter, no river labels
    add_common_overlays(
        ax, peri, main_extent,
        perimeter_color="black",
        show_rivers=False,
        thick_perimeter=True,
    )

    out_png = OUT_DIR / "overview_landcover.png"
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight", pad_inches=0.015, facecolor="white")
    print(f"Saved: {out_png}")
    plt.close(fig)


def main(dpi: int = 400):
    print("Loading Alpine Convention perimeter...")
    peri = load_alpine_perimeter()
    minx, miny, maxx, maxy = peri.total_bounds
    print(f"Alpine Convention lon/lat bounds: ({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})")

    pad = 0.45
    main_extent = (minx - pad, maxx + pad, miny - pad, maxy + pad)

    make_dem_figure(peri, main_extent, dpi)
    make_clc_figure(peri, main_extent, dpi)


if __name__ == "__main__":
    main()