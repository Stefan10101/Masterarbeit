#!/usr/bin/env python3
"""
MAP Terrain Metadata Preparation Script
Master's thesis - efficient extraction of topographic features from COP30 DEM
for weather station metadata to support terrain-aware interpolation.

Key design choices (sound logic + efficiency):
- Reproject DEM once to EPSG:32632 (UTM 32N, conformal, low distortion across study area 7.6-13.5Â°E).
- Pure numpy implementation for slope, aspect, profile_curvature, plan_curvature (Zevenbergen & Thorne finite differences) â€” no compilation needed, only rasterio + numpy.
- Build VRT for multi-tile DEM (no unnecessary mosaicking, memory efficient).
- Pre-compute attribute rasters once, then point-sample (O(1) per station).
- Full logging for reproducibility.
- Explicit handling of nodata, flat areas (aspect=-1), and stations outside DEM.
- Minimal assumptions; all paths configurable at top.

Dependencies (install once on your machine):
    pip install pandas geopandas rasterio pyproj
    (scipy optional but recommended for future; not required here)

Run on Windows with the exact folder structure you specified.
"""

import os
import sys
import logging
import tarfile
import subprocess
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject as rio_reproject, Resampling

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT


# =============================================================================
# CONFIGURATION - edit only these paths if needed
# =============================================================================
SOURCE_STATIONS_CSV = Path(DATA_ROOT / "kombiniert" / "stations_overview.csv")
DEM_TAR_GZ = Path(DATA_ROOT / "maps" / "map_source" / "rasters_cop30" / "rasters_cop30.tar.gz")

TARGET_BASE_DIR = Path(DATA_ROOT / "map_station")
LOG_DIR = Path(DATA_ROOT / "code" / "logs" / "map")

TARGET_CRS = "EPSG:32632"          # UTM 32N - best balance for the station extent
RESAMPLING = Resampling.bilinear   # good for DEM elevation

# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"map_terrain_extraction_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("=" * 80)
    logging.info("MAP Terrain Metadata Preparation started")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Target CRS: {TARGET_CRS}")
    return log_file

# =============================================================================
# DEM EXTRACTION & VRT BUILDING (efficient, no full mosaic)
# =============================================================================
def extract_and_prepare_dem(tar_path: Path, work_dir: Path) -> Path:
    """Extract .tif files from tar.gz and build a VRT if multiple tiles."""
    extract_dir = work_dir / "dem_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Extracting DEM tiles from {tar_path.name} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tif_members = [m for m in tar.getmembers() if m.name.lower().endswith(".tif")]
        if not tif_members:
            raise RuntimeError("No .tif files found inside the tar.gz")
        tar.extractall(path=extract_dir, members=tif_members)

    tifs = sorted(extract_dir.rglob("*.tif"))
    logging.info(f"Found {len(tifs)} DEM tile(s)")

    if len(tifs) == 1:
        dem_source = tifs[0]
        logging.info(f"Single tile DEM: {dem_source}")
    else:
        vrt_path = work_dir / "dem_mosaic.vrt"
        logging.info(f"Building VRT from {len(tifs)} tiles â†’ {vrt_path.name}")
        cmd = ["gdalbuildvrt", "-overwrite", str(vrt_path)] + [str(t) for t in tifs]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(result.stderr)
            raise RuntimeError("gdalbuildvrt failed. Is GDAL installed and in PATH?")
        dem_source = vrt_path

    # Quick validation
    with rasterio.open(dem_source) as src:
        logging.info(f"DEM CRS: {src.crs}, bounds: {src.bounds}, res: {src.res}")
        if src.crs.to_string() != "EPSG:4326":
            logging.warning("DEM is not in EPSG:4326 - check source!")
    return dem_source

# =============================================================================
# REPROJECT DEM (once, to meter-based CRS)
# =============================================================================
def reproject_dem_to_target(src_path: Path, dst_path: Path, dst_crs: str) -> Path:
    logging.info(f"Reprojecting DEM to {dst_crs} ... (this may take a minute)")
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": src.nodata if src.nodata is not None else -9999.0,
            "dtype": "float32"
        })
        with rasterio.open(dst_path, "w", **meta) as dst:
            for i in range(1, src.count + 1):
                rio_reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=RESAMPLING
                )
    logging.info(f"Reprojected DEM saved â†’ {dst_path}")
    return dst_path

# =============================================================================
# COMPUTE TERRAIN ATTRIBUTES (pure numpy - Zevenbergen & Thorne method)
# No richdem needed â€” works everywhere rasterio + numpy are installed.
# =============================================================================
def compute_and_save_terrain_attributes(dem_path: Path, out_dir: Path) -> dict:
    """Compute slope, aspect, profile_curvature, plan_curvature and save as GeoTIFFs.
    Uses standard finite-difference formulas (Zevenbergen & Thorne 1987).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Computing terrain attributes with numpy (Zevenbergen & Thorne) ...")

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            dem = np.where(dem == nodata, np.nan, dem)

        # Pixel size in metres (projected CRS)
        xres = src.transform[0]
        yres = abs(src.transform[4])

        # First partial derivatives
        zy, zx = np.gradient(dem, yres, xres)   # zy along y (N-S), zx along x (E-W)

        # Slope (degrees)
        slope = np.arctan(np.sqrt(zx**2 + zy**2)) * (180.0 / np.pi)

        # Aspect: 0Â° = north, clockwise to 360Â°, -1 for flat (slope == 0)
        aspect = np.arctan2(-zx, zy) * (180.0 / np.pi)
        aspect = np.where(aspect < 0, 360.0 + aspect, aspect)
        aspect = np.where((slope == 0) | np.isnan(slope), -1.0, aspect)

        # Second derivatives for curvature (Zevenbergen & Thorne)
        zxx = np.gradient(zx, xres, axis=1)
        zyy = np.gradient(zy, yres, axis=0)
        zxy = np.gradient(zx, yres, axis=0)   # cross derivative (sufficiently accurate)

        p, q = zx, zy
        r, s, t = zxx, zxy, zyy

        denom = p**2 + q**2 + 1e-12   # avoid div-by-zero

        # Profile curvature (along steepest descent, affects acceleration)
        profile_curv = (p**2 * r + 2 * p * q * s + q**2 * t) / (denom * np.sqrt(1 + p**2 + q**2)**3)

        # Plan curvature (across slope, affects flow convergence/divergence)
        plan_curv = (q**2 * r - 2 * p * q * s + p**2 * t) / (denom * np.sqrt(1 + p**2 + q**2))

        # Set curvature to 0 on flat areas
        flat_mask = (slope == 0) | np.isnan(slope)
        profile_curv = np.where(flat_mask, 0.0, profile_curv)
        plan_curv = np.where(flat_mask, 0.0, plan_curv)

        attrs = {
            "slope": slope,
            "aspect": aspect,
            "profile_curvature": profile_curv,
            "plan_curvature": plan_curv,
        }

        meta = src.meta.copy()
        meta.update(dtype="float32", nodata=np.nan)

        terrain_paths = {}
        for name, arr in attrs.items():
            out_tif = out_dir / f"dem_{name}.tif"
            with rasterio.open(out_tif, "w", **meta) as dst:
                dst.write(arr.astype(np.float32), 1)
            terrain_paths[name] = out_tif
            logging.info(f"  Saved {name} â†’ {out_tif.name}")

    logging.info("Terrain attribute rasters ready (numpy implementation).")
    return terrain_paths


# =============================================================================
# SAMPLE ALL ATTRIBUTES (elevation + slope/aspect/curvatures)
# =============================================================================
def sample_all_attributes(
    stations_df: pd.DataFrame,
    reprojected_dem_path: Path,
    terrain_paths: dict,
    target_crs: str
) -> pd.DataFrame:
    logging.info(f"Sampling all attributes at {len(stations_df)} stations ...")

    gdf = gpd.GeoDataFrame(
        stations_df.copy(),
        geometry=gpd.points_from_xy(stations_df["lon"], stations_df["lat"]),
        crs="EPSG:4326"
    ).to_crs(target_crs)

    coords = [(geom.x, geom.y) for geom in gdf.geometry]

    # Elevation from reprojected DEM
    with rasterio.open(reprojected_dem_path) as src:
        elev_samples = [s[0] if s else np.nan for s in src.sample(coords)]
        elev_samples = [float(v) if not np.isnan(v) else np.nan for v in elev_samples]

    results = pd.DataFrame({
        "elev_dem": elev_samples
    }, index=stations_df.index)

    # Other attributes
    for attr, tif_path in terrain_paths.items():
        with rasterio.open(tif_path) as src:
            samples = list(src.sample(coords))
            values = [float(s[0]) if s and not np.isnan(s[0]) else np.nan for s in samples]
            results[attr] = values

    # Quality flag
    results["elev_diff"] = stations_df["hoehe"] - results["elev_dem"]
    results["has_terrain_data"] = results["slope"].notna() & (results["slope"] >= 0)

    n_missing = (~results["has_terrain_data"]).sum()
    if n_missing > 0:
        logging.warning(f"{n_missing} stations have no valid terrain data (outside DEM or nodata)")

    logging.info("All attributes sampled.")
    return results

# =============================================================================
# MAIN
# =============================================================================
def main():
    log_file = setup_logging()

    TARGET_BASE_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = TARGET_BASE_DIR / "processing"
    work_dir.mkdir(exist_ok=True)

    # 1. Copy original stations file
    target_stations = TARGET_BASE_DIR / "stations_overview.csv"
    shutil.copy2(SOURCE_STATIONS_CSV, target_stations)
    logging.info(f"Copied stations_overview.csv â†’ {target_stations}")

    # 2. Prepare DEM (extract + VRT if needed)
    dem_source = extract_and_prepare_dem(DEM_TAR_GZ, work_dir)

    # 3. Reproject
    reprojected_dem = work_dir / "dem_reprojected_32632.tif"
    if not reprojected_dem.exists():
        reproject_dem_to_target(dem_source, reprojected_dem, TARGET_CRS)
    else:
        logging.info(f"Using existing reprojected DEM: {reprojected_dem}")

    # 4. Compute terrain attributes
    terrain_dir = TARGET_BASE_DIR / "terrain_rasters"
    terrain_paths = compute_and_save_terrain_attributes(reprojected_dem, terrain_dir)

    # 5. Load stations and sample
    stations_df = pd.read_csv(target_stations)
    logging.info(f"Loaded {len(stations_df)} station records")

    terrain_samples = sample_all_attributes(
        stations_df, reprojected_dem, terrain_paths, TARGET_CRS
    )

    # 6. Merge and save final metadata
    final_df = pd.concat([stations_df, terrain_samples], axis=1)

    output_csv = TARGET_BASE_DIR / "stations_metadata_with_terrain.csv"
    final_df.to_csv(output_csv, index=False, encoding="utf-8")
    logging.info(f"Saved augmented metadata â†’ {output_csv}")

    # Summary stats
    logging.info("=== SUMMARY ===")
    logging.info(f"Stations with valid terrain data: {final_df['has_terrain_data'].sum()} / {len(final_df)}")
    logging.info(f"Mean elevation (DEM): {final_df['elev_dem'].mean():.1f} m")
    logging.info(f"Mean slope: {final_df['slope'].mean():.2f}Â°")
    logging.info(f"Aspect distribution (non-flat): {final_df[final_df['aspect'] >= 0]['aspect'].describe()}")

    logging.info("Script finished successfully.")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()

