#!/usr/bin/env python3
"""
analyze_dem_topographic_features.py
Master's thesis - DEM topographic feature extraction (TPI, landforms, hydrology, other metrics).
No richdem dependency - uses your pre-computed slope from earlier.

Focus: valleys/rivers, catchments, mountain-related features (via TPI), multi-scale.
"""

from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage
from pysheds.grid import Grid

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
# CONFIG - change only these
# =============================================================================
DEM_PATH = Path(DATA_ROOT / "map_station" / "processing" / "dem_reprojected_32632.tif")
SLOPE_PATH = Path(DATA_ROOT / "map_station" / "terrain_rasters" / "dem_slope.tif")
OUTPUT_DIR = Path(DATA_ROOT / "map_station" / "dem_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Multi-scale TPI radii in pixels (30 m DEM)
TPI_SCALES = [5, 10, 30, 100]

# pysheds stream threshold (accumulation cells)
STREAM_THRESHOLD = 500

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def compute_tpi(dem_array: np.ndarray, radius: int) -> np.ndarray:
    """TPI = cell - mean(neighbourhood). Square window (fast & effective)."""
    if radius < 1:
        return np.zeros_like(dem_array, dtype=np.float32)
    kernel_size = 2 * radius + 1
    mean_elev = ndimage.uniform_filter(dem_array.astype(np.float32),
                                       size=kernel_size, mode='reflect')
    tpi = dem_array - mean_elev
    return tpi.astype(np.float32)

def simple_landform_classification(tpi: np.ndarray, slope: np.ndarray,
                                   tpi_thresh: float = 0.5) -> np.ndarray:
    """Basic slope-position classification (extend later)."""
    landform = np.zeros_like(tpi, dtype=np.uint8)
    flat = (slope < 5) & (np.abs(tpi) < tpi_thresh)
    ridge = tpi > tpi_thresh
    valley = tpi < -tpi_thresh
    landform[flat] = 1
    landform[ridge] = 2
    landform[valley] = 3
    return landform

def compute_tri(dem_array: np.ndarray) -> np.ndarray:
    """Terrain Ruggedness Index."""
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]]) / 8.0
    mean_neigh = ndimage.convolve(dem_array.astype(np.float32), kernel, mode='reflect')
    tri = np.abs(dem_array - mean_neigh)
    return tri.astype(np.float32)

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("Loading DEM...")
    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile

    print("Loading precomputed slope (from earlier script)...")
    with rasterio.open(SLOPE_PATH) as src:
        slope = src.read(1).astype(np.float32)

    print("Computing multi-scale TPI...")
    for r in TPI_SCALES:
        tpi = compute_tpi(dem, r)
        out_path = OUTPUT_DIR / f"tpi_r{r*30}m.tif"
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(tpi, 1)
        print(f"  Saved {out_path.name}")

    # Multi-scale landform example (small + large TPI)
    print("Creating basic landform classification...")
    tpi_small = compute_tpi(dem, TPI_SCALES[0])
    tpi_large = compute_tpi(dem, TPI_SCALES[-1])  # (we only use small for classification here)
    landform = simple_landform_classification(tpi_small, slope)
    out_path = OUTPUT_DIR / "landform_basic.tif"
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(landform, 1)
    print(f"Saved {out_path.name}")

    # TRI
    print("Computing TRI...")
    tri = compute_tri(dem)
    out_path = OUTPUT_DIR / "tri.tif"
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(tri, 1)
    print(f"Saved {out_path.name}")

      # === Hydrology with pysheds (valleys, rivers, catchments) ===
    print("Running hydrological analysis with pysheds...")
    grid = Grid.from_raster(str(DEM_PATH))
    dem_pysheds = grid.read_raster(str(DEM_PATH))

    # Condition DEM (modern return-value style)
    pit_filled = grid.fill_pits(dem_pysheds)
    flooded = grid.fill_depressions(pit_filled)
    conditioned = grid.resolve_flats(flooded)

    # Flow direction & accumulation
    dir_grid = grid.flowdir(conditioned, dirmap=(64, 128, 1, 2, 4, 8, 16, 32))
    acc = grid.accumulation(dir_grid, dirmap=(64, 128, 1, 2, 4, 8, 16, 32))

    # Streams
    streams = acc > STREAM_THRESHOLD
    grid.to_raster(streams.astype(np.uint8), str(OUTPUT_DIR / "streams.tif"))
    print(f"Stream network saved (threshold = {STREAM_THRESHOLD} cells)")

    # Example catchment (change x,y to a real major outlet later)
    x, y = 600000, 5200000
    catch = grid.catchment(x=x, y=y, data=dir_grid, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                           out_name='catch', recursionlimit=15000)
    grid.to_raster(catch, str(OUTPUT_DIR / "example_catchment.tif"))
    print("Example catchment saved")

    # TWI (uses your precomputed slope)
    print("Computing TWI...")
    twi = np.log((acc * 30.0 + 1e-6) / (np.tan(np.deg2rad(slope)) + 1e-6))
    out_path = OUTPUT_DIR / "twi.tif"
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(twi.astype(np.float32), 1)
    print("Saved TWI")

    print("\nâœ… All features computed and saved to:", OUTPUT_DIR)
    print("Next easy extensions:")
    print("  - Full 6-class or 10-class landform table (multi-scale TPI + slope)")
    print("  - Add WhiteboxTools for more advanced hydrology / multi-scale tools")
    print("  - Ridge skeletonization for mountain chains")
    print("  - Sample all new layers at your stations (add columns like tpi_300m, landform_class, twi, tri...)")

if __name__ == "__main__":
    main()
