import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT


DEM_PATH = DATA_ROOT / "map_station" / "processing" / "dem_reprojected_32632.tif"
LAYER_PATH = DATA_ROOT / "map_station" / "dem_features" / "tpi_r300m.tif"   # â† change to any layer
OUT_PNG = DATA_ROOT / "map_station" / "dem_features" / "tpi_r300m_hillshade.png"

# Load DEM and layer
with rasterio.open(DEM_PATH) as src:
    dem = src.read(1).astype(np.float32)
    transform = src.transform

with rasterio.open(LAYER_PATH) as src:
    layer = src.read(1).astype(np.float32)

# Simple hillshade (azimuth 315Â°, altitude 45Â°)
ls = LightSource(azdeg=315, altdeg=45)
hillshade = ls.hillshade(dem, vert_exag=1.0)

# Plot
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(hillshade, cmap='gray', extent=[transform[2], transform[2] + transform[0]*dem.shape[1],
                                          transform[5] + transform[4]*dem.shape[0], transform[5]])
im = ax.imshow(layer, cmap='RdBu_r', alpha=0.6, extent=...)   # adjust cmap & alpha
plt.colorbar(im, ax=ax, label='TPI (m)')
ax.set_title('TPI 300 m on hillshade')
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
print(f"Saved â†’ {OUT_PNG}")
