# ============================================================
# paths.py  (full updated file)
# ============================================================
#!/usr/bin/env python3
"""
paths.py
Central path management for the Master Thesis.
Cleaned and aligned with the new master grid architecture.
"""

from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    current = Path(__file__).resolve().parent
    while current.parent != current:
        if (current / "CODE").exists() and (current / "BSS").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not determine project root.")


# ============================================================
# BASE PATHS
# ============================================================
PROJECT_ROOT: Path = get_project_root()
DATA_ROOT = PROJECT_ROOT


def get_raw_data_dir() -> Path:
    return DATA_ROOT / "Kombiniert"


def get_metadata_path() -> Path:
    return get_raw_data_dir() / "stations_overview.csv"


def get_dem_path() -> Path:
    return DATA_ROOT / "DEM_Source" / "COP30_mosaic_EPSG31287.tif"


def get_landcover_path() -> Path:
    return DATA_ROOT / "Landcover_Source" / "Reprojected" / "CLC2018_EPSG31287_100m.tif"


# ============================================================
# METHOD OUTPUT STRUCTURE
# ============================================================
def get_method_output_dir(method: str) -> Path:
    return DATA_ROOT / method / "Output"


def get_aggregated_dir(method: str) -> Path:
    return get_method_output_dir(method) / "aggregated"


def get_stations_dir(method: str) -> Path:
    return get_method_output_dir(method) / "stations"


def get_grids_dir(method: str) -> Path:
    return get_method_output_dir(method) / "grids"


def get_maps_dir(method: str) -> Path:
    return get_method_output_dir(method) / f"{method.lower()}_maps"


def get_clipped_dems_dir(method: str) -> Path:
    return get_method_output_dir(method) / "clipped_dems"


def get_clipped_landcover_dir(method: str) -> Path:
    return get_method_output_dir(method) / "clipped_landcover"


# ============================================================
# NEW: MASTER GRID HELPER (recommended)
# ============================================================
def get_master_grid_path(method: str, resolution: int) -> Path:
    """Path to master grid for a given method and resolution."""
    return get_grids_dir(method) / "master" / f"res_{resolution}m" / "grid.nc"


# ============================================================
# AGGREGATED DATA
# ============================================================
def get_aggregated_data_path(method: str, time_resolution: str) -> Path:
    aggregated_dir = get_aggregated_dir(method)
    filename_map = {
        "weekly": "weekly_station_data.parquet",
        "daily": "daily_station_data.parquet",
        "monthly": "monthly_station_data.parquet",
        "half_hourly": "half_hourly_station_data.parquet",
    }
    filename = filename_map.get(time_resolution, f"{time_resolution}_station_data.parquet")
    return aggregated_dir / filename


# ============================================================
# DOMAIN-SPECIFIC HELPERS (kept for transition period)
# ============================================================
def get_domain_stations_path(method: str, domain: str) -> Path:
    return get_stations_dir(method) / domain / "stations_projected.parquet"


def get_domain_grid_path(method: str, domain: str, resolution: int) -> Path:
    """Legacy path - will be phased out."""
    return get_grids_dir(method) / domain / f"res_{resolution}m" / "grid.nc"


def get_clipped_landcover_path(method: str, domain: str) -> Path:
    return get_clipped_landcover_dir(method) / f"CLC2018_clipped_{domain}_EPSG31287.tif"


# ============================================================
# PLOTTING OUTPUT STRUCTURE
# ============================================================
def get_plots_root() -> Path:
    return DATA_ROOT / "Plots"


def get_method_plot_dir(method: str, domain: str = "full") -> Path:
    return get_plots_root() / method / domain


def get_validation_scatter_dir(method: str, domain: str = "full") -> Path:
    return get_method_plot_dir(method, domain) / "validation_scatters"


def get_validation_scatter_path(method: str, domain: str, variable: str, resolution: int) -> Path:
    d = get_validation_scatter_dir(method, domain) / f"{resolution}m" / variable
    ensure_dir(d)
    return d / f"{variable}_{resolution}m_validation_scatter.png"


def get_surface_map_plot_dir(method: str, domain: str, resolution: int, variable: str) -> Path:
    d = get_method_plot_dir(method, domain) / f"{resolution}m" / variable
    ensure_dir(d)
    return d


def get_surface_map_plot_path(method: str, domain: str, resolution: int,
                              variable: str, time_label: str) -> Path:
    d = get_surface_map_plot_dir(method, domain, resolution, variable)
    return d / f"{variable}_{time_label}.png"


def get_12panel_summary_dir(method: str, domain: str, resolution: int) -> Path:
    return get_method_plot_dir(method, domain) / f"{resolution}m" / "12panel_summary"


# ============================================================
# UTILITY
# ============================================================
def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# MAP OUTPUT (fixed version)
# ============================================================
def get_map_output_dir(method: str, domain: str, resolution: int, variable: str) -> Path:
    d = get_method_plot_dir(method, domain) / f"{resolution}m" / variable
    ensure_dir(d)
    return d

def get_interpolated_map_path(method: str, domain: str, variable: str, resolution: int,
                              start_date: str = None, end_date: str = None) -> Path:
    """Path to the multi-timestep interpolated NetCDF.
    When start_date/end_date are given they are embedded in the filename
    so different periods never collide.
    """
    d = get_method_output_dir(method) / "interpolated_maps" / domain / f"res_{resolution}m" / variable
    ensure_dir(d)
    if start_date and end_date:
        tag = f"{start_date.replace('-', '')}-{end_date.replace('-', '')}"
        return d / f"{variable}_{resolution}m_{tag}.nc"
    return d / f"{variable}_{resolution}m.nc"


def get_llocv_path(method: str, domain: str, variable: str, resolution: int,
                   start_date: str = None, end_date: str = None) -> Path:
    """Companion LLOCV parquet next to the interpolated map NC."""
    nc = get_interpolated_map_path(method, domain, variable, resolution, start_date, end_date)
    return nc.with_name(nc.stem + "_llocv.parquet")

def get_map_png_path(method: str, domain: str, resolution: int,
                     variable: str, time_label: str) -> Path:
    return get_map_output_dir(method, domain, resolution, variable) / f"{variable}_{time_label}.png"


def get_map_output_path(method: str, domain: str, variable: str, resolution: int, time_label: str) -> Path:
    """Path to save a single-timestep surface map PNG."""
    d = get_method_plot_dir(method, domain) / f"{resolution}m" / variable
    ensure_dir(d)
    return d / f"{variable}_{time_label}.png"