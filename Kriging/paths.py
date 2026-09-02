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


def get_stations_metadata_path() -> Path:
    """
    Rich station metadata that includes terrain attributes
    (elev_dem, slope, aspect, curvature, ...).
    Used by the regime-clustering pipeline.
    """
    return get_raw_data_dir() / "stations_metadata_with_terrain.csv"


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


def get_clusters_dir(method: str) -> Path:
    """
    Directory that stores regime-clustering artefacts for a method.
    Example: .../IDW/Output/clusters
    """
    return get_method_output_dir(method) / "clusters"


def get_medoids_path(
    method: str,
    resolution: str,
    cluster_method: str,
    variable: str,
    start_date: str = None,
    end_date: str = None,
) -> Path:
    """Path to the medoids parquet produced by identify_regimes."""
    root = get_clusters_dir(method) / resolution / cluster_method
    if start_date or end_date:
        tag = f"{start_date or 'start'}_{end_date or 'end'}".replace("-", "")
        root = root / tag
    return root / f"{variable}_medoids.parquet"


def get_cluster_params_dir(method: str, resolution: str, cluster_method: str) -> Path:
    """
    Directory for free-parameter files trained on cluster medoids.
    Example: .../IDW/Output/cluster_params/half_hourly/gmm
    """
    d = get_method_output_dir(method) / "cluster_params" / resolution / cluster_method
    ensure_dir(d)
    return d


def get_cluster_params_path(
    method: str, resolution: str, cluster_method: str, variable: str
) -> Path:
    """Path to the per-variable cluster-parameter parquet."""
    return get_cluster_params_dir(method, resolution, cluster_method) / f"{variable}_params.parquet"


def get_rfsi_model_path(variable: str, time_resolution: str) -> Path:
    """Pooled RFSI forest (one file per variable × aggregation)."""
    d = get_method_output_dir("RFSI") / "models" / time_resolution
    ensure_dir(d)
    return d / f"{variable}_rfsi_pooled.joblib"


def get_time_splits_path() -> Path:
    return PROJECT_ROOT / "CODE" / "shared" / "splits" / "time_splits.yaml"


def get_rfsi_tuned_params_path(variable: str, time_resolution: str) -> Path:
    """Best pooled RFSI hyperparameters from the DEV medoid search."""
    d = get_method_output_dir("RFSI") / "cluster_params" / time_resolution / "pooled"
    ensure_dir(d)
    return d / f"{variable}_pooled_params.yaml"


def get_rgi_model_path(variable: str, time_resolution: str) -> Path:
    """Frozen RGI weights (one file per variable × aggregation)."""
    d = get_method_output_dir("RGI") / "models" / time_resolution
    ensure_dir(d)
    return d / f"{variable}_rgi.pt"


def get_rgi_tuned_params_path(variable: str, time_resolution: str) -> Path:
    """Best RGI hyperparameters from the DEV station-fold search."""
    d = get_method_output_dir("RGI") / "cluster_params" / time_resolution / "rgi"
    ensure_dir(d)
    return d / f"{variable}_rgi_params.yaml"


def get_kriging_model_path(variable: str, time_resolution: str) -> Path:
    """Frozen RK trend + variogram pack (one file per variable × aggregation)."""
    d = get_method_output_dir("Kriging") / "models" / time_resolution
    ensure_dir(d)
    return d / f"{variable}_kriging.joblib"


def get_kriging_tuned_params_path(variable: str, time_resolution: str) -> Path:
    """Best Kriging hyperparameters from the staged DEV search."""
    d = get_method_output_dir("Kriging") / "cluster_params" / time_resolution / "kriging"
    ensure_dir(d)
    return d / f"{variable}_kriging_params.yaml"


def get_nested_llocv_path(
    method: str,
    variable: str,
    time_resolution: str,
    domain: str = "full",
) -> Path:
    """Station-out nested LLOCV predictions (not the frozen-model parquet)."""
    d = get_method_output_dir(method) / "llocv" / domain / "nested" / time_resolution
    ensure_dir(d)
    return d / f"{variable}_{time_resolution}_nested_llocv.parquet"


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


def get_clusters_plot_dir(resolution: str = "half_hourly") -> Path:
    """
    Comparison plots for regime clustering.
    Example: .../Plots/Clusters/half_hourly
    """
    d = get_plots_root() / "Clusters" / resolution
    ensure_dir(d)
    return d


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

def get_interpolated_map_path(
    method: str,
    domain: str,
    variable: str,
    resolution: int,
    start_date: str = None,
    end_date: str = None,
    time_resolution: str = None,
) -> Path:
    """Path to the multi-timestep interpolated NetCDF.
    When start_date/end_date are given they are embedded in the filename
    so different periods never collide. Optional time_resolution is also
    embedded when provided.
    """
    d = get_method_output_dir(method) / "interpolated_maps" / domain / f"res_{resolution}m" / variable
    ensure_dir(d)
    parts = [variable, f"{resolution}m"]
    if time_resolution:
        parts.append(time_resolution)
    if start_date and end_date:
        parts.append(f"{start_date.replace('-', '')}-{end_date.replace('-', '')}")
    return d / ("_".join(parts) + ".nc")


def get_llocv_path(
    method: str,
    domain: str,
    variable: str,
    resolution: int,
    start_date: str = None,
    end_date: str = None,
    time_resolution: str = None,
) -> Path:
    """Path to the leave-one-out CV predictions parquet."""
    d = get_method_output_dir(method) / "llocv" / domain / f"res_{resolution}m" / variable
    ensure_dir(d)
    parts = [variable, f"{resolution}m"]
    if time_resolution:
        parts.append(time_resolution)
    if start_date and end_date:
        parts.append(f"{start_date.replace('-', '')}-{end_date.replace('-', '')}")
    return d / ("_".join(parts) + "_llocv.parquet")

def get_map_png_path(method: str, domain: str, resolution: int,
                     variable: str, time_label: str) -> Path:
    return get_map_output_dir(method, domain, resolution, variable) / f"{variable}_{time_label}.png"


def get_map_output_path(method: str, domain: str, variable: str, resolution: int, time_label: str) -> Path:
    """Path to save a single-timestep surface map PNG."""
    d = get_method_plot_dir(method, domain) / f"{resolution}m" / variable
    ensure_dir(d)
    return d / f"{variable}_{time_label}.png"
