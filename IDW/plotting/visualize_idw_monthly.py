#!/usr/bin/env python3
"""
visualize_idw_v2.py
Time-resolution aware plotting script for the new 2D IDW output.
"""

from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from netCDF4 import Dataset, num2date
import warnings

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT


warnings.filterwarnings("ignore", category=UserWarning)

# ====================== CONFIGURATION ======================
NC_FILE = DATA_ROOT / "idw" / "output" / "idw_maps" / "inn_valley" / "rh_mean_100m.nc"
BATCH_MODE = False
INPUT_DIR = DATA_ROOT / "idw" / "output"
OUTPUT_BASE = DATA_ROOT / "plots" / "idw"

STATION_LOOKUP = DATA_ROOT / "idw" / "input" / "stations_projected.parquet"
MONTHLY_STATION_DATA = DATA_ROOT / "idw" / "input" / "monthly_station_data.parquet"

CMAP_DICT = {
    "temp_mean": "RdBu_r", "temp_min": "RdBu_r", "temp_max": "RdBu_r",
    "precip_sum": "Blues", "snow_mean": "BuPu", "snow_min": "BuPu",
    "snow_max": "BuPu", "default": "viridis"
}
CMAP_OVERRIDE = None
VMIN_VMAX = None

SHOW_TOPO = True
TOPO_LEVELS = 8
DPI = 200

MAKE_12PANEL_SUMMARY = True
TWELVE_PANEL_YEAR = 2022
MAKE_MAY_COMPARISONS = True
# ============================================================


def get_time_label(timestamp, time_resolution):
    """Generate appropriate filename label based on time resolution."""
    if time_resolution == "weekly":
        return timestamp.strftime("%YW%W")
    elif time_resolution == "daily":
        return timestamp.strftime("%Y%m%d")
    elif time_resolution in ["half_hourly", "day", "night"]:
        return timestamp.strftime("%Y%m%d_%H%M")
    else:
        return timestamp.strftime("%Y%m%d")


def get_valid_stations_for_time(var_name, timestamp, station_lookup_path, monthly_data_path):
    # Simplified version - you can expand this later for daily/half-hourly if needed
    try:
        stations = pd.read_parquet(station_lookup_path)
        return stations
    except Exception:
        return None


def plot_idw_map(ax, x, y, grid, elev_grid, cmap, norm, title,
                 stations_df=None, rmse=None, mae=None, show_topo=True,
                 params_text=None, is_summary=False):

    pcm = ax.pcolormesh(x, y, np.ma.masked_invalid(grid), cmap=cmap, norm=norm, shading="auto")

    if show_topo and elev_grid is not None:
        ax.contour(x, y, np.ma.masked_invalid(elev_grid), levels=TOPO_LEVELS,
                   colors="k", linewidths=0.5, alpha=0.4)

    if stations_df is not None and len(stations_df) > 0:
        ax.scatter(stations_df['x'], stations_df['y'], s=9, c="black", alpha=0.7, zorder=5)

    ax.set_aspect("equal")
    if not is_summary:
        ax.set_xlabel("Easting [m] (EPSG:31287)")
        ax.set_ylabel("Northing [m] (EPSG:31287)")
    ax.set_title(title, fontsize=9 if is_summary else 10)

    info_lines = []
    if params_text:
        info_lines.append(params_text)
    if rmse is not None and mae is not None:
        info_lines.append(f"RMSE={rmse:.2f}  MAE={mae:.2f}")
    if info_lines:
        ax.text(0.99, 0.01, "\n".join(info_lines), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7 if is_summary else 8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.9))
    return pcm


def process_one_nc(nc_path):
    print(f"\n=== Processing {Path(nc_path).name} ===")
    ds = Dataset(nc_path, "r")

    domain = ds.getncattr("domain") if "domain" in ds.ncattrs() else "unknown"
    resolution = ds.getncattr("resolution_m") if "resolution_m" in ds.ncattrs() else 0
    res_str = f"{resolution}m"
    time_resolution = ds.getncattr("time_resolution") if "time_resolution" in ds.ncattrs() else "unknown"

    var_name = [v for v in ds.variables if v not in ["time", "x", "y", "elev", "p", "Fz", "k", "rmse", "mae"]][0]
    long_name = getattr(ds.variables[var_name], "long_name", var_name)
    units = getattr(ds.variables[var_name], "units", "")

    time_var = ds.variables["time"]
    times = num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
    x = ds.variables["x"][:]
    y = ds.variables["y"][:]
    elev = ds.variables["elev"][:] if "elev" in ds.variables else None

    p_arr = ds.variables["p"][:] if "p" in ds.variables else None
    fz_arr = ds.variables["Fz"][:] if "Fz" in ds.variables else None
    k_arr = ds.variables["k"][:] if "k" in ds.variables else None
    rmse_arr = ds.variables["rmse"][:] if "rmse" in ds.variables else None
    mae_arr = ds.variables["mae"][:] if "mae" in ds.variables else None

    # === Organized folder: Domain / Resolution / Variable ===
    out_dir = Path(OUTPUT_BASE) / domain / res_str / var_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Colormap scaling
    cmap = CMAP_OVERRIDE or CMAP_DICT.get(var_name, CMAP_DICT["default"])
    diverging = any(kw in str(cmap).lower() for kw in ["rdbu", "coolwarm", "bwr", "seismic"])
    if VMIN_VMAX:
        vmin, vmax = VMIN_VMAX
    elif diverging:
        vmin, vmax = np.nanpercentile(ds.variables[var_name][:], [5, 95])
    else:
        vmin, vmax = np.nanpercentile(ds.variables[var_name][:], [2, 98])
    norm = Normalize(vmin=vmin, vmax=vmax)

    print(f"Variable: {var_name} | Domain: {domain} | Resolution: {resolution} m | "
          f"Time Resolution: {time_resolution} | Timesteps: {len(times)}")

    for t_idx, t in enumerate(times):
        grid = ds.variables[var_name][t_idx]
        elev_grid = elev if SHOW_TOPO else None

        stations_df = get_valid_stations_for_time(var_name, t, STATION_LOOKUP, MONTHLY_STATION_DATA)
        rmse = float(rmse_arr[t_idx]) if rmse_arr is not None else None
        mae = float(mae_arr[t_idx]) if mae_arr is not None else None

        params_text = None
        if p_arr is not None and fz_arr is not None and k_arr is not None:
            params_text = f"p={p_arr[t_idx]:.2f}  Fz={fz_arr[t_idx]:.2f}  k={int(k_arr[t_idx])}"

        time_label = get_time_label(t, time_resolution)
        title = f"{long_name} â€” {time_label} ({resolution} m)"

        fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
        plot_idw_map(ax, x, y, grid, elev_grid, cmap, norm, title,
                     stations_df=stations_df, rmse=rmse, mae=mae,
                     show_topo=SHOW_TOPO, params_text=params_text)

        cbar = fig.colorbar(ax.collections[0], ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(f"{long_name} [{units}]", fontsize=9)

        fname = f"{var_name}_{time_label}.png"
        fig.savefig(out_dir / fname, bbox_inches="tight", dpi=DPI)
        plt.close(fig)

        if (t_idx + 1) % 3 == 0 or t_idx == len(times) - 1:
            print(f"  {t_idx + 1}/{len(times)} plots done â†’ {fname}")

    ds.close()
    print(f"Finished {Path(nc_path).name}")


def create_12panel_summary(nc_path):
    """12-panel summary saved inside Domain/Resolutionm/12panel_summary/"""
    print(f"Creating 12-panel summary for {Path(nc_path).name} ...")
    ds = Dataset(nc_path, "r")

    domain = ds.getncattr("domain") if "domain" in ds.ncattrs() else "unknown"
    resolution = ds.getncattr("resolution_m") if "resolution_m" in ds.ncattrs() else 0
    res_str = f"{resolution}m"

    var_name = [v for v in ds.variables if v not in ["time", "x", "y", "elev", "p", "Fz", "k", "rmse", "mae"]][0]
    long_name = getattr(ds.variables[var_name], "long_name", var_name)
    units = getattr(ds.variables[var_name], "units", "")

    time_var = ds.variables["time"]
    times = num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
    year_indices = [i for i, t in enumerate(times) if t.year == TWELVE_PANEL_YEAR]

    if not year_indices:
        print(f"  No data for year {TWELVE_PANEL_YEAR}")
        ds.close()
        return

    out_dir = Path(OUTPUT_BASE) / domain / res_str / "12panel_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    # (Plotting logic for 12-panel can be added here - kept minimal for now)
    print(f"  12-panel summary would be saved to: {out_dir}")
    ds.close()


def main():
    if BATCH_MODE:
        nc_files = sorted(glob.glob(str(Path(INPUT_DIR) / "*_*.nc")))
        for nc in nc_files:
            process_one_nc(nc)
            if MAKE_12PANEL_SUMMARY:
                create_12panel_summary(nc)
    else:
        process_one_nc(NC_FILE)
        if MAKE_12PANEL_SUMMARY:
            create_12panel_summary(NC_FILE)

    print("\n=== All processing finished ===")
    print(f"Check output folder: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
