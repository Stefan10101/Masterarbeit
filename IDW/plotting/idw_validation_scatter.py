#!/usr/bin/env python3
"""
idw_validation_scatter.py
Master thesis â€” validation scatterplots (predicted vs actual) for modified IDW climate surfaces (Alps)

Logic:
- Default: pool ALL months + years per variable (recommended for thesis)
- x = actual station observation
- y = IDW surface value at nearest grid cell to station (x, y)
- One clean figure per variable/resolution with 1:1 line, OLS fit, n/RMSE/MAE/bias/RÂ²
- Efficient: cKDTree built once per NC, vectorized queries, pure-numpy stats
- Reuses your existing paths, parquet readers and batch/single mode

Requires: scipy (pip install scipy) for fast nearest-neighbor lookup.
"""

from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import warnings
from scipy.spatial import cKDTree
import gc

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

# ====================== CONFIGURATION (copy/adapt from visualize_idw_monthly.py) ======================
NC_FILE = DATA_ROOT / "idw" / "output" / "snow_max_500m.nc"

BATCH_MODE = True
INPUT_DIR = DATA_ROOT / "idw" / "output"
OUTPUT_BASE = DATA_ROOT / "plots" / "idw"

STATION_LOOKUP = DATA_ROOT / "idw" / "input" / "stations_projected.parquet"
MONTHLY_STATION_DATA = DATA_ROOT / "idw" / "input" / "monthly_station_data.parquet"

# Filter to one resolution only (set to None to process every *_*.nc found)
FILTER_RES = "100m"

SCATTER_DPI = 250
# =====================================================================================================


def get_stations_with_actual(var_name, year, month, station_lookup_path, monthly_data_path):
    """Return df with station_name, actual (var), x, y for all stations that have data that month."""
    if station_lookup_path is None or monthly_data_path is None:
        return None
    try:
        stations = pd.read_parquet(station_lookup_path)
        monthly = pd.read_parquet(monthly_data_path)

        if 'year_month' in monthly.columns:
            monthly['year'] = monthly['year_month'].str[:4].astype(int)
            monthly['month'] = monthly['year_month'].str[5:7].astype(int)
        else:
            return None

        monthly = monthly[(monthly['year'] == year) & (monthly['month'] == month)]
        if var_name not in monthly.columns:
            return None

        valid = monthly[monthly[var_name].notna()][['station_name', var_name]].copy()
        valid = valid.rename(columns={var_name: 'actual'})
        valid_stations = pd.merge(
            valid,
            stations[['station_name', 'x', 'y']],
            on='station_name',
            how='inner'
        )
        valid_stations = valid_stations.dropna(subset=['x', 'y', 'actual'])
        return valid_stations if len(valid_stations) > 0 else None
    except Exception as e:
        print(f"    Warning in get_stations_with_actual ({year}-{month:02d}, {var_name}): {e}")
        return None


def compute_stats(actual, predicted):
    """Pure-numpy validation metrics (no extra dependencies)."""
    mask = np.isfinite(actual) & np.isfinite(predicted)
    a = np.asarray(actual)[mask]
    p = np.asarray(predicted)[mask]
    n = len(a)
    if n < 2:
        return dict(n=n, rmse=np.nan, mae=np.nan, bias=np.nan, r2=np.nan)

    rmse = np.sqrt(np.mean((p - a)**2))
    mae = np.mean(np.abs(p - a))
    bias = np.mean(p - a)

    if np.std(a) > 0 and np.std(p) > 0:
        r2 = np.corrcoef(a, p)[0, 1]**2
    else:
        r2 = np.nan
    return dict(n=n, rmse=rmse, mae=mae, bias=bias, r2=r2)


def plot_validation_scatter(actual, predicted, var_name, long_name, units,
                            resolution, year_min, year_max, out_path, dpi):
    """Thesis-ready scatter: 1:1 line + OLS fit + stats box."""
    mask = np.isfinite(actual) & np.isfinite(predicted)
    a = np.asarray(actual)[mask]
    p = np.asarray(predicted)[mask]
    if len(a) == 0:
        print(f"  No valid pairs for {var_name} â€” skipping plot.")
        return

    stats = compute_stats(a, p)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    # scatter (alpha handles overlap)
    ax.scatter(a, p, s=10, alpha=0.4, c="#2E86AB", edgecolors="none", zorder=3)

    # data range + padding
    vmin = min(np.min(a), np.min(p))
    vmax = max(np.max(a), np.max(p))
    pad = (vmax - vmin) * 0.06

    # 1:1 reference
    ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad],
            'k--', lw=1.3, label="1:1 line", zorder=4)

    # OLS fit (numpy only)
    if stats['n'] > 2 and np.isfinite(stats['r2']):
        try:
            slope, intercept = np.polyfit(a, p, 1)
            ax.plot([vmin - pad, vmax + pad],
                    [slope * (vmin - pad) + intercept, slope * (vmax + pad) + intercept],
                    color="#E94F37", lw=1.8, alpha=0.9,
                    label=f"OLS fit (RÂ² = {stats['r2']:.3f})", zorder=5)
        except Exception:
            pass

    ax.set_xlabel(f"Actual {long_name} [{units}]", fontsize=11)
    ax.set_ylabel(f"IDW Predicted {long_name} [{units}]", fontsize=11)
    ax.set_title(f"IDW Validation Scatter â€” {long_name} ({resolution} m)\n"
                 f"{year_min}â€“{year_max} (all months/years pooled)  |  n = {stats['n']}",
                 fontsize=11, pad=8)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, zorder=0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)

    # stats box
    txt = (f"n     = {stats['n']}\n"
           f"RMSE  = {stats['rmse']:.3f}\n"
           f"MAE   = {stats['mae']:.3f}\n"
           f"Bias  = {stats['bias']:.3f}\n"
           f"RÂ²    = {stats['r2']:.3f}")
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, fontsize=9,
            ha="right", va="bottom", linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.95),
            zorder=6)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def process_one_nc_for_scatter(nc_path, output_base, station_lookup, monthly_station_data, dpi):
    print(f"\n=== Processing {Path(nc_path).name} ===")
    ds = Dataset(nc_path, "r")

    time_var = ds.variables["time"]
    x = ds.variables["x"][:]
    y = ds.variables["y"][:]

    var_name = None
    for v in ds.variables:
        if v not in ["time", "cell", "x", "y", "elev", "p", "Fz", "k", "rmse", "mae"]:
            var_name = v
            break
    if var_name is None:
        print("  No climate variable found â€” skipping.")
        ds.close()
        return

    long_name = getattr(ds.variables[var_name], "long_name", var_name)
    units = getattr(ds.variables[var_name], "units", "")
    resolution = ds.getncattr("resolution_m") if "resolution_m" in ds.ncattrs() else 0
    times = num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
    year_min, year_max = times[0].year, times[-1].year

    # KDTree once (grid geometry is constant across months)
    grid_xy = np.column_stack((x.astype(np.float64), y.astype(np.float64)))
    tree = cKDTree(grid_xy)
    print(f"  {var_name} | {resolution} m | {len(times)} months | KDTree built on {len(x)} cells")

    all_actual = []
    all_predicted = []

    for t in range(len(times)):
        year, month = times[t].year, times[t].month
        stations_df = get_stations_with_actual(var_name, year, month, station_lookup, monthly_station_data)
        if stations_df is None or len(stations_df) == 0:
            continue

        station_xy = stations_df[['x', 'y']].values.astype(np.float64)
        _, idx = tree.query(station_xy, k=1)          # nearest grid cell index
        month_slice = ds.variables[var_name][t]       # 1D array, same ordering as x/y
        predicted = month_slice[idx]
        actual = stations_df['actual'].values

        all_actual.extend(actual)
        all_predicted.extend(predicted)

        if (t + 1) % 12 == 0 or t == len(times) - 1:
            print(f"    {t + 1}/{len(times)} months  ({len(all_actual)} pairs collected)")

    ds.close()
    gc.collect()

    if len(all_actual) == 0:
        print(f"  No matching station observations found â€” no scatter created.")
        return

    out_dir = Path(output_base) / "validation_scatters"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{var_name}_{resolution}m_validation_scatter.png"

    plot_validation_scatter(np.array(all_actual), np.array(all_predicted),
                            var_name, long_name, units, resolution,
                            year_min, year_max, out_path, dpi)

    print(f"Finished {var_name} ({resolution} m) â€” total {len(all_actual)} stationâ€“month pairs")


def main():
    if BATCH_MODE:
        nc_files = sorted(glob.glob(str(Path(INPUT_DIR) / "*_*.nc")))
        if FILTER_RES:
            nc_files = [f for f in nc_files if f"_{FILTER_RES}" in Path(f).name]
        print(f"Found {len(nc_files)} files (FILTER_RES = {FILTER_RES}) in {INPUT_DIR}")

        for nc in nc_files:
            process_one_nc_for_scatter(nc, OUTPUT_BASE, STATION_LOOKUP, MONTHLY_STATION_DATA, SCATTER_DPI)
    else:
        process_one_nc_for_scatter(NC_FILE, OUTPUT_BASE, STATION_LOOKUP, MONTHLY_STATION_DATA, SCATTER_DPI)

    print("\n=== All validation scatters finished ===")
    print(f"Output folder: {Path(OUTPUT_BASE) / 'validation_scatters'}")


if __name__ == "__main__":
    main()

