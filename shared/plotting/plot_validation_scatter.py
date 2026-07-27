#!/usr/bin/env python3
"""
plot_validation_scatter.py
Pooled validation scatterplots (actual vs predicted) for IDW / BSS / RFSI

Supports:
- mode: "config"       → uses method + domain from config.yaml
- mode: "folder_loop"  → processes any folder of .nc files
- Both 1D (flattened) and 2D grids automatically
"""

from pathlib import Path
import sys
import glob
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parents[2]))
from paths import (
    get_aggregated_data_path,
    get_domain_stations_path,
    get_validation_scatter_path,
    get_maps_dir,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)["plotting"]


def get_mode():
    return cfg.get("mode", "config").lower()


def get_input_nc_files():
    mode = get_mode()

    if mode == "folder_loop":
        folder = cfg.get("input_nc_folder")
        if not folder:
            raise ValueError("input_nc_folder must be set when mode == 'folder_loop'")
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        return sorted(glob.glob(str(folder / "*.nc")))

    # mode == "config"
    method = cfg.get("method")
    domain = cfg.get("domain", "full")
    resolutions = cfg.get("resolutions")

    if not method:
        raise ValueError("method must be set when mode == 'config'")

    maps_dir = get_maps_dir(method) / domain
    if not maps_dir.exists():
        raise FileNotFoundError(f"Maps directory not found: {maps_dir}")

    nc_files = sorted(glob.glob(str(maps_dir / "*.nc")))

    if resolutions:
        res_strs = [f"_{r}m.nc" for r in resolutions]
        nc_files = [f for f in nc_files if any(r in f for r in res_strs)]

    return nc_files


def get_plotting_params():
    return {"dpi": cfg.get("scatter_dpi", 300)}


def compute_stats(actual, predicted):
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


def plot_validation_scatter(actual, predicted, var_name, units, resolution, out_path, dpi):
    mask = np.isfinite(actual) & np.isfinite(predicted)
    a = np.asarray(actual)[mask]
    p = np.asarray(predicted)[mask]
    if len(a) == 0:
        print(f"  No valid pairs for {var_name} — skipping.")
        return

    stats = compute_stats(a, p)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.scatter(a, p, s=10, alpha=0.4, c="#2E86AB", edgecolors="none", zorder=3)

    vmin = min(np.min(a), np.min(p))
    vmax = max(np.max(a), np.max(p))
    pad = (vmax - vmin) * 0.06

    ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad],
            'k--', lw=1.3, label="1:1 line", zorder=4)

    if stats['n'] > 2 and np.isfinite(stats['r2']):
        try:
            slope, intercept = np.polyfit(a, p, 1)
            ax.plot([vmin - pad, vmax + pad],
                    [slope * (vmin - pad) + intercept, slope * (vmax + pad) + intercept],
                    color="#E94F37", lw=1.8, alpha=0.9,
                    label=f"OLS fit (R² = {stats['r2']:.3f})", zorder=5)
        except Exception:
            pass

    unit_str = f" [{units}]" if units else ""
    ax.set_xlabel(f"Actual {var_name}{unit_str}", fontsize=11)
    ax.set_ylabel(f"Predicted {var_name}{unit_str}", fontsize=11)
    ax.set_title(f"Validation Scatter — {var_name} ({resolution} m)\n"
                 f"n = {stats['n']} (pooled across all timesteps)",
                 fontsize=11, pad=8)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, zorder=0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)

    txt = (f"n     = {stats['n']}\n"
           f"RMSE  = {stats['rmse']:.3f}\n"
           f"MAE   = {stats['mae']:.3f}\n"
           f"Bias  = {stats['bias']:.3f}\n"
           f"R²    = {stats['r2']:.3f}")
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, fontsize=9,
            ha="right", va="bottom", linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.95),
            zorder=6)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def process_nc_file(nc_path):
    print(f"\n=== Processing {Path(nc_path).name} ===")
    ds = xr.open_dataset(nc_path)

    # === Robust method detection ===
    method_attr = ds.attrs.get("method", "").upper()
    if "RFSI" in method_attr or "RANDOM FOREST" in method_attr:
        method = "RFSI"
    elif "BSS" in method_attr or "BILINEAR" in method_attr:
        method = "BSS"
    elif "IDW" in method_attr:
        method = "IDW"
    else:
        parent = Path(nc_path).parent.name.lower()
        if "rfsi" in parent:
            method = "RFSI"
        elif "bss" in parent:
            method = "BSS"
        else:
            method = "IDW"

    domain = ds.attrs.get("domain", "full")
    resolution = int(ds.attrs.get("resolution_m", 0))
    time_res = ds.attrs.get("time_resolution", "monthly")

    # === Robust variable detection ===
    aux_vars = {"time", "y", "x", "elev", "p", "Fz", "k", "rmse", "mae", "nse", "kge",
                "n_segments", "tau", "gcv", "effective_df", "n_obs"}

    data_vars = [v for v in ds.data_vars if v not in aux_vars]
    var_name = data_vars[0] if len(data_vars) == 1 else None

    if var_name == "prediction" or var_name is None:
        fname = Path(nc_path).stem.lower()
        for v in ["precip_sum", "temp_mean", "temp_min", "temp_max",
                  "snow_mean", "snow_max", "snow_min", "rh_mean",
                  "wind_mean", "wind_max"]:
            if v in fname:
                var_name = v
                break

    if var_name is None:
        print("  No climate variable found — skipping.")
        ds.close()
        return

    data_var = "prediction" if "prediction" in ds.data_vars else var_name
    units = ds[data_var].attrs.get("units", "") if data_var in ds else ""

    print(f"  Method: {method} | Domain: {domain} | Var: {var_name} | Resolution: {resolution} m")

    # Load station data
    try:
        stations = pd.read_parquet(get_domain_stations_path(method, domain))
        agg_path = get_aggregated_data_path(method, time_res)
        agg = pd.read_parquet(agg_path)
    except Exception as e:
        print(f"  Could not load station/aggregated data: {e}")
        ds.close()
        return

    time_col = agg.columns[1]
    if time_res == "weekly":
        agg["time"] = pd.to_datetime(agg[time_col] + "-1", format="%Y-W%W-%w")
    else:
        agg["time"] = pd.to_datetime(agg[time_col])

    # === Auto-detect grid type and do efficient nearest neighbor lookup ===
    data_sample = ds[data_var].isel(time=0).values
    is_2d = data_sample.ndim == 2

    all_actual = []
    all_predicted = []

    if is_2d:
        # 2D grid (new production files)
        ny, nx = data_sample.shape
        grid_x = ds["x"].values
        grid_y = ds["y"].values

        for t_idx, t in enumerate(ds["time"].values):
            t_pd = pd.to_datetime(t)
            stations_t = agg[agg["time"] == t_pd][["station_name", var_name]].dropna()
            if len(stations_t) == 0:
                continue
            stations_t = stations_t.merge(
                stations[["station_name", "x", "y"]], on="station_name", how="inner"
            ).dropna(subset=["x", "y"])
            if len(stations_t) == 0:
                continue

            sx = stations_t["x"].values
            sy = stations_t["y"].values

            x_idx = np.searchsorted(grid_x, sx, side="left")
            y_idx = np.searchsorted(grid_y, sy, side="left")
            x_idx = np.clip(x_idx, 0, nx - 1)
            y_idx = np.clip(y_idx, 0, ny - 1)

            flat_idx = y_idx * nx + x_idx
            predicted = ds[data_var].isel(time=t_idx).values.ravel()[flat_idx]
            actual = stations_t[var_name].values

            all_actual.extend(actual)
            all_predicted.extend(predicted)
    else:
        # 1D flattened grid (older files)
        grid_xy = np.column_stack((
            ds["x"].values.astype(np.float64),
            ds["y"].values.astype(np.float64)
        ))
        tree = cKDTree(grid_xy)

        for t_idx, t in enumerate(ds["time"].values):
            t_pd = pd.to_datetime(t)
            stations_t = agg[agg["time"] == t_pd][["station_name", var_name]].dropna()
            if len(stations_t) == 0:
                continue
            stations_t = stations_t.merge(
                stations[["station_name", "x", "y"]], on="station_name", how="inner"
            ).dropna(subset=["x", "y"])
            if len(stations_t) == 0:
                continue

            station_xy = stations_t[["x", "y"]].values.astype(np.float64)
            _, idx = tree.query(station_xy, k=1)
            predicted = ds[data_var].isel(time=t_idx).values.ravel()[idx]
            actual = stations_t[var_name].values

            all_actual.extend(actual)
            all_predicted.extend(predicted)

    ds.close()

    if len(all_actual) == 0:
        print(f"  No matching station observations found — skipping.")
        return

    out_path = get_validation_scatter_path(method, domain, var_name, resolution)

    plot_validation_scatter(
        np.array(all_actual),
        np.array(all_predicted),
        var_name,
        units,
        resolution,
        out_path,
        get_plotting_params()["dpi"]
    )

    print(f"Finished {var_name} ({resolution} m) — {len(all_actual)} station–timestep pairs")


def main():
    print("=" * 80)
    print(f"Validation Scatter Plotter | Mode: {get_mode()}")
    print("=" * 80)

    nc_files = get_input_nc_files()
    print(f"Found {len(nc_files)} NC file(s) to process.")

    for nc in nc_files:
        process_nc_file(nc)

    print("\n=== All validation scatters finished ===")


if __name__ == "__main__":
    main()