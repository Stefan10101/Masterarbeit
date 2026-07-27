#!/usr/bin/env python3
"""
plot_maps.py
Single-timestep surface maps for IDW / BSS / RFSI
Robust time handling for all aggregation periods
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
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parents[2]))

from paths import (
    get_maps_dir,
    get_domain_stations_path,
    get_aggregated_data_path,
    get_map_output_path,
    get_method_output_dir,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

MAPS_CFG = cfg.get("maps", {})
PLOTTING_CFG = cfg.get("plotting", {})


# ============================================================
# ROBUST TIME PARSING FOR ALL AGGREGATION PERIODS
# ============================================================
def parse_aggregated_time(agg_df: pd.DataFrame, time_res: str) -> pd.Series:
    """
    Convert time column from aggregated station data to datetime.
    Works for half_hourly, daily, weekly, monthly, and seasonally.
    """
    time_col = agg_df.columns[1]
    col_name = time_col.lower()

    if col_name == "timestamp":
        # half_hourly
        return pd.to_datetime(agg_df[time_col], errors="coerce")

    elif col_name == "date":
        # daily
        return pd.to_datetime(agg_df[time_col], errors="coerce")

    elif col_name == "year_week":
        # weekly: '2022-W05'
        def _parse_week(week_str):
            try:
                year, week = str(week_str).split("-W")
                week = int(week)
                return pd.to_datetime(f"{year}-W{week:02d}-1", format="%Y-W%W-%w")
            except Exception:
                return pd.NaT
        return agg_df[time_col].apply(_parse_week)

    elif col_name == "year_month":
        # monthly: '2020-05' → first day of month
        return pd.to_datetime(agg_df[time_col] + "-01", errors="coerce")

    elif col_name == "season":
        # seasonally: 'Autumn', 'Spring', etc. (limited support)
        print("  [Warning] Seasonal aggregation detected. "
              "Station overlay may not align perfectly with map timesteps.")
        return pd.Series([pd.NaT] * len(agg_df), index=agg_df.index)

    else:
        # fallback for unknown formats
        return pd.to_datetime(agg_df[time_col], errors="coerce")


def get_mode():
    return PLOTTING_CFG.get("mode", "config").lower()


def get_input_nc_files():
    mode = get_mode()
    if mode == "folder_loop":
        folder = PLOTTING_CFG.get("input_nc_folder")
        return sorted(glob.glob(str(Path(folder) / "**/*.nc"), recursive=True))

    method = PLOTTING_CFG.get("method")
    domain = PLOTTING_CFG.get("domain", "full")
    resolutions = PLOTTING_CFG.get("resolutions") or []

    base = Path(get_method_output_dir(method)) / "interpolated_maps" / domain
    pattern = str(base / "res_*m" / "*" / "*.nc")
    nc_files = sorted(glob.glob(pattern))

    if resolutions:
        res_strs = [f"res_{r}m" for r in resolutions]
        nc_files = [f for f in nc_files if any(r in f for r in res_strs)]

    return nc_files


def get_colormap(var_name):
    cmaps = MAPS_CFG.get("colormaps", {})
    return cmaps.get(var_name, cmaps.get("default", "viridis"))


def process_one_timestep(args):
    (nc_path, t_idx, t, var_name, data_var, stations_df,
     show_stations, show_provider, show_topo, show_extra,
     time_res, domain, res) = args

    ds = xr.open_dataset(nc_path)
    plot_type, X, Y, C = _normalize_spatial_data(ds, data_var, t_idx)

    # Method detection (early for title)
    method_attr = ds.attrs.get("method", "").upper()
    if "RFSI" in method_attr:
        method = "RFSI"
    elif "BSS" in method_attr:
        method = "BSS"
    elif "IDW" in method_attr:
        method = "IDW"
    else:
        method = PLOTTING_CFG.get("method", "IDW").upper()

    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)

    if plot_type == "grid":
        pcm = ax.pcolormesh(X, Y, C, cmap=get_colormap(var_name), shading="auto")
        if show_topo:
            elev_name = next((n for n in ["elev", "elevation", "topo", "dem", "altitude"]
                              if n in ds), None)
            if elev_name:
                try:
                    elev = ds[elev_name]
                    if hasattr(elev, "dims") and "time" in elev.dims:
                        elev = elev.isel(time=0)
                    ax.contour(X, Y, elev.values, levels=8,
                               colors="k", linewidths=0.4, alpha=0.45)
                except Exception:
                    pass  # silent → no topo if data problem
    else:
        sc = ax.scatter(X, Y, c=C, cmap=get_colormap(var_name),
                        s=8, edgecolors="none", alpha=0.9)
        pcm = sc

    # === Station overlay (timezone-safe) ===
    if show_stations and stations_df is not None and len(stations_df) > 0:
        t_pd = pd.to_datetime(t)

        # Remove timezone info for safe comparison
        stations_df_clean = stations_df.copy()
        stations_df_clean["time"] = pd.to_datetime(stations_df_clean["time"]).dt.tz_localize(None)
        t_pd = t_pd.tz_localize(None) if t_pd.tz is not None else t_pd

        valid = stations_df_clean[stations_df_clean["time"] == t_pd][
            ["station_name", var_name, "x", "y"] +
            (["provider"] if "provider" in stations_df_clean.columns else [])
        ].dropna()

        if len(valid) > 0:
            if show_provider and "provider" in valid.columns:
                for prov in valid["provider"].unique():
                    prov_data = valid[valid["provider"] == prov]
                    ax.scatter(prov_data["x"], prov_data["y"], s=14,
                            marker="o", edgecolors="k", linewidths=0.4,
                            label=str(prov), zorder=5)
                ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
            else:
                ax.scatter(valid["x"], valid["y"], s=12, c="black",
                        alpha=0.85, edgecolors="none", zorder=5)

    # === Title ===
    t_start = pd.to_datetime(t)
    if time_res == "weekly":
        week_start = t_start - pd.Timedelta(days=t_start.weekday())
        week_end = week_start + pd.Timedelta(days=6)
        date_str = f"{week_start.strftime('%d.%m.%Y')} - {week_end.strftime('%d.%m.%Y')}"
    else:
        date_str = t_start.strftime('%d.%m.%Y')

    title = f"{var_name} | {method} | {domain} | {res}m | {time_res} | {date_str}"
    ax.set_title(title, fontsize=11)

    ax.set_aspect("equal")
    if plot_type == "points":
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())

    # Extra info
    if show_extra:
        info = []
        for key in ["p", "Fz", "k", "tau", "n_segments", "n_obs"]:
            if key in ds:
                val = float(ds[key].isel(time=t_idx).values)
                info.append(f"{key}={val:.2f}")
        for key in ["rmse", "mae", "nse", "kge"]:
            if key in ds:
                val = float(ds[key].isel(time=t_idx).values)
                info.append(f"{key.upper()}={val:.3f}")
        if info:
            ax.text(0.99, 0.01, "\n".join(info), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.9))

    cbar = fig.colorbar(pcm, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label(var_name)

    time_label = pd.to_datetime(t).strftime('%Y%m%d_%H%M')
    out_path = get_map_output_path(method, domain, var_name, res, time_label)

    fig.savefig(out_path, dpi=MAPS_CFG.get("dpi", 250), bbox_inches="tight")
    plt.close(fig)
    ds.close()

    return out_path.name


def _normalize_spatial_data(ds, data_var, t_idx):
    da = ds[data_var].isel(time=t_idx)
    if "cell" in da.dims:
        return "points", ds["x"].values, ds["y"].values, da.values
    else:
        return "grid", ds["x"].values, ds["y"].values, da.values


def main():
    print("=" * 80)
    print("Map Plotter | Mode:", get_mode())
    print("=" * 80)

    nc_files = get_input_nc_files()
    print(f"Found {len(nc_files)} NC file(s)")

    show_stations = MAPS_CFG.get("show_stations", True)
    show_provider = MAPS_CFG.get("show_provider_symbols", True)
    show_topo = MAPS_CFG.get("show_topo", False)
    show_extra = MAPS_CFG.get("show_extra_info", True)
    timestep_step = MAPS_CFG.get("timestep_step", 1)
    n_jobs = MAPS_CFG.get("n_jobs", -1)

    all_tasks = []

    for nc_path in nc_files:
        ds = xr.open_dataset(nc_path)
        # Robust var_name: prefer physical variable (e.g. precip_sum) over "prediction"
        # This fixes station overlay lookup in aggregated data + sensible titles/cmaps
        candidates = [v for v in ds.data_vars
                      if v not in {"time", "x", "y", "elev", "mask", "prediction"}]
        if candidates:
            var_name = candidates[0]
        else:
            var_name = ds.attrs.get("variable", ds.attrs.get("var_name", "prediction"))
        data_var = "prediction" if "prediction" in ds else var_name
        times = ds["time"].values
        method = ds.attrs.get("method", "unknown").split()[0].upper()
        domain = ds.attrs.get("domain", "full")
        res = int(ds.attrs.get("resolution_m", 0))
        time_res = ds.attrs.get("time_resolution", "weekly")

        # === Load stations with robust time parsing ===
        stations_df = None
        try:
            agg_path = get_aggregated_data_path(method, time_res)
            if agg_path.exists():
                agg = pd.read_parquet(agg_path)
                agg["time"] = parse_aggregated_time(agg, time_res)

                stations_path = get_domain_stations_path(method, domain)
                if stations_path.exists():
                    stations = pd.read_parquet(stations_path)
                    stations_df = agg[["station_name", "time", var_name]].merge(
                        stations[["station_name", "x", "y"] +
                                 (["provider"] if "provider" in stations.columns else [])],
                        on="station_name", how="left"
                    )
        except Exception as e:
            print(f"  [Warning] Could not load stations for {Path(nc_path).name}: {e}")

        for t_idx in range(0, len(times), timestep_step):
            all_tasks.append((
                nc_path, t_idx, times[t_idx], var_name, data_var,
                stations_df, show_stations, show_provider,
                show_topo, show_extra, time_res, domain, res
            ))
        ds.close()

    print(f"Total maps to create: {len(all_tasks)}")

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_one_timestep)(task) for task in all_tasks
    )

    print(f"\nFinished — {len(results)} maps saved.")


if __name__ == "__main__":
    main()