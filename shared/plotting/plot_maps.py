#!/usr/bin/env python3
"""
plot_maps.py
Single-timestep surface maps for IDW / BSS / RFSI
- Configurable time filtering (months / weeks / years / step)
- Correct station overlay (only stations present at that timestep)
- Height contours every 500 m (from DEM when not stored in NC)
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
    get_domain_stations_path,
    get_aggregated_data_path,
    get_map_output_path,
    get_method_output_dir,
    get_dem_path,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

MAPS_CFG = cfg.get("maps", {})
PLOTTING_CFG = cfg.get("plotting", {})


def parse_aggregated_time(agg_df: pd.DataFrame, time_res: str) -> pd.Series:
    """Convert time column from aggregated station data to datetime."""
    time_col = agg_df.columns[1]
    col_name = time_col.lower()

    if col_name == "timestamp":
        return pd.to_datetime(agg_df[time_col], errors="coerce")
    elif col_name == "date":
        return pd.to_datetime(agg_df[time_col], errors="coerce")
    elif col_name == "year_week":
        def _parse_week(week_str):
            try:
                year, week = str(week_str).split("-W")
                week = int(week)
                return pd.to_datetime(f"{year}-W{week:02d}-1", format="%Y-W%W-%w")
            except Exception:
                return pd.NaT
        return agg_df[time_col].apply(_parse_week)
    elif col_name == "year_month":
        return pd.to_datetime(agg_df[time_col] + "-01", errors="coerce")
    elif col_name == "season":
        print("  [Warning] Seasonal aggregation – station overlay may be incomplete.")
        return pd.Series([pd.NaT] * len(agg_df), index=agg_df.index)
    else:
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
    time_period = PLOTTING_CFG.get("time_period")

    base = Path(get_method_output_dir(method)) / "interpolated_maps" / domain
    pattern = str(base / "res_*m" / "*" / "*.nc")
    nc_files = sorted(glob.glob(pattern))

    if resolutions:
        res_patterns = [f"res_{r}m" for r in resolutions]
        nc_files = [f for f in nc_files if any(r in f for r in res_patterns)]

    if time_period:
        nc_files = [f for f in nc_files if time_period in Path(f).name]

    return nc_files


def get_colormap(var_name):
    cmaps = MAPS_CFG.get("colormaps", {})
    return cmaps.get(var_name, cmaps.get("default", "viridis"))


def detect_method(ds, fallback: str = "IDW") -> str:
    attr = str(ds.attrs.get("method", "")).upper()
    if "RFSI" in attr or "RANDOM FOREST" in attr:
        return "RFSI"
    if "BSS" in attr or "BILINEAR" in attr:
        return "BSS"
    if "IDW" in attr:
        return "IDW"
    return fallback.upper()


def infer_time_resolution(ds, nc_path: Path) -> str:
    """Prefer NC attribute, then filename heuristic, then time spacing."""
    attr = ds.attrs.get("time_resolution")
    if attr:
        return str(attr).lower()

    name = nc_path.name.lower()
    for key in ("half_hourly", "hourly", "daily", "weekly", "monthly", "seasonal"):
        if key in name:
            return key

    try:
        t = pd.to_datetime(ds["time"].values)
        if len(t) < 2:
            return "weekly"
        median_days = np.median(np.diff(t).astype("timedelta64[D]").astype(float))
        if median_days < 0.6:
            return "half_hourly"
        if median_days < 2:
            return "daily"
        if median_days < 10:
            return "weekly"
        if median_days < 45:
            return "monthly"
        return "seasonal"
    except Exception:
        return "weekly"


def filter_timesteps(times, time_filter):
    """Return indices that survive the config time filter."""
    if not time_filter:
        return list(range(len(times)))

    step = int(time_filter.get("step", 1) or 1)
    months = time_filter.get("months")
    weeks = time_filter.get("weeks")
    years = time_filter.get("years")
    timestamps = time_filter.get("timestamps")

    t_pd = pd.to_datetime(times)
    mask = np.ones(len(t_pd), dtype=bool)

    if years is not None:
        mask &= np.array(t_pd.year.isin(years))
    if months is not None:
        mask &= np.array(t_pd.month.isin(months))
    if weeks is not None:
        mask &= np.array(t_pd.isocalendar().week.isin(weeks))
    if timestamps is not None:
        wanted = pd.to_datetime(timestamps)
        mask &= np.array(t_pd.normalize().isin(wanted.normalize()))

    idx = np.where(mask)[0]
    if step > 1:
        idx = idx[::step]
    return idx.tolist()


def load_elevation_on_grid(x, y):
    """
    Sample the project DEM onto the map grid.
    Returns 2-D elev array (ny, nx) or None on failure.
    Requires rasterio.
    """
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_bounds
    except ImportError:
        print("  [Warning] rasterio not available – height contours disabled.")
        return None

    dem_path = get_dem_path()
    if not dem_path.exists():
        print(f"  [Warning] DEM not found: {dem_path}")
        return None

    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1000.0
    dy = float(np.median(np.diff(y))) if len(y) > 1 else 1000.0
    y_top = float(y.max()) + abs(dy) / 2
    y_bottom = float(y.min()) - abs(dy) / 2
    x_left = float(x.min()) - abs(dx) / 2
    x_right = float(x.max()) + abs(dx) / 2

    ny, nx = len(y), len(x)
    dst_transform = from_bounds(x_left, y_bottom, x_right, y_top, nx, ny)
    elev = np.full((ny, nx), np.nan, dtype=np.float32)

    with rasterio.open(dem_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=elev,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=src.crs,
            resampling=Resampling.bilinear,
        )
    return elev


def process_one_timestep(args):
    (nc_path, t_idx, t, var_name, data_var, stations_df,
     show_stations, show_provider, show_topo, show_extra,
     time_res, domain, res, elev_grid) = args

    ds = xr.open_dataset(nc_path)
    plot_type, X, Y, C = _normalize_spatial_data(ds, data_var, t_idx)
    method = detect_method(ds, fallback=PLOTTING_CFG.get("method", "IDW"))

    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)

    if plot_type == "grid":
        pcm = ax.pcolormesh(X, Y, C, cmap=get_colormap(var_name), shading="auto")

        if show_topo:
            elev_vals = None
            elev_name = next((n for n in ["elev", "elevation", "topo", "dem", "altitude"]
                              if n in ds), None)
            if elev_name is not None:
                elev = ds[elev_name]
                if "time" in elev.dims:
                    elev = elev.isel(time=0)
                elev_vals = elev.values
            elif elev_grid is not None:
                elev_vals = elev_grid

            if elev_vals is not None:
                zmin = np.nanmin(elev_vals)
                zmax = np.nanmax(elev_vals)
                if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
                    levels = np.arange(
                        np.floor(zmin / 500.0) * 500.0,
                        np.ceil(zmax / 500.0) * 500.0 + 500.0,
                        500.0,
                    )
                    ax.contour(X, Y, elev_vals, levels=levels,
                               colors="k", linewidths=0.5, alpha=0.55)
    else:
        sc = ax.scatter(X, Y, c=C, cmap=get_colormap(var_name),
                        s=8, edgecolors="none", alpha=0.9)
        pcm = sc

    # === Station overlay (robust matching) ===
    if show_stations and stations_df is not None and len(stations_df) > 0:
        t_pd = pd.to_datetime(t)
        if getattr(t_pd, "tzinfo", None) is not None:
            t_pd = t_pd.tz_localize(None)

        stations_clean = stations_df.copy()
        stations_clean["time"] = pd.to_datetime(stations_clean["time"]).dt.tz_localize(None)

        if time_res == "weekly":
            t_year = int(t_pd.isocalendar().year)
            t_week = int(t_pd.isocalendar().week)
            stations_clean["year"] = stations_clean["time"].dt.isocalendar().year.astype(int)
            stations_clean["week"] = stations_clean["time"].dt.isocalendar().week.astype(int)
            valid = stations_clean[
                (stations_clean["year"] == t_year) & (stations_clean["week"] == t_week)
            ]
        else:
            valid = stations_clean[
                stations_clean["time"].dt.normalize() == t_pd.normalize()
            ]

        cols = ["station_name", var_name, "x", "y"]
        if "provider" in valid.columns:
            cols.append("provider")
        valid = valid[cols].dropna(subset=["x", "y", var_name])

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
    elif time_res == "monthly":
        date_str = t_start.strftime("%Y-%m")
    else:
        date_str = t_start.strftime("%d.%m.%Y")

    title = f"{var_name} | {method} | {domain} | {res}m | {time_res} | {date_str}"
    ax.set_title(title, fontsize=11)

    ax.set_aspect("equal")
    if plot_type == "points":
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())

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

    time_label = pd.to_datetime(t).strftime("%Y%m%d_%H%M")
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
    n_jobs = MAPS_CFG.get("n_jobs", -1)
    time_filter = MAPS_CFG.get("time_filter") or {}

    elev_cache = {}

    all_tasks = []

    for nc_path in nc_files:
        nc_path = Path(nc_path)
        ds = xr.open_dataset(nc_path)

        aux = {"time", "x", "y", "elev", "elevation", "topo", "dem", "mask", "prediction",
               "p", "Fz", "k", "tau", "n_segments", "n_obs", "rmse", "mae", "nse", "kge"}
        candidates = [v for v in ds.data_vars if v not in aux]
        if candidates:
            var_name = candidates[0]
        else:
            var_name = ds.attrs.get("variable", ds.attrs.get("var_name", "prediction"))
        data_var = "prediction" if "prediction" in ds else var_name

        times = ds["time"].values
        method = detect_method(ds, fallback=PLOTTING_CFG.get("method", "IDW"))
        domain = ds.attrs.get("domain", "full")
        res = int(ds.attrs.get("resolution_m", 0) or 0)
        time_res = infer_time_resolution(ds, nc_path)

        elev_grid = None
        if show_topo:
            cache_key = (domain, res)
            if cache_key not in elev_cache:
                elev_name = next((n for n in ["elev", "elevation", "topo", "dem", "altitude"]
                                  if n in ds), None)
                if elev_name is not None:
                    elev = ds[elev_name]
                    if "time" in elev.dims:
                        elev = elev.isel(time=0)
                    elev_cache[cache_key] = elev.values
                else:
                    print(f"  Sampling DEM onto grid {domain} / {res}m …")
                    elev_cache[cache_key] = load_elevation_on_grid(
                        ds["x"].values, ds["y"].values
                    )
            elev_grid = elev_cache[cache_key]

        stations_df = None
        try:
            agg_path = get_aggregated_data_path(method, time_res)
            if agg_path.exists():
                agg = pd.read_parquet(agg_path)
                agg["time"] = parse_aggregated_time(agg, time_res)

                stations_path = get_domain_stations_path(method, domain)
                if stations_path.exists():
                    stations = pd.read_parquet(stations_path)
                    keep = ["station_name", "x", "y"]
                    if "provider" in stations.columns:
                        keep.append("provider")
                    stations_df = agg[["station_name", "time", var_name]].merge(
                        stations[keep], on="station_name", how="left"
                    )
        except Exception as e:
            print(f"  [Warning] Could not load stations for {nc_path.name}: {e}")

        t_indices = filter_timesteps(times, time_filter)
        print(f"  {nc_path.name}: {len(t_indices)} / {len(times)} timesteps selected "
              f"(time_res={time_res})")

        for t_idx in t_indices:
            all_tasks.append((
                str(nc_path), t_idx, times[t_idx], var_name, data_var,
                stations_df, show_stations, show_provider,
                show_topo, show_extra, time_res, domain, res, elev_grid
            ))
        ds.close()

    print(f"Total maps to create: {len(all_tasks)}")

    if not all_tasks:
        print("Nothing to plot – check time_filter / resolutions / time_period.")
        return

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_one_timestep)(task) for task in all_tasks
    )

    print(f"\nFinished — {len(results)} maps saved.")


if __name__ == "__main__":
    main()
