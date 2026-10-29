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

def _find_code_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "paths.py").exists():
            return parent
    raise RuntimeError("Could not locate CODE/paths.py")

CODE_DIR = _find_code_dir()
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

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

    base = Path(get_method_output_dir(method)) / "interpolated_maps"
    # Standard layout: interpolated_maps/<domain>/res_*m/<time_res>/*.nc
    pattern = str(base / domain / "res_*m" / "*" / "*.nc")
    nc_files = sorted(glob.glob(pattern))
    # SMACNP layout: interpolated_maps/<time_res>/*.nc
    if not nc_files:
        pattern = str(base / "*" / "*.nc")
        nc_files = sorted(glob.glob(pattern))

    if resolutions:
        res_patterns = [f"res_{r}m" for r in resolutions] + [f"{r}m" for r in resolutions]
        nc_files = [f for f in nc_files if any(r in f for r in res_patterns)]

    if time_period:
        matched = [f for f in nc_files if time_period in Path(f).name]
        if matched:
            nc_files = matched

    # Optional: keep only one timescale (weekly, daily, monthly, ...)
    time_res_filter = PLOTTING_CFG.get("time_resolution")
    if time_res_filter:
        key = str(time_res_filter).lower()
        nc_files = [f for f in nc_files if key in Path(f).name.lower()]

    return nc_files


def get_colormap(var_name):
    cmaps = MAPS_CFG.get("colormaps", {})
    return cmaps.get(var_name, cmaps.get("default", "viridis"))


def detect_method(ds, fallback: str = "IDW") -> str:
    attr = str(ds.attrs.get("method", "")).upper()
    if "SMACNP" in attr:
        return "SMACNP"
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


def alpine_mask(x, y, buffer_km: float = 50.0):
    """Boolean mask (ny, nx): True inside Alpine Convention perimeter + buffer."""
    import geopandas as gpd
    from shapely.ops import unary_union
    import urllib.request
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    url = (
        "https://services5.arcgis.com/qjjtr5pMpt3PDiiR/ArcGIS/rest/services/"
        "Alpine_Convention_Perimeter/FeatureServer/1/query?"
        "where=1%3D1&outFields=*&f=geojson"
    )
    cache = Path(__file__).resolve().parent / "alpine_perimeter.geojson"
    if not cache.exists():
        urllib.request.urlretrieve(url, cache)
    peri = gpd.read_file(cache).to_crs("EPSG:31287")
    geom = unary_union(peri.geometry).buffer(buffer_km * 1000.0)

    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1000.0
    dy = float(np.median(np.diff(y))) if len(y) > 1 else 1000.0
    ny, nx = len(y), len(x)
    # from_bounds expects west, south, east, north; y may be descending
    y_top = float(y.max()) + abs(dy) / 2
    y_bot = float(y.min()) - abs(dy) / 2
    x_left = float(x.min()) - abs(dx) / 2
    x_right = float(x.max()) + abs(dx) / 2
    transform = from_bounds(x_left, y_bot, x_right, y_top, nx, ny)
    mask = rasterize(
        [(geom, 1)], out_shape=(ny, nx), transform=transform, fill=0, dtype=np.uint8
    )
    # rasterize puts north at row 0. Flip if grid y is ascending (south at row 0).
    if y[0] < y[-1]:
        mask = mask[::-1]
    return mask.astype(bool)


def _load_lines_31287():
    """Coastline and major rivers in EPSG:31287, plus label positions."""
    import cartopy.io.shapereader as shpreader
    import geopandas as gpd

    coast_path = shpreader.natural_earth(resolution="10m", category="physical", name="coastline")
    river_path = shpreader.natural_earth(resolution="10m", category="physical", name="rivers_lake_centerlines")
    coast = gpd.read_file(coast_path).to_crs("EPSG:31287")
    rivers = gpd.read_file(river_path).to_crs("EPSG:31287")
    # Label anchors in lon/lat, converted once
    labels_ll = [
        ("Rhine", 8.3, 47.6), ("Inn", 11.5, 47.3), ("Etsch", 11.15, 46.35),
        ("Rhône", 6.8, 46.2), ("Drau", 13.4, 46.7), ("Po", 10.5, 45.1),
        ("Danube", 13.5, 48.4),
    ]
    lab = gpd.GeoDataFrame(
        {"name": [n for n, _, _ in labels_ll]},
        geometry=gpd.points_from_xy([x for _, x, _ in labels_ll], [y for _, _, y in labels_ll]),
        crs="EPSG:4326",
    ).to_crs("EPSG:31287")
    return coast, rivers, lab


def process_one_timestep(args):
    (nc_path, t_idx, t, var_name, data_var, stations_df,
     show_stations, show_provider, show_topo, show_extra,
     time_res, domain, res, elev_grid) = args

    import geopandas as gpd

    ds = xr.open_dataset(nc_path)
    plot_type, X, Y, C = _normalize_spatial_data(ds, data_var, t_idx)
    method = detect_method(ds, fallback=PLOTTING_CFG.get("method", "IDW"))

    if not hasattr(process_one_timestep, "_mask") or process_one_timestep._mask_shape != C.shape:
        process_one_timestep._mask = alpine_mask(X, Y, buffer_km=50.0)
        process_one_timestep._mask_shape = C.shape
        process_one_timestep._lines = _load_lines_31287()
    mask = process_one_timestep._mask
    C_plot = np.where(mask, C, np.nan)

    peri_path = Path(__file__).resolve().parent / "alpine_perimeter.geojson"
    peri = gpd.read_file(peri_path).to_crs("EPSG:31287")
    coast, rivers, labels = process_one_timestep._lines

    fig, ax = plt.subplots(figsize=(12, 8))

    # Grayscale DEM everywhere
    elev_vals = ds["elev"].values if "elev" in ds else elev_grid
    if elev_vals is not None:
        ax.pcolormesh(X, Y, elev_vals, cmap="gray_r", vmin=0, vmax=3200, shading="auto", zorder=1)

    # Interpolation only inside perimeter + buffer
    pcm = ax.pcolormesh(X, Y, C_plot, cmap=get_colormap(var_name), shading="auto", alpha=0.75, zorder=2)

    # Coast and rivers
    coast.plot(ax=ax, color="0.35", linewidth=0.4, zorder=3)
    rivers.plot(ax=ax, color="#4a90d9", linewidth=0.35, zorder=3)
    for _, row in labels.iterrows():
        ax.text(row.geometry.x, row.geometry.y, row["name"], fontsize=7,
                fontstyle="italic", color="#1a5f8a", zorder=5,
                bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=0.4))

    # Perimeter
    peri.boundary.plot(ax=ax, edgecolor="black", linewidth=1.1, zorder=4)

    # Crop to perimeter bounds with pad
    minx, miny, maxx, maxy = peri.total_bounds
    pad = 40000
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")

    # Black frame, no title, no axis labels
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("")

    # Small colour legend, raised
    cax = ax.inset_axes([0.70, 0.10, 0.25, 0.028])
    cb = fig.colorbar(pcm, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=6)
    cb.set_label(var_name, fontsize=6)

    t_start = pd.to_datetime(t)
    ax.set_title(f"{method}  |  {t_start.strftime('%d.%m.%Y')}", fontsize=11)
    time_label = t_start.strftime("%Y%m%d_%H%M")
    # Sanitize in case of unexpected characters
    time_label = "".join(c if c.isalnum() or c in "_-" else "_" for c in time_label)
    out_path = get_map_output_path(method, domain, var_name, res, time_label)
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Saving {out_path}")
    try:
        fig.savefig(str(out_path), dpi=MAPS_CFG.get("dpi", 200), bbox_inches="tight", facecolor="white")
    except OSError as e:
        print(f"  Failed to save {out_path}: {e}")
        print(f"  Parent exists: {out_path.parent.exists()}, is_dir: {out_path.parent.is_dir()}")
        raise
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
        candidates = [v for v in ds.data_vars if v not in aux and not str(v).endswith("_variance")]
        if candidates:
            raw = candidates[0]
            data_var = raw
            # SMACNP stores <var>_mean + <var>_variance (e.g. precip_sum_mean,
            # temp_mean_mean). Canonical name is the base variable so colormaps,
            # labels and station lookup match the other methods.
            base = raw[:-5] if str(raw).endswith("_mean") else None
            if base is not None and f"{base}_variance" in ds.data_vars:
                var_name = base
            else:
                var_name = raw
        else:
            var_name = ds.attrs.get("variable", ds.attrs.get("var_name", "prediction"))
            data_var = "prediction" if "prediction" in ds else var_name

        times = ds["time"].values
        method = detect_method(ds, fallback=PLOTTING_CFG.get("method", "IDW"))
        domain = ds.attrs.get("domain", "full")
        res = int(ds.attrs.get("resolution_m", 0) or 0)
        if res == 0:
            # SMACNP filename: temp_mean_daily_1000m.nc
            import re
            m = re.search(r"(\d+)m", nc_path.name)
            res = int(m.group(1)) if m else 1000
        time_res = infer_time_resolution(ds, nc_path)
        # Station overlay: SMACNP has no own station table; use RFSI sources
        station_method = "RFSI" if method == "SMACNP" else method

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
            agg_path = get_aggregated_data_path(station_method, time_res)
            if agg_path.exists():
                agg = pd.read_parquet(agg_path)
                agg["time"] = parse_aggregated_time(agg, time_res)

                stations_path = get_domain_stations_path(station_method, domain)
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
