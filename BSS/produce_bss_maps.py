#!/usr/bin/env python3
"""
produce_bss_maps.py
BSS/BSSE production using pre-trained cluster parameters.

Fixes vs previous version:
  - all timesteps are processed (no random half-hourly sample)
  - NetCDF written via get_interpolated_map_path
  - tau_d / tau_e / n_segments stored correctly
  - master grid preferred over legacy domain grid
  - elev column aliases handled
  - fixed domain bounds for the knot grid
"""

from __future__ import annotations

from pathlib import Path
import sys
import gc
import pandas as pd
import numpy as np
import yaml
import xarray as xr
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from paths import (
    get_aggregated_data_path,
    get_cluster_params_path,
    get_clusters_dir,
    get_domain_stations_path,
    get_interpolated_map_path,
    get_llocv_path,
    get_master_grid_path,
    get_domain_grid_path,
)
from bss_core import (
    create_knot_grid,
    fit_bss,
    fit_bsse,
    predict_surface,
    predict_bsse,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

DOMAIN = cfg["domain"]["preset"]
TIME_RES = cfg["time_resolution"]
START_DATE = pd.to_datetime(cfg["start_date"])
END_DATE = pd.to_datetime(cfg["end_date"])
RESOLUTIONS = cfg["resolutions_to_process"]
N_JOBS = cfg.get("n_jobs", -1)
BSS_METHOD = cfg["bss"]["method"]
NON_NEGATIVE_VARS = cfg["bss"].get("non_negative_vars", [])
CLUSTER_METHOD = cfg.get("cluster_method", "gmm")
MIN_STATIONS = cfg.get("min_stations_per_field", 10)
SAVE_LLOCV = bool(cfg.get("loocv", {}).get("enable", True))
OVERWRITE = bool(cfg.get("overwrite_maps", True))

COL_TO_CANONICAL = {
    "temp_mean": "temperature",
    "temp_min": "temperature",
    "temp_max": "temperature",
    "precip_sum": "precipitation",
    "wind_mean": "wind_speed",
    "wind_max": "wind_speed",
    "rh_mean": "relative_humidity",
    "snow_mean": "snow_height",
    "snow_max": "snow_height",
    "snow_min": "snow_height",
}

FALLBACK_PARAMS = {
    "n_segments": 10,
    "tau_d": 0.1,
    "tau_e": 0.1,
    "gcv": np.nan,
    "effective_df": np.nan,
}


def domain_bounds(cfg: dict, domain: str) -> tuple:
    domain_cfg = cfg.get("domain", {})
    buffer_m = float(domain_cfg.get("buffer_m", 15000))
    predefined = cfg.get("predefined_bboxes", {})
    custom = domain_cfg.get("custom_bbox", [100000, 275000, 395000, 400000])
    bbox = predefined.get(domain, custom)
    xmin, ymin, xmax, ymax = [float(v) for v in bbox]
    x_range = xmax - xmin
    y_range = ymax - ymin
    margin = 0.06
    return (
        xmin - buffer_m - x_range * margin,
        xmax + buffer_m + x_range * margin,
        ymin - buffer_m - y_range * margin,
        ymax + buffer_m + y_range * margin,
    )


def load_aggregated_data() -> pd.DataFrame:
    file_path = get_aggregated_data_path("BSS", TIME_RES)
    if not file_path.exists():
        raise FileNotFoundError(f"Aggregated file not found: {file_path}")
    df = pd.read_parquet(file_path)

    time_candidates = ["timestamp", "time", "date", "year_week", "year_month"]
    time_col = next((c for c in time_candidates if c in df.columns), df.columns[1])

    if TIME_RES == "weekly":
        df["time"] = pd.to_datetime(
            df[time_col].astype(str) + "-1", format="%Y-W%W-%w", utc=True
        )
    elif TIME_RES == "monthly":
        df["time"] = pd.to_datetime(df[time_col].astype(str) + "-01", utc=True)
    else:
        df["time"] = pd.to_datetime(df[time_col], utc=True)

    start = START_DATE if START_DATE.tzinfo else START_DATE.tz_localize("UTC")
    end = END_DATE if END_DATE.tzinfo else END_DATE.tz_localize("UTC")
    return df[(df["time"] >= start) & (df["time"] <= end)]


def load_stations() -> pd.DataFrame:
    path = get_domain_stations_path("BSS", DOMAIN)
    stations = pd.read_parquet(path)
    rename = {}
    if "elev" not in stations.columns and "elev_dem" in stations.columns:
        rename["elev_dem"] = "elev"
    if "elev" not in stations.columns and "hoehe" in stations.columns:
        rename["hoehe"] = "elev"
    if rename:
        stations = stations.rename(columns=rename)
    return stations


def load_cluster_assignments(canonical_var: str) -> pd.Series:
    path = (
        get_clusters_dir("BSS") / TIME_RES / CLUSTER_METHOD
        / f"{canonical_var}_assignments.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Cluster assignments not found: {path}\n"
            f"Run identify_regimes.py with --method BSS first."
        )
    df = pd.read_parquet(path, columns=["timestamp", "cluster_id"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")["cluster_id"]


def load_cluster_params(canonical_var: str) -> dict:
    path = get_cluster_params_path("BSS", TIME_RES, CLUSTER_METHOD, canonical_var)
    if not path.exists():
        raise FileNotFoundError(
            f"Cluster parameters not found: {path}\n"
            f"Run train_cluster_params.py first."
        )
    df = pd.read_parquet(path)
    out = {}
    for _, row in df.iterrows():
        out[int(row["cluster_id"])] = {
            "n_segments": int(row["n_segments"]),
            "tau_d": float(row["tau_d"]),
            "tau_e": float(row["tau_e"]),
            "gcv": float(row["gcv"]) if pd.notna(row["gcv"]) else np.nan,
            "effective_df": float(row["effective_df"]) if pd.notna(row["effective_df"]) else np.nan,
        }
    return out


def load_grid(res: int):
    """Prefer master grid; fall back to legacy domain grid."""
    master = get_master_grid_path("BSS", res)
    if master.exists():
        grid = xr.open_dataset(master)
        if "mask" in grid:
            mask = grid["mask"].values.astype(bool)
        else:
            mask = np.isfinite(grid["elev"].values)
        return grid, mask

    legacy = get_domain_grid_path("BSS", DOMAIN, res)
    if legacy.exists():
        grid = xr.open_dataset(legacy)
        mask = grid["mask"].values.astype(bool)
        return grid, mask

    raise FileNotFoundError(
        f"No grid found for BSS @ {res}m\n  tried: {master}\n  tried: {legacy}"
    )


def llocv_field(df_t, var, knot_x, knot_y, tau_d, tau_e, bss_method, clip_nn):
    recs = []
    n = len(df_t)
    coords = df_t[["x", "y"]].to_numpy(float)
    elev = df_t["elev"].to_numpy(float)
    vals = df_t[var].to_numpy(float)
    names = df_t["station_name"].to_numpy()
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if mask.sum() < 5:
            continue
        if bss_method == "bsse":
            fit = fit_bsse(vals[mask], coords[mask], elev[mask], knot_x, knot_y,
                           tau_d=tau_d, tau_e=tau_e)
            pred = predict_bsse(fit["d"], fit["e"], coords[[i]], elev[[i]], knot_x, knot_y)
        else:
            fit = fit_bss(vals[mask], coords[mask], knot_x, knot_y, tau_d, tau_d)
            pred = predict_surface(fit["d"], coords[[i]], knot_x, knot_y)
        yhat = float(pred[0])
        if clip_nn:
            yhat = max(yhat, 0.0)
        recs.append({
            "station_name": names[i],
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "observed": float(vals[i]),
            "predicted": yhat,
        })
    return pd.DataFrame(recs)


def process_one_time_step(args):
    (t, df_t, var, target_coords, target_elev, mask,
     bounds, params, bss_method) = args

    if len(df_t) < 5:
        return None

    station_coords = df_t[["x", "y"]].values.astype(float)
    station_values = df_t[var].values.astype(float)
    station_elev = df_t["elev"].values.astype(float)

    n_seg = int(params["n_segments"])
    tau_d = float(params["tau_d"])
    tau_e = float(params["tau_e"])
    xmin, xmax, ymin, ymax = bounds

    knot_x, knot_y = create_knot_grid(xmin, xmax, ymin, ymax, n_seg)

    if bss_method == "bsse":
        result = fit_bsse(
            station_values, station_coords, station_elev,
            knot_x, knot_y, tau_d=tau_d, tau_e=tau_e,
        )
        interp_1d = predict_bsse(
            result["d"], result["e"], target_coords, target_elev, knot_x, knot_y
        )
    else:
        result = fit_bss(
            station_values, station_coords, knot_x, knot_y, tau_d, tau_d
        )
        interp_1d = predict_surface(result["d"], target_coords, knot_x, knot_y)

    if var in NON_NEGATIVE_VARS:
        interp_1d = np.clip(interp_1d, 0, None)

    llocv_df = None
    if SAVE_LLOCV:
        llocv_df = llocv_field(
            df_t, var, knot_x, knot_y, tau_d, tau_e, bss_method,
            clip_nn=var in NON_NEGATIVE_VARS,
        )
        if len(llocv_df):
            llocv_df = llocv_df.copy()
            llocv_df["time"] = t
            llocv_df["n_segments"] = n_seg
            llocv_df["tau_d"] = tau_d
            llocv_df["tau_e"] = tau_e

    return {
        "time": t,
        "data": interp_1d,
        "n_segments": n_seg,
        "tau_d": tau_d,
        "tau_e": tau_e,
        "gcv": float(result.get("gcv", params.get("gcv", np.nan))),
        "effective_df": float(result.get("effective_df", params.get("effective_df", np.nan))),
        "cluster_id": int(params.get("cluster_id", -1)),
        "llocv": llocv_df,
    }


def main():
    print("=" * 80)
    print(f"BSS/BSSE Production (cluster params) | {DOMAIN} | {BSS_METHOD.upper()}")
    print(f"Time resolution: {TIME_RES} | Period: {START_DATE.date()} → {END_DATE.date()}")
    print(f"cluster_method={CLUSTER_METHOD}")
    print("=" * 80)

    station_data = load_aggregated_data()
    stations = load_stations()
    bounds = domain_bounds(cfg, DOMAIN)
    print(f"Knot bounds: {tuple(round(v, 1) for v in bounds)}")

    assignment_cache = {}
    params_cache = {}

    for res in RESOLUTIONS:
        print(f"\n{'='*70}\nRESOLUTION: {res} m")

        try:
            grid, mask = load_grid(res)
        except FileNotFoundError as e:
            print(f"  {e}")
            continue

        yy, xx = np.where(mask)
        target_coords = np.column_stack([
            grid["x"].values[xx],
            grid["y"].values[yy],
        ])
        target_elev = grid["elev"].values[mask]
        print(f"  Valid target cells: {len(target_coords):,}")

        for var in ["precip_sum", "temp_mean", "temp_min", "temp_max",
                    "wind_mean", "wind_max", "rh_mean",
                    "snow_mean", "snow_max", "snow_min"]:
            if var not in station_data.columns:
                continue

            canonical = COL_TO_CANONICAL.get(var)
            if canonical is None:
                print(f"  [SKIP] no cluster mapping for {var}")
                continue

            if canonical not in assignment_cache:
                try:
                    assignments = load_cluster_assignments(canonical)
                    params = load_cluster_params(canonical)
                except FileNotFoundError as e:
                    print(f"  [SKIP] {var}: {e}")
                    assignment_cache[canonical] = None
                    params_cache[canonical] = None
                    continue
                assignment_cache[canonical] = assignments
                params_cache[canonical] = params
                print(
                    f"  loaded clusters for {canonical}: "
                    f"{len(params)} clusters, {len(assignments):,} assignments"
                )

            assignments = assignment_cache[canonical]
            cluster_params = params_cache[canonical]
            if assignments is None or cluster_params is None:
                continue

            out_file = get_interpolated_map_path(
                "BSS", DOMAIN, var, res,
                start_date=str(START_DATE.date()),
                end_date=str(END_DATE.date()),
                time_resolution=TIME_RES,
            )
            if out_file.exists() and not OVERWRITE:
                print(f"  Skipping (exists): {out_file.name}")
                continue
            if out_file.exists():
                print(f"  overwriting {out_file.name}")

            print(f"\n>>> {var}  (cluster var={canonical})")

            valid = station_data[["station_name", "time", var]].dropna()
            valid = valid.merge(
                stations[["station_name", "x", "y", "elev"]],
                on="station_name", how="left",
            ).dropna(subset=["x", "y", "elev", var])

            if len(valid) < MIN_STATIONS:
                print("  Too few stations — skipping")
                continue

            time_steps = sorted(valid["time"].unique())
            tasks = []
            n_fallback = 0
            for t in time_steps:
                cid = None
                if t in assignments.index:
                    cid = int(assignments.loc[t])
                else:
                    diffs = (assignments.index - t).abs()
                    if len(diffs) and diffs.min() <= pd.Timedelta("1min"):
                        cid = int(assignments.iloc[diffs.argmin()])

                if cid is not None and cid in cluster_params:
                    params = dict(cluster_params[cid])
                    params["cluster_id"] = cid
                else:
                    params = dict(FALLBACK_PARAMS)
                    params["cluster_id"] = -1
                    n_fallback += 1

                tasks.append(
                    (t, valid[valid["time"] == t], var,
                     target_coords, target_elev, mask,
                     bounds, params, BSS_METHOD)
                )

            if n_fallback:
                print(
                    f"  warning: {n_fallback}/{len(time_steps)} timesteps "
                    f"used fallback params"
                )

            results = Parallel(n_jobs=N_JOBS)(
                delayed(process_one_time_step)(task)
                for task in tqdm(tasks, desc=f"  {var}", leave=False)
            )
            results = [r for r in results if r is not None]
            if not results:
                continue

            if SAVE_LLOCV:
                parts = [r["llocv"] for r in results
                         if r.get("llocv") is not None and len(r["llocv"])]
                if parts:
                    llocv_all = pd.concat(parts, ignore_index=True)
                    import importlib.util as _ilu
                    _sp = Path(__file__).resolve().parents[1] / "shared" / "splits" / "splits.py"
                    _spec = _ilu.spec_from_file_location("thesis_time_splits", _sp)
                    _mod = _ilu.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    llocv_all["split"] = _mod.label_times(llocv_all["time"]).to_numpy()
                    llocv_path = get_llocv_path(
                        "BSS", DOMAIN, var, res,
                        start_date=str(START_DATE.date()),
                        end_date=str(END_DATE.date()),
                        time_resolution=TIME_RES,
                    )
                    llocv_all.to_parquet(llocv_path, index=False)
                    print(f"  LLOCV: {llocv_path.name} ({len(llocv_all)} rows)")

            n_y, n_x = mask.shape
            data_3d = np.full((len(results), n_y, n_x), np.nan, dtype=np.float32)
            for i, r in enumerate(results):
                data_3d[i][mask] = r["data"]

            time_coord = pd.DatetimeIndex(
                pd.to_datetime([r["time"] for r in results], utc=True)
            ).tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
            ds = xr.Dataset(
                {var: (("time", "y", "x"), data_3d)},
                coords={
                    "time": time_coord,
                    "y": grid["y"].values,
                    "x": grid["x"].values,
                },
            )
            if "elev" in grid:
                ds["elev"] = (("y", "x"), grid["elev"].values.astype(np.float32))

            for key in ["n_segments", "tau_d", "tau_e", "gcv", "effective_df", "cluster_id"]:
                ds[key] = ("time", [r[key] for r in results])

            ds.attrs.update({
                "title": f"{var} - {TIME_RES} - {DOMAIN} - {BSS_METHOD.upper()}",
                "domain": DOMAIN,
                "time_resolution": TIME_RES,
                "resolution_m": int(res),
                "method": f"Bilinear Surface Smoothing ({BSS_METHOD})",
                "method_full": f"BSS/BSSE with cluster-based free parameters",
                "cluster_method": CLUSTER_METHOD,
                "start_date": str(START_DATE.date()),
                "end_date": str(END_DATE.date()),
                "crs": "EPSG:31287",
                "primary_validation": "gcv",
                "non_negative_clipping_applied": int(var in NON_NEGATIVE_VARS),
                "created": datetime.now().isoformat(),
            })

            ds.to_netcdf(out_file, engine="netcdf4")
            print(f"  Saved: {out_file.name} ({len(results)} timesteps)")
            del data_3d, ds, results, tasks
            gc.collect()

        grid.close()
        gc.collect()

    print("\n" + "=" * 80)
    print("BSS/BSSE Production finished successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
