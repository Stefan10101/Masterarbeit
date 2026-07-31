#!/usr/bin/env python3
"""
produce_idw_maps.py
Modified IDW production using pre-trained cluster parameters.

Flow:
  1. Load cluster assignments + per-cluster free parameters
     (produced by train_cluster_params.py).
  2. For every timestep look up its cluster_id → (p, Fz, k).
  3. Interpolate with those fixed parameters (no per-timestep LOOCV).
  4. Optionally still write leave-one-out predictions for validation.
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
from sklearn.neighbors import KDTree
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
)
from idw_core import modified_idw, get_valid_targets
from loocv_optimizer import loocv_predictions

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

DOMAIN = cfg["domain"]["preset"]
TIME_RES = cfg["time_resolution"]
START_DATE = pd.to_datetime(cfg["start_date"])
END_DATE = pd.to_datetime(cfg["end_date"])
RESOLUTIONS = cfg["resolutions_to_process"]
PRIMARY_METRIC = cfg.get("primary_metric", "rmse")
N_JOBS = cfg.get("n_jobs", -1)
SAVE_LLOCV = bool(cfg.get("idw", {}).get("save_llocv", cfg.get("save_llocv", True)))
CLUSTER_METHOD = cfg.get("cluster_method", "gmm")
MIN_STATIONS = cfg.get("min_stations_per_field", 10)

# Aggregated column → canonical clustering variable
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

# Fallback parameters if a cluster is missing (should not happen after training)
FALLBACK_PARAMS = {"p": 2.0, "Fz": 0.3, "k": 12,
                   "rmse": np.nan, "mae": np.nan, "nse": np.nan, "kge": np.nan}


def get_param_group(res_m: int) -> str:
    if res_m >= 1000:
        return "coarse"
    if res_m >= 100:
        return "medium"
    return "fine"


def load_aggregated_data() -> pd.DataFrame:
    file_path = get_aggregated_data_path("IDW", TIME_RES)
    if not file_path.exists():
        raise FileNotFoundError(f"Aggregated file not found: {file_path}")
    df = pd.read_parquet(file_path)
    time_col = df.columns[1]
    if TIME_RES == "weekly":
        df["time"] = pd.to_datetime(df[time_col] + "-1", format="%Y-W%W-%w", utc=True)
    elif TIME_RES == "monthly":
        df["time"] = pd.to_datetime(df[time_col] + "-01", utc=True)
    else:
        df["time"] = pd.to_datetime(df[time_col], utc=True)

    # make START/END tz-aware for safe comparison
    start = START_DATE if START_DATE.tzinfo else START_DATE.tz_localize("UTC")
    end = END_DATE if END_DATE.tzinfo else END_DATE.tz_localize("UTC")
    return df[(df["time"] >= start) & (df["time"] <= end)]


def load_cluster_assignments(canonical_var: str) -> pd.DataFrame:
    """Return DataFrame with columns [timestamp, cluster_id] for one variable."""
    path = get_clusters_dir("IDW") / TIME_RES / CLUSTER_METHOD / f"{canonical_var}_assignments.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Cluster assignments not found: {path}\n"
            f"Run identify_regimes.py first."
        )
    df = pd.read_parquet(path, columns=["timestamp", "cluster_id"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")["cluster_id"]


def load_cluster_params(canonical_var: str) -> dict:
    """
    Return {cluster_id: {p, Fz, k, rmse, mae, nse, kge}} for one variable.
    """
    path = get_cluster_params_path("IDW", TIME_RES, CLUSTER_METHOD, canonical_var)
    if not path.exists():
        raise FileNotFoundError(
            f"Cluster parameters not found: {path}\n"
            f"Run train_cluster_params.py first."
        )
    df = pd.read_parquet(path)
    out = {}
    for _, row in df.iterrows():
        out[int(row["cluster_id"])] = {
            "p": float(row["p"]),
            "Fz": float(row["Fz"]),
            "k": int(row["k"]),
            "rmse": float(row["rmse"]) if pd.notna(row["rmse"]) else np.nan,
            "mae": float(row["mae"]) if pd.notna(row["mae"]) else np.nan,
            "nse": float(row["nse"]) if pd.notna(row["nse"]) else np.nan,
            "kge": float(row["kge"]) if pd.notna(row["kge"]) else np.nan,
        }
    return out


def process_one_time_step(args):
    t, df_t, var, target_coords, target_elev, param_group, params = args

    if len(df_t) < 5:
        return None

    p = float(params["p"])
    Fz = float(params["Fz"])
    k = min(int(params["k"]), len(df_t))

    llocv_df = None
    if SAVE_LLOCV:
        llocv_df = loocv_predictions(df_t, var, p, Fz, k)
        if len(llocv_df) > 0:
            llocv_df = llocv_df.copy()
            llocv_df["time"] = t
            llocv_df["p"] = p
            llocv_df["Fz"] = Fz
            llocv_df["k"] = k

    tree = KDTree(df_t[["x", "y"]].values)
    chunk_size = 1_000_000 if param_group != "coarse" else None
    interp_1d = modified_idw(
        station_values=df_t[var].values,
        station_coords=df_t[["x", "y"]].values,
        station_elev=df_t["elev"].values,
        target_coords=target_coords,
        target_elev=target_elev,
        tree=tree,
        p=p, Fz=Fz, k=k,
        chunk_size=chunk_size,
    )

    return {
        "time": t,
        "data": interp_1d,
        "p": p, "Fz": Fz, "k": k,
        "rmse": params.get("rmse", np.nan),
        "mae": params.get("mae", np.nan),
        "nse": params.get("nse", np.nan),
        "kge": params.get("kge", np.nan),
        "cluster_id": params.get("cluster_id", -1),
        "llocv": llocv_df,
    }


def main():
    print("=" * 80)
    print(f"IDW Production (cluster params) | {DOMAIN} | {TIME_RES}")
    print(f"Period: {START_DATE.date()} → {END_DATE.date()} | cluster={CLUSTER_METHOD}")
    print(f"save_llocv={SAVE_LLOCV}")
    print("=" * 80)

    station_data = load_aggregated_data()
    stations = pd.read_parquet(get_domain_stations_path("IDW", DOMAIN))
    if "elev" not in stations.columns:
        if "elev_dem" in stations.columns:
            stations = stations.rename(columns={"elev_dem": "elev"})
        elif "hoehe" in stations.columns:
            stations = stations.rename(columns={"hoehe": "elev"})

    # cache assignments + params per canonical variable (load once)
    assignment_cache = {}
    params_cache = {}

    for res in RESOLUTIONS:
        param_group = get_param_group(res)
        print(f"\n{'='*60}\nRESOLUTION: {res} m ({param_group})")

        grid = xr.open_dataset(get_master_grid_path("IDW", res))
        target_coords, target_elev, mask = get_valid_targets(grid)
        if len(target_coords) == 0:
            print("  0 valid target cells — skipping")
            continue
        print(f"  Valid target cells: {len(target_coords):,}")

        elev_2d = grid["elev"].values.astype(np.float32) if "elev" in grid else None

        for var in ["precip_sum", "temp_mean", "temp_min", "temp_max",
                    "wind_mean", "wind_max", "rh_mean", "snow_mean", "snow_max", "snow_min"]:
            if var not in station_data.columns:
                continue

            canonical = COL_TO_CANONICAL.get(var)
            if canonical is None:
                print(f"  [SKIP] no cluster mapping for column {var}")
                continue

            # load cluster artefacts (cached)
            if canonical not in assignment_cache:
                try:
                    assignment_cache[canonical] = load_cluster_assignments(canonical)
                    params_cache[canonical] = load_cluster_params(canonical)
                    print(f"  loaded clusters for {canonical}: "
                          f"{len(params_cache[canonical])} clusters, "
                          f"{len(assignment_cache[canonical]):,} assignments")
                except FileNotFoundError as e:
                    print(f"  [SKIP] {var}: {e}")
                    continue

            assignments = assignment_cache[canonical]
            cluster_params = params_cache[canonical]

            out_file = get_interpolated_map_path(
                "IDW", DOMAIN, var, res,
                start_date=str(START_DATE.date()),
                end_date=str(END_DATE.date()),
                time_resolution=TIME_RES,
            )
            if out_file.exists():
                print(f"  Skipping (exists): {out_file.name}")
                continue

            print(f"\n>>> {var}  (cluster var={canonical})")

            valid = station_data[["station_name", "time", var]].dropna()
            valid = valid.merge(
                stations[["station_name", "x", "y", "elev"]], on="station_name"
            ).dropna(subset=["x", "y", "elev", var])

            if len(valid) < MIN_STATIONS:
                print("  Too few stations — skipping")
                continue

            time_steps = sorted(valid["time"].unique())
            tasks = []
            n_fallback = 0
            for t in time_steps:
                # look up cluster
                cid = None
                if t in assignments.index:
                    cid = int(assignments.loc[t])
                else:
                    # nearest assignment within 1 min (tz safety)
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
                     target_coords, target_elev, param_group, params)
                )

            if n_fallback:
                print(f"  warning: {n_fallback}/{len(time_steps)} timesteps used fallback params")

            results = Parallel(n_jobs=N_JOBS)(
                delayed(process_one_time_step)(task)
                for task in tqdm(tasks, desc=f"  {var}", leave=False)
            )
            results = [r for r in results if r is not None]
            if not results:
                continue

            if SAVE_LLOCV:
                parts = [r["llocv"] for r in results
                         if r["llocv"] is not None and len(r["llocv"]) > 0]
                if parts:
                    llocv_all = pd.concat(parts, ignore_index=True)
                    llocv_path = get_llocv_path(
                        "IDW", DOMAIN, var, res,
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

            ds = xr.Dataset(
                {var: (("time", "y", "x"), data_3d)},
                coords={
                    "time": [r["time"] for r in results],
                    "y": grid["y"].values,
                    "x": grid["x"].values,
                },
            )
            if elev_2d is not None:
                ds["elev"] = (("y", "x"), elev_2d)

            for key in ["p", "Fz", "k", "rmse", "mae", "nse", "kge", "cluster_id"]:
                ds[key] = ("time", [r[key] for r in results])

            ds.attrs.update({
                "title": f"{var} - IDW - {TIME_RES} - {DOMAIN}",
                "domain": DOMAIN,
                "time_resolution": TIME_RES,
                "variable": var,
                "resolution_m": int(res),
                "method": "IDW",
                "method_full": "Modified IDW with cluster-based free parameters",
                "cluster_method": CLUSTER_METHOD,
                "start_date": str(START_DATE.date()),
                "end_date": str(END_DATE.date()),
                "crs": "EPSG:31287",
                "primary_metric": PRIMARY_METRIC,
                "created": datetime.now().isoformat(),
            })

            ds.to_netcdf(out_file, engine="netcdf4")
            print(f"  Saved: {out_file.name} ({len(results)} timesteps)")
            del data_3d, ds, results, tasks
            gc.collect()

        grid.close()
        gc.collect()

    print("\n" + "=" * 80)
    print("IDW Production finished successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
