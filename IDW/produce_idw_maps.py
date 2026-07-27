# ============================================================
# produce_idw_maps.py  (full updated file)
# ============================================================
#!/usr/bin/env python3
"""
produce_idw_maps.py
Main production script for Modified IDW interpolation.
Uses the central paths.py for all file locations.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import yaml
import xarray as xr
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.neighbors import KDTree
import random
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# IMPORT CENTRAL PATHS
# ============================================================
sys.path.append(str(Path(__file__).resolve().parents[1]))
from paths import (
    get_aggregated_data_path,
    get_domain_stations_path,
    get_master_grid_path,
    get_interpolated_map_path,
)

from idw_core import modified_idw, get_valid_targets
from loocv_optimizer import optimize_idw_params_loocv


# ============================================================
# CONFIGURATION
# ============================================================
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
HALFHOURLY_SAMPLE_N = cfg["loocv"].get("half_hourly_sample_n", 10)


def get_param_group(res_m: int) -> str:
    if res_m >= 1000:
        return "coarse"
    elif res_m >= 100:
        return "medium"
    else:
        return "fine"


def load_aggregated_data() -> pd.DataFrame:
    """Load aggregated station data using central paths."""
    file_path = get_aggregated_data_path("IDW", TIME_RES)

    if not file_path.exists():
        raise FileNotFoundError(f"Aggregated file not found: {file_path}")

    df = pd.read_parquet(file_path)
    time_col = df.columns[1]

    if TIME_RES == "weekly":
        df["time"] = pd.to_datetime(df[time_col] + "-1", format="%Y-W%W-%w")
    else:
        df["time"] = pd.to_datetime(df[time_col])

    df = df[(df["time"] >= START_DATE) & (df["time"] <= END_DATE)]
    return df


def process_one_time_step(args):
    t, df_t, var, target_coords, target_elev, param_group = args

    if len(df_t) < 5:
        return None

    # Optimization strategy
    if TIME_RES in ["half_hourly", "day", "night"]:
        all_times = df_t["time"].unique()
        sample_times = random.sample(list(all_times), min(HALFHOURLY_SAMPLE_N, len(all_times)))
        sample_df = df_t[df_t["time"].isin(sample_times)]

        best_params, best_scores = optimize_idw_params_loocv(
            sample_df, var, cfg["param_grid"]["power"],
            primary_metric=PRIMARY_METRIC, verbose=False, n_jobs=1
        )
    else:
        best_params, best_scores = optimize_idw_params_loocv(
            df_t, var, cfg["param_grid"]["power"],
            primary_metric=PRIMARY_METRIC, verbose=False, n_jobs=1
        )

    p, Fz, k = best_params["p"], best_params["Fz"], best_params["k"]
    k = min(k, len(df_t))  # safety clamp: never request more neighbors than available stations

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
        chunk_size=chunk_size
    )

    return {
        "time": t,
        "data": interp_1d,
        "p": p, "Fz": Fz, "k": k,
        "rmse": best_scores.get("rmse", np.nan),
        "mae": best_scores.get("mae", np.nan),
        "nse": best_scores.get("nse", np.nan),
        "kge": best_scores.get("kge", np.nan),
    }


def main():
    print("=" * 80)
    print(f"IDW Thesis Production | Domain: {DOMAIN} | {TIME_RES}")
    print(f"Period: {START_DATE.date()} → {END_DATE.date()}")
    print("=" * 80)

    station_data = load_aggregated_data()
    stations_path = get_domain_stations_path("IDW", DOMAIN)
    stations = pd.read_parquet(stations_path)

    for res in RESOLUTIONS:
        param_group = get_param_group(res)
        print(f"\n{'='*60}\nRESOLUTION: {res} m ({param_group})")

        grid_path = get_master_grid_path("IDW", res)
        grid = xr.open_dataset(grid_path)
        target_coords, target_elev, mask = get_valid_targets(grid)

        if len(target_coords) == 0:
            print(f"  Master grid @ {res}m has 0 valid target cells (empty mask) — skipping resolution")
            continue

        for var in ["precip_sum", "temp_mean", "temp_min", "temp_max",
                    "wind_mean", "wind_max", "rh_mean", "snow_mean", "snow_max", "snow_min"]:

            if var not in station_data.columns:
                continue

            out_file = get_interpolated_map_path(
                "IDW", DOMAIN, var, res,
                start_date=str(START_DATE.date()),
                end_date=str(END_DATE.date())
            )
            if out_file.exists():
                print(f"  Skipping {var} @ {res}m (already exists)")
                continue

            print(f"\n>>> {var}")

            valid = station_data[["station_name", "time", var]].dropna()
            valid = valid.merge(stations[["station_name", "x", "y", "elev"]], on="station_name")

            if len(valid) < cfg.get("min_stations_per_field", 10):
                print(f"  Too few stations — skipping")
                continue

            time_steps = sorted(valid["time"].unique())

            tasks = [(t, valid[valid["time"] == t], var, target_coords, target_elev, param_group)
                     for t in time_steps]

            results = Parallel(n_jobs=N_JOBS)(
                delayed(process_one_time_step)(task) for task in tqdm(tasks, desc=f"  {var}", leave=False)
            )
            results = [r for r in results if r is not None]

            if not results:
                continue

            n_y, n_x = mask.shape
            data_3d = np.full((len(results), n_y, n_x), np.nan, dtype=np.float32)

            for i, res_dict in enumerate(results):
                data_3d[i][mask] = res_dict["data"]

            times = [r["time"] for r in results]

            ds = xr.Dataset(
                {var: (("time", "y", "x"), data_3d)},
                coords={
                    "time": times,
                    "y": grid["y"].values,
                    "x": grid["x"].values,
                }
            )

            ds["p"] = ("time", [r["p"] for r in results])
            ds["Fz"] = ("time", [r["Fz"] for r in results])
            ds["k"] = ("time", [r["k"] for r in results])
            ds["rmse"] = ("time", [r["rmse"] for r in results])
            ds["mae"] = ("time", [r["mae"] for r in results])
            ds["nse"] = ("time", [r["nse"] for r in results])
            ds["kge"] = ("time", [r["kge"] for r in results])

            ds.attrs.update({
                "title": f"{var} - {TIME_RES} - {DOMAIN}",
                "domain": DOMAIN,
                "time_resolution": TIME_RES,
                "resolution_m": res,
                "method": "Modified IDW with separate XY + Z weighting",
                "created": datetime.now().isoformat(),
                "primary_metric": PRIMARY_METRIC,
            })

            ds.to_netcdf(out_file, engine="netcdf4")
            print(f"  Saved: {out_file.name} ({len(results)} timesteps)")

    print("\n" + "=" * 80)
    print("IDW Production finished successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()