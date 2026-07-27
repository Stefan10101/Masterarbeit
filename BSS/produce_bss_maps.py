#!/usr/bin/env python3
"""
produce_bss_maps.py
Main production script for BSS and BSSE interpolation.
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
import random
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# IMPORT CENTRAL PATHS
# ============================================================
# Add CODE folder to path so we can import paths.py
sys.path.append(str(Path(__file__).resolve().parents[1]))
from paths import (
    get_aggregated_data_path,
    get_domain_stations_path,
    get_domain_grid_path,
    get_map_output_path,
)

from bss_core import optimize_bss_gcv, predict_surface, predict_bsse


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
N_JOBS = cfg.get("n_jobs", -1)
BSS_METHOD = cfg["bss"]["method"]
NON_NEGATIVE_VARS = cfg["bss"].get("non_negative_vars", [])
SEGMENT_RANGE = range(cfg["bss"]["segment_min"], cfg["bss"]["segment_max"] + 1, cfg["bss"]["segment_step"])

# Tau handling (supports both old tau_search and new separate tau_d/tau_e)
TAU_VALUES = cfg["bss"].get("tau_search")
TAU_D_VALUES = cfg["bss"].get("tau_d_search") or TAU_VALUES
TAU_E_VALUES = cfg["bss"].get("tau_e_search") or TAU_VALUES


def load_aggregated_data() -> pd.DataFrame:
    """Load aggregated station data using central paths."""
    file_path = get_aggregated_data_path("BSS", TIME_RES)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Aggregated file not found: {file_path}\n"
            f"Expected: BSS/Output/aggregated/{TIME_RES}_station_data.parquet"
        )

    df = pd.read_parquet(file_path)
    time_col = df.columns[1]

    if TIME_RES == "weekly":
        df["time"] = pd.to_datetime(df[time_col] + "-1", format="%Y-W%W-%w")
    else:
        df["time"] = pd.to_datetime(df[time_col])

    df = df[(df["time"] >= START_DATE) & (df["time"] <= END_DATE)]
    return df


def process_one_time_step(args):
    t, df_t, var, target_coords, target_elev, method = args
    if len(df_t) < 5:
        return None

    station_coords = df_t[["x", "y"]].values
    station_values = df_t[var].values.astype(float)
    station_elev = df_t["elev"].values.astype(float) if method == "bsse" else None

    # === ROBUST KNOT GRID BOUNDS ===
    # Use union of stations + target grid points + buffer.
    # This guarantees both fitting and prediction points are inside the knot grid.
    if len(station_coords) > 0 and len(target_coords) > 0:
        all_x = np.concatenate([station_coords[:, 0], target_coords[:, 0]])
        all_y = np.concatenate([station_coords[:, 1], target_coords[:, 1]])
        buffer = 0.06  # 6% buffer - adjust if needed
        x_range = all_x.max() - all_x.min()
        y_range = all_y.max() - all_y.min()
        xmin = all_x.min() - x_range * buffer
        xmax = all_x.max() + x_range * buffer
        ymin = all_y.min() - y_range * buffer
        ymax = all_y.max() + y_range * buffer
    else:
        # Fallback (should rarely happen)
        xmin, xmax, ymin, ymax = target_coords[:, 0].min(), target_coords[:, 0].max(), \
                                 target_coords[:, 1].min(), target_coords[:, 1].max()

    result = optimize_bss_gcv(
        station_values=station_values,
        station_coords=station_coords,
        station_elev=station_elev,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        segment_range=SEGMENT_RANGE,
        tau_d_values=TAU_D_VALUES,
        tau_e_values=TAU_E_VALUES,
        method=method
    )

    d = result["d"]
    knot_x = result["knot_x"]
    knot_y = result["knot_y"]

    if method == "bsse":
        e = result["e"]
        interp_1d = predict_bsse(d, e, target_coords, target_elev, knot_x, knot_y)
    else:
        interp_1d = predict_surface(d, target_coords, knot_x, knot_y)

    if var in NON_NEGATIVE_VARS:
        interp_1d = np.clip(interp_1d, 0, None)

    best = result.get("best_params", {})
    return {
        "time": t,
        "data": interp_1d,
        "n_segments": best.get("n_segments", np.nan),
        "tau": best.get("tau", np.nan),
        "gcv": result.get("gcv", np.nan),
        "effective_df": result.get("effective_df", np.nan),
    }


def main():
    print("=" * 85)
    print(f"BSS/BSSE Thesis Production | Domain: {DOMAIN} | Method: {BSS_METHOD.upper()}")
    print(f"Time resolution: {TIME_RES} | Period: {START_DATE.date()} → {END_DATE.date()}")
    print("=" * 85)

    station_data = load_aggregated_data()
    stations_path = get_domain_stations_path("BSS", DOMAIN)
    stations = pd.read_parquet(stations_path)

    for res in RESOLUTIONS:
        print(f"\n{'='*70}\nRESOLUTION: {res} m")

        grid_path = get_domain_grid_path("BSS", DOMAIN, res)
        if not grid_path.exists():
            print(f"  Grid not found for {res}m — skipping")
            continue

        grid = xr.open_dataset(grid_path)
        target_coords = np.column_stack([
            grid["x"].values[np.where(grid["mask"])[1]],
            grid["y"].values[np.where(grid["mask"])[0]]
        ])
        target_elev = grid["elev"].values[grid["mask"].values]

        for var in ["precip_sum", "temp_mean", "temp_min", "temp_max",
                    "wind_mean", "wind_max", "rh_mean", "snow_mean", "snow_max", "snow_min"]:

            if var not in station_data.columns:
                continue

            out_file = get_map_output_path("BSS", DOMAIN, var, res, TIME_RES)
            if out_file.exists():
                print(f"  Skipping {var} @ {res}m (already exists)")
                continue

            print(f"\n>>> {var} ({BSS_METHOD})")

            valid = station_data[["station_name", "time", var]].dropna()
            valid = valid.merge(
                stations[["station_name", "x", "y", "elev"]],
                on="station_name", how="left"
            ).dropna(subset=["x", "y", "elev"])

            if len(valid) < cfg.get("min_stations_per_field", 10):
                print(f"  Too few stations — skipping")
                continue

            time_steps = sorted(valid["time"].unique())

            if TIME_RES in ["half_hourly", "day", "night"]:
                sample_n = cfg["loocv"].get("half_hourly_sample_n", 5)
                sample_times = random.sample(list(time_steps), min(sample_n, len(time_steps)))
                tasks = [(t, valid[valid["time"] == t], var, target_coords, target_elev, BSS_METHOD)
                         for t in sample_times]
            else:
                tasks = [(t, valid[valid["time"] == t], var, target_coords, target_elev, BSS_METHOD)
                         for t in time_steps]

            results = Parallel(n_jobs=N_JOBS)(
                delayed(process_one_time_step)(task) for task in tqdm(tasks, desc=f"  {var}", leave=False)
            )
            results = [r for r in results if r is not None]

            if not results:
                continue

            n_y, n_x = grid["mask"].shape
            data_3d = np.full((len(results), n_y, n_x), np.nan, dtype=np.float32)
            for i, res_dict in enumerate(results):
                data_3d[i][grid["mask"].values] = res_dict["data"]

            times = [r["time"] for r in results]

            ds = xr.Dataset(
                {var: (("time", "y", "x"), data_3d)},
                coords={"time": times, "y": grid["y"].values, "x": grid["x"].values}
            )

            ds["n_segments"] = ("time", [r["n_segments"] for r in results])
            ds["tau"] = ("time", [r["tau"] for r in results])
            ds["gcv"] = ("time", [r["gcv"] for r in results])
            ds["effective_df"] = ("time", [r["effective_df"] for r in results])

            ds.attrs.update({
                "title": f"{var} - {TIME_RES} - {DOMAIN} - {BSS_METHOD.upper()}",
                "domain": DOMAIN,
                "time_resolution": TIME_RES,
                "resolution_m": res,
                "method": f"Bilinear Surface Smoothing ({BSS_METHOD})",
                "created": datetime.now().isoformat(),
                "primary_validation": "gcv",
                "non_negative_clipping_applied": int(var in NON_NEGATIVE_VARS),
            })

            ds.to_netcdf(out_file, engine="netcdf4")
            print(f"  Saved: {out_file.name} ({len(results)} timesteps)")

    print("\n" + "=" * 85)
    print("BSS/BSSE Production finished successfully.")
    print("=" * 85)


if __name__ == "__main__":
    main()