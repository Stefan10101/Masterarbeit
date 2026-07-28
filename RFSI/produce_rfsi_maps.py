#!/usr/bin/env python3
"""
produce_rfsi_maps.py
RFSI production – fully patched:
- time_resolution in output filenames
- two-phase hyper-parameter search (subsample + cheap RF)
- true LLOCV parquet
- richer NC attrs + elev
- memory-safe chunked prediction + early grid crop
"""

from pathlib import Path
import sys
import gc
import pandas as pd
import numpy as np
import xarray as xr
import yaml
import rasterio
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from paths import (
    get_aggregated_data_path,
    get_domain_stations_path,
    get_master_grid_path,
    get_landcover_path,
    get_interpolated_map_path,
    get_llocv_path,
)
from rfsi_core import RFSI
from rfsi_optimizer import (
    optimize_rfsi_params_loocv,
    compute_metrics,
    run_full_llocv,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

DOMAIN = cfg["domain"]["preset"]
TIME_RES = cfg["time_resolution"]
START_DATE = pd.to_datetime(cfg["start_date"])
END_DATE = pd.to_datetime(cfg["end_date"])
RESOLUTIONS = cfg.get("resolutions_to_process", [1000])

rfsi_cfg = cfg.get("rfsi", {})
N_OBS_LIST = rfsi_cfg.get("n_obs_list", [8, 10, 12, 15])
RF_FIXED = rfsi_cfg.get("rf_fixed", {"n_estimators": 400, "random_state": 31, "n_jobs": -1})
RF_TUNABLE = rfsi_cfg.get("rf_tunable", {})
PRIMARY_METRIC = rfsi_cfg.get("primary_metric", "rmse")
CHUNK_SIZE = int(rfsi_cfg.get("predict_chunk_size", 250_000))
FINE_RES_THRESHOLD = int(rfsi_cfg.get("fine_res_threshold", 200))
N_JOBS_FINE = int(rfsi_cfg.get("n_jobs_fine", 2))
N_JOBS_COARSE = int(rfsi_cfg.get("n_jobs_coarse", -1))
SAVE_LLOCV = bool(rfsi_cfg.get("save_llocv", True))
N_OPT_STATIONS = int(rfsi_cfg.get("n_opt_stations", 50))
N_ESTIMATORS_SEARCH = int(rfsi_cfg.get("n_estimators_search", 100))
N_ESTIMATORS_FINAL = int(RF_FIXED.get("n_estimators", 400))


def get_time_column(time_resolution: str) -> str:
    try:
        return cfg["aggregation"][time_resolution]["time_col"]
    except Exception:
        return {
            "weekly": "year_week",
            "daily": "date",
            "monthly": "year_month",
            "half_hourly": "timestamp",
            "seasonally": "season",
        }.get(time_resolution, "time")


def get_domain_bbox(cfg: dict, domain_name: str) -> tuple:
    domain_cfg = cfg.get("domain", {})
    buffer_m = domain_cfg.get("buffer_m", 15000)
    custom = domain_cfg.get("custom_bbox", [100000, 275000, 395000, 400000])
    predefined = cfg.get("predefined_bboxes", {})
    bbox = predefined.get(domain_name, custom)
    xmin, ymin, xmax, ymax = bbox
    return xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m


def get_domain_mask(grid_ds: xr.Dataset, bbox: tuple) -> np.ndarray:
    xmin, ymin, xmax, ymax = bbox
    x_mask = (grid_ds.x >= xmin) & (grid_ds.x <= xmax)
    y_mask = (grid_ds.y >= ymin) & (grid_ds.y <= ymax)
    return (y_mask.values[:, None] & x_mask.values[None, :])


def load_aggregated_data():
    df = pd.read_parquet(get_aggregated_data_path("RFSI", TIME_RES))
    time_col = get_time_column(TIME_RES)
    if TIME_RES == "weekly":
        df["time"] = pd.to_datetime(df[time_col] + "-1", format="%Y-W%W-%w")
    elif TIME_RES == "monthly":
        df["time"] = pd.to_datetime(df[time_col] + "-01")
    else:
        df["time"] = pd.to_datetime(df[time_col])
    return df[(df["time"] >= START_DATE) & (df["time"] <= END_DATE)]


def get_all_stations():
    return pd.read_parquet(get_domain_stations_path("RFSI", "full"))


def prepare_covariates(df, encoder=None, fit_encoder=False):
    X_list = []
    if "elev" in df.columns:
        X_list.append(df[["elev"]].values)
    if "clc_code" in df.columns:
        clc = df[["clc_code"]].values.reshape(-1, 1)
        if fit_encoder or encoder is None:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            clc_onehot = encoder.fit_transform(clc)
        else:
            clc_onehot = encoder.transform(clc)
        X_list.append(clc_onehot)
    return (np.hstack(X_list) if X_list else None), encoder


def predict_in_chunks(model, grid_points, X_grid, chunk_size=CHUNK_SIZE):
    n = len(grid_points)
    preds = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk_size):
        sl = slice(i, i + chunk_size)
        preds[sl] = model.predict(
            coords_pred=grid_points[sl],
            X_cov_pred=X_grid[sl] if X_grid is not None else None,
        )
    return preds


def process_one_time_step(args):
    (t, df_t, var_name, grid_points, X_grid, ny, nx,
     n_obs, rf_params, encoder, domain_mask_flat) = args

    if len(df_t) < 8:
        return None

    df_t = df_t.drop(columns=["time"], errors="ignore")
    coords = df_t[["x", "y"]].values
    z = df_t[var_name].values
    X_cov, _ = prepare_covariates(df_t, encoder=encoder)

    model = RFSI(n_obs=n_obs, rf_params=rf_params)
    model.fit(coords=coords, z=z, X_cov=X_cov)

    preds = predict_in_chunks(model, grid_points, X_grid)
    pred_map = preds.reshape(ny, nx).copy()
    pred_map.ravel()[~domain_mask_flat] = np.nan

    train_pred = model.predict(coords, X_cov_pred=X_cov)
    metrics = compute_metrics(z, train_pred)

    return {"time": t, "pred_map": pred_map, "n_obs": n_obs, "metrics": metrics}


def main():
    print("=" * 80)
    print(f"RFSI Production | {DOMAIN} | {TIME_RES}")
    print(f"Period: {START_DATE.date()} → {END_DATE.date()} | save_llocv={SAVE_LLOCV}")
    print(f"Phase A: {N_OPT_STATIONS} stations, n_estimators={N_ESTIMATORS_SEARCH}")
    print(f"Phase B: all stations, n_estimators={N_ESTIMATORS_FINAL}")
    print("=" * 80)

    station_data = load_aggregated_data()
    stations = get_all_stations()
    domain_bbox = get_domain_bbox(cfg, DOMAIN)

    for res in RESOLUTIONS:
        print(f"\n{'='*60}\nRESOLUTION: {res} m")
        n_jobs = N_JOBS_FINE if res <= FINE_RES_THRESHOLD else N_JOBS_COARSE
        print(f"  Parallel jobs: {n_jobs}")

        grid_ds = xr.open_dataset(get_master_grid_path("RFSI", res))
        x_sel = (grid_ds.x >= domain_bbox[0]) & (grid_ds.x <= domain_bbox[2])
        y_sel = (grid_ds.y >= domain_bbox[1]) & (grid_ds.y <= domain_bbox[3])
        grid_crop = grid_ds.isel(x=x_sel, y=y_sel)
        ny = int(grid_crop.sizes["y"])
        nx = int(grid_crop.sizes["x"])
        print(f"  Cropped grid: {ny} x {nx} = {ny * nx:,} cells")

        domain_mask = get_domain_mask(grid_crop, domain_bbox)
        domain_mask_flat = domain_mask.ravel()

        gx, gy = np.meshgrid(grid_crop.x.values, grid_crop.y.values)
        grid_points = np.column_stack([gx.ravel(), gy.ravel()])
        with rasterio.open(get_landcover_path()) as src:
            grid_clc = np.array([val[0] for val in src.sample(grid_points)], dtype=np.float32)

        grid_elev = grid_crop.elev.values.astype(np.float32)
        grid_df = pd.DataFrame({"elev": grid_elev.ravel(), "clc_code": grid_clc})

        for var in ["precip_sum", "temp_mean", "temp_min", "temp_max",
                    "wind_mean", "wind_max", "rh_mean", "snow_mean", "snow_max", "snow_min"]:
            if var not in station_data.columns:
                continue

            print(f"\n>>> {var}")

            out_file = get_interpolated_map_path(
                "RFSI", DOMAIN, var, res,
                start_date=str(START_DATE.date()),
                end_date=str(END_DATE.date()),
                time_resolution=TIME_RES,
            )
            if out_file.exists():
                print(f"  Skipping (exists): {out_file.name}")
                continue

            valid = station_data[["station_name", "time", var]].dropna()
            station_cols = ["station_name", "x", "y", "elev"]
            if "clc_code" in stations.columns:
                station_cols.append("clc_code")
            valid = valid.merge(stations[station_cols], on="station_name", how="left")
            valid = valid.dropna(subset=["x", "y", var])

            if len(valid) < cfg.get("min_stations_per_field", 10):
                print("  Too few stations — skipping")
                continue

            time_steps = sorted(valid["time"].unique())

            # Phase A – fast search
            sample_t = time_steps[len(time_steps) // 2]
            df_sample = valid[valid["time"] == sample_t].drop(columns=["time"], errors="ignore")
            n_opt = min(N_OPT_STATIONS, len(df_sample))
            df_opt = (df_sample.sample(n=n_opt, random_state=31).reset_index(drop=True)
                      if len(df_sample) > n_opt else df_sample.reset_index(drop=True))

            rf_fixed_search = {**RF_FIXED, "n_estimators": N_ESTIMATORS_SEARCH, "n_jobs": 1}
            cov_cols = [c for c in ["elev", "clc_code"] if c in valid.columns]
            _, encoder = prepare_covariates(valid[cov_cols].dropna(), fit_encoder=True)
            X_grid, _ = prepare_covariates(grid_df, encoder=encoder)

            print(f"  Phase A: {len(df_opt)}/{len(df_sample)} stations, "
                  f"n_estimators={N_ESTIMATORS_SEARCH}...")
            best_params, best_scores = optimize_rfsi_params_loocv(
                df_opt, var, N_OBS_LIST, rf_fixed_search, RF_TUNABLE,
                primary_metric=PRIMARY_METRIC, n_jobs=n_jobs,
            )

            n_obs = int(best_params["n_obs"])
            rf_params = {**RF_FIXED, "n_estimators": N_ESTIMATORS_FINAL}
            for k in ["max_depth", "min_samples_leaf", "max_features"]:
                if k in best_params:
                    val = best_params[k]
                    if isinstance(val, (np.integer, np.floating)):
                        val = val.item()
                    rf_params[k] = val
            if pd.isna(rf_params.get("max_depth")) or rf_params.get("max_depth") is None:
                rf_params["max_depth"] = None
            else:
                rf_params["max_depth"] = int(rf_params["max_depth"])

            print(f"  Best params: n_obs={n_obs}, {rf_params}")
            print(f"  Opt scores (subsample): {best_scores}")

            # Phase B – full LLOCV parquet
            if SAVE_LLOCV:
                print("  Phase B: full LLOCV parquet...")
                llocv_df = run_full_llocv(
                    valid, var, n_obs, rf_params, encoder=encoder, time_col="time"
                )
                llocv_path = get_llocv_path(
                    "RFSI", DOMAIN, var, res,
                    start_date=str(START_DATE.date()),
                    end_date=str(END_DATE.date()),
                    time_resolution=TIME_RES,
                )
                llocv_df.to_parquet(llocv_path, index=False)
                print(f"  LLOCV: {llocv_path.name} ({len(llocv_df)} rows)")

            # Phase B – maps
            tasks = [
                (t, valid[valid["time"] == t], var, grid_points, X_grid,
                 ny, nx, n_obs, rf_params, encoder, domain_mask_flat)
                for t in time_steps
            ]
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_one_time_step)(task) for task in tqdm(tasks, desc=f"  {var}")
            )
            results = [r for r in results if r is not None]
            if not results:
                continue

            data_3d = np.full((len(results), ny, nx), np.nan, dtype=np.float32)
            for i, r in enumerate(results):
                data_3d[i] = r["pred_map"]

            ds = xr.Dataset(
                {var: (("time", "y", "x"), data_3d)},
                coords={
                    "time": [r["time"] for r in results],
                    "y": grid_crop.y.values,
                    "x": grid_crop.x.values,
                },
            )
            ds["elev"] = (("y", "x"), grid_elev)
            for key in ["n_obs", "rmse", "mae", "nse", "kge"]:
                if key == "n_obs":
                    ds[key] = ("time", [r["n_obs"] for r in results])
                else:
                    ds[key] = ("time", [r["metrics"].get(key, np.nan) for r in results])

            ds.attrs.update({
                "title": f"{var} - RFSI - {TIME_RES} - {DOMAIN}",
                "domain": DOMAIN,
                "resolution_m": int(res),
                "method": "RFSI",
                "method_full": "Random Forest Spatial Interpolation (RFSI)",
                "time_resolution": TIME_RES,
                "variable": var,
                "start_date": str(START_DATE.date()),
                "end_date": str(END_DATE.date()),
                "crs": "EPSG:31287",
                "n_obs": int(n_obs),
                "rf_n_estimators": int(rf_params.get("n_estimators", 0)),
                "rf_max_depth": str(rf_params.get("max_depth")),
                "rf_min_samples_leaf": str(rf_params.get("min_samples_leaf")),
                "rf_max_features": str(rf_params.get("max_features")),
                "opt_rmse": float(best_scores.get("rmse", np.nan)),
                "opt_mae": float(best_scores.get("mae", np.nan)),
                "opt_nse": float(best_scores.get("nse", np.nan)),
                "opt_kge": float(best_scores.get("kge", np.nan)),
                "n_opt_stations": int(n_opt),
                "created": datetime.now().isoformat(),
            })

            ds.to_netcdf(out_file, engine="netcdf4")
            print(f"  Saved: {out_file.name}")
            del data_3d, ds, results, tasks
            gc.collect()

        grid_ds.close()
        gc.collect()

    print("\nRFSI production finished.")


if __name__ == "__main__":
    main()
