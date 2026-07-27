#!/usr/bin/env python3
"""
produce_rfsi_maps.py
RFSI production script using master grids + consistent output structure.
"""

from pathlib import Path
import sys
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
import math
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from paths import (
    get_aggregated_data_path,
    get_domain_stations_path,
    get_master_grid_path,
    get_landcover_path,
    get_interpolated_map_path,
)
from rfsi_core import RFSI
from rfsi_optimizer import optimize_rfsi_params_loocv, compute_metrics


# ================== CONFIG ==================
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

DOMAIN = cfg["domain"]["preset"]
TIME_RES = cfg["time_resolution"]
START_DATE = pd.to_datetime(cfg["start_date"])
END_DATE = pd.to_datetime(cfg["end_date"])
RESOLUTIONS = cfg.get("resolutions_to_process", [100])

rfsi_cfg = cfg.get("rfsi", {})
N_OBS_LIST = rfsi_cfg.get("n_obs_list", [8, 10, 12])
RF_FIXED = rfsi_cfg.get("rf_fixed", {"n_estimators": 400, "random_state": 42})
RF_TUNABLE = rfsi_cfg.get("rf_tunable", {})
PRIMARY_METRIC = rfsi_cfg.get("primary_metric", "rmse")


def get_time_column(time_resolution: str) -> str:
    try:
        return cfg["aggregation"][time_resolution]["time_col"]
    except:
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


def process_one_time_step(args):
    t, df_t, var_name, grid_ds, grid_clc, n_obs, rf_params, encoder, domain_mask = args

    if len(df_t) < 8:
        return None

    df_t = df_t.drop(columns=["time"], errors="ignore")
    coords = df_t[["x", "y"]].values
    z = df_t[var_name].values
    X_cov, _ = prepare_covariates(df_t, encoder=encoder)

    model = RFSI(n_obs=n_obs, rf_params=rf_params)
    model.fit(coords=coords, z=z, X_cov=X_cov)

    grid_elev = grid_ds.elev.values.ravel()
    grid_df = pd.DataFrame({"elev": grid_elev, "clc_code": grid_clc})
    X_grid, _ = prepare_covariates(grid_df, encoder=encoder)

    gx, gy = np.meshgrid(grid_ds.x.values, grid_ds.y.values)
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])

    preds = model.predict(coords_pred=grid_points, X_cov_pred=X_grid)
    pred_map = preds.reshape(grid_ds.sizes["y"], grid_ds.sizes["x"])
    pred_map[~domain_mask] = np.nan

    train_pred = model.predict(coords, X_cov_pred=X_cov)
    metrics = compute_metrics(z, train_pred)

    return {"time": t, "pred_map": pred_map, "n_obs": n_obs, "metrics": metrics}


def main():
    print("=" * 80)
    print(f"RFSI Production | Master Grid + {DOMAIN} domain")
    print("=" * 80)

    station_data = load_aggregated_data()
    stations = get_all_stations()
    domain_bbox = get_domain_bbox(cfg, DOMAIN)

    for res in RESOLUTIONS:
        print(f"\n{'='*60}\nRESOLUTION: {res} m")

        grid_ds = xr.open_dataset(get_master_grid_path("RFSI", res))
        domain_mask = get_domain_mask(grid_ds, domain_bbox)

        # Sample landcover from full file onto master grid
        lc_path = get_landcover_path()
        with rasterio.open(lc_path) as src:
            gx, gy = np.meshgrid(grid_ds.x.values, grid_ds.y.values)
            grid_points = np.column_stack([gx.ravel(), gy.ravel()])
            grid_clc = np.array([val[0] for val in src.sample(grid_points)])

        for var in ["precip_sum", "temp_mean", "temp_min", "temp_max",
                    "wind_mean", "wind_max", "rh_mean", "snow_mean", "snow_max", "snow_min"]:

            if var not in station_data.columns:
                continue

            print(f"\n>>> {var}")

            valid = station_data[["station_name", "time", var]].dropna()
            station_cols = ["station_name", "x", "y", "elev"]
            if "clc_code" in stations.columns:
                station_cols.append("clc_code")

            valid = valid.merge(
                stations[station_cols], on="station_name", how="left"
            ).dropna(subset=["x", "y", var])

            if len(valid) < cfg.get("min_stations_per_field", 10):
                print("  Too few stations — skipping")
                continue

            time_steps = sorted(valid["time"].unique())

            # Hyperparameter optimization on one representative timestep
            sample_t = time_steps[len(time_steps) // 2]
            df_sample = valid[valid["time"] == sample_t].drop(columns=["time"], errors="ignore")

            _, encoder = prepare_covariates(valid[["elev", "clc_code"]].dropna(), fit_encoder=True)

            print("  Optimizing hyperparameters...")
            best_params, _ = optimize_rfsi_params_loocv(
                df_sample, var, N_OBS_LIST, RF_FIXED, RF_TUNABLE,
                primary_metric=PRIMARY_METRIC, n_jobs=-1
            )

            n_obs = int(best_params["n_obs"])
            rf_params = {**RF_FIXED}

            for k in ["max_depth", "min_samples_leaf", "max_features"]:
                if k in best_params:
                    val = best_params[k]
                    # Convert numpy scalar types (np.float64, np.int64, etc.) to native Python types
                    if isinstance(val, (np.integer, np.floating)):
                        val = val.item()
                    rf_params[k] = val

            # Ensure max_depth is a valid type for RandomForestRegressor (int or None)
            if pd.isna(rf_params.get("max_depth")) or rf_params.get("max_depth") is None:
                rf_params["max_depth"] = None
            else:
                rf_params["max_depth"] = int(rf_params["max_depth"])

            tasks = [(t, valid[valid["time"] == t], var, grid_ds, grid_clc,
                      n_obs, rf_params, encoder, domain_mask) for t in time_steps]
            results = Parallel(n_jobs=-1)(
                delayed(process_one_time_step)(task) for task in tqdm(tasks, desc=f"  {var}")
            )
            results = [r for r in results if r is not None]
            if not results:
                continue

            # === Fixed cropping section ===
            x_sel = (grid_ds.x >= domain_bbox[0]) & (grid_ds.x <= domain_bbox[2])
            y_sel = (grid_ds.y >= domain_bbox[1]) & (grid_ds.y <= domain_bbox[3])

            n_y = int(y_sel.sum().item())
            n_x = int(x_sel.sum().item())

            data_3d = np.full((len(results), n_y, n_x), np.nan, dtype=np.float32)

            for i, r in enumerate(results):
                data_3d[i] = r["pred_map"][y_sel.values][:, x_sel.values]

            ds = xr.Dataset(
                {var: (("time", "y", "x"), data_3d)},
                coords={
                    "time": [r["time"] for r in results],
                    "y": grid_ds.y.values[y_sel],
                    "x": grid_ds.x.values[x_sel]
                }
            )

            for key in ["n_obs", "rmse", "mae", "nse", "kge"]:
                ds[key] = ("time", [r.get(key, np.nan) if key == "n_obs" else r["metrics"].get(key, np.nan) for r in results])

            ds.attrs.update({
                "title": f"{var} - RFSI - {TIME_RES} - {DOMAIN}",
                "domain": DOMAIN,
                "resolution_m": res,
                "method": "Random Forest Spatial Interpolation (RFSI)",
                "created": datetime.now().isoformat(),
            })

            # === Consistent output path (same structure as IDW) ===
            out_file = get_interpolated_map_path(
                "RFSI", DOMAIN, var, res,
                start_date=str(START_DATE.date()),
                end_date=str(END_DATE.date())
            )
            if out_file.exists():
                print(f"  Skipping {var} @ {res}m (already exists)")
                continue
            ds.to_netcdf(out_file, engine="netcdf4")
            print(f"  Saved: {out_file}")
    print("\nRFSI production finished.")


if __name__ == "__main__":
    main()