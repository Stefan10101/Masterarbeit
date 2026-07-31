#!/usr/bin/env python3
"""
produce_rfsi_maps.py
RFSI production using pre-trained cluster parameters.

Flow:
  1. Load cluster assignments + per-cluster free parameters
     (produced by train_cluster_params.py).
  2. For every timestep look up its cluster_id → (n_obs, RF hyperparams).
  3. Fit RFSI once and predict (no per-variable Phase-A search).
  4. Optionally write leave-location-out predictions for validation.
"""

from __future__ import annotations

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
    get_cluster_params_path,
    get_clusters_dir,
    get_domain_stations_path,
    get_interpolated_map_path,
    get_landcover_path,
    get_llocv_path,
    get_master_grid_path,
)
from rfsi_core import RFSI
from rfsi_optimizer import compute_metrics, run_full_llocv

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
RF_FIXED = dict(rfsi_cfg.get("rf_fixed", {"n_estimators": 400, "random_state": 31}))
PRIMARY_METRIC = rfsi_cfg.get("primary_metric", "rmse")
CHUNK_SIZE = int(rfsi_cfg.get("predict_chunk_size", 250_000))
FINE_RES_THRESHOLD = int(rfsi_cfg.get("fine_res_threshold", 200))
N_JOBS_FINE = int(rfsi_cfg.get("n_jobs_fine", 2))
N_JOBS_COARSE = int(rfsi_cfg.get("n_jobs_coarse", -1))
SAVE_LLOCV = bool(rfsi_cfg.get("save_llocv", True))
N_ESTIMATORS_FINAL = int(RF_FIXED.get("n_estimators", 400))
CLUSTER_METHOD = cfg.get("cluster_method", "gmm")
MIN_STATIONS = cfg.get("min_stations_per_field", 10)

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
    "n_obs": 10,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "rmse": np.nan, "mae": np.nan, "nse": np.nan, "kge": np.nan,
}


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


def load_aggregated_data() -> pd.DataFrame:
    df = pd.read_parquet(get_aggregated_data_path("RFSI", TIME_RES))
    time_col = get_time_column(TIME_RES)
    if TIME_RES == "weekly":
        df["time"] = pd.to_datetime(df[time_col] + "-1", format="%Y-W%W-%w", utc=True)
    elif TIME_RES == "monthly":
        df["time"] = pd.to_datetime(df[time_col] + "-01", utc=True)
    else:
        df["time"] = pd.to_datetime(df[time_col], utc=True)

    start = START_DATE if START_DATE.tzinfo else START_DATE.tz_localize("UTC")
    end = END_DATE if END_DATE.tzinfo else END_DATE.tz_localize("UTC")
    return df[(df["time"] >= start) & (df["time"] <= end)]


def get_all_stations() -> pd.DataFrame:
    stations = pd.read_parquet(get_domain_stations_path("RFSI", "full"))
    if "elev" not in stations.columns:
        if "elev_dem" in stations.columns:
            stations = stations.rename(columns={"elev_dem": "elev"})
        elif "hoehe" in stations.columns:
            stations = stations.rename(columns={"hoehe": "elev"})
    return stations


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


def load_cluster_assignments(canonical_var: str) -> pd.Series:
    path = (
        get_clusters_dir("RFSI") / TIME_RES / CLUSTER_METHOD
        / f"{canonical_var}_assignments.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Cluster assignments not found: {path}\n"
            f"Run identify_regimes.py with --method RFSI first."
        )
    df = pd.read_parquet(path, columns=["timestamp", "cluster_id"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")["cluster_id"]


def load_cluster_params(canonical_var: str) -> dict:
    path = get_cluster_params_path("RFSI", TIME_RES, CLUSTER_METHOD, canonical_var)
    if not path.exists():
        raise FileNotFoundError(
            f"Cluster parameters not found: {path}\n"
            f"Run train_cluster_params.py first."
        )
    df = pd.read_parquet(path)
    out = {}
    for _, row in df.iterrows():
        md = row["max_depth"]
        if md is None or (isinstance(md, float) and np.isnan(md)):
            md = None
        else:
            md = int(md)
        mf = row["max_features"]
        if isinstance(mf, (float, np.floating)) and not np.isnan(mf):
            mf = float(mf)
        out[int(row["cluster_id"])] = {
            "n_obs": int(row["n_obs"]),
            "max_depth": md,
            "min_samples_leaf": int(row["min_samples_leaf"]),
            "max_features": mf,
            "rmse": float(row["rmse"]) if pd.notna(row["rmse"]) else np.nan,
            "mae": float(row["mae"]) if pd.notna(row["mae"]) else np.nan,
            "nse": float(row["nse"]) if pd.notna(row["nse"]) else np.nan,
            "kge": float(row["kge"]) if pd.notna(row["kge"]) else np.nan,
        }
    return out


def build_rf_params(cluster_params: dict) -> dict:
    """Merge cluster-specific tunable params with production RF_FIXED."""
    return {
        **RF_FIXED,
        "n_estimators": N_ESTIMATORS_FINAL,
        "max_depth": cluster_params["max_depth"],
        "min_samples_leaf": cluster_params["min_samples_leaf"],
        "max_features": cluster_params["max_features"],
        "n_jobs": 1,
    }


def process_one_time_step(args):
    (t, df_t, var_name, grid_points, X_grid, ny, nx,
     n_obs, rf_params, encoder, domain_mask_flat, cluster_id, scores) = args

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

    return {
        "time": t,
        "pred_map": pred_map,
        "n_obs": n_obs,
        "cluster_id": cluster_id,
        "metrics": metrics,
        "scores": scores,
        "rf_params": rf_params,
    }


def main():
    print("=" * 80)
    print(f"RFSI Production (cluster params) | {DOMAIN} | {TIME_RES}")
    print(f"Period: {START_DATE.date()} → {END_DATE.date()} | cluster={CLUSTER_METHOD}")
    print(f"save_llocv={SAVE_LLOCV}  n_estimators={N_ESTIMATORS_FINAL}")
    print("=" * 80)

    station_data = load_aggregated_data()
    stations = get_all_stations()
    domain_bbox = get_domain_bbox(cfg, DOMAIN)

    assignment_cache = {}
    params_cache = {}

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
            grid_clc = np.array(
                [val[0] for val in src.sample(grid_points)], dtype=np.float32
            )

        grid_elev = grid_crop.elev.values.astype(np.float32)
        grid_df = pd.DataFrame({"elev": grid_elev.ravel(), "clc_code": grid_clc})

        for var in ["precip_sum", "temp_mean", "temp_min", "temp_max",
                    "wind_mean", "wind_max", "rh_mean",
                    "snow_mean", "snow_max", "snow_min"]:
            if var not in station_data.columns:
                continue

            canonical = COL_TO_CANONICAL.get(var)
            if canonical is None:
                print(f"  [SKIP] no cluster mapping for column {var}")
                continue

            if canonical not in assignment_cache:
                try:
                    assignment_cache[canonical] = load_cluster_assignments(canonical)
                    params_cache[canonical] = load_cluster_params(canonical)
                    print(
                        f"  loaded clusters for {canonical}: "
                        f"{len(params_cache[canonical])} clusters, "
                        f"{len(assignment_cache[canonical]):,} assignments"
                    )
                except FileNotFoundError as e:
                    print(f"  [SKIP] {var}: {e}")
                    continue

            assignments = assignment_cache[canonical]
            cluster_params = params_cache[canonical]

            out_file = get_interpolated_map_path(
                "RFSI", DOMAIN, var, res,
                start_date=str(START_DATE.date()),
                end_date=str(END_DATE.date()),
                time_resolution=TIME_RES,
            )
            if out_file.exists():
                print(f"  Skipping (exists): {out_file.name}")
                continue

            print(f"\n>>> {var}  (cluster var={canonical})")

            valid = station_data[["station_name", "time", var]].dropna()
            station_cols = ["station_name", "x", "y", "elev"]
            if "clc_code" in stations.columns:
                station_cols.append("clc_code")
            valid = valid.merge(stations[station_cols], on="station_name", how="left")
            valid = valid.dropna(subset=["x", "y", var])

            if len(valid) < MIN_STATIONS:
                print("  Too few stations — skipping")
                continue

            cov_cols = [c for c in ["elev", "clc_code"] if c in valid.columns]
            _, encoder = prepare_covariates(valid[cov_cols].dropna(), fit_encoder=True)
            X_grid, _ = prepare_covariates(grid_df, encoder=encoder)

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
                    cp = cluster_params[cid]
                else:
                    cp = dict(FALLBACK_PARAMS)
                    n_fallback += 1
                    cid = -1

                n_obs = int(cp["n_obs"])
                rf_params = build_rf_params(cp)
                scores = {
                    "rmse": cp.get("rmse", np.nan),
                    "mae": cp.get("mae", np.nan),
                    "nse": cp.get("nse", np.nan),
                    "kge": cp.get("kge", np.nan),
                }
                tasks.append(
                    (t, valid[valid["time"] == t], var, grid_points, X_grid,
                     ny, nx, n_obs, rf_params, encoder, domain_mask_flat,
                     cid, scores)
                )

            if n_fallback:
                print(
                    f"  warning: {n_fallback}/{len(time_steps)} timesteps "
                    f"used fallback params"
                )

            results = Parallel(n_jobs=n_jobs)(
                delayed(process_one_time_step)(task)
                for task in tqdm(tasks, desc=f"  {var}")
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
            ds["n_obs"] = ("time", [r["n_obs"] for r in results])
            ds["cluster_id"] = ("time", [r["cluster_id"] for r in results])
            for key in ["rmse", "mae", "nse", "kge"]:
                ds[key] = ("time", [r["metrics"].get(key, np.nan) for r in results])

            ds.attrs.update({
                "title": f"{var} - RFSI - {TIME_RES} - {DOMAIN}",
                "domain": DOMAIN,
                "resolution_m": int(res),
                "method": "RFSI",
                "method_full": "Random Forest Spatial Interpolation (cluster-based params)",
                "cluster_method": CLUSTER_METHOD,
                "time_resolution": TIME_RES,
                "variable": var,
                "start_date": str(START_DATE.date()),
                "end_date": str(END_DATE.date()),
                "crs": "EPSG:31287",
                "rf_n_estimators": int(N_ESTIMATORS_FINAL),
                "created": datetime.now().isoformat(),
            })

            ds.to_netcdf(out_file, engine="netcdf4")
            print(f"  Saved: {out_file.name} ({len(results)} timesteps)")
            del data_3d, ds, results, tasks
            gc.collect()

        grid_ds.close()
        gc.collect()

    print("\nRFSI production finished.")


if __name__ == "__main__":
    main()
