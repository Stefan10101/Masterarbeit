#!/usr/bin/env python3
"""
produce_rfsi_maps.py
Pooled RFSI (Sekulić et al. 2020): one forest per variable.

Flow:
  1. Stack all station–time rows (neighbour values + distances at that time,
     plus elevation / landcover at the target).
  2. Fit one RandomForestRegressor.
  3. For every timestamp, predict the grid from that timestamp's stations
     with the frozen forest.
  4. Optional station predictions (self excluded). These are not nested LLOCV.
"""

from __future__ import annotations

from pathlib import Path
import sys
import gc
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from paths import (
    get_aggregated_data_path,
    get_domain_stations_path,
    get_interpolated_map_path,
    get_llocv_path,
    get_master_grid_path,
    get_rfsi_model_path,
)
from rfsi_core import RFSI, build_covariates, extras_for_var, neighbor_width, two_step_var
from rfsi_optimizer import compute_metrics

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
RF_FIXED = dict(rfsi_cfg.get("rf_fixed", {"n_estimators": 250, "random_state": 22}))
N_OBS = int(rfsi_cfg.get("n_obs", 10))
CHUNK_SIZE = int(rfsi_cfg.get("predict_chunk_size", 250_000))
SAVE_LLOCV = bool(rfsi_cfg.get("save_llocv", True))
SAVE_MODEL = bool(rfsi_cfg.get("save_model", True))
USE_ELEV = bool(rfsi_cfg.get("use_elevation", True))
USE_LC = bool(rfsi_cfg.get("use_landcover", True))
MIN_STATIONS = cfg.get("min_stations_per_field", 10)
VARIABLES = rfsi_cfg.get("variables_to_process")  # None = all present columns

ALL_VARS = [
    "precip_sum", "temp_mean", "temp_min", "temp_max",
    "wind_mean", "wind_max", "rh_mean",
    "snow_mean", "snow_max", "snow_min",
]


def get_time_column(time_resolution: str) -> str:
    try:
        return cfg["aggregation"][time_resolution]["time_col"]
    except Exception:
        return {
            "weekly": "year_week",
            "daily": "date",
            "monthly": "year_month",
            "half_hourly": "timestamp",
        }.get(time_resolution, "time")


def get_domain_bbox(cfg: dict, domain_name: str) -> tuple:
    domain_cfg = cfg.get("domain", {})
    buffer_m = domain_cfg.get("buffer_m", 15000)
    custom = domain_cfg.get("custom_bbox", [100000, 275000, 395000, 400000])
    predefined = cfg.get("predefined_bboxes", {})
    bbox = predefined.get(domain_name, custom)
    xmin, ymin, xmax, ymax = bbox
    return xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m


def get_domain_mask(grid_ds, bbox: tuple) -> np.ndarray:
    xmin, ymin, xmax, ymax = bbox
    x_mask = (grid_ds.x >= xmin) & (grid_ds.x <= xmax)
    y_mask = (grid_ds.y >= ymin) & (grid_ds.y <= ymax)
    return (y_mask.values[:, None] & x_mask.values[None, :])


def as_naive_utc(values) -> np.ndarray:
    """NetCDF cannot store tz-aware or Python datetime objects."""
    idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    return idx.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")


def load_aggregated_data() -> pd.DataFrame:
    df = pd.read_parquet(get_aggregated_data_path("RFSI", TIME_RES))
    time_col = get_time_column(TIME_RES)
    if TIME_RES == "weekly":
        raw = pd.to_datetime(df[time_col] + "-1", format="%Y-W%W-%w", utc=True)
    elif TIME_RES == "monthly":
        raw = pd.to_datetime(df[time_col].astype(str) + "-01", utc=True)
    else:
        raw = pd.to_datetime(df[time_col], utc=True)
    df["time"] = as_naive_utc(raw)

    start = pd.Timestamp(START_DATE).tz_localize(None)
    end = pd.Timestamp(END_DATE).tz_localize(None)
    return df[(df["time"] >= start) & (df["time"] <= end)].copy()


def get_all_stations() -> pd.DataFrame:
    stations = pd.read_parquet(get_domain_stations_path("RFSI", "full"))
    if "elev" not in stations.columns:
        if "elev_dem" in stations.columns:
            stations = stations.rename(columns={"elev_dem": "elev"})
        elif "hoehe" in stations.columns:
            stations = stations.rename(columns={"hoehe": "elev"})
    return stations


def prepare_covariates(df, encoder=None, fit_encoder=False, use_elev=True, use_lc=True,
                      extras=None):
    X = build_covariates(
        elev=df["elev"].to_numpy() if use_elev and "elev" in df.columns else None,
        clc_code=df["clc_code"].to_numpy() if use_lc and "clc_code" in df.columns else None,
        use_elev=use_elev and "elev" in df.columns,
        use_lc=use_lc and "clc_code" in df.columns,
        extras=extras,
    )
    return X, None


def predict_in_chunks(model, coords_obs, z_obs, grid_points, X_grid, chunk_size=CHUNK_SIZE):
    n = len(grid_points)
    preds = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk_size):
        sl = slice(i, i + chunk_size)
        preds[sl] = model.predict_field(
            coords_obs, z_obs,
            grid_points[sl],
            X_cov_pred=X_grid[sl] if X_grid is not None else None,
        )
    return preds


def grid_landcover_flat(grid_crop) -> np.ndarray:
    """CLC already stored on the master grid. No rasterio at production time."""
    for name in ("clc", "clc_code", "landcover"):
        if name in grid_crop:
            return np.asarray(grid_crop[name].values, dtype=np.float32).ravel()
    raise RuntimeError(
        "Master grid has no landcover variable ('clc'). Attach it once with:\n"
        "  cd CODE/shared/grids\n"
        "  python create_grids.py --method RFSI --attach-landcover "
        "--resolutions 1000 --config ../../RFSI/config.yaml"
    )


def main():
    import argparse
    import xarray as xr
    from joblib import dump
    p = argparse.ArgumentParser()
    p.add_argument("--variables", nargs="*", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--months", default=None)
    args = p.parse_args()

    print("=" * 80)
    print(f"RFSI pooled production | {DOMAIN} | {TIME_RES}")
    print(f"Period: {START_DATE.date()} → {END_DATE.date()}")
    print(f"n_obs={N_OBS}  n_estimators={RF_FIXED.get('n_estimators')}  save_llocv={SAVE_LLOCV}")
    print("=" * 80)

    station_data = load_aggregated_data()
    stations = get_all_stations()
    domain_bbox = get_domain_bbox(cfg, DOMAIN)

    use_elev = USE_ELEV
    use_lc = USE_LC and ("clc_code" in stations.columns)
    if USE_LC and not use_lc:
        raise RuntimeError(
            "use_landcover=true but stations_projected.parquet has no clc_code. "
            "Re-run create_grids.py --method RFSI --master so stations get CLC."
        )
    print(f"  covariates: elev={use_elev}  landcover={use_lc}")

    station_cols = ["station_name", "x", "y"]
    if use_elev and "elev" in stations.columns:
        station_cols.append("elev")
    if use_lc and "clc_code" in stations.columns:
        station_cols.append("clc_code")
    stations = stations[station_cols].drop_duplicates("station_name")

    wanted = args.variables or VARIABLES or ALL_VARS
    wanted = [v for v in wanted if v in station_data.columns]
    if not wanted:
        raise RuntimeError(f"None of {VARIABLES} found in aggregated data.")

    rf_params = {
        **RF_FIXED,
        "n_jobs": -1,
        "max_depth": RF_FIXED.get("max_depth"),
        "min_samples_leaf": RF_FIXED.get("min_samples_leaf", 5),
        "max_features": RF_FIXED.get("max_features", "sqrt"),
    }
    # drop keys sklearn does not accept if None-only extras slipped in
    rf_params = {k: v for k, v in rf_params.items() if k in {
        "n_estimators", "max_depth", "min_samples_leaf", "max_features",
        "random_state", "n_jobs", "min_samples_split",
    }}

    for var in wanted:
        print(f"\n{'=' * 60}\nVARIABLE: {var}")
        valid = station_data[["station_name", "time", var]].dropna()
        valid = valid.merge(stations, on="station_name", how="inner")
        valid = valid.dropna(subset=["x", "y", var])
        if "elev" in valid.columns:
            valid = valid.dropna(subset=["elev"])
        rh_t_mode = str(rfsi_cfg.get("rh_t_mode", "none"))
        if str(var).lower().startswith("rh") and rh_t_mode in ("predicted", "observed") and "temp_mean" in station_data.columns:
            tjoin = station_data[["station_name", "time", "temp_mean"]]
            valid = valid.merge(tjoin, on=["station_name", "time"], how="left")
        if args.quick or args.months:
            from Kriging.kriging_data import subset_times, subset_splits
            valid = subset_times(valid, args.months or "seasonal4")
            valid = subset_splits(valid.assign(split="test"), "test") if False else valid
            if args.quick:
                spec_start = pd.Timestamp("2024-01-01")
                valid = valid[valid["time"] >= spec_start]

        n_times = valid["time"].nunique()
        n_stat = valid["station_name"].nunique()
        print(f"  {len(valid):,} rows | {n_stat} stations | {n_times} timestamps")
        if n_stat < MIN_STATIONS:
            print("  too few stations — skip")
            continue

        X_cov, encoder = prepare_covariates(
            valid, fit_encoder=True, use_elev=use_elev, use_lc=use_lc,
            extras=extras_for_var(valid, var, rh_t_mode),
        )
        model = RFSI(
            n_obs=N_OBS, rf_params=rf_params,
            two_step=bool(rfsi_cfg.get("two_step", True)) and two_step_var(var),
            tau_wet=float(rfsi_cfg.get("tau_wet", 0.5)),
            var_name=var,
        )
        print("  fitting pooled forest …")
        model.fit_pooled(
            times=valid["time"].to_numpy(),
            coords=valid[["x", "y"]].to_numpy(),
            z=valid[var].to_numpy(),
            X_cov=X_cov,
            min_stations=MIN_STATIONS,
        )
        print(f"  trained on {model.n_train_rows_:,} rows, {model.n_features_} features")

        if SAVE_MODEL:
            model_path = get_rfsi_model_path(var, TIME_RES)
            dump({"model": model, "encoder": encoder, "variable": var,
                  "n_obs": N_OBS, "rf_params": rf_params}, model_path)
            print(f"  saved model → {model_path}")

        llocv_records = []

        for res in RESOLUTIONS:
            print(f"\n  resolution {res} m")
            grid_ds = xr.open_dataset(get_master_grid_path("RFSI", res))
            x_sel = (grid_ds.x >= domain_bbox[0]) & (grid_ds.x <= domain_bbox[2])
            y_sel = (grid_ds.y >= domain_bbox[1]) & (grid_ds.y <= domain_bbox[3])
            grid_crop = grid_ds.isel(x=x_sel, y=y_sel)
            ny = int(grid_crop.sizes["y"])
            nx = int(grid_crop.sizes["x"])
            print(f"  cropped grid: {ny} x {nx} = {ny * nx:,} cells")

            domain_mask = get_domain_mask(grid_crop, domain_bbox)
            domain_mask_flat = domain_mask.ravel()
            gx, gy = np.meshgrid(grid_crop.x.values, grid_crop.y.values)
            grid_points = np.column_stack([gx.ravel(), gy.ravel()])

            grid_df = pd.DataFrame()
            if use_elev and "elev" in grid_crop:
                grid_df["elev"] = grid_crop.elev.values.astype(np.float32).ravel()
            if use_lc:
                grid_df["clc_code"] = grid_landcover_flat(grid_crop)
            if len(grid_df.columns):
                X_grid, _ = prepare_covariates(
                    grid_df, encoder=encoder, use_elev=use_elev, use_lc=use_lc
                )
            else:
                X_grid = None
            need_t_grid = (
                str(var).lower().startswith("rh")
                and rh_t_mode in ("predicted", "observed")
                and "temp_mean" in valid.columns
            )
            t_model = None
            if need_t_grid:
                Xt, _ = prepare_covariates(
                    valid, encoder=encoder, use_elev=use_elev, use_lc=use_lc
                )
                t_model = RFSI(n_obs=N_OBS, rf_params=rf_params, var_name="temp_mean")
                t_model.fit_pooled(
                    times=valid["time"].to_numpy(),
                    coords=valid[["x", "y"]].to_numpy(),
                    z=valid["temp_mean"].to_numpy(),
                    X_cov=Xt,
                    min_stations=MIN_STATIONS,
                )
                print(f"  RH T-model trained on {t_model.n_train_rows_:,} rows")
            if X_grid is not None and model.n_features_ is not None and t_model is None:
                expect = model.n_features_
                got = neighbor_width(N_OBS) + X_grid.shape[1]
                if got != expect:
                    raise RuntimeError(
                        f"covariate mismatch: train features={expect}, "
                        f"predict features={got} (n_obs={N_OBS}, "
                        f"X_grid={X_grid.shape[1]})"
                    )

            out_file = get_interpolated_map_path(
                "RFSI", DOMAIN, var, res,
                start_date=str(START_DATE.date()),
                end_date=str(END_DATE.date()),
                time_resolution=TIME_RES,
            )

            time_steps = sorted(valid["time"].unique())
            data_3d = np.full((len(time_steps), ny, nx), np.nan, dtype=np.float32)
            kept_times = []
            station_rmse = []

            for i, t in enumerate(tqdm(time_steps, desc=f"  {var} {res}m")):
                df_t = valid[valid["time"] == t]
                if len(df_t) < MIN_STATIONS:
                    continue
                coords = df_t[["x", "y"]].to_numpy()
                z = df_t[var].to_numpy()
                X_t, _ = prepare_covariates(
                    df_t, encoder=encoder, use_elev=use_elev, use_lc=use_lc,
                    extras=extras_for_var(df_t, var, rh_t_mode),
                )
                X_grid_t = X_grid
                if t_model is not None and X_grid is not None:
                    t_hat = t_model.predict_field(
                        coords, df_t["temp_mean"].to_numpy(), grid_points, X_cov_pred=X_grid,
                    )
                    grid_df_t = grid_df.copy()
                    grid_df_t["temp_mean"] = t_hat
                    X_grid_t, _ = prepare_covariates(
                        grid_df_t, encoder=encoder, use_elev=use_elev, use_lc=use_lc,
                        extras=extras_for_var(grid_df_t, var, rh_t_mode),
                    )
                    expect = model.n_features_
                    got = neighbor_width(N_OBS) + X_grid_t.shape[1]
                    if got != expect:
                        raise RuntimeError(
                            f"RH covariate mismatch: train={expect} predict={got}"
                        )

                preds = predict_in_chunks(model, coords, z, grid_points, X_grid_t)
                pred_map = preds.reshape(ny, nx)
                pred_map.ravel()[~domain_mask_flat] = np.nan
                idx = len(kept_times)
                data_3d[idx] = pred_map
                kept_times.append(t)

                st_pred = model.predict_stations(coords, z, X_cov_obs=X_t)
                met = compute_metrics(z, st_pred)
                station_rmse.append(met["rmse"])
                if SAVE_LLOCV:
                    for name, obs, pred in zip(df_t["station_name"].to_numpy(), z, st_pred):
                        llocv_records.append({
                            "time": t, "station_name": name, "variable": var,
                            "observed": float(obs), "predicted": float(pred),
                            "resolution_m": int(res),
                        })

            n_kept = len(kept_times)
            data_3d = data_3d[:n_kept]
            time_coord = as_naive_utc(kept_times)
            ds = xr.Dataset(
                {var: (("time", "y", "x"), data_3d)},
                coords={
                    "time": time_coord,
                    "y": grid_crop.y.values,
                    "x": grid_crop.x.values,
                },
            )
            if "elev" in grid_crop:
                ds["elev"] = (("y", "x"), grid_crop.elev.values.astype(np.float32))
            ds["n_obs"] = ("time", np.full(n_kept, N_OBS, dtype=np.int16))
            ds["station_rmse"] = ("time", np.asarray(station_rmse, dtype=np.float32))
            ds.attrs.update({
                "title": f"{var} - RFSI pooled - {TIME_RES} - {DOMAIN}",
                "domain": DOMAIN,
                "resolution_m": int(res),
                "method": "RFSI",
                "method_full": "Random Forest Spatial Interpolation (pooled space-time)",
                "architecture": "pooled",
                "time_resolution": TIME_RES,
                "variable": var,
                "n_obs": N_OBS,
                "n_estimators": int(rf_params.get("n_estimators", 250)),
                "n_train_rows": int(model.n_train_rows_),
                "start_date": str(START_DATE.date()),
                "end_date": str(END_DATE.date()),
                "crs": "EPSG:31287",
                "created": datetime.now().isoformat(),
            })
            out_file.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(out_file, engine="netcdf4")
            print(f"  saved {n_kept} timesteps → {out_file}")
            grid_ds.close()
            del data_3d, ds
            gc.collect()

        if SAVE_LLOCV and llocv_records:
            llocv_path = get_llocv_path(
                "RFSI", DOMAIN, var, RESOLUTIONS[0],
                start_date=str(START_DATE.date()),
                end_date=str(END_DATE.date()),
                time_resolution=TIME_RES,
            )
            pd.DataFrame(llocv_records).to_parquet(llocv_path, index=False)
            print(f"  station predictions → {llocv_path}")
            print("  note: frozen-model, self excluded; not nested station LLOCV")

    print("\nRFSI pooled production finished.")


if __name__ == "__main__":
    main()
