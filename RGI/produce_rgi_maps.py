#!/usr/bin/env python3
"""
RGI production maps.

Fit one model per variable on TRAIN+DEV station times (all stations).
Predict every timestamp on the master grid. Query cells never send values.
Station parquet from this script is a frozen-model check, not nested LLOCV.
"""

from __future__ import annotations

from pathlib import Path
import sys
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import (
    get_interpolated_map_path,
    get_llocv_path,
    get_master_grid_path,
    get_rgi_model_path,
    get_rgi_tuned_params_path,
)
from rgi_core import RGI
from rgi_data import (
    attach_extras,
    compute_metrics,
    data_sources,
    extra_cols_for_var,
    load_config,
    load_panel,
)
from llocv_rgi import cfg_to_rgi

try:
    import xarray as xr
except ImportError:
    xr = None


def load_tuned(var: str, time_res: str) -> dict:
    path = get_rgi_tuned_params_path(var, time_res)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def grid_clc(grid) -> np.ndarray:
    for name in ("clc_code", "clc", "landcover", "CLC"):
        if name in grid:
            return np.nan_to_num(grid[name].values.astype(np.float32), nan=0.0)
    return None


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--variables", nargs="*", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--months", default=None)
    args = p.parse_args()
    cfg = load_config()
    if args.quick:
        cfg.setdefault("rgi", {})
        cfg["rgi"]["epochs"] = min(int(cfg["rgi"].get("epochs", 80)), 15)
        cfg["rgi"]["patience"] = min(int(cfg["rgi"].get("patience", 12)), 4)
        cfg["rgi"]["save_llocv"] = False
        cfg["_quick_months"] = args.months or "seasonal4"
    if xr is None:
        raise RuntimeError("xarray is required for produce_rgi_maps.py")

    domain = cfg["domain"]["preset"]
    time_res = cfg["time_resolution"]
    start = pd.Timestamp(cfg["start_date"])
    end = pd.Timestamp(cfg["end_date"])
    resolutions = cfg.get("resolutions_to_process", [1000])
    rgi_cfg = cfg.get("rgi", {})
    variables = args.variables or rgi_cfg.get("variables_to_process") or [
        "temp_mean", "precip_sum", "wind_mean", "rh_mean", "snow_mean",
    ]
    save_llocv = bool(rgi_cfg.get("save_llocv", True))
    save_model = bool(rgi_cfg.get("save_model", True))
    tile = int(rgi_cfg.get("predict_tile", 120000))
    min_stations = int(cfg.get("min_stations_per_field", 10))
    _, _, grid_method = data_sources(cfg)

    from rgi_core import describe_device, resolve_device
    print("=" * 72)
    print(f"RGI production | {domain} | {time_res}")
    print(f"Period: {start.date()} → {end.date()}")
    print(f"  device {describe_device(resolve_device(rgi_cfg.get('device', 'auto')))}")
    print("=" * 72)

    for var in variables:
        panel = attach_extras(load_panel(cfg, var), cfg, var)
        if cfg.get("_quick_months"):
            from Kriging.kriging_data import subset_times, subset_splits
            panel = subset_times(panel, cfg["_quick_months"])
        fit = panel[panel["split"].isin(("train", "dev"))]
        if fit.empty:
            print(f"  [SKIP] {var}: no train/dev rows")
            continue
        tuned = load_tuned(var, time_res)
        tuned["extra_cols"] = extra_cols_for_var(var, cfg)
        tuned["variable"] = var
        rcfg = cfg_to_rgi(cfg, tuned)
        rcfg.two_step = bool(rgi_cfg.get("two_step", True)) and (
            str(var).lower().startswith("precip") or str(var).lower().startswith("snow")
        )
        print(f"\n>>> {var}  k={rcfg.k} L={rcfg.n_layers} "
              f"alpha={rcfg.alpha} az={rcfg.alpha_z}")
        print(f"  fitting on {len(fit):,} train+dev rows")
        val = fit[fit["split"] == "dev"]
        model = RGI(rcfg)
        info = model.fit(fit, var, val_df=val if len(val) else None)
        print(f"  fit done  val_mse={info['best_val_mse']:.4f}  epochs={info['epochs']}")
        if save_model:
            path = get_rgi_model_path(var, time_res)
            model.save(path)
            print(f"  model → {path}")

        llocv_records = []
        for res in resolutions:
            grid_path = get_master_grid_path(grid_method, res)
            grid = xr.open_dataset(grid_path)
            ny, nx = grid.y.size, grid.x.size
            xx, yy = np.meshgrid(grid.x.values, grid.y.values)
            elev = grid["elev"].values if "elev" in grid else np.full((ny, nx), np.nan)
            clc = grid_clc(grid)
            flat_x = xx.ravel()
            flat_y = yy.ravel()
            # COP30 / master-grid nodata is a finite sentinel (often << 0), not NaN.
            elev_f = elev.astype(np.float32)
            valid_cell = np.isfinite(elev_f.ravel()) & (elev_f.ravel() > -100.0)
            flat_e = np.where(valid_cell, elev_f.ravel(), 0.0).astype(np.float32)
            flat_c = (
                np.zeros(flat_x.size, dtype=np.int32)
                if clc is None
                else np.nan_to_num(clc.ravel(), nan=0.0).astype(np.int32)
            )

            out_file = get_interpolated_map_path(
                "RGI", domain, var, res,
                start_date=str(start.date()),
                end_date=str(end.date()),
                time_resolution=time_res,
            )
            if out_file.exists() and not cfg.get("overwrite_maps", True):
                print(f"  skip existing {out_file.name}")
                grid.close()
                continue

            times = sorted(panel["time"].unique())
            data_3d = np.full((len(times), ny, nx), np.nan, dtype=np.float32)
            kept = []
            station_rmse = []
            from tqdm import tqdm
            for t in tqdm(times, desc=f"  {var} {res}m"):
                df_t = panel[panel["time"] == t]
                if len(df_t) < min_stations:
                    continue
                pred_flat = np.full(flat_x.size, np.nan, dtype=np.float32)
                idx = np.flatnonzero(valid_cell)
                for s in range(0, len(idx), tile):
                    sl = idx[s:s + tile]
                    qry = pd.DataFrame({
                        "x": flat_x[sl],
                        "y": flat_y[sl],
                        "elev": flat_e[sl],
                        "clc_code": flat_c[sl],
                        var: np.zeros(len(sl), dtype=np.float32),
                    })
                    pred_flat[sl] = model.predict_frame(df_t, qry, var)
                i = len(kept)
                data_3d[i] = pred_flat.reshape(ny, nx)
                kept.append(t)
                preds_st = []
                for name in df_t["station_name"].tolist():
                    y1 = model.predict_stations_at_time(df_t, var, [name])
                    preds_st.append(float(y1[0]) if len(y1) else np.nan)
                preds_st = np.asarray(preds_st, dtype=np.float32)
                met = compute_metrics(df_t[var].to_numpy(), preds_st)
                station_rmse.append(met["rmse"])
                if save_llocv:
                    for name, obs, pr in zip(df_t["station_name"], df_t[var], preds_st):
                        llocv_records.append({
                            "time": t, "station_name": name, "variable": var,
                            "observed": float(obs), "predicted": float(pr),
                            "resolution_m": int(res),
                        })

            n_kept = len(kept)
            data_3d = data_3d[:n_kept]
            time_coord = pd.DatetimeIndex(pd.to_datetime(kept, utc=True)).tz_convert(
                "UTC"
            ).tz_localize(None).to_numpy(dtype="datetime64[ns]")
            ds = xr.Dataset(
                {var: (("time", "y", "x"), data_3d)},
                coords={"time": time_coord, "y": grid.y.values, "x": grid.x.values},
            )
            if "elev" in grid:
                ds["elev"] = (("y", "x"), grid.elev.values.astype(np.float32))
            ds["k"] = ("time", np.full(n_kept, rcfg.k, dtype=np.int16))
            ds["station_rmse"] = ("time", np.asarray(station_rmse, dtype=np.float32))
            ds.attrs.update({
                "title": f"{var} - RGI - {time_res} - {domain}",
                "domain": domain,
                "resolution_m": int(res),
                "method": "RGI",
                "method_full": "Residual Graph Interpolator (initial residual, spatial-only)",
                "time_resolution": time_res,
                "variable": var,
                "k": rcfg.k,
                "n_layers": rcfg.n_layers,
                "alpha": rcfg.alpha,
                "alpha_z": rcfg.alpha_z,
                "start_date": str(start.date()),
                "end_date": str(end.date()),
                "crs": "EPSG:31287",
                "created": datetime.now().isoformat(),
            })
            out_file.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(out_file, engine="netcdf4")
            print(f"  saved {n_kept} timesteps → {out_file}")
            grid.close()
            del data_3d, ds
            gc.collect()

        if save_llocv and llocv_records:
            llocv_path = get_llocv_path(
                "RGI", domain, var, resolutions[0],
                start_date=str(start.date()),
                end_date=str(end.date()),
                time_resolution=time_res,
            )
            pd.DataFrame(llocv_records).to_parquet(llocv_path, index=False)
            print(f"  frozen station check → {llocv_path}")
            print("  note: not nested station LLOCV (see llocv_rgi.py)")

    print("\nRGI production finished.")


if __name__ == "__main__":
    main()
