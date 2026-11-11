#!/usr/bin/env python3
"""
Kriging production maps.

Fit one RK (or OK) model per variable on TRAIN+DEV station times.
Predict every timestamp on the master grid.
Station parquet from this script is a frozen-model check, not nested LLOCV.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from shared.map_io import save_map_with_partials
from paths import (
    get_interpolated_map_path,
    get_kriging_model_path,
    get_kriging_tuned_params_path,
    get_master_grid_path,
)
from kriging_core import KrigingInterpolator
from kriging_data import (
    attach_pack_terrain,
    attach_temperature,
    clc_group,
    compute_block,
    data_sources,
    load_config,
    load_panel,
    pack_from_master,
    subset_splits,
    subset_times,
    subset_years,
)
from llocv_kriging import cfg_to_kriging
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res, variables_from_cfg

try:
    import xarray as xr
except ImportError:
    xr = None


def load_tuned(var: str, time_res: str) -> dict:
    path = get_kriging_tuned_params_path(var, time_res)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def grid_clc(grid) -> np.ndarray | None:
    for name in ("clc_code", "clc", "landcover", "CLC"):
        if name in grid:
            return np.nan_to_num(grid[name].values.astype(np.float32), nan=0.0)
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variables", nargs="*", default=None)
    p.add_argument("--years", default=None)
    p.add_argument("--months", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--jobs", type=int, default=None,
                   help="parallel timestamps (default: half the logical cores)")
    p.add_argument("--rh-t-mode", default=None, choices=["none", "predicted", "observed"])
    add_hours_arg(p)
    add_time_res_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    time_res = apply_time_res(cfg, args)
    if xr is None:
        raise RuntimeError("xarray is required for produce_kriging_maps.py")

    domain = cfg["domain"]["preset"]
    start = pd.Timestamp(cfg["start_date"])
    end = pd.Timestamp(cfg["end_date"])
    resolutions = cfg.get("resolutions_to_process", [1000])
    kblock = cfg.get("kriging", {})
    variables = variables_from_cfg(cfg, args.variables)
    comp = compute_block(cfg)
    months = args.months or ("seasonal4" if args.quick else None)
    splits = args.split or ("test" if args.quick else comp.get("produce_splits", "all"))
    pack = pack_from_master(cfg, int(kblock.get("n_regions", 6)))
    save_model = bool(kblock.get("save_model", True))
    min_stations = int(cfg.get("min_stations_per_field", 10))
    _, _, grid_method = data_sources(cfg)

    print("=" * 72)
    print(f"Kriging production | {domain} | {time_res}")
    print(f"Period: {start.date()} → {end.date()}")
    print(f"months={months or 'all'} split={splits} pack={'yes' if pack else 'no'}")
    print("=" * 72)

    for var in variables:
        panel = load_panel(cfg, var)
        tuned = load_tuned(var, time_res)
        if args.rh_t_mode:
            tuned["rh_t_mode"] = args.rh_t_mode
        kcfg_preview = cfg_to_kriging(cfg, tuned)
        if str(var).lower().startswith("rh") and kcfg_preview.rh_t_mode in ("predicted", "observed"):
            panel = attach_temperature(panel, cfg)
            if kcfg_preview.rh_t_mode == "observed":
                print("produce: rh_t_mode=observed has no grid T; using predicted")
                tuned["rh_t_mode"] = "predicted"
        panel = attach_pack_terrain(panel, pack)
        panel = subset_splits(panel, splits) if splits and splits != "all" else panel
        # keep train+dev for the pooled fit even if maps are TEST-only
        fit_panel = attach_pack_terrain(load_panel(cfg, var), pack)
        if str(var).lower().startswith("rh") and kcfg_preview.rh_t_mode in ("predicted", "observed"):
            fit_panel = attach_temperature(fit_panel, cfg)
        fit = fit_panel[fit_panel["split"].isin(("train", "dev"))]
        # months filters which maps to write, not the training data.
        # Fit stays on the full train+dev window so a 2025-only produce still has rows.
        if months:
            panel = subset_times(panel, months)
        panel, _hours = apply_hour_cut(panel, time_res, args.hours)
        fit, _ = apply_hour_cut(fit, time_res, args.hours)
        panel = subset_years(panel, args.years)
        if fit.empty:
            print(f"  [SKIP] {var}: no train/dev rows")
            continue
        kcfg = cfg_to_kriging(cfg, tuned)
        print(f"\n>>> {var}  trend={kcfg.trend}  k={kcfg.k}  family={kcfg.family}  "
              f"az={kcfg.alpha_z}  r={kcfg.aniso_ratio}  int={kcfg.interactions}  rh_t={kcfg.rh_t_mode}")
        print(f"  fitting on {len(fit):,} train+dev rows")
        model = KrigingInterpolator(kcfg)
        model.pack = pack
        info = model.fit(fit, var, time_res, cfg)
        t_model = None
        if str(var).lower().startswith("rh") and kcfg.rh_t_mode == "predicted" and "temp_mean" in fit.columns:
            t_model = KrigingInterpolator(kcfg)
            t_model.pack = pack
            t_model.fit(fit, "temp_mean", time_res, cfg)
        print(f"  fit done  n_regimes={info['n_regimes']}  "
              f"vgm_pairs={info['vgm'].get('n_pairs')}  "
              f"range={info['vgm'].get('range'):.0f}")
        if save_model:
            path = get_kriging_model_path(var, time_res)
            joblib.dump(model.state_dict(), path)
            print(f"  model → {path}")

        for res in resolutions:
            grid_path = get_master_grid_path(grid_method, res)
            grid = xr.open_dataset(grid_path)
            ny, nx = grid.y.size, grid.x.size
            xx, yy = np.meshgrid(grid.x.values, grid.y.values)
            elev = grid["elev"].values if "elev" in grid else np.full((ny, nx), np.nan)
            clc = grid_clc(grid)
            flat_x = xx.ravel()
            flat_y = yy.ravel()
            elev_f = elev.astype(np.float32)
            valid_cell = np.isfinite(elev_f.ravel()) & (elev_f.ravel() > -100.0)
            flat_e = np.where(valid_cell, elev_f.ravel(), 0.0).astype(np.float32)
            flat_c = (
                np.zeros(flat_x.size, dtype=np.int32)
                if clc is None
                else clc_group(np.nan_to_num(clc.ravel(), nan=0.0).astype(np.int32))
            )

            out_file = get_interpolated_map_path(
                "Kriging", domain, var, res,
                start_date=str(start.date()),
                end_date=str(end.date()),
                time_resolution=time_res,
            )
            if out_file.exists() and not cfg.get("overwrite_maps", True):
                print(f"  skip existing {out_file.name}")
                grid.close()
                continue

            # Grid covariates so the map query matches the station fit.
            from shared.terrain import enabled_covariates
            grid_cov = {}
            try:
                from paths import get_grid_terrain_path
                tds = xr.open_dataset(get_grid_terrain_path(int(res)))
                for col in enabled_covariates(cfg):
                    if col in ("elev", "clc_level2") or col not in tds:
                        continue
                    grid_cov[col] = np.nan_to_num(tds[col].values, nan=0.0).ravel().astype(np.float64)
                tds.close()
            except Exception as exc:
                print(f"  grid covariates unavailable ({exc})", flush=True)

            times = sorted(panel["time"].unique())
            n_jobs = int(args.jobs) if args.jobs is not None else max(1, (os.cpu_count() or 1) // 2)
            n_valid = int(valid_cell.sum())
            print(f"  {var} {res}m: {n_valid:,} cells x {len(times)} times, jobs={n_jobs}", flush=True)

            def one(t):
                df_t = panel[panel["time"] == t]
                if len(df_t) < min_stations:
                    return None
                pred_flat = np.full(flat_x.size, np.nan, dtype=np.float32)
                idx = np.flatnonzero(valid_cell)
                tile = kcfg.predict_tile
                for s in range(0, len(idx), tile):
                    sl = idx[s:s + tile]
                    qry = pd.DataFrame({
                        "x": flat_x[sl],
                        "y": flat_y[sl],
                        "elev": flat_e[sl],
                        "clc_group": flat_c[sl],
                        "cluster_id": np.full(len(sl), int(df_t["cluster_id"].iloc[0])),
                        var: np.zeros(len(sl), dtype=np.float32),
                    })
                    qry = attach_pack_terrain(qry, pack)
                    for col, arr in grid_cov.items():
                        qry[col] = arr[sl]
                    if t_model is not None:
                        qry["temp_mean"] = t_model.predict_frame(df_t, qry, "temp_mean")
                    pred_flat[sl] = model.predict_frame(df_t, qry, var)
                return t, pred_flat.reshape(ny, nx)

            # Bar must count finished months, not submitted ones. Wrapping the
            # generator that feeds Parallel hits 100% before any kriging runs.
            if n_jobs == 1:
                results = [one(t) for t in tqdm(times, desc=f"  {var} {res}m")]
            else:
                from threading import Lock
                bar = tqdm(total=len(times), desc=f"  {var} {res}m")
                lock = Lock()

                def one_counted(t):
                    try:
                        return one(t)
                    finally:
                        with lock:
                            bar.update(1)

                results = Parallel(n_jobs=n_jobs, backend="threading")(
                    delayed(one_counted)(t) for t in times
                )
                bar.close()
            results = [r for r in results if r is not None]
            if not results:
                print(f"  {var}: no fields")
                grid.close()
                continue
            kept, stacks = zip(*results)
            data_3d = np.stack(stacks).astype(np.float32)
            n_kept = len(kept)
            # data_3d already stacked to n_kept; no slice needed
            time_coord = pd.DatetimeIndex(pd.to_datetime(kept, utc=True)).tz_convert(
                "UTC"
            ).tz_localize(None).to_numpy(dtype="datetime64[ns]")
            ds = xr.Dataset(
                {var: (("time", "y", "x"), data_3d)},
                coords={"time": time_coord, "y": grid.y.values, "x": grid.x.values},
            )
            if "elev" in grid:
                ds["elev"] = (("y", "x"), grid.elev.values.astype(np.float32))
            ds["k"] = ("time", np.full(n_kept, kcfg.k, dtype=np.int16))
            ds.attrs.update({
                "title": f"{var} - Kriging - {time_res} - {domain}",
                "domain": domain,
                "resolution_m": int(res),
                "method": "Kriging",
                "method_full": "Regression-Kriging (linear trend + residual OK)"
                if kcfg.trend == "linear" else "Ordinary Kriging",
                "time_resolution": time_res,
                "variable": var,
                "trend": kcfg.trend,
                "interactions": kcfg.interactions,
                "family": kcfg.family,
                "k": kcfg.k,
                "k_indicator": kcfg.k_indicator,
                "theta_deg": kcfg.theta_deg,
                "aniso_ratio": kcfg.aniso_ratio,
                "alpha_z": kcfg.alpha_z,
                "start_date": str(start.date()),
                "end_date": str(end.date()),
                "crs": "EPSG:31287",
                "created": datetime.now().isoformat(),
            })
            out_file.parent.mkdir(parents=True, exist_ok=True)
            save_map_with_partials(ds, out_file, time_res)
            print(f"  saved {n_kept} timesteps → {out_file}")
            grid.close()
            del data_3d, ds
            gc.collect()

    print("\nKriging production finished.")


if __name__ == "__main__":
    main()
