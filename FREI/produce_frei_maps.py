#!/usr/bin/env python3
"""Frei production maps on the master grid."""

from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from shared.map_io import save_map_with_partials
from paths import get_interpolated_map_path, get_master_grid_path
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res, variables_from_cfg
from shared.splits.evaluation_sets import apply_evaluation_protocol
from frei_core import FreiInterpolator, two_step_var
from frei_data import (
    load_idw_residual_params,
    attach_temperature,
    compute_block,
    data_sources,
    load_frei_config,
    load_panel,
    precip_trace,
    subset_splits,
    subset_times,
    subset_years,
)
from llocv_frei import cfg_to_frei, load_tuned, pack_from_master

try:
    import xarray as xr
except ImportError:
    xr = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default=None, help="single variable (prefer --variables)")
    p.add_argument("--variables", nargs="+", default=None, help="list of variables")
    p.add_argument("--years", default=None, help="e.g. 2024,2025")
    p.add_argument("--months", default=None, help="all | seasonal4 | YYYY-MM,YYYY-MM")
    p.add_argument("--split", default=None, help="all | test | dev,test")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--jobs", type=int, default=None,
                   help="parallel timestamps (default: half the logical cores)")
    p.add_argument("--rh-t-mode", default=None, choices=["none", "predicted", "observed"])
    add_hours_arg(p)
    add_time_res_arg(p)
    args = p.parse_args()

    cfg = load_frei_config()
    time_res = apply_time_res(cfg, args)
    if xr is None:
        raise RuntimeError("xarray required")
    domain = cfg["domain"]["preset"]
    _, _, grid_method = data_sources(cfg)
    if args.variables:
        variables = args.variables
    elif args.variable:
        variables = [args.variable]
    else:
        variables = variables_from_cfg(cfg)
    min_stations = int(cfg.get("min_stations_per_field", 10))
    comp = compute_block(cfg)
    months = args.months or ("seasonal4" if args.quick else None)
    years = args.years
    splits = args.split or ("test" if args.quick else comp.get("produce_splits", "all"))

    for res in cfg.get("resolutions_to_process", [1000]):
        grid = xr.open_dataset(get_master_grid_path(grid_method, res))
        x = grid["x"].values.astype(np.float64)
        y = grid["y"].values.astype(np.float64)
        z = grid["elev"].values.astype(np.float64) if "elev" in grid else np.zeros((y.size, x.size))
        xx, yy = np.meshgrid(x, y)
        zz = z if z.ndim == 2 else np.zeros_like(xx)
        xq, yq, zq = xx.ravel(), yy.ravel(), zz.ravel()

        for var in variables:
            out = get_interpolated_map_path("Frei", domain, var, res, cfg["start_date"], cfg["end_date"], time_res)
            if out.exists() and not cfg.get("overwrite_maps", True):
                print(f"exists, skip {out.name}", flush=True)
                continue
            panel = load_panel(cfg, var)
            # Tag with shared quarter splits so subset_splits can select train/tune/test.
            # Produce keeps all stations (including held-out) as donors; only time is filtered.
            panel, _, _, _ = apply_evaluation_protocol(panel, variable=var)
            tuned = load_tuned(var, time_res)
            tuned["trace"] = precip_trace(cfg, time_res, var)
            tuned.update(load_idw_residual_params(time_res, var))
            fcfg = cfg_to_frei(cfg, tuned)
            if var.lower().startswith("rh") and fcfg.rh_t_mode in ("predicted", "observed"):
                panel = attach_temperature(panel, cfg)
                if fcfg.rh_t_mode == "observed":
                    print("produce: rh_t_mode=observed has no grid T; using predicted", flush=True)
                    fcfg.rh_t_mode = "predicted"
            panel = subset_splits(panel, splits)
            panel = subset_years(panel, years)
            if months:
                panel = subset_times(panel, months)
            panel, _hours = apply_hour_cut(panel, time_res, args.hours)
            pack = pack_from_master(cfg, fcfg.n_regions)
            model = FreiInterpolator(fcfg, pack=pack)
            use_profile = not two_step_var(var)
            print(
                f"{var} produce times={panel['time'].nunique()} split={splits} "
                f"years={years or 'all'} months={months or 'all'}",
                flush=True,
            )
            # Grid covariates for the linear background, same keys as LLOCV.
            from shared.terrain import enabled_covariates
            cov_keys = [c for c in enabled_covariates(cfg)
                        if c not in ("elev", "slope", "northness", "clc_level2")]
            grid_cov = {}
            try:
                from paths import get_grid_terrain_path
                tds = xr.open_dataset(get_grid_terrain_path(int(res)))
                for col in cov_keys:
                    if col in tds:
                        grid_cov[col] = np.nan_to_num(tds[col].values, nan=0.0).ravel().astype(np.float64)
                tds.close()
            except Exception as exc:
                print(f"  grid covariates unavailable ({exc})", flush=True)

            groups = list(panel.groupby("time"))
            n_jobs = int(args.jobs) if args.jobs is not None else max(1, (os.cpu_count() or 1) // 2)

            def one(ts_sl):
                ts, sl = ts_sl
                if sl["station_name"].nunique() < min_stations:
                    return None
                hat, _ = model.predict_timestamp(
                    sl["x"].to_numpy(), sl["y"].to_numpy(), sl["elev"].to_numpy(),
                    sl[var].to_numpy(), xq, yq, zq, use_profile=True, var=var,
                )
                stamp = pd.Timestamp(ts)
                if stamp.tzinfo is not None:
                    stamp = stamp.tz_convert("UTC").tz_localize(None)
                return np.datetime64(stamp, "ns"), hat.reshape(yy.shape).astype(np.float32)

            if n_jobs == 1:
                results = [one(g) for g in tqdm(groups, desc=f"  {var}", unit="month")]
            else:
                results = Parallel(n_jobs=n_jobs, backend="threading")(
                    delayed(one)(g) for g in tqdm(groups, desc=f"  {var}", unit="month")
                )
            results = [r for r in results if r is not None]
            if not results:
                print(f"{var} no fields written", flush=True)
                continue
            used, fields = zip(*results)
            da = xr.DataArray(np.stack(fields), dims=("time", "y", "x"),
                              coords={"time": np.array(used), "y": y, "x": x}, name=var)
            out = get_interpolated_map_path("Frei", domain, var, res, cfg["start_date"], cfg["end_date"], time_res)
            save_map_with_partials(xr.Dataset({var: da}), out, time_res)
            print(f"wrote {out} n={len(fields)}", flush=True)


if __name__ == "__main__":
    main()
