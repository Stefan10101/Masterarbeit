#!/usr/bin/env python3
"""
IDW map production.

Loads the global (p, Fz, k) from tune_idw.py and interpolates every
timestamp onto the master grid. No scoring — that is llocv_idw.py.
"""

from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from sklearn.neighbors import KDTree
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from idw_core import get_valid_targets, interpolate_idw
from idw_data import load_config, load_global_params, load_panel
from paths import get_interpolated_map_path, get_master_grid_path
from shared.time_res import add_time_res_arg, apply_time_res


def interpolate_one(args):
    t, df_t, params, target_coords, target_elev, chunk_size = args
    if len(df_t) < 5:
        return None
    k = min(int(params["k"]), len(df_t))
    tree = KDTree(df_t[["x", "y"]].to_numpy())
    data = interpolate_idw(
        station_values=df_t["value"].to_numpy(),
        station_coords=df_t[["x", "y"]].to_numpy(),
        station_elev=df_t["elev"].to_numpy(),
        target_coords=target_coords,
        target_elev=target_elev,
        tree=tree,
        p=float(params["p"]), Fz=float(params["Fz"]), k=k,
        chunk_size=chunk_size,
    )
    return {"time": t, "data": data}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variables", nargs="+", default=None)
    p.add_argument("--months", default=None)
    p.add_argument("--jobs", type=int, default=None)
    add_time_res_arg(p)
    args = p.parse_args()

    cfg = load_config()
    time_res = apply_time_res(cfg, args)
    domain = cfg.get("domain", {}).get("preset", "full")
    resolutions = cfg.get("resolutions_to_process", [1000])
    # Sequential by default. joblib process workers crash on Windows.
    n_jobs = int(args.jobs) if args.jobs is not None else 1
    variables = args.variables or ["temp_mean", "precip_sum"]

    print("=" * 72)
    print(f"IDW maps | {domain} | {time_res} | {cfg['start_date']} → {cfg['end_date']}")
    print("=" * 72)

    for res in resolutions:
        grid = xr.open_dataset(get_master_grid_path("IDW", res))
        target_coords, target_elev, mask = get_valid_targets(grid)
        if len(target_coords) == 0:
            print(f"  {res} m: 0 valid cells — skipping")
            grid.close()
            continue
        chunk_size = 1_000_000 if res < 1000 else None
        elev_2d = grid["elev"].values.astype(np.float32) if "elev" in grid else None
        print(f"\n{res} m  valid cells={len(target_coords):,}")

        for var in variables:
            panel = load_panel(cfg, var)
            if args.months and args.months != "all":
                from idw_data import subset_times
                panel = subset_times(panel, args.months)
            params = load_global_params(time_res, var)

            out_file = get_interpolated_map_path(
                "IDW", domain, var, res,
                start_date=str(cfg["start_date"]),
                end_date=str(cfg["end_date"]),
                time_resolution=time_res,
            )
            times = sorted(panel["time"].unique())
            tasks = [
                (t, panel[panel["time"] == t], params,
                 target_coords, target_elev, chunk_size)
                for t in times
            ]
            if n_jobs == 1:
                results = [interpolate_one(task) for task in tqdm(tasks, desc=f"  {var}", leave=False)]
            else:
                results = Parallel(n_jobs=n_jobs, backend="threading")(
                    delayed(interpolate_one)(task)
                    for task in tqdm(tasks, desc=f"  {var}", leave=False)
                )
            results = [r for r in results if r is not None]
            if not results:
                print(f"  {var}: no results")
                continue

            n_y, n_x = mask.shape
            data_3d = np.full((len(results), n_y, n_x), np.nan, dtype=np.float32)
            for i, r in enumerate(results):
                data_3d[i][mask] = r["data"]

            time_coord = pd.DatetimeIndex(
                pd.to_datetime([r["time"] for r in results], utc=True)
            ).tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")

            ds = xr.Dataset(
                {var: (("time", "y", "x"), data_3d)},
                coords={"time": time_coord, "y": grid["y"].values, "x": grid["x"].values},
            )
            if elev_2d is not None:
                ds["elev"] = (("y", "x"), elev_2d)
            ds["p"] = float(params["p"])
            ds["Fz"] = float(params["Fz"])
            ds["k"] = int(params["k"])
            ds.attrs.update({
                "title": f"{var} - IDW - {time_res} - {domain}",
                "domain": domain,
                "time_resolution": time_res,
                "variable": var,
                "resolution_m": int(res),
                "method": "IDW",
                "method_full": "Modified IDW, global parameters",
                "start_date": str(cfg["start_date"]),
                "end_date": str(cfg["end_date"]),
                "crs": "EPSG:31287",
                "created": datetime.now().isoformat(),
            })
            ds.to_netcdf(out_file, engine="netcdf4")
            print(f"  saved {out_file.name}  ({len(results)} timesteps)")
            del data_3d, ds, results, tasks
            gc.collect()

        grid.close()
        gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
