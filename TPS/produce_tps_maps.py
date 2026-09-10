#!/usr/bin/env python3
"""TPS / E-OBS production maps on the master grid."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_interpolated_map_path, get_master_grid_path, get_tps_tuned_params_path
from tps_core import TPSInterpolator, attach_watershed
from tps_data import (
    attach_pack_terrain,
    attach_temperature,
    data_sources,
    load_panel,
    load_tps_config,
    pack_from_master,
    precip_trace,
    subset_splits,
    subset_times,
    subset_years,
    uses_two_step,
)
from llocv_tps import cfg_to_tps, load_tuned, month_key, pack_from_master, predict_rows
from shared.time_res import add_time_res_arg, apply_time_res

try:
    import xarray as xr
except ImportError:
    xr = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default=None)
    p.add_argument("--years", default=None)
    p.add_argument("--months", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--rh-t-mode", default=None, choices=["none", "predicted", "observed"])
    add_time_res_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_tps_config()
    time_res = apply_time_res(cfg, args)
    if xr is None:
        raise RuntimeError("xarray required")
    domain = cfg["domain"]["preset"]
    start = str(cfg["start_date"]).replace("-", "")
    end = str(cfg["end_date"]).replace("-", "")
    _, _, grid_method = data_sources(cfg)
    variables = [args.variable] if args.variable else (cfg["tps"].get("variables_to_process") or ["temp_mean"])
    months = args.months or ("seasonal4" if args.quick else None)
    splits = args.split or ("test" if args.quick else "all")
    pack = pack_from_master(cfg, int(cfg.get("tps", {}).get("n_regions", 6)))
    min_stations = int(cfg.get("min_stations_per_field", 10))

    for res in cfg.get("resolutions_to_process", [1000]):
        grid = xr.open_dataset(get_master_grid_path(grid_method, res))
        x = grid["x"].values.astype(np.float64)
        y = grid["y"].values.astype(np.float64)
        if "elev" in grid:
            z = grid["elev"].values.astype(np.float64)
        else:
            z = np.zeros_like(x)
        xx, yy = np.meshgrid(x, y)
        if z.ndim == 2:
            zz = z
        else:
            zz = np.meshgrid(x, y)[0] * 0.0
        xq, yq, zq = xx.ravel(), yy.ravel(), zz.ravel()

        for var in variables:
            panel = load_panel(cfg, var)
            tuned = load_tuned(var, time_res)
            tuned["trace"] = precip_trace(cfg, time_res, var)
            if args.rh_t_mode:
                tuned["rh_t_mode"] = args.rh_t_mode
            tcfg = cfg_to_tps(cfg, tuned)
            if str(var).lower().startswith("rh") and tcfg.rh_t_mode in ("predicted", "observed"):
                panel = attach_temperature(panel, cfg)
                if tcfg.rh_t_mode == "observed":
                    print("produce: rh_t_mode=observed has no grid T; using predicted")
                    tcfg.rh_t_mode = "predicted"
            panel = attach_pack_terrain(panel, pack)
            panel = subset_splits(panel, splits)
            if months:
                panel = subset_times(panel, months)
            panel = subset_years(panel, args.years)
            model = TPSInterpolator(tcfg)
            attach_watershed(
                model,
                pack_from_master(cfg, tcfg.n_regions),
                panel["x"].to_numpy(), panel["y"].to_numpy(), panel["elev"].to_numpy(),
            )
            times = sorted(panel["time"].unique())
            fields = []
            used = []
            for ts in times:
                sl = panel[panel["time"] == ts]
                if sl["station_name"].nunique() < min_stations:
                    continue
                # fake query frame so predict_rows can reuse E-OBS monthly logic
                q = pd.DataFrame({"x": xq, "y": yq, "elev": zq, "time": ts})
                q = attach_pack_terrain(q, pack)
                hat = predict_rows(model, sl, q, var, monthly_panel=panel, time_res=time_res)
                fields.append(hat.reshape(yy.shape).astype(np.float32))
                used.append(np.datetime64(ts, "ns"))
            if not fields:
                print(f"skip {var}: no timestamps")
                continue
            da = xr.DataArray(
                np.stack(fields),
                dims=("time", "y", "x"),
                coords={"time": np.array(used), "y": y, "x": x},
                name=var,
            )
            ds = xr.Dataset({var: da})
            out = get_interpolated_map_path("TPS", domain, var, res, cfg["start_date"], cfg["end_date"], time_res)
            ds.to_netcdf(out)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
