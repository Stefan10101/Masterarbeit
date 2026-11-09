#!/usr/bin/env python3
"""
CNN hyperparameter search.

Protocol (pooled): fit on full pool × train quarters;
score fixed tune-targets on tune quarters (donors exclude them only at prediction).
Trains a U-Net per (n_levels, base_ch) combo.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import itertools
import sys

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_cnn_tuned_params_path, get_master_grid_path
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res, variables_from_cfg
from shared.splits.evaluation_sets import apply_evaluation_protocol
from cnn_core import grid_terrain, two_step_var
from cnn_data import clc_group, data_sources, load_cnn_config, load_panel, precip_trace, attach_temperature, attach_station_tpi
from train_cnn import cfg_to_cnn, train_one

try:
    import xarray as xr
except ImportError:
    xr = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default=None, help="single variable (prefer --variables)")
    p.add_argument("--variables", nargs="+", default=None, help="list of variables")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--jobs", type=int, default=4)
    add_hours_arg(p)
    add_time_res_arg(p)
    args = p.parse_args()
    cfg = load_cnn_config()
    apply_time_res(cfg, args)
    if args.quick:
        cfg.setdefault("cnn", {})
        cfg["cnn"]["epochs"] = min(int(cfg["cnn"].get("epochs", 40)), 8)
        cfg["cnn"]["patience"] = min(int(cfg["cnn"].get("patience", 8)), 3)
        cfg["cnn"]["search"] = {
            "n_levels": [int(cfg["cnn"].get("n_levels", 4))],
            "base_ch": [int(cfg["cnn"].get("base_ch", 32))],
            "train_mask_frac": [float(cfg["cnn"].get("train_mask_frac", 0.40))],
            "tau_wet": [float(cfg["cnn"].get("tau_wet", 0.5))],
        }
    if xr is None:
        raise RuntimeError("xarray required")
    time_res = cfg["time_resolution"]
    block = cfg["cnn"]
    search = block.get("search", {})
    _, _, grid_method = data_sources(cfg)
    res = int(cfg.get("resolutions_to_process", [1000])[0])
    grid = xr.open_dataset(get_master_grid_path(grid_method, res))
    gx = grid["x"].values.astype(np.float64)
    gy = grid["y"].values.astype(np.float64)
    elev = grid["elev"].values.astype(np.float64) if "elev" in grid else np.zeros((gy.size, gx.size))
    clc = np.zeros_like(elev)
    for name in ("clc_code", "clc", "landcover"):
        if name in grid:
            clc = clc_group(np.nan_to_num(grid[name].values, nan=0).astype(np.int32)).astype(np.float64)
            break
    slope, sinasp, cosasp = grid_terrain(elev, gx, gy)
    if args.variables:
        variables = args.variables
    elif args.variable:
        variables = [args.variable]
    else:
        variables = variables_from_cfg(cfg)
    for var in variables:
        panel = load_panel(cfg, var)
        # Fit on full pool × train. Score fixed tune-targets on tune.
        # train_one reads split=='dev' as the score set.
        from shared.splits.evaluation_sets import apply_evaluation_protocol
        tagged, _, _, _ = apply_evaluation_protocol(panel, variable=var)
        train = tagged[(~tagged["is_heldout"]) & (tagged["split"] == "train")].copy()
        # Score set: fixed tune-targets on tune quarters
        dev = tagged[(~tagged["is_heldout"]) & tagged["is_tune_target"]
                     & (tagged["split"] == "tune")].copy()
        dev["split"] = "dev"
        # Temporary donors for prediction: pool minus tune-targets
        pool_tune = tagged[(~tagged["is_heldout"]) & (~tagged["is_tune_target"])
                           & (tagged["split"] == "tune")].copy()
        pool_tune["split"] = "pool"
        panel = pd.concat([train, dev, pool_tune], ignore_index=True)
        if str(var).lower().startswith("rh"):
            panel = attach_temperature(panel, cfg)
        if str(var).lower().startswith("wind"):
            panel = attach_station_tpi(panel)
        print(f"{var} train rows={len(train)} tune-target rows={len(dev)} "
              f"tune times={dev['time'].nunique()}", flush=True)
        panel, _hours = apply_hour_cut(panel, time_res, args.hours)
        trace = precip_trace(cfg, time_res, var)
        best = None
        trials = []
        for levels, ch in itertools.product(
            search.get("n_levels", [block.get("n_levels", 4)]),
            search.get("base_ch", [block.get("base_ch", 32)]),
        ):
            ccfg = cfg_to_cnn(cfg, {"n_levels": levels, "base_ch": ch})
            _, dev = train_one(
                panel, var, ccfg, elev, clc, gx, gy,
                slope, sinasp, cosasp, time_res=time_res, trace=trace, jobs=max(1, args.jobs),
                cfg=cfg,
            )
            print(f"{var} levels={levels} ch={ch} dev={dev:.4f}", flush=True)
            trial = {
                "n_levels": int(levels),
                "base_ch": int(ch),
                "dev_mse": float(dev),
                "dev_rmse": float(dev) ** 0.5,
                "dev": float(dev),
            }
            trials.append(trial)
            if best is None or dev < best["dev"]:
                best = dict(trial)
        path = get_cnn_tuned_params_path(var, time_res)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(best, f)
        if trials:
            sp = path.with_name(f"{var}_search.csv")
            pd.DataFrame(trials).to_csv(sp, index=False)
            print(f"wrote {sp}  ({len(trials)} trials)")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
