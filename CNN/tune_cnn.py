#!/usr/bin/env python3
"""DEV search for CNN: base_ch × mask_frac. Trains a U-Net per combo."""

from __future__ import annotations

from pathlib import Path
import argparse
import itertools
import sys

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_cnn_tuned_params_path, get_master_grid_path
from shared.time_res import add_time_res_arg, apply_time_res
from cnn_core import grid_terrain, two_step_var
from cnn_data import clc_group, data_sources, load_cnn_config, load_panel, precip_trace
from train_cnn import cfg_to_cnn, train_one

try:
    import xarray as xr
except ImportError:
    xr = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default=None)
    p.add_argument("--quick", action="store_true")
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
    variables = [args.variable] if args.variable else block.get("variables_to_process", ["temp_mean"])
    for var in variables:
        panel = load_panel(cfg, var)
        trace = precip_trace(cfg, time_res, var)
        taus = list(search.get("tau_wet", [block.get("tau_wet", 0.5)])) if two_step_var(var) else [block.get("tau_wet", 0.5)]
        best = None
        for levels, ch, frac, tau in itertools.product(
            search.get("n_levels", [block.get("n_levels", 4)]),
            search.get("base_ch", [block.get("base_ch", 32)]),
            search.get("train_mask_frac", [block.get("train_mask_frac", 0.25)]),
            taus,
        ):
            ccfg = cfg_to_cnn(cfg, {
                "n_levels": levels, "base_ch": ch,
                "train_mask_frac": frac, "tau_wet": tau,
            })
            _, dev = train_one(
                panel, var, ccfg, elev, clc, gx, gy,
                slope, sinasp, cosasp, time_res=time_res, trace=trace,
            )
            print(f"{var} levels={levels} ch={ch} mask={frac} tau={tau} dev={dev:.4f}", flush=True)
            if best is None or dev < best["dev"]:
                best = {
                    "n_levels": int(levels),
                    "base_ch": int(ch),
                    "train_mask_frac": float(frac),
                    "tau_wet": float(tau),
                    "dev": float(dev),
                }
        path = get_cnn_tuned_params_path(var, time_res)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(best, f)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
