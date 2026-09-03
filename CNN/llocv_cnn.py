#!/usr/bin/env python3
"""CNN LLOCV.

Default: frozen pooled model, score held-out station pixels (same protocol
as produce_* station parquet — not nested).

--nested retrains one U-Net per station fold on TRAIN times of the other
stations and scores DEV+TEST of the held-out stations.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_cnn_model_path, get_master_grid_path, get_nested_llocv_path
from cnn_core import CNNInterpolator, cyclic_time, idw_raster, station_to_raster, two_step_var
from cnn_data import data_sources, load_cnn_config, load_panel, print_split_metrics
from train_cnn import cfg_to_cnn, grid_index, load_tuned, train_one

try:
    import xarray as xr
except ImportError:
    xr = None


def station_folds(names, n_folds, seed):
    rng = np.random.default_rng(seed)
    names = list(names)
    rng.shuffle(names)
    folds = [[] for _ in range(n_folds)]
    for i, n in enumerate(names):
        folds[i % n_folds].append(n)
    return folds


def predict_stations(model, part, var, gx, gy, elev, clc, ccfg):
    rows, cols = grid_index(part["x"].to_numpy(), part["y"].to_numpy(), gx, gy)
    val = part[var].to_numpy()
    ny, nx = elev.shape
    field, mask = station_to_raster(rows, cols, val, ny, nx)
    base = idw_raster(rows, cols, val, ny, nx, k=ccfg.idw_k) if ccfg.residual_idw else None
    # zero the query pixels so the net cannot copy them
    field_q = field.copy()
    mask_q = mask.copy()
    field_q[rows, cols] = 0.0
    mask_q[rows, cols] = 0.0
    # keep donor pixels: rebuild from everyone except we only have `part`
    # caller must pass donors+queries separately
    return field_q, mask_q, base, rows, cols


def eval_parts(model, donors, queries, var, gx, gy, elev, clc, ccfg):
    rows_d, cols_d = grid_index(donors["x"].to_numpy(), donors["y"].to_numpy(), gx, gy)
    val_d = donors[var].to_numpy()
    ny, nx = elev.shape
    field, mask = station_to_raster(rows_d, cols_d, val_d, ny, nx)
    base = idw_raster(rows_d, cols_d, val_d, ny, nx, k=ccfg.idw_k) if ccfg.residual_idw else None
    inp = np.where(mask > 0, field - base, 0.0).astype(np.float32) if base is not None else field
    ts = donors["time"].iloc[0] if "time" in donors.columns else queries["time"].iloc[0]
    tch = cyclic_time(ts, False) if ccfg.cyclic_time else None
    hat = model.predict_raster(
        inp, mask, elev, clc, time_ch=tch,
        two_step=ccfg.two_step and two_step_var(var),
        base=base,
    )
    rows_q, cols_q = grid_index(queries["x"].to_numpy(), queries["y"].to_numpy(), gx, gy)
    return hat[rows_q, cols_q]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default=None)
    p.add_argument("--nested", action="store_true")
    p.add_argument("--folds", type=int, default=None)
    args = p.parse_args()
    cfg = load_cnn_config()
    if xr is None:
        raise RuntimeError("xarray required")
    time_res = cfg["time_resolution"]
    _, _, grid_method = data_sources(cfg)
    res = int(cfg.get("resolutions_to_process", [1000])[0])
    grid = xr.open_dataset(get_master_grid_path(grid_method, res))
    gx = grid["x"].values.astype(np.float64)
    gy = grid["y"].values.astype(np.float64)
    elev = grid["elev"].values.astype(np.float64) if "elev" in grid else np.zeros((gy.size, gx.size))
    clc = np.zeros_like(elev)
    from cnn_data import clc_group
    for name in ("clc_code", "clc", "landcover"):
        if name in grid:
            clc = clc_group(np.nan_to_num(grid[name].values, nan=0).astype(np.int32)).astype(np.float64)
            break
    variables = [args.variable] if args.variable else cfg["cnn"].get("variables_to_process", ["temp_mean"])
    n_folds = args.folds or 5

    for var in variables:
        panel = load_panel(cfg, var)
        ccfg = cfg_to_cnn(cfg, load_tuned(var, time_res))
        rows = []
        if args.nested:
            names = sorted(panel["station_name"].unique())
            folds = station_folds(names, n_folds, ccfg.seed)
            for hold in folds:
                hold_set = set(hold)
                don = panel[~panel["station_name"].isin(hold_set)]
                q = panel[panel["station_name"].isin(hold_set)]
                model, _ = train_one(don, var, ccfg, elev, clc, gx, gy)
                for ts, qq in q[q["split"].isin(["dev", "test"])].groupby("time"):
                    dd = don[don["time"] == ts]
                    if dd.empty:
                        continue
                    hat = eval_parts(model, dd, qq, var, gx, gy, elev, clc, ccfg)
                    part = qq[["station_name", "time", "split", var]].copy()
                    part["predicted"] = hat
                    part = part.rename(columns={var: "observed"})
                    rows.append(part)
        else:
            blob = torch.load(get_cnn_model_path(var, time_res), map_location="cpu")
            model = CNNInterpolator(ccfg)
            model.load_state_dict(blob)
            for ts, sl in panel.groupby("time"):
                # leave-one-station in the raster: use all others as donors
                names = sl["station_name"].tolist()
                if len(names) < ccfg.min_stations + 1:
                    continue
                for i, name in enumerate(names):
                    qq = sl.iloc[[i]]
                    dd = sl.drop(sl.index[i])
                    hat = eval_parts(model, dd, qq, var, gx, gy, elev, clc, ccfg)
                    part = qq[["station_name", "time", "split", var]].copy()
                    part["predicted"] = hat
                    part = part.rename(columns={var: "observed"})
                    rows.append(part)
        pred = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        out = get_nested_llocv_path("CNN", var, time_res, domain="full")
        pred.to_parquet(out, index=False)
        print(f"{var} -> {out}")
        if not pred.empty:
            print_split_metrics(pred)


if __name__ == "__main__":
    main()
