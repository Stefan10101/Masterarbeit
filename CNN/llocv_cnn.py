#!/usr/bin/env python3
"""CNN LLOCV.

Default: frozen pooled model, 5 station folds, score held-out stations
on every timestamp (same folds as GAM/Kriging/RGI).

--nested retrains one U-Net per station fold on TRAIN times of the other
stations and scores DEV+TEST of the held-out stations.

--loo is the slow full leave-one-station-per-timestamp path.
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
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res
from cnn_core import CNNInterpolator, cyclic_time, grid_terrain, idw_fz_points, idw_fz_raster, idw_raster, sample_at_points, station_to_raster, two_step_var
from shared.splits.evaluation_sets import apply_evaluation_protocol
from cnn_data import clc_group, data_sources, load_cnn_config, load_panel, precip_trace, print_split_metrics, subset_times, attach_temperature, attach_station_tpi
from train_cnn import cfg_to_cnn, grid_index, is_subdaily, load_tuned, train_one

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


def eval_parts(model, donors, queries, var, gx, gy, elev, clc, ccfg, slope, sinasp, cosasp, subdaily):
    rows_d, cols_d = grid_index(donors["x"].to_numpy(), donors["y"].to_numpy(), gx, gy)
    val_d = donors[var].to_numpy()
    ny, nx = elev.shape
    field, mask = station_to_raster(rows_d, cols_d, val_d, ny, nx)
    residual = bool(model.cfg.residual_idw)
    log_amt = bool(getattr(model, "log_amount", False))
    if residual:
        base = idw_fz_raster(
            donors["x"].to_numpy(), donors["y"].to_numpy(), donors["elev"].to_numpy(), val_d,
            gx, gy, elev, p=model.cfg.idw_p, Fz=model.cfg.idw_fz, k=model.cfg.idw_k, exclude_self=False,
        )
        qbase = idw_fz_points(
            donors["x"].to_numpy(), donors["y"].to_numpy(), donors["elev"].to_numpy(), val_d,
            queries["x"].to_numpy(), queries["y"].to_numpy(), queries["elev"].to_numpy(),
            p=model.cfg.idw_p, Fz=model.cfg.idw_fz, k=model.cfg.idw_k, exclude_self=False,
        )
    else:
        base = None
        qbase = None
    if log_amt:
        inp = np.log1p(np.maximum(field, 0.0)).astype(np.float32)
    elif base is not None:
        inp = np.where(mask > 0, field - base, 0.0).astype(np.float32)
    else:
        inp = field.astype(np.float32)
    ts = donors["time"].iloc[0] if "time" in donors.columns else queries["time"].iloc[0]
    tch = cyclic_time(ts, subdaily) if model.cfg.cyclic_time else None
    extras = None
    if str(var).lower().startswith("wind"):
        from paths import get_grid_terrain_path
        tpath = get_grid_terrain_path(1000)
        if tpath.exists():
            import xarray as xr
            tg = xr.open_dataset(tpath)
            if "tpi" in tg:
                extras = [tg["tpi"].values.astype(np.float32)]
    elif str(var).lower().startswith("rh") and "temp_mean" in donors.columns:
        tfield = np.zeros((ny, nx), dtype=np.float32)
        tr, tc = grid_index(donors["x"].to_numpy(), donors["y"].to_numpy(), gx, gy)
        tfield[tr, tc] = donors["temp_mean"].to_numpy(dtype=np.float32)
        extras = [tfield / 20.0]
    donor_wet = float(np.mean(val_d > 0.1)) if len(val_d) else 0.0
    hat = model.predict_raster(
        inp, mask, elev, clc,
        slope=slope, sinasp=sinasp, cosasp=cosasp,
        time_ch=tch,
        two_step=model.cfg.two_step and two_step_var(var),
        base=base,
        extras=extras,
        var=var,
        donor_wet=donor_wet,
    )
    pred = sample_at_points(hat, queries["x"].to_numpy(), queries["y"].to_numpy(), gx, gy)
    if qbase is not None:
        # predict_raster already added the cell baseline; swap in the point baseline
        cell = sample_at_points(base, queries["x"].to_numpy(), queries["y"].to_numpy(), gx, gy)
        pred = pred - cell + qbase
    return pred


def load_frozen(ccfg, var, time_res):
    path = get_cnn_model_path(var, time_res)
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    model = CNNInterpolator(
        ccfg,
        in_ch=int(blob.get("in_ch", 9)),
        out_ch=int(blob.get("out_ch", 1)),
    )
    model.load_state_dict(blob)
    model.net.to(model.device)
    print(
        f"loaded {path.name} in={model.in_ch} out={model.out_ch} "
        f"levels={model.cfg.n_levels} y_mean={model.y_mean:.3f} y_std={model.y_std:.3f} "
        f"residual_idw={model.cfg.residual_idw} device={model.device}",
        flush=True,
    )
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default=None)
    p.add_argument("--nested", action="store_true")
    p.add_argument("--loo", action="store_true")
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--score", default="dev,test")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--months", default=None)
    add_hours_arg(p)
    add_time_res_arg(p)
    args = p.parse_args()
    cfg = load_cnn_config()
    time_res = apply_time_res(cfg, args)
    if xr is None:
        raise RuntimeError("xarray required")
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
    variables = [args.variable] if args.variable else cfg["cnn"].get("variables_to_process", ["temp_mean"])
    n_folds = args.folds or int(cfg["cnn"].get("n_folds", 5))
    if args.quick:
        n_folds = args.folds or 2
        cfg.setdefault("cnn", {})
        cfg["cnn"]["epochs"] = min(int(cfg["cnn"].get("epochs", 40)), 8)
        cfg["cnn"]["patience"] = min(int(cfg["cnn"].get("patience", 8)), 3)
    score = {s.strip() for s in args.score.split(",") if s.strip()}

    for var in variables:
        panel = load_panel(cfg, var)
        panel, held_out, pool, qsplits = apply_evaluation_protocol(panel)
        if str(var).lower().startswith("rh"):
            panel = attach_temperature(panel, cfg)
        if str(var).lower().startswith("wind"):
            panel = attach_station_tpi(panel)
        # Restrict scoring to held-out stations on test quarters
        panel = panel[(panel["is_heldout"] & (panel["split"] == "test")) | (~panel["is_heldout"])]
        if args.quick or args.months:
            from cnn_data import subset_times
            panel = subset_times(panel, args.months or "seasonal4")
        panel, _hours = apply_hour_cut(panel, time_res, args.hours)
        ccfg = cfg_to_cnn(cfg, load_tuned(var, time_res))
        subdaily = is_subdaily(panel, time_res)
        trace = precip_trace(cfg, time_res, var)
        rows = []
        if args.nested:
            names = sorted(panel["station_name"].unique())
            folds = station_folds(names, n_folds, ccfg.seed)
            for hold in folds:
                hold_set = set(hold)
                don = panel[~panel["station_name"].isin(hold_set)]
                q = panel[panel["station_name"].isin(hold_set)]
                model, _ = train_one(
                    don, var, ccfg, elev, clc, gx, gy,
                    slope, sinasp, cosasp, time_res=time_res, trace=trace,
                )
                q_score = q if score == {"all"} else q[q["split"].isin(score)]
                for ts, qq in q_score.groupby("time"):
                    dd = don[don["time"] == ts]
                    if dd.empty or dd["station_name"].nunique() < ccfg.min_stations:
                        continue
                    hat = eval_parts(model, dd, qq, var, gx, gy, elev, clc, ccfg, slope, sinasp, cosasp, subdaily)
                    part = qq[["station_name", "time", "split", var]].copy()
                    part["predicted"] = hat
                    part = part.rename(columns={var: "observed"})
                    rows.append(part)
        else:
            # Protocol: score the already-trained net on held-out x test.
            score_df = panel[(panel["is_heldout"]) & (panel["split"].isin(score if score != {"all"} else {"test"}))]
            print(f"{var} score rows={len(score_df)}", flush=True)
            model = load_frozen(ccfg, var, time_res)
            n_ts = score_df["time"].nunique()
            for i, (ts, qq) in enumerate(score_df.groupby("time"), start=1):
                dd = panel[(~panel["is_heldout"]) & (panel["time"] == ts)]
                if qq.empty or dd["station_name"].nunique() < ccfg.min_stations:
                    print(f"    [{i}/{n_ts}] {ts} skipped", flush=True)
                    continue
                print(f"    [{i}/{n_ts}] {ts} donors={dd['station_name'].nunique()} targets={len(qq)}", flush=True)
                hat = eval_parts(model, dd, qq, var, gx, gy, elev, clc, ccfg, slope, sinasp, cosasp, subdaily)
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
