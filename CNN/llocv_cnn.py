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
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_cnn_model_path, get_master_grid_path, get_nested_llocv_path
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res, variables_from_cfg
from cnn_core import (
    CNNInterpolator, cyclic_time, grid_terrain, idw_fz_points,
    sample_at_points, station_to_raster, two_step_var, clip_var,
)
from shared.splits.evaluation_sets import apply_evaluation_protocol
from shared.metrics import compute_metrics as eval_metrics, print_eval
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


def bind_static(model, elev, clc, slope, sinasp, cosasp, core_extras, subdaily, var):
    """Terrain/CLC/extras are identical every timestamp. Pack them once."""
    extras = list(core_extras) if core_extras else []
    if str(var).lower().startswith("rh"):
        extras.append(np.zeros(elev.shape, dtype=np.float32))
    n_time = 0
    if model.cfg.cyclic_time:
        n_time = 4 if subdaily else 2
    model.bind_static(
        elev, clc, slope=slope, sinasp=sinasp, cosasp=cosasp,
        extras=extras or None, n_time=n_time,
    )


def prep_frame(model, donors, queries, var, gx, gy, subdaily):
    """CPU side: residual at donor stations (point IDW), packed channels.

    The full-grid IDW raster is not a network input. Building it per month
    was the GPU idle gap. Point IDW at the donors matches the training residual.
    """
    rows_d, cols_d = grid_index(donors["x"].to_numpy(), donors["y"].to_numpy(), gx, gy)
    val_d = donors[var].to_numpy()
    ny, nx = model._static.shape[-2:]
    field, mask = station_to_raster(rows_d, cols_d, val_d, ny, nx)
    residual = bool(model.cfg.residual_idw) and not bool(getattr(model, "log_amount", False))
    log_res = bool(getattr(model, "log_residual", False))
    qbase = None
    p, fz, k = model.cfg.idw_p, model.cfg.idw_fz, model.cfg.idw_k
    sx, sy, sz = donors["x"].to_numpy(), donors["y"].to_numpy(), donors["elev"].to_numpy()
    if log_res:
        dlog = np.log1p(np.maximum(val_d, 0.0))
        inp, _ = station_to_raster(rows_d, cols_d, dlog, ny, nx)
        qbase = idw_fz_points(
            sx, sy, sz, dlog,
            queries["x"].to_numpy(), queries["y"].to_numpy(), queries["elev"].to_numpy(),
            p=p, Fz=fz, k=k, exclude_self=False,
        )
    elif residual:
        p, fz, k = model.cfg.idw_p, model.cfg.idw_fz, model.cfg.idw_k
        sx, sy, sz = donors["x"].to_numpy(), donors["y"].to_numpy(), donors["elev"].to_numpy()
        dbase = idw_fz_points(sx, sy, sz, val_d, sx, sy, sz, p=p, Fz=fz, k=k, exclude_self=True)
        qbase = idw_fz_points(
            sx, sy, sz, val_d,
            queries["x"].to_numpy(), queries["y"].to_numpy(), queries["elev"].to_numpy(),
            p=p, Fz=fz, k=k, exclude_self=False,
        )
        inp, _ = station_to_raster(rows_d, cols_d, val_d - dbase, ny, nx)
    elif bool(getattr(model, "log_amount", False)):
        inp = np.log1p(np.maximum(field, 0.0)).astype(np.float32)
    else:
        inp = field
    ts = donors["time"].iloc[0] if "time" in donors.columns else queries["time"].iloc[0]
    tch = cyclic_time(ts, subdaily) if model.cfg.cyclic_time else None
    rh_field = None
    if str(var).lower().startswith("rh") and "temp_mean" in donors.columns:
        rh_field = np.zeros((ny, nx), dtype=np.float32)
        tr, tc = grid_index(donors["x"].to_numpy(), donors["y"].to_numpy(), gx, gy)
        rh_field[tr, tc] = donors["temp_mean"].to_numpy(dtype=np.float32) / 20.0
    packed = model.pack_dynamic(inp, mask, tch, rh_field)
    return {
        "packed": packed,
        "qbase": qbase,
        "qx": queries["x"].to_numpy(),
        "qy": queries["y"].to_numpy(),
        "donor_wet": float(np.mean(val_d > 0.1)) if len(val_d) else 0.0,
        "two_step": bool(model.cfg.two_step) and two_step_var(var),
        "residual": residual,
        "log_residual": log_res,
    }


def score_frame(model, pre, gx, gy, var):
    hat = model.predict_packed(
        pre["packed"], two_step=pre["two_step"], base=None,
        var=var, donor_wet=pre["donor_wet"], apply_clip=not pre["residual"],
    )
    pred = sample_at_points(hat, pre["qx"], pre["qy"], gx, gy)
    if pre.get("log_residual") and pre["qbase"] is not None:
        # Amount head is a log residual. Add the point log-IDW, invert, gate.
        # predict_packed with base=None returned expm1(residual); undo that.
        cap = float(getattr(model, "_amt_clip", 1e3))
        # Re-read the amount head so we add the baseline in log space.
        x = torch.from_numpy(np.ascontiguousarray(pre["packed"])).unsqueeze(0).to(model.device)
        with torch.no_grad():
            raw = model.net(x).cpu().numpy()[0]
        amt = raw[1] * model.y_std + model.y_mean
        qy = sample_at_points(amt, pre["qx"], pre["qy"], gx, gy)
        p = 1.0 / (1.0 + np.exp(-sample_at_points(raw[0], pre["qx"], pre["qy"], gx, gy)))
        val = np.expm1(np.minimum(pre["qbase"] + qy, np.log1p(cap)))
        pred = clip_var(var, np.where(p >= model.cfg.tau_wet, np.maximum(val, 0.0), 0.0))
    elif pre["qbase"] is not None:
        # Residual field. Score is the point baseline plus the proximity-weighted residual.
        w = sample_at_points(pre["packed"][model._weight_idx], pre["qx"], pre["qy"], gx, gy)
        pred = clip_var(var, pre["qbase"] + w * pred)
    return pred


def eval_parts(model, donors, queries, var, gx, gy, subdaily):
    return score_frame(model, prep_frame(model, donors, queries, var, gx, gy, subdaily), gx, gy, var)


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
    p.add_argument("--variable", default=None, help="single variable (prefer --variables)")
    p.add_argument("--variables", nargs="+", default=None, help="list of variables")
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
    if args.variables:
        variables = args.variables
    elif args.variable:
        variables = [args.variable]
    else:
        variables = variables_from_cfg(cfg)
    n_folds = args.folds or int(cfg["cnn"].get("n_folds", 5))
    if args.quick:
        n_folds = args.folds or 2
        cfg.setdefault("cnn", {})
        cfg["cnn"]["epochs"] = min(int(cfg["cnn"].get("epochs", 40)), 8)
        cfg["cnn"]["patience"] = min(int(cfg["cnn"].get("patience", 8)), 3)
    score = {s.strip() for s in args.score.split(",") if s.strip()}

    from cnn_core import load_extra_covariates
    core_extras, _ = load_extra_covariates(cfg, int(cfg.get("resolutions_to_process", [1000])[0]))
    print(f"core extras: {len(core_extras)} channels", flush=True)

    for var in variables:
        panel = load_panel(cfg, var)
        panel, held_out, pool, qsplits = apply_evaluation_protocol(panel, variable=var)
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
                    cfg=cfg,
                )
                q_score = q if score == {"all"} else q[q["split"].isin(score)]
                bind_static(model, elev, clc, slope, sinasp, cosasp, core_extras, subdaily, var)
                for ts, qq in q_score.groupby("time"):
                    dd = don[don["time"] == ts]
                    if dd.empty or dd["station_name"].nunique() < ccfg.min_stations:
                        continue
                    hat = eval_parts(model, dd, qq, var, gx, gy, subdaily)
                    part = qq[["station_name", "time", "split", var]].copy()
                    part["predicted"] = hat
                    part = part.rename(columns={var: "observed"})
                    rows.append(part)
        else:
            # Protocol: score the already-trained net on held-out x test.
            # Prep of month i+1 runs on a thread while the GPU forwards month i.
            score_df = panel[(panel["is_heldout"]) & (panel["split"].isin(score if score != {"all"} else {"test"}))]
            print(f"{var} score rows={len(score_df)}", flush=True)
            model = load_frozen(ccfg, var, time_res)
            bind_static(model, elev, clc, slope, sinasp, cosasp, core_extras, subdaily, var)
            n_ts = score_df["time"].nunique()
            items = []
            for i, (ts, qq) in enumerate(score_df.groupby("time"), start=1):
                dd = panel[(~panel["is_heldout"]) & (panel["time"] == ts)]
                if qq.empty or dd["station_name"].nunique() < ccfg.min_stations:
                    print(f"    [{i}/{n_ts}] {ts} skipped", flush=True)
                    continue
                items.append((i, ts, qq, dd))

            def _prep(item):
                i, ts, qq, dd = item
                t0 = time.perf_counter()
                pre = prep_frame(model, dd, qq, var, gx, gy, subdaily)
                return i, ts, qq, dd, pre, time.perf_counter() - t0

            def _consume(i, ts, qq, dd, pre, dt_prep):
                t0 = time.perf_counter()
                hat = score_frame(model, pre, gx, gy, var)
                dt_gpu = time.perf_counter() - t0
                part = qq[["station_name", "time", "split", var]].copy()
                part["predicted"] = hat
                part = part.rename(columns={var: "observed"})
                rows.append(part)
                print(
                    f"    [{i}/{n_ts}] {ts} donors={dd['station_name'].nunique()} "
                    f"targets={len(qq)} prep={dt_prep:.1f}s gpu={dt_gpu:.1f}s",
                    flush=True,
                )

            if items:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    it = iter(items)
                    fut = ex.submit(_prep, next(it))
                    for nxt in it:
                        i, ts, qq, dd, pre, dt_prep = fut.result()
                        fut = ex.submit(_prep, nxt)
                        _consume(i, ts, qq, dd, pre, dt_prep)
                    i, ts, qq, dd, pre, dt_prep = fut.result()
                    _consume(i, ts, qq, dd, pre, dt_prep)
        pred = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        out = get_nested_llocv_path("CNN", var, time_res, domain="full")
        pred.to_parquet(out, index=False)
        print(f"{var} -> {out}")
        if not pred.empty:
            print_split_metrics(pred)
            if "split" in pred.columns and (pred["split"] == "test").any():
                part = pred[pred["split"] == "test"]
                print_eval(eval_metrics(part["observed"], part["predicted"], var, time_res), prefix="  test ")


if __name__ == "__main__":
    main()
