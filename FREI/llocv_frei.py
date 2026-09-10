#!/usr/bin/env python3
"""Station-fold LLOCV for Frei."""

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

from paths import get_frei_tuned_params_path, get_master_grid_path, get_nested_llocv_path
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res
from frei_core import FreiConfig, FreiInterpolator, two_step_var
from frei_data import (
    attach_temperature,
    data_sources,
    load_frei_config,
    load_panel,
    precip_trace,
    print_split_metrics,
    subset_times,
)


def parse_split_arg(text: str) -> set[str]:
    text = text.strip().lower()
    if text == "all":
        return {"all"}
    return {s.strip() for s in text.split(",") if s.strip()}


def station_folds(names, n_folds, seed):
    rng = np.random.default_rng(seed)
    names = list(names)
    rng.shuffle(names)
    folds = [[] for _ in range(n_folds)]
    for i, n in enumerate(names):
        folds[i % n_folds].append(n)
    return folds


def cfg_to_frei(cfg, overrides=None) -> FreiConfig:
    f = dict(cfg.get("frei", {}))
    o = overrides or {}
    return FreiConfig(
        n_regions=int(o.get("n_regions", f.get("n_regions", 6))),
        shrink=float(o.get("shrink", f.get("shrink", 0.3))),
        blend_km=float(f.get("blend_km", 15.0)),
        summit_tpi=float(f.get("summit_tpi", 80.0)),
        coldpool_tpi=float(f.get("coldpool_tpi", -60.0)),
        accum_pct=float(f.get("accum_pct", 96.0)),
        k=int(o.get("k", f.get("k", 16))),
        power=float(o.get("power", f.get("power", 2.0))),
        across_w=float(o.get("across_w", f.get("across_w", 4.0))),
        select_metric=bool(o.get("select_metric", f.get("select_metric", True))),
        lam_z=float(o.get("lam_z", f.get("lam_z", 150.0))),
        two_step=bool(f.get("two_step", True)),
        tau_wet=float(o.get("tau_wet", f.get("tau_wet", 0.5))),
        trace=float(o.get("trace", f.get("trace", 0.1))),
        min_stations=int(cfg.get("min_stations_per_field", 10)),
        predict_tile=int(f.get("predict_tile", 20000)),
        seed=int(f.get("seed", 22)),
        across_w_grid=tuple(f.get("search", {}).get("across_w", [2.0, 4.0, 8.0])),
        rh_t_mode=str(o.get("rh_t_mode", f.get("rh_t_mode", "none"))),
        snow_terrain=bool(f.get("snow_terrain", True)),
        wind_elev_bg=bool(f.get("wind_elev_bg", True)),
    )


def load_tuned(var, time_res):
    path = get_frei_tuned_params_path(var, time_res)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def pack_from_master(cfg, n_regions: int = 6):
    try:
        import xarray as xr
    except ImportError:
        return None
    from shared.watersheds import WatershedConfig, build_pack
    _, _, grid_method = data_sources(cfg)
    res = int(cfg.get("resolutions_to_process", [1000])[0])
    path = get_master_grid_path(grid_method, res)
    if not path.exists():
        return None
    g = xr.open_dataset(path)
    if "elev" not in g:
        return None
    f = cfg.get("frei", {})
    wcfg = WatershedConfig(
        n_regions=int(n_regions),
        accum_pct=float(f.get("accum_pct", 96.0)),
        summit_tpi=float(f.get("summit_tpi", 80.0)),
        coldpool_tpi=float(f.get("coldpool_tpi", -60.0)),
    )
    return build_pack(g["elev"].values.astype(np.float64), g["x"].values, g["y"].values, wcfg)


def run_llocv(panel, var, fcfg: FreiConfig, n_folds, fit_splits, score_splits, pack=None):
    names = sorted(panel["station_name"].unique())
    folds = station_folds(names, n_folds, fcfg.seed)
    model = FreiInterpolator(fcfg, pack=pack)
    use_profile = not two_step_var(var)
    need_t = var.lower().startswith("rh") and fcfg.rh_t_mode in ("predicted", "observed")
    rows = []
    for hold in folds:
        hold_set = set(hold)
        don_all = panel[~panel["station_name"].isin(hold_set)]
        q_all = panel[panel["station_name"].isin(hold_set)]
        if score_splits != {"all"}:
            q_all = q_all[q_all["split"].isin(score_splits)]
        q_all = q_all.copy()
        don_all = don_all.copy()
        q_all["_t"] = pd.to_datetime(q_all["time"]).astype("datetime64[ns]")
        don_all["_t"] = pd.to_datetime(don_all["time"]).astype("datetime64[ns]")
        for ts, q in q_all.groupby("_t"):
            don = don_all[don_all["_t"] == ts]
            if don["station_name"].nunique() < fcfg.min_stations:
                continue
            t_obs = don["temp_mean"].to_numpy() if need_t and "temp_mean" in don.columns else None
            t_obs_q = q["temp_mean"].to_numpy() if need_t and "temp_mean" in q.columns else None
            try:
                hat, _ = model.predict_timestamp(
                    don["x"].to_numpy(), don["y"].to_numpy(), don["elev"].to_numpy(),
                    don[var].to_numpy(),
                    q["x"].to_numpy(), q["y"].to_numpy(), q["elev"].to_numpy(),
                    use_profile=use_profile,
                    var=var,
                    t_obs=t_obs,
                    t_obs_q=t_obs_q,
                )
            except Exception as exc:
                print(f"  frei predict failed {ts}: {type(exc).__name__}: {exc}", flush=True)
                continue
            part = q[["station_name", "time", "split", var]].copy()
            part["predicted"] = hat
            part = part.rename(columns={var: "observed"})
            rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["station_name", "time", "split", "observed", "predicted"])
    return pd.concat(rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default=None)
    p.add_argument("--fit", default="train")
    p.add_argument("--score", default="dev,test")
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--months", default=None, help="all | seasonal4 | YYYY-MM,YYYY-MM")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--rh-t-mode", default=None, choices=["none", "predicted", "observed"])
    add_hours_arg(p)
    add_time_res_arg(p)
    args = p.parse_args()
    cfg = load_frei_config()
    time_res = apply_time_res(cfg, args)
    variables = [args.variable] if args.variable else cfg["frei"].get("variables_to_process", ["temp_mean"])
    n_folds = args.folds or int(cfg["frei"].get("n_folds", 5))
    months = args.months
    if args.quick:
        n_folds = args.folds or 2
        months = months or "seasonal4"
    for var in variables:
        tuned = load_tuned(var, time_res)
        tuned["trace"] = precip_trace(cfg, time_res, var)
        if args.rh_t_mode:
            tuned["rh_t_mode"] = args.rh_t_mode
        fcfg = cfg_to_frei(cfg, tuned)
        if args.quick:
            fcfg.select_metric = False
        pack = pack_from_master(cfg, fcfg.n_regions)
        panel = load_panel(cfg, var)
        if var.lower().startswith("rh") and fcfg.rh_t_mode in ("predicted", "observed"):
            panel = attach_temperature(panel, cfg)
        if months:
            panel = subset_times(panel, months)
        panel, hours = apply_hour_cut(panel, time_res, args.hours)
        pred = run_llocv(panel, var, fcfg, n_folds, parse_split_arg(args.fit), parse_split_arg(args.score), pack)
        out = get_nested_llocv_path("Frei", var, time_res, domain="full")
        pred.to_parquet(out, index=False)
        print(f"{var} folds={n_folds} months={months or 'all'} hours={hours} rh_t={fcfg.rh_t_mode} -> {out}")
        print_split_metrics(pred)


if __name__ == "__main__":
    main()
