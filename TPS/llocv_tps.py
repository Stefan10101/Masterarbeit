#!/usr/bin/env python3
"""Station-fold LLOCV for TPS / E-OBS protocol."""

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

from paths import get_nested_llocv_path, get_tps_tuned_params_path
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res
from tps_core import TPSConfig, TPSInterpolator, attach_watershed, two_step_var
from tps_data import (
    attach_pack_terrain,
    attach_temperature,
    data_sources,
    load_panel,
    load_tps_config,
    pack_from_master,
    precip_trace,
    print_split_metrics,
    subset_times,
    uses_two_step,
)


def parse_split_arg(text: str) -> set[str]:
    text = text.strip().lower()
    if text == "all":
        return {"all"}
    names = {s.strip() for s in text.split(",") if s.strip()}
    bad = names - {"train", "dev", "test"}
    if bad:
        raise ValueError(f"unknown splits {bad}")
    return names


def station_folds(names, n_folds, seed):
    rng = np.random.default_rng(seed)
    names = list(names)
    rng.shuffle(names)
    folds = [[] for _ in range(n_folds)]
    for i, n in enumerate(names):
        folds[i % n_folds].append(n)
    return folds


def cfg_to_tps(cfg, overrides=None) -> TPSConfig:
    t = dict(cfg.get("tps", {}))
    o = overrides or {}
    return TPSConfig(
        lam=float(o.get("lam", t.get("lam", 1.0))),
        alpha_z=float(o.get("alpha_z", t.get("alpha_z", 100.0))),
        alpha_z_mode=str(o.get("alpha_z_mode", t.get("alpha_z_mode", "global"))),
        n_regions=int(o.get("n_regions", t.get("n_regions", 6))),
        kernel=str(o.get("kernel", t.get("kernel", "3d"))),
        nugget=float(t.get("nugget", 1e-6)),
        protocol=str(o.get("protocol", t.get("protocol", "eobs"))),
        k=int(o.get("k", t.get("k", 32))),
        family=str(t.get("family", "exponential")),
        k_indicator=int(t.get("k_indicator", 16)),
        tau_wet=float(o.get("tau_wet", t.get("tau_wet", 0.5))),
        two_step=bool(t.get("two_step", True)),
        trace=float(o.get("trace", t.get("trace", 0.1))),
        min_stations=int(cfg.get("min_stations_per_field", 10)),
        predict_tile=int(t.get("predict_tile", 20000)),
        seed=int(t.get("seed", 22)),
        rh_t_mode=str(o.get("rh_t_mode", t.get("rh_t_mode", "none"))),
        wind_watershed=bool(t.get("wind_watershed", True)),
        snow_terrain=bool(t.get("snow_terrain", True)),
    )


def load_tuned(var, time_res):
    path = get_tps_tuned_params_path(var, time_res)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# pack_from_master imported from tps_data (adds slope/northness)


def month_key(times) -> np.ndarray:
    t = pd.DatetimeIndex(times)
    return (t.year * 100 + t.month).to_numpy()


def monthly_from_panel(panel: pd.DataFrame, var: str, yearmonth: int) -> pd.DataFrame:
    mk = month_key(panel["time"])
    mon = panel[mk == yearmonth]
    if mon.empty:
        return mon
    agg = "sum" if two_step_var(var) else "mean"
    return (
        mon.groupby("station_name")
        .agg(x=("x", "first"), y=("y", "first"), elev=("elev", "first"), val=(var, agg))
        .reset_index()
    )


def _extras(df: pd.DataFrame, var: str, tcfg: TPSConfig):
    cols = []
    v = str(var).lower()
    if v.startswith("snow") and tcfg.snow_terrain:
        for c, fill in (("slope", 0.0), ("northness", 1.0)):
            if c in df.columns:
                x = df[c].to_numpy(dtype=float)
                cols.append(np.where(np.isfinite(x), x, fill))
            else:
                cols.append(np.full(len(df), fill))
    if v.startswith("rh") and tcfg.rh_t_mode in ("predicted", "observed") and "temp_mean" in df.columns:
        x = df["temp_mean"].to_numpy(dtype=float)
        mu = float(np.nanmean(x)) if np.isfinite(x).any() else 0.0
        cols.append(np.where(np.isfinite(x), x, mu))
    if not cols:
        return None
    return np.column_stack(cols)


def predict_rows(model: TPSInterpolator, donors: pd.DataFrame, query: pd.DataFrame, var: str,
                 monthly_panel: pd.DataFrame | None = None, time_res: str = "monthly"):
    if donors.empty or query.empty:
        return np.full(len(query), np.nan)
    if (
        str(var).lower().startswith("rh")
        and model.cfg.rh_t_mode == "predicted"
        and "temp_mean" in donors.columns
    ):
        t_hat = model.predict_timestamp(
            donors[["x", "y"]].to_numpy(),
            donors["temp_mean"].to_numpy(),
            donors["elev"].to_numpy(),
            query["x"].to_numpy(),
            query["y"].to_numpy(),
            query["elev"].to_numpy(),
            var="temp_mean",
        )
        query = query.copy()
        query["temp_mean"] = t_hat
    extra = _extras(donors, var, model.cfg)
    extra_q = _extras(query, var, model.cfg)
    region = donors["region"].to_numpy() if "region" in donors.columns else None
    region_q = query["region"].to_numpy() if "region" in query.columns else None
    use_eobs = (
        model.cfg.protocol == "eobs"
        and time_res in ("daily", "weekly", "half_hourly")
        and monthly_panel is not None
        and len(monthly_panel)
    )
    if not use_eobs:
        return model.predict_timestamp(
            donors[["x", "y"]].to_numpy(),
            donors[var].to_numpy(),
            donors["elev"].to_numpy(),
            query["x"].to_numpy(),
            query["y"].to_numpy(),
            query["elev"].to_numpy(),
            var=var,
            extra=extra, extra_q=extra_q, region=region, region_q=region_q,
        )
    qk = int(month_key(query["time"])[0])
    monthly = monthly_from_panel(monthly_panel, var, qk)
    if len(monthly) < model.cfg.min_stations:
        return model.predict_timestamp(
            donors[["x", "y"]].to_numpy(),
            donors[var].to_numpy(),
            donors["elev"].to_numpy(),
            query["x"].to_numpy(),
            query["y"].to_numpy(),
            query["elev"].to_numpy(),
            var=var,
            extra=extra, extra_q=extra_q, region=region, region_q=region_q,
        )
    clc_s = donors["clc_group"].to_numpy() if "clc_group" in donors.columns else None
    clc_q = query["clc_group"].to_numpy() if "clc_group" in query.columns else None
    return model.predict_eobs_timestamp(
        donors[["x", "y"]].to_numpy(),
        donors[var].to_numpy(),
        donors["elev"].to_numpy(),
        monthly[["x", "y"]].to_numpy(),
        monthly["val"].to_numpy(),
        monthly["elev"].to_numpy(),
        query["x"].to_numpy(),
        query["y"].to_numpy(),
        query["elev"].to_numpy(),
        var=var,
        clc_s=clc_s,
        clc_q=clc_q,
        time=query["time"].iloc[0],
    )


def run_llocv(panel, var, tcfg: TPSConfig, n_folds: int, fit_splits, score_splits, pack=None,
              time_res: str = "monthly"):
    names = sorted(panel["station_name"].unique())
    folds = station_folds(names, n_folds, tcfg.seed)
    model = TPSInterpolator(tcfg)
    attach_watershed(model, pack, panel["x"].to_numpy(), panel["y"].to_numpy(), panel["elev"].to_numpy())
    if tcfg.alpha_z_mode == "watershed" and not model.cfg.region_alpha:
        print("  warning: watershed pack missing, falling back to global αz")
    rows = []
    for hold in folds:
        hold_set = set(hold)
        don_all = panel[~panel["station_name"].isin(hold_set)]
        q_all = panel[panel["station_name"].isin(hold_set)]
        # per-timestamp method: donors are other stations at the same time,
        # including DEV/TEST times. fit_splits is ignored for donors.
        if score_splits != {"all"}:
            q_all = q_all[q_all["split"].isin(score_splits)]
        for ts, q in q_all.groupby("time"):
            don = don_all[don_all["time"] == ts]
            if don["station_name"].nunique() < tcfg.min_stations:
                continue
            hat = predict_rows(model, don, q, var, monthly_panel=don_all, time_res=time_res)
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
    p.add_argument("--months", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--rh-t-mode", default=None, choices=["none", "predicted", "observed"])
    add_hours_arg(p)
    add_time_res_arg(p)
    args = p.parse_args()

    cfg = load_tps_config()
    time_res = apply_time_res(cfg, args)
    variables = [args.variable] if args.variable else cfg["tps"].get("variables_to_process", ["temp_mean"])
    n_folds = args.folds or int(cfg["tps"].get("n_folds", 5))
    months = args.months
    if args.quick:
        n_folds = args.folds or 2
        months = months or "seasonal4"
    fit_splits = parse_split_arg(args.fit)
    score_splits = parse_split_arg(args.score)

    for var in variables:
        tuned = load_tuned(var, time_res)
        tuned["trace"] = precip_trace(cfg, time_res, var)
        if args.rh_t_mode:
            tuned["rh_t_mode"] = args.rh_t_mode
        tcfg = cfg_to_tps(cfg, tuned)
        panel = load_panel(cfg, var)
        if str(var).lower().startswith("rh") and tcfg.rh_t_mode in ("predicted", "observed"):
            panel = attach_temperature(panel, cfg)
        pack = pack_from_master(cfg, tcfg.n_regions)
        panel = attach_pack_terrain(panel, pack)
        if months:
            panel = subset_times(panel, months)
        panel, _hours = apply_hour_cut(panel, time_res, args.hours)
        pred = run_llocv(panel, var, tcfg, n_folds, fit_splits, score_splits, pack, time_res)
        out = get_nested_llocv_path("TPS", var, time_res, domain="full")
        pred.to_parquet(out, index=False)
        print(f"{var} -> {out}")
        print_split_metrics(pred)


if __name__ == "__main__":
    main()
