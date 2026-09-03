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

from paths import get_master_grid_path, get_nested_llocv_path, get_tps_tuned_params_path
from tps_core import TPSConfig, TPSInterpolator, attach_watershed
from tps_data import data_sources, load_panel, load_tps_config, print_split_metrics, uses_two_step


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
        min_stations=int(cfg.get("min_stations_per_field", 10)),
        predict_tile=int(t.get("predict_tile", 20000)),
        seed=int(t.get("seed", 22)),
    )


def load_tuned(var, time_res):
    path = get_tps_tuned_params_path(var, time_res)
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
    return build_pack(
        g["elev"].values.astype(np.float64),
        g["x"].values,
        g["y"].values,
        WatershedConfig(n_regions=int(n_regions)),
    )


def month_key(times) -> np.ndarray:
    t = pd.DatetimeIndex(times)
    return (t.year * 100 + t.month).to_numpy()


def predict_rows(model: TPSInterpolator, donors: pd.DataFrame, query: pd.DataFrame, var: str):
    if donors.empty or query.empty:
        return np.full(len(query), np.nan)
    precip = uses_two_step(var)
    if model.cfg.protocol == "tps" or donors["time"].nunique() <= 1:
        return model.predict_timestamp(
            donors[["x", "y"]].to_numpy(),
            donors[var].to_numpy(),
            donors["elev"].to_numpy(),
            query["x"].to_numpy(),
            query["y"].to_numpy(),
            query["elev"].to_numpy(),
        )
    mk = month_key(donors["time"])
    qk = month_key(query["time"])[0]
    mon = donors[mk == qk]
    if len(mon) < model.cfg.min_stations:
        mon = donors
    agg = "sum" if precip else "mean"
    monthly = mon.groupby("station_name").agg(x=("x", "first"), y=("y", "first"),
                                              elev=("elev", "first"), val=(var, agg)).reset_index()
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
        is_precip=precip,
    )


def run_llocv(panel, var, tcfg: TPSConfig, n_folds: int, fit_splits, score_splits, pack=None):
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
            hat = predict_rows(model, don, q, var)
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
    args = p.parse_args()

    cfg = load_tps_config()
    time_res = cfg["time_resolution"]
    variables = [args.variable] if args.variable else cfg["tps"].get("variables_to_process", ["temp_mean"])
    n_folds = args.folds or int(cfg["tps"].get("n_folds", 5))
    fit_splits = parse_split_arg(args.fit)
    score_splits = parse_split_arg(args.score)

    for var in variables:
        tuned = load_tuned(var, time_res)
        tcfg = cfg_to_tps(cfg, tuned)
        panel = load_panel(cfg, var)
        pred = run_llocv(panel, var, tcfg, n_folds, fit_splits, score_splits)
        out = get_nested_llocv_path("TPS", var, time_res, domain="full")
        pred.to_parquet(out, index=False)
        print(f"{var} -> {out}")
        print_split_metrics(pred)


if __name__ == "__main__":
    main()
