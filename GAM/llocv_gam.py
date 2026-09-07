#!/usr/bin/env python3
"""Station-fold LLOCV for per-timestamp GAM."""

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

from paths import get_gam_tuned_params_path, get_nested_llocv_path
from gam_core import GAMConfig, GAMInterpolator
from gam_data import load_gam_config, load_panel, precip_trace, print_split_metrics


def parse_split_arg(text: str) -> set[str]:
    text = text.strip().lower()
    if text == "all":
        return {"all"}
    names = {s.strip() for s in text.split(",") if s.strip()}
    return names


def station_folds(names, n_folds, seed):
    rng = np.random.default_rng(seed)
    names = list(names)
    rng.shuffle(names)
    folds = [[] for _ in range(n_folds)]
    for i, n in enumerate(names):
        folds[i % n_folds].append(n)
    return folds


def cfg_to_gam(cfg, overrides=None) -> GAMConfig:
    g = dict(cfg.get("gam", {}))
    o = overrides or {}
    return GAMConfig(
        formula=str(o.get("formula", g.get("formula", "te_xy_s_elev"))),
        n_splines=int(o.get("n_splines", g.get("n_splines", 10))),
        spline_order=int(g.get("spline_order", 3)),
        use_clc=bool(g.get("use_clc", True)),
        use_terrain=bool(g.get("use_terrain", True)),
        min_clc_count=int(g.get("min_clc_count", 15)),
        two_step=bool(g.get("two_step", True)),
        tau_wet=float(o.get("tau_wet", g.get("tau_wet", 0.5))),
        rscript=str(g.get("rscript", "Rscript")),
        fit_r=str(g.get("fit_r", "fit_gam.R")),
        min_stations=int(cfg.get("min_stations_per_field", 10)),
        predict_tile=int(g.get("predict_tile", 50000)),
        seed=int(g.get("seed", 22)),
        trace=float(o.get("trace", g.get("trace", 0.1))),
        r_timeout=float(g.get("r_timeout", 600.0)),
    )


def load_tuned(var, time_res):
    path = get_gam_tuned_params_path(var, time_res)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_llocv(panel, var, gcfg: GAMConfig, n_folds, fit_splits, score_splits, progress: str | None = None):
    import time

    names = sorted(panel["station_name"].unique())
    folds = station_folds(names, n_folds, gcfg.seed)
    rows = []
    tag = f"[{progress}] " if progress else ""
    t0 = time.time()
    n_done = 0
    n_fail = 0
    backend = None
    for fi, hold in enumerate(folds, start=1):
        hold_set = set(hold)
        don_all = panel[~panel["station_name"].isin(hold_set)]
        q_all = panel[panel["station_name"].isin(hold_set)]
        if score_splits != {"all"}:
            q_all = q_all[q_all["split"].isin(score_splits)]
        q_all = q_all.copy()
        don_all = don_all.copy()
        q_all["_t"] = pd.to_datetime(q_all["time"]).astype("datetime64[ns]")
        don_all["_t"] = pd.to_datetime(don_all["time"]).astype("datetime64[ns]")
        times = list(q_all.groupby("_t"))
        print(
            f"{tag}fold {fi}/{n_folds} hold={len(hold_set)} score_times={len(times)}",
            flush=True,
        )
        for ti, (ts, q) in enumerate(times, start=1):
            don = don_all[don_all["_t"] == ts]
            if don["station_name"].nunique() < gcfg.min_stations:
                continue
            model = GAMInterpolator(gcfg)
            try:
                hat = model.predict_timestamp(
                    don["x"].to_numpy(), don["y"].to_numpy(), don["elev"].to_numpy(),
                    don[var].to_numpy(),
                    q["x"].to_numpy(), q["y"].to_numpy(), q["elev"].to_numpy(),
                    clc=don["clc_group"].to_numpy(),
                    clc_q=q["clc_group"].to_numpy(),
                    slope=don["slope"].to_numpy() if "slope" in don.columns else None,
                    slope_q=q["slope"].to_numpy() if "slope" in q.columns else None,
                    sinasp=don["sinasp"].to_numpy() if "sinasp" in don.columns else None,
                    sinasp_q=q["sinasp"].to_numpy() if "sinasp" in q.columns else None,
                    cosasp=don["cosasp"].to_numpy() if "cosasp" in don.columns else None,
                    cosasp_q=q["cosasp"].to_numpy() if "cosasp" in q.columns else None,
                    var=var,
                    trace=gcfg.trace,
                )
            except Exception as exc:
                n_fail += 1
                print(f"{tag}FAIL {ts} {type(exc).__name__}: {exc}", flush=True)
                continue
            n_done += 1
            be = getattr(model, "_backend", None)
            if backend is None and be:
                backend = be
                print(f"{tag}backend={backend} first_ok {ts}", flush=True)
            if ti == 1 or ti % 6 == 0 or ti == len(times):
                dt = time.time() - t0
                print(
                    f"{tag}fold {fi}/{n_folds} time {ti}/{len(times)} "
                    f"ok={n_done} fail={n_fail} {dt:.0f}s backend={backend}",
                    flush=True,
                )
            part = q[["station_name", "time", "split", var]].copy()
            part["predicted"] = hat
            part = part.rename(columns={var: "observed"})
            rows.append(part)
    dt = time.time() - t0
    print(f"{tag}llocv done ok={n_done} fail={n_fail} rows={len(rows)} {dt:.0f}s", flush=True)
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
    cfg = load_gam_config()
    time_res = cfg["time_resolution"]
    variables = [args.variable] if args.variable else cfg["gam"].get("variables_to_process", ["temp_mean"])
    n_folds = args.folds or int(cfg["gam"].get("n_folds", 5))
    for var in variables:
        tuned = load_tuned(var, time_res)
        tuned["trace"] = precip_trace(cfg, time_res, var)
        gcfg = cfg_to_gam(cfg, tuned)
        panel = load_panel(cfg, var)
        pred = run_llocv(panel, var, gcfg, n_folds, parse_split_arg(args.fit), parse_split_arg(args.score))
        out = get_nested_llocv_path("GAM", var, time_res, domain="full")
        pred.to_parquet(out, index=False)
        print(f"{var} -> {out}")
        print_split_metrics(pred)


if __name__ == "__main__":
    main()
