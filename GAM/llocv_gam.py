#!/usr/bin/env python3
"""Station-fold LLOCV for per-timestamp GAM."""

from __future__ import annotations

from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_gam_tuned_params_path, get_nested_llocv_path
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res
from gam_core import GAMConfig, GAMInterpolator
from shared.splits.evaluation_sets import apply_evaluation_protocol
from gam_data import (
    attach_pack_terrain,
    attach_temperature,
    compute_block,
    load_gam_config,
    load_panel,
    pack_from_master,
    precip_trace,
    print_split_metrics,
    subset_times,
)


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
        rh_t_mode=str(o.get("rh_t_mode", g.get("rh_t_mode", "none"))),
        wind_watershed=bool(g.get("wind_watershed", True)),
    )


def load_tuned(var, time_res):
    path = get_gam_tuned_params_path(var, time_res)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}



def _score_one(payload):
    """One timestamp. Own R process, so it is safe to run several at once."""
    i, n, ts, don, q, var, gcfg = payload
    model = GAMInterpolator(gcfg)
    hat = model.predict_timestamp(
        don["x"].to_numpy(), don["y"].to_numpy(), don["elev"].to_numpy(),
        don[var].to_numpy(),
        q["x"].to_numpy(), q["y"].to_numpy(), q["elev"].to_numpy(),
        var=var,
        clc=don["clc_group"].to_numpy() if "clc_group" in don.columns else None,
        clc_q=q["clc_group"].to_numpy() if "clc_group" in q.columns else None,
        slope=don["slope"].to_numpy() if "slope" in don.columns else None,
        slope_q=q["slope"].to_numpy() if "slope" in q.columns else None,
        sinasp=don["sinasp"].to_numpy() if "sinasp" in don.columns else None,
        sinasp_q=q["sinasp"].to_numpy() if "sinasp" in q.columns else None,
        cosasp=don["cosasp"].to_numpy() if "cosasp" in don.columns else None,
        cosasp_q=q["cosasp"].to_numpy() if "cosasp" in q.columns else None,
    )
    part = q[["station_name", "time", "split", var]].copy()
    part["observed"] = part[var]
    part["predicted"] = hat
    return i, str(ts), len(don), len(q), part.drop(columns=[var])


def run_llocv(panel, var, gcfg: GAMConfig, n_folds, fit_splits, score_splits, progress: str | None = None,
              pack=None, jobs: int = 1):
    """Protocol-aware. Final scores: held-out x test. Tuning: pool x requested splits."""
    panel, held_out, pool, qsplits = apply_evaluation_protocol(panel)
    score_ok = {"tune" if s == "dev" else s for s in score_splits}
    # Only the final test uses the held-out stations. Tune scores the pool.
    if score_splits == {"all"} or "test" in score_ok:
        score_df = panel[(panel["is_heldout"]) & (panel["split"].isin(score_ok))]
    else:
        score_df = panel[(~panel["is_heldout"]) & (panel["split"].isin(score_ok))]
    if score_df.empty:
        print(f"  {var} no score rows for {sorted(score_ok)}", flush=True)
        return pd.DataFrame(columns=["station_name", "time", "split", "observed", "predicted"])

    groups = list(score_df.groupby("time", sort=False))
    print(f"  {var} scoring {len(groups)} timestamps, {len(score_df)} rows, "
          f"form={gcfg.formula} ns={gcfg.n_splines} jobs={jobs}", flush=True)
    payloads = []
    for i, (ts, q) in enumerate(groups, start=1):
        don = panel[(~panel["is_heldout"]) & (panel["time"] == ts)]
        if don["station_name"].nunique() < gcfg.min_stations:
            print(f"    [{i}/{len(groups)}] {ts} skipped, too few donors", flush=True)
            continue
        payloads.append((i, len(groups), ts, don, q, var, gcfg))
    rows = []
    first_err = None
    if jobs <= 1 or len(payloads) <= 1:
        for payload in payloads:
            i = payload[0]
            try:
                i, ts, nd, nq, part = _score_one(payload)
                print(f"    [{i}/{len(groups)}] {ts} donors={nd} targets={nq}", flush=True)
                rows.append(part)
            except Exception as exc:
                if first_err is None:
                    first_err = exc
                print(f"    [{i}/{len(groups)}] failed: {type(exc).__name__}: {exc}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_score_one, p): p[0] for p in payloads}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    i, ts, nd, nq, part = fut.result()
                    print(f"    [{i}/{len(groups)}] {ts} donors={nd} targets={nq}", flush=True)
                    rows.append(part)
                except Exception as exc:
                    if first_err is None:
                        first_err = exc
                    print(f"    [{i}/{len(groups)}] failed: {type(exc).__name__}: {exc}", flush=True)
    if not rows:
        if first_err is not None:
            print(f"  {var} no rows, first error: {first_err}", flush=True)
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
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--rh-t-mode", default=None, choices=["none", "predicted", "observed"])
    add_hours_arg(p)
    add_time_res_arg(p)
    args = p.parse_args()
    cfg = load_gam_config()
    time_res = apply_time_res(cfg, args)
    variables = [args.variable] if args.variable else cfg["gam"].get("variables_to_process", ["temp_mean"])
    n_folds = args.folds or int(cfg["gam"].get("n_folds", 5))
    months = args.months
    if args.quick:
        n_folds = args.folds or 2
        months = months or "seasonal4"
    print(f"GAM LLOCV | {time_res} | vars={variables} | score will use held-out x requested split", flush=True)
    pack = pack_from_master(cfg, int(cfg.get("gam", {}).get("n_regions", 6)))
    for var in variables:
        tuned = load_tuned(var, time_res)
        tuned["trace"] = precip_trace(cfg, time_res, var)
        if args.rh_t_mode:
            tuned["rh_t_mode"] = args.rh_t_mode
        gcfg = cfg_to_gam(cfg, tuned)
        print(f"{var} loading panel", flush=True)
        panel = load_panel(cfg, var)
        print(f"{var} panel rows={len(panel)}", flush=True)
        if str(var).lower().startswith("rh") and gcfg.rh_t_mode in ("predicted", "observed"):
            panel = attach_temperature(panel, cfg)
        panel = attach_pack_terrain(panel, pack)
        if months:
            panel = subset_times(panel, months)
        panel, _hours = apply_hour_cut(panel, time_res, args.hours)
        pred = run_llocv(panel, var, gcfg, n_folds, parse_split_arg(args.fit), parse_split_arg(args.score), pack=pack, jobs=max(1, args.jobs))
        out = get_nested_llocv_path("GAM", var, time_res, domain="full")
        pred.to_parquet(out, index=False)
        print(f"{var} -> {out}")
        print_split_metrics(pred)


if __name__ == "__main__":
    main()
