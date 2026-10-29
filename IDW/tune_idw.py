#!/usr/bin/env python3
"""
Global IDW hyperparameter tune.

One (p, Fz, k) per variable.
Score: pool stations at tune quarters only (prepare_for_tuning drops held-out
and test quarters; this script then keeps split=='dev').
Proof runs: --max-times caps the tune timestamps (default 400).

Writes:
  Methods/IDW/Output/global_params/{time_res}/{var}_params.yaml
  Methods/IDW/Output/global_params/{time_res}/{var}_search.csv   (every combo)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import KDTree
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from idw_core import clip_var, two_step_var
from idw_data import (
    build_field,
    compute_metrics,
    load_config,
    load_panel,
    save_global_params,
)
from shared.splits.evaluation_sets import load_holdout, prepare_for_tuning
from shared.time_res import add_time_res_arg, apply_time_res


TRACE = {
    "precip_sum": {"half_hourly": 0.05, "daily": 0.1, "weekly": 0.7, "monthly": 1.0},
    "snow_mean": {"half_hourly": 0.5, "daily": 0.5, "weekly": 1.0, "monthly": 1.0},
}
MAX_TIMES_DEFAULT = 400


def _predict_loo(dists, idxs, elev, vals, p, Fz):
    dists = np.maximum(dists, 1e-6)
    d_z = np.abs(elev[idxs] - elev[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        w_xy = dists ** (-p)
        w_z = np.where(d_z > 0, d_z ** (-p), 0.0)
    w = (1.0 - Fz) * w_xy + Fz * w_z
    nb = vals[idxs]
    valid = np.isfinite(nb)
    w = np.where(valid, w, 0.0)
    wsum = w.sum(axis=1)
    return np.divide((w * np.where(valid, nb, 0.0)).sum(axis=1), wsum,
                      out=np.full(len(vals), np.nan), where=wsum > 0)


def _score_one(field: pd.DataFrame, p: float, Fz: float, k: int,
               tree: KDTree | None = None, var: str = "", trace: float = 1.0) -> dict:
    """Leave-one-out score for one (p, Fz, k). Two-step for precip/snow."""
    coords = field[["x", "y"]].to_numpy()
    elev = field["elev"].to_numpy()
    vals = field["value"].to_numpy()
    n = len(field)
    k_use = min(int(k), n - 1)
    if k_use < 1:
        return {"rmse": np.nan, "mae": np.nan, "nse": np.nan, "kge": np.nan, "n": 0}
    if tree is None:
        tree = KDTree(coords)
    dists, idxs = tree.query(coords, k=k_use + 1)
    dists, idxs = dists[:, 1:], idxs[:, 1:]
    if two_step_var(var):
        wet = (vals > trace).astype(float)
        p_wet = np.clip(_predict_loo(dists, idxs, elev, wet, p, Fz), 0.0, 1.0)
        # Amount only from wet stations
        amt_vals = np.where(wet > 0, vals, np.nan)
        amt = _predict_loo(dists, idxs, elev, amt_vals, p, Fz)
        pred = np.where((p_wet >= 0.5) & np.isfinite(amt), np.maximum(amt, 0.0), 0.0)
        pred = clip_var(var, pred)
    else:
        pred = clip_var(var, _predict_loo(dists, idxs, elev, vals, p, Fz))
    return compute_metrics(vals, pred)


def _eval_combo(args):
    field, p, Fz, k, tree, var, trace = args
    return {"p": p, "Fz": Fz, "k": k, **_score_one(field, p, Fz, k, tree, var, trace)}


def optimize_field(field, param_grid, primary="rmse", n_jobs=1,
                   show_progress=False, var: str = "", trace: float = 1.0):
    p_list = param_grid.get("p", [2.0])
    Fz_list = param_grid.get("Fz", [0.0, 0.5])
    k_list = param_grid.get("k_neighbors", [16])
    tree = KDTree(field[["x", "y"]].to_numpy())
    combos = [(field, p, Fz, k, tree, var, trace)
              for p in p_list for Fz in Fz_list for k in k_list]
    if n_jobs == 1:
        iterator = tqdm(combos, leave=False, desc="    combos") if show_progress else combos
        results = [_eval_combo(c) for c in iterator]
    else:
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_eval_combo)(c) for c in combos
        )
    df = pd.DataFrame(results)
    if df[primary].isna().all():
        return {"p": 2.0, "Fz": 0.5, "k": 16}, {"rmse": np.nan, "mae": np.nan, "nse": np.nan, "kge": np.nan}, df
    best_idx = df[primary].idxmax() if primary in ("kge", "nse") else df[primary].idxmin()
    row = df.loc[best_idx]
    return (
        {"p": float(row["p"]), "Fz": float(row["Fz"]), "k": int(row["k"])},
        {"rmse": float(row["rmse"]), "mae": float(row["mae"]),
         "nse": float(row["nse"]), "kge": float(row["kge"])},
        df,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variables", nargs="+", default=None)
    p.add_argument("--months", default=None, help="all | seasonal4 | YYYY-MM,YYYY-MM")
    p.add_argument("--max-times", type=int, default=MAX_TIMES_DEFAULT,
                   help="cap tune timestamps (sampled within the tune split)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--early-stop", action="store_true", help="stop once the winner is stable")
    p.add_argument("--patience", type=int, default=40, help="timestamps the winner must hold")
    p.add_argument("--jobs", type=int, default=None,
                   help="threads over (p, Fz, k) combos (default: half the logical cores)")
    add_time_res_arg(p)
    args = p.parse_args()

    cfg = load_config()
    time_res = apply_time_res(cfg, args)
    param_grid = cfg["param_grid"]["power"]
    primary = cfg.get("primary_metric", "rmse")
    min_stations = int(cfg.get("min_stations_per_field", 10))
    n_jobs = int(args.jobs) if args.jobs is not None else max(1, (os.cpu_count() or 1) // 2)
    variables = args.variables or [
        "temp_mean", "precip_sum", "wind_mean", "rh_mean", "snow_mean",
    ]

    held_out, _pool = load_holdout()
    held_set = set(held_out)

    print("=" * 72)
    print(f"IDW global tune | {time_res} | vars={variables}")
    print("=" * 72)

    for var in variables:
        panel = load_panel(cfg, var)
        panel = prepare_for_tuning(panel)
        # Tune quarters only. prepare_for_tuning remaps tune -> dev.
        panel = panel[panel["split"] == "dev"].copy()
        if args.months and args.months != "all":
            from idw_data import subset_times
            panel = subset_times(panel, args.months)
        trace = TRACE.get(var, {}).get(time_res, 1.0)
        two = two_step_var(var)
        print(f"  {var}  two_step={two}  trace={trace}  "
              f"tune times={panel['time'].nunique()}  "
              f"stations={panel['station_name'].nunique()}")

        out_exists = (SCRIPT_DIR / "Output" / "global_params" / time_res / f"{var}_params.yaml")
        if out_exists.exists() and not args.force:
            print(f"[{var}] exists → {out_exists.name}  (use --force)")
            continue

        times = sorted(panel["time"].unique())
        if args.max_times and len(times) > args.max_times:
            rng = np.random.default_rng(int(cfg.get("random_seed", 31)))
            times = sorted(rng.choice(times, size=args.max_times, replace=False))
        print(f"\n----- {var}  {len(times)} timestamps  "
              f"{panel['station_name'].nunique()} stations -----")

        n_combos = (len(param_grid.get("p", []))
                    * len(param_grid.get("Fz", []))
                    * len(param_grid.get("k_neighbors", [])))
        print(f"  grid={n_combos} combos/timestamp  jobs={n_jobs}")
        rows = []
        search_parts = []
        # Daily and finer: stop once the most common (p, Fz, k) has not changed.
        early = args.early_stop or time_res in ("daily", "half_hourly")
        patience = args.patience
        min_times = max(30, patience)
        stable = None
        stable_for = 0
        if early:
            print(f"  early stop on, patience={patience} after {min_times} timestamps", flush=True)
        for i, ts in enumerate(tqdm(times, desc=f"  {var}"), 1):
            field = build_field(panel, ts, min_stations, held_set)
            if field is None:
                continue
            best_params, best_scores, combo_df = optimize_field(
                field, param_grid, primary=primary, n_jobs=n_jobs,
                show_progress=(n_jobs == 1), var=var, trace=trace,
            )
            rows.append({**best_params, **best_scores, "time": str(ts)})
            combo_df = combo_df.copy()
            combo_df["time"] = str(ts)
            search_parts.append(combo_df)
            if early and len(rows) >= min_times:
                keys = [(r["p"], r["Fz"], r["k"]) for r in rows]
                mode = max(set(keys), key=keys.count)
                if mode == stable:
                    stable_for += 1
                else:
                    stable, stable_for = mode, 1
                if stable_for >= patience:
                    print(f"    early stop at {i}/{len(times)}  "
                          f"winner p={mode[0]} Fz={mode[1]} k={mode[2]} "
                          f"unchanged for {patience}", flush=True)
                    break
            if i == 1 or i % 5 == 0:
                print(f"    {i}/{len(times)}  best p={best_params['p']} "
                      f"Fz={best_params['Fz']} k={best_params['k']} "
                      f"rmse={best_scores['rmse']:.3f}", flush=True)

        if not rows:
            print(f"  [WARN] no usable fields for {var}")
            continue

        df = pd.DataFrame(rows)
        result = {
            "variable": var,
            "time_resolution": time_res,
            "p": float(df["p"].median()),
            "Fz": float(df["Fz"].median()),
            "k": int(round(df["k"].median())),
            "rmse": float(df["rmse"].median()),
            "mae": float(df["mae"].median()),
            "nse": float(df["nse"].median()),
            "kge": float(df["kge"].median()),
            "n_times_used": int(len(df)),
        }
        path = save_global_params(time_res, var, result)
        search = pd.concat(search_parts, ignore_index=True)
        summary = (search.groupby(["p", "Fz", "k"], as_index=False)
                   .agg(rmse=("rmse", "median"), mae=("mae", "median"),
                        nse=("nse", "median"), kge=("kge", "median"),
                        n_times=("rmse", "size")))
        search_path = path.with_name(f"{var}_search.csv")
        summary.to_csv(search_path, index=False)
        print(f"  p={result['p']:.2f}  Fz={result['Fz']:.2f}  k={result['k']}  "
              f"rmse={result['rmse']:.3f}  (n={result['n_times_used']})")
        print(f"  wrote {path}")
        print(f"  wrote {search_path}  ({len(summary)} combos)")

    print("\nDone.")


if __name__ == "__main__":
    main()
