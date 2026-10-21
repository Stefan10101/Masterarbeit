#!/usr/bin/env python3
"""
Global IDW hyperparameter tune.

One (p, Fz, k) per variable on the designated subset:
  held-out stations excluded, test quarters removed (prepare_for_tuning).
Optional --months / --max-times thin the timestamps for a dry run.

Writes IDW/Output/global_params/{time_resolution}/{var}_params.yaml
"""

from __future__ import annotations

import argparse
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

from idw_data import (
    build_field,
    compute_metrics,
    load_config,
    load_panel,
    save_global_params,
)
from shared.splits.evaluation_sets import load_holdout, prepare_for_tuning
from shared.time_res import add_time_res_arg, apply_time_res


def _score_one(field: pd.DataFrame, p: float, Fz: float, k: int,
               tree: KDTree | None = None) -> dict:
    """Vectorized leave-one-out score for one (p, Fz, k)."""
    coords = field[["x", "y"]].to_numpy()
    elev = field["elev"].to_numpy()
    vals = field["value"].to_numpy()
    n = len(field)
    k_use = min(int(k), n - 1)
    if k_use < 1:
        return {"rmse": np.nan, "mae": np.nan, "nse": np.nan, "kge": np.nan}
    if tree is None:
        tree = KDTree(coords)
    # k+1 so we can drop self (distance 0)
    dists, idxs = tree.query(coords, k=k_use + 1)
    # drop the self column (smallest distance)
    dists = np.maximum(dists[:, 1:], 1e-6)
    idxs = idxs[:, 1:]
    d_z = np.abs(elev[idxs] - elev[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        w_xy = dists ** (-p)
        w_z = np.where(d_z > 0, d_z ** (-p), 0.0)
    w = ((1.0 - Fz) * w_xy ** 2 + Fz * w_z ** 2) / 0.5
    nb_vals = vals[idxs]
    valid = np.isfinite(nb_vals)
    w = np.where(valid, w, 0.0)
    wsum = w.sum(axis=1)
    pred = np.divide((w * np.where(valid, nb_vals, 0.0)).sum(axis=1), wsum,
                      out=np.full(n, np.nan), where=wsum > 0)
    return compute_metrics(vals, pred)


def _eval_combo(args):
    field, p, Fz, k, tree = args
    return {"p": p, "Fz": Fz, "k": k, **_score_one(field, p, Fz, k, tree)}


def optimize_field(field, param_grid, primary="rmse", n_jobs=1,
                   show_progress=False) -> tuple[dict, dict]:
    p_list = param_grid.get("p", [2.0])
    Fz_list = param_grid.get("Fz", [0.0, 0.3])
    k_list = param_grid.get("k_neighbors", [12])
    tree = KDTree(field[["x", "y"]].to_numpy())
    combos = [(field, p, Fz, k, tree) for p in p_list for Fz in Fz_list for k in k_list]
    # Threads only. Process workers crash on Windows (0xc0000139).
    if n_jobs == 1:
        iterator = tqdm(combos, leave=False, desc="    combos") if show_progress else combos
        results = [_eval_combo(c) for c in iterator]
    else:
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_eval_combo)(c) for c in combos
        )
    df = pd.DataFrame(results)
    if df[primary].isna().all():
        return {"p": 2.0, "Fz": 0.3, "k": 12}, {"rmse": np.nan, "mae": np.nan, "nse": np.nan, "kge": np.nan}
    best_idx = df[primary].idxmax() if primary in ("kge", "nse") else df[primary].idxmin()
    row = df.loc[best_idx]
    return (
        {"p": float(row["p"]), "Fz": float(row["Fz"]), "k": int(row["k"])},
        {"rmse": float(row["rmse"]), "mae": float(row["mae"]),
         "nse": float(row["nse"]), "kge": float(row["kge"])},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variables", nargs="+", default=None)
    p.add_argument("--months", default=None, help="all | seasonal4 | YYYY-MM,YYYY-MM")
    p.add_argument("--max-times", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--early-stop", action="store_true", help="stop once the winner is stable")
    p.add_argument("--patience", type=int, default=40, help="timestamps the winner must hold")
    p.add_argument("--jobs", type=int, default=1,
                   help="1 = sequential (safe on Windows). >1 uses threads.")
    add_time_res_arg(p)
    args = p.parse_args()

    cfg = load_config()
    time_res = apply_time_res(cfg, args)
    param_grid = cfg["param_grid"]["power"]
    primary = cfg.get("primary_metric", "rmse")
    min_stations = int(cfg.get("min_stations_per_field", 10))
    n_jobs = int(args.jobs)
    variables = args.variables or ["temp_mean", "precip_sum"]

    held_out, _pool = load_holdout()
    held_set = set(held_out)

    print("=" * 72)
    print(f"IDW global tune | {time_res} | vars={variables}")
    print("=" * 72)

    for var in variables:
        panel = load_panel(cfg, var)
        panel = prepare_for_tuning(panel)
        if args.months and args.months != "all":
            from idw_data import subset_times
            panel = subset_times(panel, args.months)

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
            best_params, best_scores = optimize_field(
                field, param_grid, primary=primary, n_jobs=n_jobs,
                show_progress=(n_jobs == 1),
            )
            rows.append({**best_params, **best_scores})
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
        print(f"  p={result['p']:.2f}  Fz={result['Fz']:.2f}  k={result['k']}  "
              f"rmse={result['rmse']:.3f}  (n={result['n_times_used']})")
        print(f"  wrote {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
