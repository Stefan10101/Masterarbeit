#!/usr/bin/env python3
"""
Global IDW hyperparameter tune.

One (p, Fz, k) per variable.
Protocol (per-timestamp): train + tune quarters, score on the fixed
tune-targets (donors = pool minus those targets during scoring).
Proof runs: --max-times caps the timestamps (default 400).

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
    hyper_source,
    build_field,
    compute_metrics,
    load_config,
    load_panel,
    save_global_params,
)
from shared.splits.evaluation_sets import (
    apply_evaluation_protocol,
    load_holdout,
    load_tune_targets,
    protocol_masks,
    load_early_stop_times,
    early_stop_kind,
)
from shared.time_res import add_hours_arg, add_time_res_arg, apply_time_res, variables_from_cfg, cap_timestamps


TRACE = {
    "precip_sum": {"half_hourly": 0.05, "daily": 0.1, "weekly": 0.7, "monthly": 1.0},
    "snow_mean": {"half_hourly": 0.5, "daily": 0.5, "weekly": 1.0, "monthly": 1.0},
}
MAX_TIMES_DEFAULT = 400
PREFIX_MARGIN = 0.20  # drop combos > 20 % worse than best after the prefix


def _match_times(panel_times: pd.Series, wanted: list[str], time_res: str) -> list:
    """Return panel timestamps that match the yaml early-stop list.

    yaml stores native aggregated values (year_month, year_week, date, timestamp).
    panel time is UTC datetime.
    """
    if not wanted:
        return []
    pt = pd.to_datetime(panel_times, utc=True)
    want = set(str(w) for w in wanted)
    if time_res == "monthly":
        key = pt.dt.strftime("%Y-%m")
    elif time_res == "weekly":
        # match the year_week strings stored in the yaml (e.g. 2020-W03)
        key = pt.dt.strftime("%Y-W%W")
    elif time_res == "daily":
        key = pt.dt.strftime("%Y-%m-%d")
    else:
        key = pt.dt.strftime("%Y-%m-%dT%H:%M:%S")
    mask = key.isin(want)
    return sorted(pt[mask].unique())



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


def _score_targets(donors, targets, p, Fz, k, var, trace):
    """Score one (p, Fz, k) on fixed targets using donors (no LOO)."""
    if len(donors) < 5 or len(targets) < 1:
        return {"rmse": np.nan, "mae": np.nan, "nse": np.nan, "kge": np.nan, "n": 0}
    from idw_core import interpolate_idw
    tree = KDTree(donors[["x", "y"]].to_numpy())
    pred = interpolate_idw(
        station_values=donors["value"].to_numpy(),
        station_coords=donors[["x", "y"]].to_numpy(),
        station_elev=donors["elev"].to_numpy(),
        target_coords=targets[["x", "y"]].to_numpy(),
        target_elev=targets["elev"].to_numpy(),
        tree=tree,
        p=float(p), Fz=float(Fz), k=min(int(k), len(donors)),
        var=var, trace=trace,
    )
    return compute_metrics(targets["value"].to_numpy(), pred)


def optimize_on_targets(donors, targets, param_grid, primary="rmse", n_jobs=1,
                        var: str = "", trace: float = 1.0, allowed=None):
    p_list = param_grid.get("p", [2.0])
    Fz_list = param_grid.get("Fz", [0.0, 0.5])
    k_list = param_grid.get("k_neighbors", [16])
    combos = [(p, Fz, k) for p in p_list for Fz in Fz_list for k in k_list]
    if allowed is not None:
        combos = [c for c in combos if c in allowed]
    if not combos:
        return {"p": 2.0, "Fz": 0.5, "k": 16}, {"rmse": np.nan, "mae": np.nan, "nse": np.nan, "kge": np.nan}, pd.DataFrame()
    def one(c):
        p, Fz, k = c
        return {"p": p, "Fz": Fz, "k": k, **_score_targets(donors, targets, p, Fz, k, var, trace)}
    if n_jobs == 1:
        results = [one(c) for c in combos]
    else:
        results = Parallel(n_jobs=n_jobs, backend="threading")(delayed(one)(c) for c in combos)
    df = pd.DataFrame(results)
    if df.empty or df[primary].isna().all():
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
    add_hours_arg(p)
    args = p.parse_args()

    cfg = load_config()
    time_res = apply_time_res(cfg, args)
    param_grid = cfg["param_grid"]["power"]
    primary = "rmse"
    min_stations = int(cfg.get("min_stations_per_field", 10))
    n_jobs = int(args.jobs) if args.jobs is not None else max(1, (os.cpu_count() or 1) // 2)
    variables = variables_from_cfg(cfg, args.variables)

    print("=" * 72)
    print(f"IDW global tune | {time_res} | vars={variables}")
    print(f"  protocol: train+tune quarters, score fixed tune-targets")
    print("=" * 72)

    parents = cfg.get("variable_parents") or {}
    for var in variables:
        src = hyper_source(cfg, var)
        if src != var:
            # child: copy parent's tuned file and skip search
            parent_path = (SCRIPT_DIR / "Output" / "global_params" / time_res / f"{src}_params.yaml")
            child_path = (SCRIPT_DIR / "Output" / "global_params" / time_res / f"{var}_params.yaml")
            if parent_path.exists():
                child_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(parent_path, child_path)
                print(f"[{var}] inherited hypers from {src} → {child_path.name}")
            else:
                print(f"[{var}] parent {src} has no tuned file yet — skipping")
            continue
        held_out, _pool = load_holdout(variable=var)
        tune_targets, _ = load_tune_targets(variable=var)
        held_set = set(held_out)
        tt_set = set(tune_targets)
        panel = load_panel(cfg, var, hours=getattr(args, 'hours', None))
        tagged, _, _, _ = apply_evaluation_protocol(panel, variable=var)
        # Per-timestamp: train + tune quarters, drop held-out
        panel = tagged[
            (~tagged["is_heldout"])
            & (tagged["split"].isin(["train", "tune"]))
        ].copy()
        if args.months and args.months != "all":
            from idw_data import subset_times
            panel = subset_times(panel, args.months)
        trace = TRACE.get(var, {}).get(time_res, 1.0)
        two = two_step_var(var)
        print(f"  {var}  two_step={two}  trace={trace}  "
              f"times={panel['time'].nunique()}  "
              f"stations={panel['station_name'].nunique()}  "
              f"tune-targets present={panel['is_tune_target'].sum()}")

        out_exists = (SCRIPT_DIR / "Output" / "global_params" / time_res / f"{var}_params.yaml")
        if out_exists.exists() and not args.force:
            print(f"[{var}] exists → {out_exists.name}  (use --force)")
            continue

        times = sorted(panel["time"].unique())
        # Early-stop protocol: score only the eval set (max 52). Prefix (20) filters combos.
        kind = early_stop_kind("IDW")
        eval_raw = load_early_stop_times(time_res, kind, which="eval")
        prefix_raw = load_early_stop_times(time_res, kind, which="prefix")
        eval_times = _match_times(pd.Series(times), eval_raw, time_res)
        prefix_times = _match_times(pd.Series(times), prefix_raw, time_res)
        prefix_set = {pd.Timestamp(t).isoformat() for t in prefix_times}
        if eval_times:
            times = eval_times
            print(f"  early-stop eval={len(times)}  prefix={len(prefix_set)}")
        elif args.max_times and len(times) > args.max_times:
            capped = cap_timestamps(panel, args.max_times, time_res,
                                    seed=int(cfg.get("idw", {}).get("seed", 31)))
            times = sorted(capped["time"].unique())
            print(f"  no early-stop yaml; capped to {len(times)}")
        print(f"\n----- {var}  {len(times)} timestamps  "
              f"{panel['station_name'].nunique()} stations -----")

        n_combos = (len(param_grid.get("p", []))
                    * len(param_grid.get("Fz", []))
                    * len(param_grid.get("k_neighbors", [])))
        print(f"  grid={n_combos} combos/timestamp  jobs={n_jobs}")
        rows = []
        search_parts = []
        combo_scores: dict[tuple, list] = {}
        alive = None  # None = all combos; set = survivors after prefix filter
        prefix_done = False
        # Mode-stability patience stays active independently of the combo filter.
        early = args.early_stop or time_res in ("daily", "half_hourly")
        patience = args.patience
        min_times = max(30, patience)
        stable = None
        stable_for = 0
        if early:
            print(f"  mode-stability patience={patience} after {min_times} timestamps", flush=True)
        for i, ts in enumerate(tqdm(times, desc=f"  {var}"), 1):
            field = build_field(panel, ts, min_stations, held_set)
            if field is None:
                continue
            donors = field[~field["station_name"].isin(tt_set)]
            targets = field[field["station_name"].isin(tt_set)]
            if len(targets) < 1 or len(donors) < 5:
                continue
            best_params, best_scores, combo_df = optimize_on_targets(
                donors, targets, param_grid, primary=primary, n_jobs=n_jobs,
                var=var, trace=trace, allowed=alive,
            )
            rows.append({**best_params, **best_scores, "time": str(ts)})
            if not combo_df.empty:
                combo_df = combo_df.copy()
                combo_df["time"] = str(ts)
                search_parts.append(combo_df)
                for _, r in combo_df.iterrows():
                    if np.isfinite(r["rmse"]):
                        combo_scores.setdefault((r["p"], r["Fz"], r["k"]), []).append(float(r["rmse"]))
            # Prefix filter: once all prefix timestamps are scored, drop bad combos.
            scored_prefix = sum(
                1 for r in rows
                if pd.to_datetime(r["time"], utc=True).isoformat() in prefix_set
            )
            if not prefix_done and prefix_set and scored_prefix >= len(prefix_set) and combo_scores:
                means = {k: float(np.mean(v)) for k, v in combo_scores.items() if v}
                if means:
                    best_mean = min(means.values())
                    alive = {k for k, m in means.items() if m <= best_mean * (1.0 + PREFIX_MARGIN)}
                    dropped = set(means) - alive
                    prefix_done = True
                    print(f"    prefix filter: kept {len(alive)} / {len(means)}  "
                          f"dropped {len(dropped)}  best_mean={best_mean:.3f}", flush=True)
            if early and len(rows) >= min_times:
                keys = [(r["p"], r["Fz"], r["k"]) for r in rows]
                mode = max(set(keys), key=keys.count)
                if mode == stable:
                    stable_for += 1
                else:
                    stable, stable_for = mode, 1
                if stable_for >= patience:
                    print(f"    mode-stability stop at {i}/{len(times)}  "
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
        search = pd.concat(search_parts, ignore_index=True) if search_parts else pd.DataFrame()
        if not search.empty:
            summary = (search.groupby(["p", "Fz", "k"], as_index=False)
                       .agg(rmse=("rmse", "median"), mae=("mae", "median"),
                            nse=("nse", "median"), kge=("kge", "median"),
                            n_times=("rmse", "size")))
            # mark combos dropped by the prefix filter
            if alive is not None:
                summary["early_stopped"] = [
                    (r.p, r.Fz, r.k) not in alive for r in summary.itertuples()
                ]
                summary["stop_reason"] = summary["early_stopped"].map(
                    lambda x: "prefix_margin" if x else "full"
                )
            else:
                summary["early_stopped"] = False
                summary["stop_reason"] = "full"
            summary["n_units_scored"] = summary["n_times"]
        else:
            summary = pd.DataFrame()
        search_path = path.with_name(f"{var}_search.csv")
        summary.to_csv(search_path, index=False)
        print(f"  p={result['p']:.2f}  Fz={result['Fz']:.2f}  k={result['k']}  "
              f"rmse={result['rmse']:.3f}  (n={result['n_times_used']})")
        print(f"  wrote {path}")
        print(f"  wrote {search_path}  ({len(summary)} combos)")

    print("\nDone.")


if __name__ == "__main__":
    main()
