#!/usr/bin/env python3
"""
Tune pooled RFSI hyperparameters.

Protocol (pooled): fit on full pool × train quarters;
score fixed tune-targets on tune quarters (donors exclude them only at prediction).
Writes RFSI/Output/.../{var}_pooled_params.yaml and the search CSV.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util as _ilu
import itertools
import sys
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import (
    get_aggregated_data_path,
    get_domain_stations_path,
    get_medoids_path,
    get_rfsi_tuned_params_path,
)
from rfsi_core import RFSI, attach_region, build_covariates, build_region_pack, cov_switches, extras_for_var, two_step_var
from rfsi_optimizer import compute_metrics

def _depth(d):
    """max_depth must be int or None for sklearn; pandas turns it into float."""
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return None
    return int(d)
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res, variables_from_cfg, cap_timestamps
from shared.splits.evaluation_sets import prepare_for_tuning

_splits_path = CODE_DIR / "shared" / "splits" / "splits.py"
_spec = _ilu.spec_from_file_location("thesis_time_splits", _splits_path)
_splits_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_splits_mod)
load_time_splits = _splits_mod.load_time_splits


def _in_dev(timestamps) -> pd.DatetimeIndex:
    spec = load_time_splits()
    start, end = spec["windows"]["dev"]
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).tz_convert("UTC").tz_localize(None)
    return ts[(ts >= start) & (ts <= end)]

COL_TO_CANONICAL = {
    "temp_mean": "temperature",
    "precip_sum": "precipitation",
    "wind_mean": "wind_speed",
    "rh_mean": "relative_humidity",
    "snow_mean": "snow_height",
}


def as_naive_utc(values, time_res=None):
    from shared.time_res import to_naive_utc
    return to_naive_utc(values, time_res)


def load_config():
    with open(SCRIPT_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_covariates(df, encoder=None, fit_encoder=False, use_elev=True, use_lc=True,
                      extras=None):
    X = build_covariates(
        elev=df["elev"].to_numpy() if use_elev and "elev" in df.columns else None,
        clc_code=df["clc_code"].to_numpy() if use_lc and "clc_code" in df.columns else None,
        use_elev=use_elev and "elev" in df.columns,
        use_lc=use_lc and "clc_code" in df.columns,
        extras=extras,
    )
    return X, None



SEASONAL4_MONTHS = (2, 5, 8, 11)


def _subset_times(panel, spec):
    if spec is None or spec in ("", "all"):
        return panel
    times = pd.to_datetime(panel["time"])
    if str(spec).strip().lower() == "seasonal4":
        return panel.loc[times.dt.month.isin(SEASONAL4_MONTHS)].copy()
    keys = {s.strip() for s in str(spec).split(",") if s.strip()}
    return panel.loc[times.dt.strftime("%Y-%m").isin(keys)].copy()


def load_panel(cfg, var):
    time_res = cfg["time_resolution"]
    df = pd.read_parquet(get_aggregated_data_path("RFSI", time_res))
    time_col = cfg["aggregation"][time_res]["time_col"]
    df["time"] = as_naive_utc(df[time_col], time_res)

    stations = pd.read_parquet(get_domain_stations_path("RFSI", "full"))
    from shared.terrain import attach_enabled_terrain, enabled_covariates
    stations = attach_enabled_terrain(stations, cfg)
    if "elev" not in stations.columns:
        if "elev_dem" in stations.columns:
            stations = stations.rename(columns={"elev_dem": "elev"})
        elif "hoehe" in stations.columns:
            stations = stations.rename(columns={"hoehe": "elev"})
    if "clc_code" not in stations.columns and "clc_level2" in stations.columns:
        stations["clc_code"] = stations["clc_level2"]
    keep = ["station_name", "x", "y"]
    for extra in ("elev", "clc_code", *enabled_covariates(cfg)):
        if extra in stations.columns and extra not in keep:
            keep.append(extra)
    stations = stations[keep].drop_duplicates("station_name")

    valid = df[["station_name", "time", var]].dropna()
    valid = valid.merge(stations, on="station_name", how="inner")
    valid = valid.dropna(subset=["x", "y", var])
    if "elev" in valid.columns:
        valid = valid.dropna(subset=["elev"])
    return valid


def load_medoid_times(cfg, var, cluster_method: str):
    time_res = cfg["time_resolution"]
    spec = load_time_splits()
    dev_start, dev_end = spec["windows"]["dev"]
    start, end = str(dev_start.date()), str(dev_end.date())
    candidates = [var, COL_TO_CANONICAL.get(var, var)]
    medoids = None
    used = None
    for name in candidates:
        for path in (
            get_medoids_path("RFSI", time_res, cluster_method, name, start, end),
            get_medoids_path("RFSI", time_res, cluster_method, name),
        ):
            if path.exists():
                medoids = pd.read_parquet(path)
                used = path
                break
        if medoids is not None:
            break
    if medoids is None:
        raise FileNotFoundError(
            "No medoid file. Run identify_regimes on DEV first:\n"
            "  python identify_regimes.py --method RFSI --resolution monthly "
            "--variables temperature precipitation "
            f"--start-date {start} --end-date {end}"
        )
    ts = pd.to_datetime(medoids["timestamp"], utc=True)
    dev_ts = _in_dev(ts)
    print(f"  medoids {used}: {pd.Index(ts).nunique()} total, {pd.Index(dev_ts).nunique()} in DEV")
    return pd.DatetimeIndex(dev_ts).unique()


def score_combo(panel, var, n_obs, rf_params, encoder, use_elev, use_lc,
                min_stations, n_splits, primary, rh_t_mode="none", two_step=False, tau_wet=0.5,
                cfg=None, champion_rmse=None):
    """Fit on full pool × train; score fixed tune-targets on tune quarters.
    Donors exclude tune-targets only at prediction time.
    If champion_rmse is set, probe with 40 % of trees and abort if still
    > 20 % worse and not improving."""
    if "is_tune_target" not in panel.columns or not panel["is_tune_target"].any():
        return None
    train = panel[(~panel["is_heldout"]) & (panel["split"] == "train")]
    hold = panel[(~panel["is_heldout"]) & panel["is_tune_target"]
                 & (panel["split"] == "tune")]
    if train["station_name"].nunique() < min_stations or hold.empty:
        return None

    def _score(model):
        obs_all, pred_all = [], []
        for t, hold_t in hold.groupby("time", sort=False):
            others = panel[(~panel["is_heldout"]) & (~panel["is_tune_target"])
                           & (panel["time"] == t)]
            if len(others) < max(min_stations, n_obs):
                continue
            X_hold, _ = prepare_covariates(
                hold_t, encoder=encoder, use_elev=use_elev, use_lc=use_lc,
                extras=extras_for_var(hold_t, var, rh_t_mode, cfg),
            )
            pred = model.predict_field(
                others[["x", "y"]].to_numpy(),
                others[var].to_numpy(),
                hold_t[["x", "y"]].to_numpy(),
                X_hold,
            )
            obs_all.append(hold_t[var].to_numpy())
            pred_all.append(np.asarray(pred, dtype=float))
        if not obs_all:
            return None
        met = compute_metrics(np.concatenate(obs_all), np.concatenate(pred_all))
        met["n"] = int(sum(len(a) for a in obs_all))
        return met

    n_full = int(rf_params.get("n_estimators", 100))
    n_probe = max(50, int(0.40 * n_full))
    # Probe with 40 % of trees when a champion exists.
    if champion_rmse is not None and n_probe < n_full:
        probe_params = {**rf_params, "n_estimators": n_probe}
        model = RFSI(n_obs=n_obs, rf_params=probe_params,
                     two_step=two_step, tau_wet=tau_wet, var_name=var)
        try:
            model.fit_pooled(
                times=train["time"].to_numpy(), coords=train[["x", "y"]].to_numpy(),
                z=train[var].to_numpy(),
                X_cov=prepare_covariates(train, encoder=encoder, use_elev=use_elev,
                                         use_lc=use_lc, extras=extras_for_var(train, var, rh_t_mode, cfg))[0],
                min_stations=min_stations,
            )
        except RuntimeError:
            return None
        probe = _score(model)
        if probe and probe["rmse"] > champion_rmse * 1.20:
            probe["early_stopped"] = True
            probe["stop_reason"] = "stalled_margin"
            probe["score_at_stop"] = probe["rmse"]
            probe["n_units_scored"] = n_probe
            return probe
    # Full forest
    model = RFSI(n_obs=n_obs, rf_params=rf_params,
                 two_step=two_step, tau_wet=tau_wet, var_name=var)
    try:
        model.fit_pooled(
            times=train["time"].to_numpy(), coords=train[["x", "y"]].to_numpy(),
            z=train[var].to_numpy(),
            X_cov=prepare_covariates(train, encoder=encoder, use_elev=use_elev,
                                     use_lc=use_lc, extras=extras_for_var(train, var, rh_t_mode, cfg))[0],
            min_stations=min_stations,
        )
    except RuntimeError:
        return None
    met = _score(model)
    if met:
        met["early_stopped"] = False
        met["stop_reason"] = "full"
        met["score_at_stop"] = met["rmse"]
        met["n_units_scored"] = n_full
    return met


def param_grid(cfg):
    rfsi = cfg.get("rfsi", {})
    n_obs_list = [int(x) for x in rfsi.get("n_obs_list", [5, 10, 20])]
    tun = rfsi.get("rf_tunable", {})
    depths = tun.get("max_depth", [None])
    leaves = tun.get("min_samples_leaf", [1])
    feats = tun.get("max_features", ["sqrt"])
    return list(itertools.product(n_obs_list, depths, leaves, feats))


def run_variable(cfg, var, cluster_method, n_splits, quick=False, months=None, max_times=400):
    rfsi = cfg.get("rfsi", {})
    use_elev, use_lc = cov_switches(cfg)
    from shared.terrain import enabled_covariates
    print(f"  covariates: {enabled_covariates(cfg)}")
    min_stations = int(cfg.get("min_stations_per_field", 10))
    primary = "rmse"
    n_est_search = int(rfsi.get("n_estimators_search", 100))
    n_est_final = int(rfsi.get("rf_fixed", {}).get("n_estimators", 250))
    seed = int(rfsi.get("rf_fixed", {}).get("random_state", 22))

    panel = load_panel(cfg, var)
    from shared.splits.evaluation_sets import apply_evaluation_protocol
    tagged, _, _, _ = apply_evaluation_protocol(panel, variable=var)
    panel = tagged[(~tagged["is_heldout"]) & (tagged["split"].isin(["train", "tune"]))].copy()
    panel = attach_region(panel, build_region_pack(cfg, int(cfg.get("rfsi", {}).get("n_regions", 6))))
    rh_t_mode = str(rfsi.get("rh_t_mode", "none"))
    two_step = bool(rfsi.get("two_step", True)) and two_step_var(var)
    tau_wet = float(rfsi.get("tau_wet", 0.5))
    # Global tune on the prepared panel (held-out dropped, test quarters removed).
    # Optional month cut for dry runs; no regime/medoid subset.
    if months or quick:
        panel = _subset_times(panel, months or "seasonal4")
    panel, _hours = apply_hour_cut(panel, cfg["time_resolution"], cfg.get("_hours"))
    # Early-stop protocol: score only the eval set (max 52). Training stays full.
    from shared.splits.evaluation_sets import load_early_stop_times, early_stop_kind
    time_res = cfg["time_resolution"]
    eval_raw = load_early_stop_times(time_res, early_stop_kind("RFSI"), which="eval")
    if eval_raw:
        pt = pd.to_datetime(panel["time"], utc=True)
        want = set(str(w) for w in eval_raw)
        if time_res == "monthly":
            key = pt.dt.strftime("%Y-%m")
        elif time_res == "weekly":
            key = pt.dt.strftime("%Y-W%W")
        elif time_res == "daily":
            key = pt.dt.strftime("%Y-%m-%d")
        else:
            key = pt.dt.strftime("%Y-%m-%dT%H:%M:%S")
        is_tune = panel["split"] == "tune"
        keep = (~is_tune) | key.isin(want)
        if keep.any():
            panel = panel.loc[keep].copy()
            print(f"  early-stop eval tune times="
                  f"{panel.loc[panel['split']=='tune','time'].nunique()}", flush=True)
    elif max_times:
        panel = cap_timestamps(panel, max_times, time_res,
                               seed=int(rfsi.get("seed", 31)))
    print(f"  tune panel: {len(panel):,} rows | "
          f"{panel['station_name'].nunique()} stations | "
          f"{panel['time'].nunique()} times | "
          f"tune-targets={int(panel['is_tune_target'].sum())}")

    X_all, encoder = prepare_covariates(
        panel, fit_encoder=True, use_elev=use_elev, use_lc=use_lc,
        extras=extras_for_var(panel, var, rh_t_mode, cfg),
    )
    del X_all

    if quick:
        n_est_search = min(n_est_search, 80)
        grid = [(int(rfsi.get("n_obs", 10)), rfsi.get("rf_fixed", {}).get("max_depth"),
                 int(rfsi.get("rf_fixed", {}).get("min_samples_leaf", 5)),
                 rfsi.get("rf_fixed", {}).get("max_features", "sqrt"))]
    else:
        # Coordinate descent: 4 parameters, so not a full product.
        n_obs_list = [int(x) for x in rfsi.get("n_obs_list", [8, 16, 32])]
        tun = rfsi.get("rf_tunable", {})
        depths = tun.get("max_depth", [None])
        leaves = tun.get("min_samples_leaf", [5])
        feats = tun.get("max_features", ["sqrt"])
        locked = {
            "n_obs": n_obs_list[len(n_obs_list) // 2],
            "depth": depths[len(depths) // 2],
            "leaf": leaves[len(leaves) // 2],
            "feat": feats[len(feats) // 2],
        }
        grid = []
        for n in n_obs_list:
            grid.append((n, locked["depth"], locked["leaf"], locked["feat"]))
        for d in depths:
            grid.append((locked["n_obs"], d, locked["leaf"], locked["feat"]))
        for lf in leaves:
            grid.append((locked["n_obs"], locked["depth"], lf, locked["feat"]))
        for ft in feats:
            grid.append((locked["n_obs"], locked["depth"], locked["leaf"], ft))
        # unique, keep order
        seen, uniq = set(), []
        for g in grid:
            if g not in seen:
                seen.add(g)
                uniq.append(g)
        grid = uniq
    print(f"  search {len(grid)} combos, fit on train × pool, "
          f"score fixed tune-targets × tune, {n_est_search} trees", flush=True)
    rows = []
    t0 = time.perf_counter()
    n_grid = len(grid)
    champion_rmse = None
    for i, (n_obs, depth, leaf, feat) in enumerate(grid, start=1):
        depth = _depth(depth)
        print(f"  [{i}/{n_grid}] n={n_obs} depth={depth} leaf={leaf} feat={feat}", flush=True)
        rf_params = {
            "n_estimators": n_est_search,
            "max_depth": depth,
            "min_samples_leaf": int(leaf),
            "max_features": feat,
            "random_state": seed,
            "n_jobs": -1,
        }
        met = score_combo(
            panel, var, int(n_obs), rf_params, encoder,
            use_elev, use_lc, min_stations, n_splits, primary,
            rh_t_mode=rh_t_mode, two_step=two_step, tau_wet=tau_wet, cfg=cfg,
            champion_rmse=champion_rmse,
        )
        if met is None:
            print(f"    [{i}/{n_grid}] no score", flush=True)
            continue
        rows.append({
            "n_obs": int(n_obs),
            "max_depth": depth,
            "min_samples_leaf": int(leaf),
            "max_features": feat,
            **met,
        })
        if not met.get("early_stopped") and (champion_rmse is None or met["rmse"] < champion_rmse):
            champion_rmse = met["rmse"]
        elapsed = time.perf_counter() - t0
        eta = elapsed / i * (n_grid - i)
        print(
            f"    [{i}/{n_grid}] n={n_obs:2d} depth={str(depth):4s} leaf={leaf} "
            f"feat={feat!s:5s}  RMSE={met['rmse']:.3f} NSE={met['nse']:.3f}  "
            f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min",
            flush=True,
        )
    if not rows:
        raise RuntimeError(f"Search produced no scores for {var}")

    table = pd.DataFrame(rows)
    if "early_stopped" in table.columns:
        finished = table[~table["early_stopped"].fillna(False)]
        if not finished.empty:
            table = finished
    ascending = primary in {"rmse", "mae"}
    table = table.sort_values(primary, ascending=ascending)
    best = table.iloc[0].to_dict()
    boundary = False

    # November winners for temperature, precip and humidity sat on n_obs=4,
    # the bottom of the grid, and the test RMSE did not match the tune RMSE.
    # Score one step past that edge. Do not loop if the new end wins too.
    if not quick:
        n_now = int(best["n_obs"])
        grid_n = sorted({int(x) for x in n_obs_list})
        step = None
        if n_now == grid_n[0] and n_now > 1:
            step = max(1, n_now // 2)
        elif n_now == grid_n[-1]:
            step = n_now * 2
        already = set(table["n_obs"].astype(int))
        if step is not None and step not in already:
            depth = _depth(best["max_depth"])
            leaf = int(best["min_samples_leaf"])
            feat = best["max_features"]
            print(f"  boundary: n_obs={n_now} is an endpoint. scoring n_obs={step}", flush=True)
            rf_params = {
                "n_estimators": n_est_search,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "max_features": feat,
                "random_state": seed,
                "n_jobs": -1,
            }
            met = score_combo(
                panel, var, int(step), rf_params, encoder,
                use_elev, use_lc, min_stations, n_splits, primary,
                rh_t_mode=rh_t_mode, two_step=two_step, tau_wet=tau_wet, cfg=cfg,
            )
            if met is not None:
                rows.append({
                    "n_obs": int(step),
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    "max_features": feat,
                    **met,
                })
                table = pd.DataFrame(rows).sort_values(primary, ascending=ascending)
                best = table.iloc[0].to_dict()
        boundary = step is not None
    elapsed = time.perf_counter() - t0

    payload = {
        "variable": var,
        "time_resolution": cfg["time_resolution"],
        "architecture": "pooled",
        "split": "dev",
        "cluster_method": cluster_method,
        "n_medoid_months": int(panel["time"].nunique()),
        "n_obs": int(best["n_obs"]),
        "max_depth": _depth(best["max_depth"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "max_features": best["max_features"],
        "n_estimators": n_est_final,
        "n_estimators_search": n_est_search,
        "primary_metric": primary,
        "boundary": bool(boundary),
        "dev_rmse": float(best["rmse"]),
        "dev_mae": float(best["mae"]),
        "dev_nse": float(best["nse"]),
        "dev_kge": float(best["kge"]),
        "dev_ccc": float(best["ccc"]),
        "search_seconds": float(elapsed),
    }
    out = get_rfsi_tuned_params_path(var, cfg["time_resolution"])
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    table_path = out.with_suffix(".csv")
    table.to_csv(table_path, index=False)
    print(f"  BEST n_obs={payload['n_obs']} max_depth={payload['max_depth']} "
          f"leaf={payload['min_samples_leaf']} feat={payload['max_features']} "
          f"RMSE={payload['dev_rmse']:.3f}")
    print(f"  wrote {out}")
    return payload


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variables", nargs="*", default=None)
    p.add_argument("--cluster-method", default="gmm")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--months", default=None, help="all | seasonal4 | YYYY-MM,YYYY-MM")
    p.add_argument("--max-times", type=int, default=400,
                   help="cap timestamps, sampled within each split")
    add_hours_arg(p)
    add_time_res_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    apply_time_res(cfg, args)
    cfg["_hours"] = args.hours
    wanted = variables_from_cfg(cfg, args.variables)
    print("=" * 72)
    print("RFSI pooled hyperparameter search (global, no regimes)")
    print(f"  variables {wanted}")
    print("=" * 72)
    for var in wanted:
        print(f"\n{'=' * 60}\n{var}")
        n_splits = 2 if args.quick else args.n_splits
        run_variable(cfg, var, args.cluster_method, n_splits, quick=args.quick,
                     months=args.months, max_times=args.max_times)
    print("\nSearch finished.")


if __name__ == "__main__":
    main()
