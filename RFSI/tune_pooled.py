#!/usr/bin/env python3
"""
Tune pooled RFSI hyperparameters on DEV medoid months.

Search space from config.yaml (n_obs_list, rf_tunable).
Score: 5-fold station CV on the pooled medoid-month panel.
Writes RFSI/Output/cluster_params/{res}/pooled/{var}_pooled_params.yaml
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
from rfsi_core import RFSI, build_covariates, extras_for_var, two_step_var
from rfsi_optimizer import compute_metrics

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


def as_naive_utc(values) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    return idx.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")


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


def load_panel(cfg, var):
    time_res = cfg["time_resolution"]
    df = pd.read_parquet(get_aggregated_data_path("RFSI", time_res))
    time_col = cfg["aggregation"][time_res]["time_col"]
    if time_res == "monthly":
        raw = pd.to_datetime(df[time_col].astype(str) + "-01", utc=True)
    elif time_res == "weekly":
        raw = pd.to_datetime(df[time_col].astype(str) + "-1", format="%Y-W%W-%w", utc=True)
    else:
        raw = pd.to_datetime(df[time_col], utc=True)
    df["time"] = as_naive_utc(raw)

    stations = pd.read_parquet(get_domain_stations_path("RFSI", "full"))
    if "elev" not in stations.columns:
        if "elev_dem" in stations.columns:
            stations = stations.rename(columns={"elev_dem": "elev"})
        elif "hoehe" in stations.columns:
            stations = stations.rename(columns={"hoehe": "elev"})
    keep = ["station_name", "x", "y"]
    for extra in ("elev", "clc_code"):
        if extra in stations.columns:
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
                min_stations, n_splits, primary, rh_t_mode="none", two_step=False, tau_wet=0.5):
    names = panel["station_name"].to_numpy()
    uniq = np.unique(names)
    if len(uniq) < n_splits * 2:
        return None
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=22)
    obs_all, pred_all = [], []
    for train_idx, test_idx in kf.split(uniq):
        train_names = set(uniq[train_idx])
        test_names = set(uniq[test_idx])
        train = panel[panel["station_name"].isin(train_names)]
        hold = panel[panel["station_name"].isin(test_names)]
        if train["station_name"].nunique() < min_stations:
            continue
        X_train, _ = prepare_covariates(
            train, encoder=encoder, use_elev=use_elev, use_lc=use_lc,
            extras=extras_for_var(train, var, rh_t_mode),
        )
        model = RFSI(
            n_obs=n_obs, rf_params=rf_params,
            two_step=two_step, tau_wet=tau_wet, var_name=var,
        )
        try:
            model.fit_pooled(
                times=train["time"].to_numpy(),
                coords=train[["x", "y"]].to_numpy(),
                z=train[var].to_numpy(),
                X_cov=X_train,
                min_stations=min_stations,
            )
        except RuntimeError:
            continue
        for t, hold_t in hold.groupby("time", sort=False):
            others = train[train["time"] == t]
            if len(others) < max(min_stations, n_obs):
                continue
            X_hold, _ = prepare_covariates(
                hold_t, encoder=encoder, use_elev=use_elev, use_lc=use_lc,
                extras=extras_for_var(hold_t, var, rh_t_mode),
            )
            pred = model.predict_field(
                others[["x", "y"]].to_numpy(),
                others[var].to_numpy(),
                hold_t[["x", "y"]].to_numpy(),
                X_cov_pred=X_hold,
            )
            obs_all.append(hold_t[var].to_numpy())
            pred_all.append(pred)
    if not obs_all:
        return None
    o = np.concatenate(obs_all)
    p = np.concatenate(pred_all)
    met = compute_metrics(o, p)
    met["n"] = int(len(o))
    return met


def param_grid(cfg):
    rfsi = cfg.get("rfsi", {})
    n_obs_list = [int(x) for x in rfsi.get("n_obs_list", [5, 10, 20])]
    tun = rfsi.get("rf_tunable", {})
    depths = tun.get("max_depth", [None])
    leaves = tun.get("min_samples_leaf", [1])
    feats = tun.get("max_features", ["sqrt"])
    return list(itertools.product(n_obs_list, depths, leaves, feats))


def run_variable(cfg, var, cluster_method, n_splits, quick=False, months=None):
    rfsi = cfg.get("rfsi", {})
    use_elev = bool(rfsi.get("use_elevation", True))
    use_lc = bool(rfsi.get("use_landcover", True))
    min_stations = int(cfg.get("min_stations_per_field", 10))
    primary = rfsi.get("primary_metric", "rmse")
    n_est_search = int(rfsi.get("n_estimators_search", 100))
    n_est_final = int(rfsi.get("rf_fixed", {}).get("n_estimators", 250))
    seed = int(rfsi.get("rf_fixed", {}).get("random_state", 22))

    panel = load_panel(cfg, var)
    rh_t_mode = str(rfsi.get("rh_t_mode", "none"))
    two_step = bool(rfsi.get("two_step", True)) and two_step_var(var)
    tau_wet = float(rfsi.get("tau_wet", 0.5))
    if quick:
        from Kriging.kriging_data import subset_times
        panel = subset_times(panel, months or "seasonal4")
        spec = load_time_splits()
        dev_start, dev_end = spec["windows"]["dev"]
        panel = panel[(panel["time"] >= dev_start) & (panel["time"] <= dev_end)].copy()
        print(f"  QUICK DEV panel: {len(panel):,} rows | "
              f"{panel['station_name'].nunique()} stations | "
              f"{panel['time'].nunique()} months")
    else:
        medoid_times = load_medoid_times(cfg, var, cluster_method)
        if len(medoid_times) == 0:
            raise RuntimeError(
                f"No DEV medoid months for {var}. "
                "Re-run identify_regimes with --start-date/--end-date = DEV."
            )
        panel_months = pd.to_datetime(panel["time"]).dt.to_period("M")
        med_months = pd.DatetimeIndex(medoid_times).tz_localize(None).to_period("M")
        panel = panel[panel_months.isin(set(med_months))].copy()
        print(f"  DEV medoid panel: {len(panel):,} rows | "
              f"{panel['station_name'].nunique()} stations | "
              f"{panel['time'].nunique()} months")

    X_all, encoder = prepare_covariates(
        panel, fit_encoder=True, use_elev=use_elev, use_lc=use_lc,
        extras=extras_for_var(panel, var, rh_t_mode),
    )
    del X_all

    if quick:
        n_est_search = min(n_est_search, 80)
        grid = [(int(rfsi.get("n_obs", 10)), rfsi.get("rf_fixed", {}).get("max_depth"),
                 int(rfsi.get("rf_fixed", {}).get("min_samples_leaf", 5)),
                 rfsi.get("rf_fixed", {}).get("max_features", "sqrt"))]
    else:
        grid = param_grid(cfg)
    print(f"  search {len(grid)} combos × {n_splits} station folds, "
          f"{n_est_search} trees")
    rows = []
    t0 = time.perf_counter()
    for n_obs, depth, leaf, feat in grid:
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
            rh_t_mode=rh_t_mode, two_step=two_step, tau_wet=tau_wet,
        )
        if met is None:
            continue
        rows.append({
            "n_obs": int(n_obs),
            "max_depth": depth,
            "min_samples_leaf": int(leaf),
            "max_features": feat,
            **met,
        })
        print(
            f"    n={n_obs:2d} depth={str(depth):4s} leaf={leaf} feat={feat!s:5s}  "
            f"RMSE={met['rmse']:.3f} NSE={met['nse']:.3f}"
        )
    if not rows:
        raise RuntimeError(f"Search produced no scores for {var}")

    table = pd.DataFrame(rows)
    ascending = primary in {"rmse", "mae"}
    table = table.sort_values(primary, ascending=ascending)
    best = table.iloc[0].to_dict()
    elapsed = time.perf_counter() - t0

    payload = {
        "variable": var,
        "time_resolution": cfg["time_resolution"],
        "architecture": "pooled",
        "split": "dev",
        "cluster_method": cluster_method,
        "n_medoid_months": int(panel["time"].nunique()),
        "n_obs": int(best["n_obs"]),
        "max_depth": None if pd.isna(best["max_depth"]) else best["max_depth"],
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "max_features": best["max_features"],
        "n_estimators": n_est_final,
        "n_estimators_search": n_est_search,
        "primary_metric": primary,
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
    p.add_argument("--months", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    wanted = args.variables or cfg.get("rfsi", {}).get("variables_to_process") or [
        "temp_mean", "precip_sum",
    ]
    spec = load_time_splits()
    dev = spec["windows"]["dev"]
    print("=" * 72)
    print("RFSI pooled hyperparameter search")
    print(f"  DEV {dev[0].date()} → {dev[1].date()}")
    print(f"  variables {wanted}")
    print("=" * 72)
    for var in wanted:
        print(f"\n{'=' * 60}\n{var}")
        n_splits = 2 if args.quick else args.n_splits
        run_variable(cfg, var, args.cluster_method, n_splits, quick=args.quick, months=args.months)
    print("\nSearch finished.")


if __name__ == "__main__":
    main()
