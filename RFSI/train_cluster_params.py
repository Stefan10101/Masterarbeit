#!/usr/bin/env python3
"""
train_cluster_params.py
Train RFSI free parameters on cluster medoids via Phase-A style LOOCV.

For every variable and every cluster the script:
  1. loads the medoid timestamps produced by identify_regimes,
  2. for each medoid runs the existing cheap LOOCV grid search
     (station subsample + reduced n_estimators),
  3. aggregates the best parameters across medoids
     (median for numeric, mode for categorical),
  4. writes one parquet per variable under
     RFSI/Output/cluster_params/{resolution}/{cluster_method}/.

Production (produce_rfsi_maps.py) later reads these files and applies
the cluster-specific parameters to every timestep.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import (
    get_aggregated_data_path,
    get_cluster_params_path,
    get_domain_stations_path,
    get_medoids_path,
)
from rfsi_optimizer import optimize_rfsi_params_loocv

CONFIG_PATH = SCRIPT_DIR / "config.yaml"

VAR_TO_COL = {
    "temperature": "temp_mean",
    "precipitation": "precip_sum",
    "wind_speed": "wind_mean",
    "relative_humidity": "rh_mean",
    "snow_height": "snow_mean",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Train RFSI free parameters on cluster medoids (Phase-A LOOCV)"
    )
    p.add_argument("--method", default="RFSI")
    p.add_argument(
        "--resolution",
        default="half_hourly",
        choices=["half_hourly", "daily", "weekly", "monthly", "seasonal"],
    )
    p.add_argument("--cluster-method", default="gmm", choices=["gmm", "kmeans", "som"])
    p.add_argument("--variables", nargs="+", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--min-stations", type=int, default=None)
    return p.parse_args()


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_aggregated(method: str, resolution: str) -> pd.DataFrame:
    path = get_aggregated_data_path(method, resolution)
    if not path.exists():
        raise FileNotFoundError(f"Aggregated file not found: {path}")
    df = pd.read_parquet(path)

    time_candidates = ["timestamp", "time", "date", "year_week", "year_month"]
    time_col = next((c for c in time_candidates if c in df.columns), None)
    if time_col is None:
        raise RuntimeError(f"No time column in {path}. Columns: {list(df.columns)}")

    if resolution == "weekly":
        df["time"] = pd.to_datetime(
            df[time_col].astype(str) + "-1", format="%Y-W%W-%w", utc=True
        )
    elif resolution == "monthly":
        df["time"] = pd.to_datetime(df[time_col].astype(str) + "-01", utc=True)
    else:
        df["time"] = pd.to_datetime(df[time_col], utc=True)
    return df


def load_stations(method: str, domain: str) -> pd.DataFrame:
    path = get_domain_stations_path(method, domain)
    if not path.exists():
        raise FileNotFoundError(f"Stations file not found: {path}")
    stations = pd.read_parquet(path)

    rename = {}
    if "elev" not in stations.columns and "elev_dem" in stations.columns:
        rename["elev_dem"] = "elev"
    if "elev" not in stations.columns and "hoehe" in stations.columns:
        rename["hoehe"] = "elev"
    if rename:
        stations = stations.rename(columns=rename)

    need = {"station_name", "x", "y", "elev"}
    missing = need - set(stations.columns)
    if missing:
        raise RuntimeError(f"Stations file missing columns: {missing}")

    cols = ["station_name", "x", "y", "elev"]
    if "clc_code" in stations.columns:
        cols.append("clc_code")
    return stations[cols].drop_duplicates("station_name")


def build_field(
    agg: pd.DataFrame,
    stations: pd.DataFrame,
    value_col: str,
    timestamp: pd.Timestamp,
    min_stations: int,
) -> Optional[pd.DataFrame]:
    t = pd.Timestamp(timestamp)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")

    mask = agg["time"] == t
    if not mask.any():
        mask = (agg["time"] - t).abs() <= pd.Timedelta("1min")
    if not mask.any():
        return None

    df_t = agg.loc[mask, ["station_name", value_col]].dropna(subset=[value_col])
    merged = (
        df_t.merge(stations, on="station_name", how="inner")
        .dropna(subset=["x", "y", "elev", value_col])
        .drop_duplicates("station_name")
    )
    if len(merged) < min_stations:
        return None
    return merged.reset_index(drop=True)


def _mode(series: pd.Series):
    """Most frequent value; preserves None/NaN as a valid category."""
    vals = []
    for v in series:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            vals.append(None)
        elif isinstance(v, (np.integer, np.floating)):
            vals.append(v.item() if hasattr(v, "item") else v)
        else:
            vals.append(v)
    if not vals:
        return None
    counts = Counter(vals)
    return counts.most_common(1)[0][0]


def train_one_cluster(
    cluster_id: int,
    medoid_times: List[pd.Timestamp],
    agg: pd.DataFrame,
    stations: pd.DataFrame,
    value_col: str,
    n_obs_list: list,
    rf_fixed_search: dict,
    rf_tunable: dict,
    primary_metric: str,
    n_opt_stations: int,
    min_stations: int,
    n_jobs: int,
) -> Optional[dict]:
    rows = []
    for ts in medoid_times:
        valid_df = build_field(agg, stations, value_col, ts, min_stations)
        if valid_df is None:
            continue

        # Phase-A style: subsample stations for the search
        if len(valid_df) > n_opt_stations:
            valid_df = (
                valid_df.sample(n=n_opt_stations, random_state=31)
                .reset_index(drop=True)
            )

        best_params, best_scores = optimize_rfsi_params_loocv(
            valid_df,
            value_col,
            n_obs_list,
            rf_fixed_search,
            rf_tunable,
            primary_metric=primary_metric,
            n_jobs=n_jobs,
            verbose=False,
        )
        rows.append({**best_params, **best_scores})

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # numeric → median; categorical / nullable → mode
    n_obs = int(round(df["n_obs"].median()))
    max_depth = _mode(df["max_depth"]) if "max_depth" in df.columns else None
    min_samples_leaf = (
        int(round(df["min_samples_leaf"].median()))
        if "min_samples_leaf" in df.columns
        else 1
    )
    max_features = _mode(df["max_features"]) if "max_features" in df.columns else "sqrt"

    # normalise max_depth
    if max_depth is not None and not (isinstance(max_depth, float) and np.isnan(max_depth)):
        try:
            max_depth = int(max_depth)
        except (TypeError, ValueError):
            max_depth = None
    else:
        max_depth = None

    return {
        "cluster_id": int(cluster_id),
        "n_obs": n_obs,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "rmse": float(df["rmse"].median()),
        "mae": float(df["mae"].median()),
        "nse": float(df["nse"].median()),
        "kge": float(df["kge"].median()),
        "n_medoids_used": int(len(df)),
    }


def main():
    args = parse_args()
    cfg = load_config()

    method = args.method.upper()
    resolution = args.resolution
    cluster_method = args.cluster_method
    domain = cfg.get("domain", {}).get("preset", "full")

    rfsi_cfg = cfg.get("rfsi", {})
    n_obs_list = rfsi_cfg.get("n_obs_list", [5, 10, 20, 30])
    rf_fixed = dict(rfsi_cfg.get("rf_fixed", {"n_estimators": 400, "random_state": 22}))
    rf_tunable = dict(rfsi_cfg.get("rf_tunable", {}))
    primary_metric = rfsi_cfg.get("primary_metric", "rmse")
    n_opt_stations = int(rfsi_cfg.get("n_opt_stations", 50))
    n_estimators_search = int(rfsi_cfg.get("n_estimators_search", 100))

    n_jobs = args.n_jobs if args.n_jobs is not None else cfg.get("n_jobs", -1)
    min_stations = (
        args.min_stations
        if args.min_stations is not None
        else cfg.get("min_stations_per_field", 10)
    )

    # cheap RF for the search phase
    rf_fixed_search = {**rf_fixed, "n_estimators": n_estimators_search, "n_jobs": 1}

    variables = args.variables or list(VAR_TO_COL.keys())

    print("=" * 78)
    print("Train RFSI cluster parameters (Phase-A LOOCV on medoids)")
    print(f"  method={method}  resolution={resolution}  cluster={cluster_method}")
    print(f"  domain={domain}  primary_metric={primary_metric}  n_jobs={n_jobs}")
    print(f"  n_opt_stations={n_opt_stations}  n_estimators_search={n_estimators_search}")
    print(f"  n_obs_list={n_obs_list}")
    print(f"  rf_tunable={rf_tunable}")
    print(f"  variables={variables}")
    print("=" * 78)

    print("Loading aggregated station data …")
    agg = load_aggregated(method, resolution)
    print(f"  {len(agg):,} rows, {agg['time'].nunique():,} timestamps")

    print("Loading station coordinates …")
    stations = load_stations(method, domain)
    print(f"  {len(stations):,} stations"
          + (" (with clc_code)" if "clc_code" in stations.columns else ""))

    for var in variables:
        value_col = VAR_TO_COL.get(var)
        if value_col is None:
            print(f"\n[SKIP] unknown variable '{var}'")
            continue
        if value_col not in agg.columns:
            print(f"\n[SKIP] column '{value_col}' not in aggregated data")
            continue

        out_path = get_cluster_params_path(method, resolution, cluster_method, var)
        if out_path.exists() and not args.force:
            print(f"\n[{var}] parameters already exist → {out_path.name}  "
                  f"(use --force to overwrite)")
            continue

        medoids_path = get_medoids_path(method, resolution, cluster_method, var)
        if not medoids_path.exists():
            print(f"\n[SKIP] medoids not found: {medoids_path}")
            print("  Run identify_regimes.py with --method RFSI (or COMMON) first.")
            continue

        medoids = pd.read_parquet(medoids_path)
        medoids["timestamp"] = pd.to_datetime(medoids["timestamp"], utc=True)
        cluster_ids = sorted(medoids["cluster_id"].unique())
        print(f"\n----- {var} (column={value_col})  {len(cluster_ids)} clusters -----")

        results = []
        for cid in tqdm(cluster_ids, desc=f"  {var}"):
            times = medoids.loc[medoids["cluster_id"] == cid, "timestamp"].tolist()
            row = train_one_cluster(
                cluster_id=cid,
                medoid_times=times,
                agg=agg,
                stations=stations,
                value_col=value_col,
                n_obs_list=n_obs_list,
                rf_fixed_search=rf_fixed_search,
                rf_tunable=rf_tunable,
                primary_metric=primary_metric,
                n_opt_stations=n_opt_stations,
                min_stations=min_stations,
                n_jobs=n_jobs,
            )
            if row is None:
                print(f"  cluster {cid}: no usable medoids — skipped")
                continue
            row["variable"] = var
            results.append(row)
            print(
                f"  cluster {cid:2d}: n_obs={row['n_obs']:2d}  "
                f"max_depth={row['max_depth']}  "
                f"min_samples_leaf={row['min_samples_leaf']}  "
                f"max_features={row['max_features']}  "
                f"rmse={row['rmse']:.3f}  (n={row['n_medoids_used']})"
            )

        if not results:
            print(f"  [WARN] no parameters produced for {var}")
            continue

        out_df = pd.DataFrame(results)
        out_df["trained_at"] = datetime.now(timezone.utc).isoformat()
        cols = [
            "variable", "cluster_id",
            "n_obs", "max_depth", "min_samples_leaf", "max_features",
            "rmse", "mae", "nse", "kge",
            "n_medoids_used", "trained_at",
        ]
        out_df = out_df[cols].sort_values("cluster_id").reset_index(drop=True)
        # store max_depth as object so None survives parquet
        out_df["max_depth"] = out_df["max_depth"].astype(object)
        out_df.to_parquet(out_path, index=False)
        print(f"  wrote {len(out_df)} clusters → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
