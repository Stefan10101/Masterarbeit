#!/usr/bin/env python3
"""
train_cluster_params.py
Train IDW free parameters (p, Fz, k) on cluster medoids via full LOOCV.

For every variable and every cluster the script:
  1. loads the 20 medoid timestamps produced by identify_regimes,
  2. runs the existing LOOCV grid search on each medoid field,
  3. aggregates the best parameters by median across medoids,
  4. writes one parquet per variable under
     IDW/Output/cluster_params/{resolution}/{cluster_method}/.

Production (produce_idw_maps.py) later reads these files and applies
the cluster-specific parameters to every timestep.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import (
    get_aggregated_data_path,
    get_cluster_params_path,
    get_domain_stations_path,
    get_medoids_path,
    ensure_dir,
)
from loocv_optimizer import optimize_idw_params_loocv

CONFIG_PATH = SCRIPT_DIR / "config.yaml"

# Canonical clustering name → preferred column in the aggregated parquet
VAR_TO_COL = {
    "temperature": "temp_mean",
    "precipitation": "precip_sum",
    "wind_speed": "wind_mean",
    "relative_humidity": "rh_mean",
    "snow_height": "snow_mean",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train IDW free parameters on cluster medoids (full LOOCV)"
    )
    p.add_argument("--method", default="IDW", help="Method folder (default: IDW)")
    p.add_argument(
        "--resolution",
        default="half_hourly",
        choices=["half_hourly", "daily", "weekly", "monthly", "seasonal"],
    )
    p.add_argument(
        "--time-resolution",
        default=None,
        choices=["half_hourly", "daily", "weekly", "monthly", "seasonal"],
        help="Alias for --resolution.",
    )
    p.add_argument("--cluster-method", default="gmm", choices=["gmm", "kmeans", "som"])
    p.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Canonical variable names (default = all known)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-train even if a parameter file already exists",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel jobs for the LOOCV grid (default = config n_jobs)",
    )
    p.add_argument(
        "--min-stations",
        type=int,
        default=None,
        help="Skip a medoid field with fewer stations (default = config)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_aggregated(method: str, resolution: str) -> pd.DataFrame:
    path = get_aggregated_data_path(method, resolution)
    if not path.exists():
        raise FileNotFoundError(f"Aggregated file not found: {path}")
    df = pd.read_parquet(path)

    # normalise time column to a tz-aware UTC Timestamp series named "time"
    time_candidates = ["timestamp", "time", "date", "year_week", "year_month"]
    time_col = next((c for c in time_candidates if c in df.columns), None)
    if time_col is None:
        raise RuntimeError(f"No time column found in {path}. Columns: {list(df.columns)}")

    if resolution == "weekly":
        df["time"] = pd.to_datetime(df[time_col].astype(str) + "-1", format="%Y-W%W-%w", utc=True)
    elif resolution == "monthly":
        df["time"] = pd.to_datetime(df[time_col].astype(str) + "-01", utc=True)
    else:
        ts = pd.to_datetime(df[time_col], utc=True)
        df["time"] = ts

    return df


def load_stations(method: str, domain: str) -> pd.DataFrame:
    path = get_domain_stations_path(method, domain)
    if not path.exists():
        raise FileNotFoundError(f"Stations file not found: {path}")
    stations = pd.read_parquet(path)
    need = {"station_name", "x", "y", "elev"}
    missing = need - set(stations.columns)
    if missing:
        # try common aliases
        rename = {}
        if "elev" not in stations.columns and "elev_dem" in stations.columns:
            rename["elev_dem"] = "elev"
        if "elev" not in stations.columns and "hoehe" in stations.columns:
            rename["hoehe"] = "elev"
        stations = stations.rename(columns=rename)
        missing = need - set(stations.columns)
        if missing:
            raise RuntimeError(f"Stations file missing columns: {missing}")
    return stations[["station_name", "x", "y", "elev"]].drop_duplicates("station_name")


def build_field(
    agg: pd.DataFrame,
    stations: pd.DataFrame,
    value_col: str,
    timestamp: pd.Timestamp,
    min_stations: int,
) -> Optional[pd.DataFrame]:
    """Return valid_df for one timestamp or None if too few stations."""
    # robust timestamp match (handle tz-naive vs tz-aware)
    t = pd.Timestamp(timestamp)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")

    mask = agg["time"] == t
    if not mask.any():
        # try without sub-second / slight offset tolerance
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
    return merged


def train_one_cluster(
    cluster_id: int,
    medoid_times: List[pd.Timestamp],
    agg: pd.DataFrame,
    stations: pd.DataFrame,
    value_col: str,
    param_grid: dict,
    primary_metric: str,
    min_stations: int,
    n_jobs: int,
) -> Optional[dict]:
    """
    Run LOOCV on every medoid of one cluster, return median parameters.
    """
    rows = []
    for ts in medoid_times:
        valid_df = build_field(agg, stations, value_col, ts, min_stations)
        if valid_df is None:
            continue
        best_params, best_scores = optimize_idw_params_loocv(
            valid_df,
            value_col,
            param_grid,
            primary_metric=primary_metric,
            verbose=False,
            n_jobs=n_jobs,
        )
        rows.append({**best_params, **best_scores})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    # median aggregation (robust to occasional outlier medoids)
    return {
        "cluster_id": int(cluster_id),
        "p": float(df["p"].median()),
        "Fz": float(df["Fz"].median()),
        "k": int(round(df["k"].median())),
        "rmse": float(df["rmse"].median()),
        "mae": float(df["mae"].median()),
        "nse": float(df["nse"].median()),
        "kge": float(df["kge"].median()),
        "n_medoids_used": int(len(df)),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config()

    method = args.method.upper()
    resolution = args.time_resolution or args.resolution
    cluster_method = args.cluster_method
    domain = cfg.get("domain", {}).get("preset", "full")
    param_grid = cfg["param_grid"]["power"]
    primary_metric = cfg.get("primary_metric", "rmse")
    n_jobs = args.n_jobs if args.n_jobs is not None else cfg.get("n_jobs", -1)
    min_stations = (
        args.min_stations
        if args.min_stations is not None
        else cfg.get("min_stations_per_field", 10)
    )

    variables = args.variables or list(VAR_TO_COL.keys())

    print("=" * 78)
    print(f"Train IDW cluster parameters")
    print(f"  method={method}  resolution={resolution}  cluster={cluster_method}")
    print(f"  domain={domain}  primary_metric={primary_metric}  n_jobs={n_jobs}")
    print(f"  variables={variables}")
    print("=" * 78)

    # load shared data once
    print("Loading aggregated station data …")
    agg = load_aggregated(method, resolution)
    print(f"  {len(agg):,} rows, {agg['time'].nunique():,} timestamps")

    print("Loading station coordinates …")
    stations = load_stations(method, domain)
    print(f"  {len(stations):,} stations")

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
            print(f"\n[{var}] parameters already exist → {out_path.name}  (use --force to overwrite)")
            continue

        import importlib.util as _ilu
        _sp = CODE_DIR / "shared" / "splits" / "splits.py"
        _spec = _ilu.spec_from_file_location("thesis_time_splits", _sp)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        spec = _mod.load_time_splits()
        dev_start, dev_end = spec["windows"]["dev"]
        start, end = str(dev_start.date()), str(dev_end.date())
        dated = get_medoids_path(method, resolution, cluster_method, var, start, end)
        undated = get_medoids_path(method, resolution, cluster_method, var)
        medoids_path = dated if dated.exists() else undated
        if not medoids_path.exists():
            print(f"\n[SKIP] medoids not found: {dated} or {undated}")
            continue

        medoids = pd.read_parquet(medoids_path)
        medoids["timestamp"] = pd.to_datetime(medoids["timestamp"], utc=True)
        labels = _mod.label_times(
            pd.DatetimeIndex(medoids["timestamp"]).tz_convert("UTC").tz_localize(None)
        )
        before = len(medoids)
        medoids = medoids.loc[labels.to_numpy() == "dev"].copy()
        print(f"  medoids {medoids_path}")
        print(f"  medoids in DEV: {len(medoids)} / {before}")
        if medoids.empty:
            print(f"\n[SKIP] no DEV medoids for {var}")
            continue

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
                param_grid=param_grid,
                primary_metric=primary_metric,
                min_stations=min_stations,
                n_jobs=n_jobs,
            )
            if row is None:
                print(f"  cluster {cid}: no usable medoids — skipped")
                continue
            row["variable"] = var
            results.append(row)
            print(
                f"  cluster {cid:2d}: p={row['p']:.2f}  Fz={row['Fz']:.2f}  k={row['k']:2d}  "
                f"rmse={row['rmse']:.3f}  (n={row['n_medoids_used']})"
            )

        if not results:
            print(f"  [WARN] no parameters produced for {var}")
            continue

        out_df = pd.DataFrame(results)
        out_df["trained_at"] = datetime.now(timezone.utc).isoformat()
        # stable column order
        cols = [
            "variable", "cluster_id", "p", "Fz", "k",
            "rmse", "mae", "nse", "kge", "n_medoids_used", "trained_at",
        ]
        out_df = out_df[cols].sort_values("cluster_id").reset_index(drop=True)
        out_df.to_parquet(out_path, index=False)
        print(f"  wrote {len(out_df)} clusters → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
