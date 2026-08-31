#!/usr/bin/env python3
"""
train_cluster_params.py
Train BSS/BSSE free parameters on cluster medoids via GCV.

For every variable and every cluster the script:
  1. loads the medoid timestamps produced by identify_regimes,
  2. runs optimize_bss_gcv on each medoid field (fixed domain bounds),
  3. aggregates best parameters by median across medoids,
  4. writes one parquet per variable under
     BSS/Output/cluster_params/{resolution}/{cluster_method}/.

Production later reads these files and applies cluster-specific parameters.
"""

from __future__ import annotations

import argparse
import sys
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
from bss_core import optimize_bss_gcv

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
        description="Train BSS/BSSE free parameters on cluster medoids (GCV)"
    )
    p.add_argument("--method", default="BSS")
    p.add_argument(
        "--resolution",
        default="half_hourly",
        choices=["half_hourly", "daily", "weekly", "monthly", "seasonal"],
    )
    p.add_argument("--cluster-method", default="gmm", choices=["gmm", "kmeans", "som"])
    p.add_argument("--variables", nargs="+", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--min-stations", type=int, default=None)
    return p.parse_args()


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def domain_bounds(cfg: dict, domain: str) -> tuple:
    """Fixed knot-grid bounds from the configured domain bbox (+ buffer)."""
    domain_cfg = cfg.get("domain", {})
    buffer_m = float(domain_cfg.get("buffer_m", 15000))
    predefined = cfg.get("predefined_bboxes", {})
    custom = domain_cfg.get("custom_bbox", [100000, 275000, 395000, 400000])
    bbox = predefined.get(domain, custom)
    xmin, ymin, xmax, ymax = [float(v) for v in bbox]
    # extra relative margin so stations near the edge stay inside the knot grid
    x_range = xmax - xmin
    y_range = ymax - ymin
    margin = 0.06
    return (
        xmin - buffer_m - x_range * margin,
        xmax + buffer_m + x_range * margin,
        ymin - buffer_m - y_range * margin,
        ymax + buffer_m + y_range * margin,
    )


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
    return stations[["station_name", "x", "y", "elev"]].drop_duplicates("station_name")


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


def train_one_cluster(
    cluster_id: int,
    medoid_times: List[pd.Timestamp],
    agg: pd.DataFrame,
    stations: pd.DataFrame,
    value_col: str,
    bounds: tuple,
    segment_range: range,
    tau_d_values: list,
    tau_e_values: list,
    bss_method: str,
    min_stations: int,
) -> Optional[dict]:
    xmin, xmax, ymin, ymax = bounds
    rows = []

    for ts in medoid_times:
        valid_df = build_field(agg, stations, value_col, ts, min_stations)
        if valid_df is None:
            continue

        station_coords = valid_df[["x", "y"]].values.astype(float)
        station_values = valid_df[value_col].values.astype(float)
        station_elev = valid_df["elev"].values.astype(float)

        result = optimize_bss_gcv(
            station_values=station_values,
            station_coords=station_coords,
            station_elev=station_elev,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            segment_range=segment_range,
            tau_d_values=tau_d_values,
            tau_e_values=tau_e_values,
            method=bss_method,
        )
        best = result.get("best_params", {})
        rows.append({
            "n_segments": best.get("n_segments", np.nan),
            "tau_d": best.get("tau_d", best.get("tau_x", np.nan)),
            "tau_e": best.get("tau_e", best.get("tau_y", np.nan)),
            "gcv": float(result.get("gcv", np.nan)),
            "effective_df": float(result.get("effective_df", np.nan)),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    return {
        "cluster_id": int(cluster_id),
        "n_segments": int(round(df["n_segments"].median())),
        "tau_d": float(df["tau_d"].median()),
        "tau_e": float(df["tau_e"].median()),
        "gcv": float(df["gcv"].median()),
        "effective_df": float(df["effective_df"].median()),
        "n_medoids_used": int(len(df)),
    }


def main():
    args = parse_args()
    cfg = load_config()

    method = args.method.upper()
    resolution = args.resolution
    cluster_method = args.cluster_method
    domain = cfg.get("domain", {}).get("preset", "full")

    bss_cfg = cfg.get("bss", {})
    bss_method = bss_cfg.get("method", "bsse")
    segment_range = range(
        int(bss_cfg.get("segment_min", 4)),
        int(bss_cfg.get("segment_max", 18)) + 1,
        int(bss_cfg.get("segment_step", 2)),
    )
    tau_values = bss_cfg.get("tau_search")
    tau_d_values = bss_cfg.get("tau_d_search") or tau_values or [0.1]
    tau_e_values = bss_cfg.get("tau_e_search") or tau_values or [0.1]

    min_stations = (
        args.min_stations
        if args.min_stations is not None
        else cfg.get("min_stations_per_field", 10)
    )
    bounds = domain_bounds(cfg, domain)
    variables = args.variables or list(VAR_TO_COL.keys())

    print("=" * 78)
    print("Train BSS/BSSE cluster parameters (GCV on medoids)")
    print(f"  method={method}  resolution={resolution}  cluster={cluster_method}")
    print(f"  domain={domain}  bss_method={bss_method}")
    print(f"  segment_range={list(segment_range)}")
    print(f"  tau_d={tau_d_values}  tau_e={tau_e_values}")
    print(f"  knot bounds (xmin,xmax,ymin,ymax)={tuple(round(v, 1) for v in bounds)}")
    print(f"  variables={variables}")
    print("=" * 78)

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
            print(f"\n[{var}] parameters already exist → {out_path.name}  "
                  f"(use --force to overwrite)")
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
            print("  Run identify_regimes.py with --method BSS first.")
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
                bounds=bounds,
                segment_range=segment_range,
                tau_d_values=tau_d_values,
                tau_e_values=tau_e_values,
                bss_method=bss_method,
                min_stations=min_stations,
            )
            if row is None:
                print(f"  cluster {cid}: no usable medoids — skipped")
                continue
            row["variable"] = var
            results.append(row)
            print(
                f"  cluster {cid:2d}: n_seg={row['n_segments']:2d}  "
                f"tau_d={row['tau_d']:.3f}  tau_e={row['tau_e']:.3f}  "
                f"gcv={row['gcv']:.4g}  (n={row['n_medoids_used']})"
            )

        if not results:
            print(f"  [WARN] no parameters produced for {var}")
            continue

        out_df = pd.DataFrame(results)
        out_df["trained_at"] = datetime.now(timezone.utc).isoformat()
        cols = [
            "variable", "cluster_id",
            "n_segments", "tau_d", "tau_e",
            "gcv", "effective_df",
            "n_medoids_used", "trained_at",
        ]
        out_df = out_df[cols].sort_values("cluster_id").reset_index(drop=True)
        out_df.to_parquet(out_path, index=False)
        print(f"  wrote {len(out_df)} clusters → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
