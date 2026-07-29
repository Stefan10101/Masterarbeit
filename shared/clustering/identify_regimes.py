#!/usr/bin/env python3
"""
identify_regimes.py
Identify meteorological regimes by clustering half-hourly (or coarser)
station fields.  Produces regime assignments and rich diagnostics that
later interpolation methods (BSS, IDW, RFSI, ...) can reuse.

Designed to live in CODE/shared/clustering/ and to be driven entirely
through paths.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from typing import List, Optional

# ---------------------------------------------------------------------------
# Make the shared package and the project root importable
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parents[1]          # .../CODE
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(THIS_DIR))

from paths import (
    get_aggregated_dir,
    get_clusters_dir,
    get_stations_metadata_path,
)

# local package
from features import get_feature_func, NAME_MAP, FEATURE_FUNCS, PREFERRED_COLUMNS
from clustering_core import build_feature_matrix, run_clustering_for_variable


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Cluster station fields into meteorological regimes"
    )
    p.add_argument("--method", required=True,
                   choices=["BSS", "IDW", "RFSI", "COMMON"],
                   help="Determines input aggregated dir and output clusters dir")
    p.add_argument("--resolution", required=True,
                   choices=["half_hourly", "daily", "weekly", "monthly", "seasonal"],
                   help="Which aggregated file to read")
    p.add_argument("--variables", nargs="+", default=None,
                   help="Subset of variables (default = all known)")
    p.add_argument("--k-min", type=int, default=3)
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--n-medoids", type=int, default=8)
    p.add_argument("--filter", default="all", choices=["all", "day", "night"])
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def _time_col_name(resolution: str) -> str:
    mapping = {
        "daily": "date",
        "weekly": "year_week",
        "monthly": "year_month",
        "half_hourly": "timestamp",
        "seasonal": "season",
    }
    return mapping[resolution]


def _load_metadata() -> pd.DataFrame:
    path = get_stations_metadata_path()
    meta = pd.read_csv(path)
    # normalise column names that appear in the sample
    if "hoehe" in meta.columns and "elev_dem" not in meta.columns:
        meta = meta.rename(columns={"hoehe": "elev_dem"})
    return meta


def _load_aggregated(
    method: str,
    resolution: str,
    filter_: str,
    start: Optional[str],
    end: Optional[str],
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    agg_dir = get_aggregated_dir(method)
    if filter_ == "all":
        fname = f"{resolution}_station_data.parquet"
    else:
        fname = f"{filter_}_{resolution}_station_data.parquet"
    path = agg_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Aggregated file not found: {path}")

    # only read columns that actually exist (saves RAM, avoids ArrowInvalid)
    time_col = _time_col_name(resolution)
    if columns is not None:
        import pyarrow.parquet as pq
        schema_names = set(pq.read_schema(path).names)
        requested = ["station_name", time_col] + list(columns)
        cols = [c for c in dict.fromkeys(requested) if c in schema_names]
        df = pd.read_parquet(path, columns=cols)
    else:
        df = pd.read_parquet(path)

    # robust date filter
    if start or end:
        ts = df[time_col]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            ts = pd.to_datetime(ts, errors="coerce", utc=True)
            # drop timezone for simple comparisons if present
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_convert(None)
        else:
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_convert(None)

        n_before = len(df)
        mask = pd.Series(True, index=df.index)
        if start:
            mask &= ts >= pd.to_datetime(start)
        if end:
            mask &= ts <= pd.to_datetime(end)
        df = df.loc[mask].copy()
        print(f"Date filter: {n_before:,} → {len(df):,} rows "
              f"({df[time_col].nunique():,} timestamps)")

    return df


def _discover_value_columns(df: pd.DataFrame, requested: Optional[List[str]]) -> Dict[str, str]:
    """
    Map canonical variable keys -> actual column name present in the parquet.
    Prefers mean/sum columns when several candidates exist.
    """
    from features import PREFERRED_COLUMNS

    available = set(df.columns)
    mapping = {}
    candidates = requested or list(FEATURE_FUNCS.keys())

    for var in candidates:
        # 1. preferred columns in order
        prefs = PREFERRED_COLUMNS.get(var, [var])
        found = next((c for c in prefs if c in available), None)
        if found:
            mapping[var] = found
            continue

        # 2. any alias that maps to this canonical name
        for alias, canonical in NAME_MAP.items():
            if canonical == var and alias in available:
                mapping[var] = alias
                break
        else:
            # 3. case-insensitive fallback
            for col in available:
                if col.lower() == var.lower():
                    mapping[var] = col
                    break

    if not mapping:
        raise RuntimeError(
            f"None of the requested variables found in the aggregated file.\n"
            f"Requested: {candidates}\nAvailable columns: {sorted(available)}"
        )
    return mapping


def main():
    args = parse_args()
    method = args.method.upper()
    resolution = args.resolution
    time_col = _time_col_name(resolution)

    print("=" * 78)
    print(f"Regime clustering | method={method} | resolution={resolution}")
    print(f"k-range = [{args.k_min}, {args.k_max}]  n_medoids = {args.n_medoids}")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Load data (only the columns we need – keeps RAM down)
    # ------------------------------------------------------------------
    meta = _load_metadata()

    # collect every preferred column name so we never load unused min/max etc.
    from features import PREFERRED_COLUMNS
    wanted_value_cols = []
    for prefs in PREFERRED_COLUMNS.values():
        wanted_value_cols.extend(prefs)
    # unique, preserve order
    wanted_value_cols = list(dict.fromkeys(wanted_value_cols))

    data = _load_aggregated(
        method, resolution, args.filter, args.start_date, args.end_date,
        columns=wanted_value_cols,
    )
    print(f"Loaded {len(data):,} rows, {data[time_col].nunique():,} unique timestamps")

    var_to_col = _discover_value_columns(data, args.variables)
    print(f"Variables to process: {list(var_to_col.keys())}")

    # temperature companion column (for humidity temp_corr)
    temp_col = None
    for cand in ("temp_mean", "temperature", "temp", "TL"):
        if cand in data.columns:
            temp_col = cand
            break

    # ------------------------------------------------------------------
    # Output directory (subfolder when a date range is given)
    # ------------------------------------------------------------------
    out_root = get_clusters_dir(method) / resolution
    if args.start_date or args.end_date:
        tag = f"{args.start_date or 'start'}_{args.end_date or 'end'}".replace("-", "")
        out_root = out_root / tag
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {out_root}")

    all_assignments = []

    for var, value_col in var_to_col.items():
        print(f"\n----- {var} (column={value_col}) -----")

        feat_df = build_feature_matrix(
            data=data,
            meta=meta,
            var=var,
            time_col=time_col,
            value_col=value_col,
            resolution=resolution,
            temp_col=temp_col,
        )
        if feat_df.empty:
            print(f"  [SKIP] no valid feature rows for {var}")
            continue

        print(f"  Feature matrix: {feat_df.shape[0]} timestamps × {feat_df.shape[1]-1} features")

        result = run_clustering_for_variable(
            feat_df=feat_df,
            var=var,
            k_min=args.k_min,
            k_max=args.k_max,
            n_medoids=args.n_medoids,
            random_state=args.random_state,
        )

        # ---- write artefacts ----
        assign = result["assignments"]
        assign.to_parquet(out_root / f"{var}_assignments.parquet", index=False)

        result["features"].to_parquet(out_root / f"{var}_features.parquet")

        # summary as YAML (human + machine readable)
        summary_path = out_root / f"{var}_summary.yaml"
        with open(summary_path, "w", encoding="utf-8") as f:
            yaml.dump(result["summary"], f, sort_keys=False, allow_unicode=True)

        # medoid timestamps only (lightweight)
        medoid_rows = []
        for cid, ts_list in result["medoids"].items():
            for ts in ts_list:
                medoid_rows.append({"variable": var, "cluster_id": cid, "timestamp": ts})
        pd.DataFrame(medoid_rows).to_parquet(
            out_root / f"{var}_medoids.parquet", index=False
        )

        all_assignments.append(assign)

        print(f"  best k = {result['summary']['best_k']}  "
              f"silhouette = {result['summary']['best_silhouette']:.3f}")
        print(f"  wrote {summary_path.name} and companions")

    # combined assignment table (handy for production)
    if all_assignments:
        combined = pd.concat(all_assignments, ignore_index=True)
        combined.to_parquet(out_root / "all_regime_assignments.parquet", index=False)
        print(f"\nCombined assignments → {out_root / 'all_regime_assignments.parquet'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
