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
import importlib.util as _ilu
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

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
    p.add_argument("--method", required=True, nargs="+",
                   choices=["BSS", "IDW", "RFSI", "COMMON"],
                   help="One or more methods. Clustering runs once; results are "
                        "written into every listed method's clusters folder.")
    p.add_argument("--resolution", required=False, default=None,
                   choices=["half_hourly", "daily", "weekly", "monthly", "seasonal"],
                   help="Which aggregated file to read")
    p.add_argument("--time-resolution", default=None,
                   choices=["half_hourly", "daily", "weekly", "monthly", "seasonal"],
                   help="Alias for --resolution")
    p.add_argument("--variables", nargs="+", default=None,
                   help="Subset of variables (default = all known)")
    p.add_argument("--k-min", type=int, default=3)
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--n-medoids", type=int, default=8)
    p.add_argument("--cluster-method", default="gmm",
                   choices=["kmeans", "gmm", "som"],
                   help="Clustering algorithm (default: gmm with BIC)")
    p.add_argument("--pca-variance", type=float, default=0.95,
                   help="PCA variance to retain before clustering "
                        "(0 to disable, default 0.95)")
    p.add_argument("--filter", default="all", choices=["all", "day", "night"])
    p.add_argument("--start-date", type=str, default=None,
                   help="Override window start (default = DEV from time_splits.yaml)")
    p.add_argument("--end-date", type=str, default=None,
                   help="Override window end (default = DEV from time_splits.yaml)")
    p.add_argument("--split", default="dev", choices=["train", "dev", "test"],
                   help="Which time_splits.yaml window to use when dates are omitted")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def _window_from_splits(split: str) -> tuple[str, str]:
    spec_path = CODE_DIR / "shared" / "splits" / "splits.py"
    spec = _ilu.spec_from_file_location("thesis_time_splits", spec_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    start, end = mod.load_time_splits()["windows"][split]
    return str(start.date()), str(end.date())


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

    # robust date filter (works for weekly / monthly / seasonal labels too)
    if start or end:
        from clustering_core import parse_time_label

        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            ts = df[time_col]
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_convert(None)
        else:
            ts = df[time_col].map(lambda x: parse_time_label(x, resolution))

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


def _output_roots(methods, resolution, cluster_method, start_date, end_date):
    """Build output dirs for every method (same relative structure)."""
    roots = []
    for m in methods:
        root = get_clusters_dir(m) / resolution / cluster_method
        if start_date or end_date:
            tag = f"{start_date or 'start'}_{end_date or 'end'}".replace("-", "")
            root = root / tag
        root.mkdir(parents=True, exist_ok=True)
        roots.append(root)
    return roots


def _write_variable_outputs(result, var, out_roots):
    """Write the same artefacts into every method folder."""
    medoid_rows = []
    for cid, ts_list in result["medoids"].items():
        for ts in ts_list:
            medoid_rows.append({"variable": var, "cluster_id": cid, "timestamp": ts})
    medoid_df = pd.DataFrame(medoid_rows)

    for root in out_roots:
        result["assignments"].to_parquet(root / f"{var}_assignments.parquet", index=False)
        result["features"].to_parquet(root / f"{var}_features.parquet")
        with open(root / f"{var}_summary.yaml", "w", encoding="utf-8") as f:
            yaml.dump(result["summary"], f, sort_keys=False, allow_unicode=True)
        medoid_df.to_parquet(root / f"{var}_medoids.parquet", index=False)
        import joblib
        joblib.dump(
            {
                "model": result["model"],
                "scaler": result["scaler"],
                "pca": result["pca"],
                "summary": result["summary"],
                "feature_columns": result["summary"].get("feature_columns"),
            },
            root / f"{var}_cluster_model.joblib",
        )


def main():
    args = parse_args()
    methods = [m.upper() for m in args.method]
    resolution = args.time_resolution or args.resolution
    if not resolution:
        raise SystemExit("need --resolution or --time-resolution")
    if not args.start_date or not args.end_date:
        d0, d1 = _window_from_splits(args.split)
        args.start_date = args.start_date or d0
        args.end_date = args.end_date or d1
        print(f"Using {args.split} window from time_splits.yaml: {args.start_date} → {args.end_date}")
    time_col = _time_col_name(resolution)

    print("=" * 78)
    print(f"Regime clustering | methods={methods} | resolution={resolution}")
    print(f"cluster={args.cluster_method} | PCA={args.pca_variance} | "
          f"k=[{args.k_min},{args.k_max}] | n_medoids={args.n_medoids}")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Load data once (from the first method that has the aggregated file)
    # ------------------------------------------------------------------
    meta = _load_metadata()

    from features import PREFERRED_COLUMNS
    wanted_value_cols = []
    for prefs in PREFERRED_COLUMNS.values():
        wanted_value_cols.extend(prefs)
    wanted_value_cols = list(dict.fromkeys(wanted_value_cols))

    data = None
    source_method = None
    for m in methods:
        try:
            data = _load_aggregated(
                m, resolution, args.filter, args.start_date, args.end_date,
                columns=wanted_value_cols,
            )
            source_method = m
            break
        except FileNotFoundError as e:
            print(f"  no aggregated file for {m}: {e}")
    if data is None:
        raise FileNotFoundError(
            f"No aggregated {resolution} file found for any of {methods}"
        )
    print(f"Loaded data from {source_method}: "
          f"{len(data):,} rows, {data[time_col].nunique():,} timestamps")

    var_to_col = _discover_value_columns(data, args.variables)
    print(f"Variables to process: {list(var_to_col.keys())}")

    temp_col = None
    for cand in ("temp_mean", "temperature", "temp", "TL"):
        if cand in data.columns:
            temp_col = cand
            break

    out_roots = _output_roots(
        methods, resolution, args.cluster_method,
        args.start_date, args.end_date,
    )
    for r in out_roots:
        print(f"Output root: {r}")

    # ------------------------------------------------------------------
    # Cluster once per variable, write into every method folder
    # ------------------------------------------------------------------
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

        pca_var = args.pca_variance if args.pca_variance > 0 else None
        result = run_clustering_for_variable(
            feat_df=feat_df,
            var=var,
            k_min=args.k_min,
            k_max=args.k_max,
            n_medoids=args.n_medoids,
            random_state=args.random_state,
            cluster_method=args.cluster_method,
            pca_variance=pca_var,
        )

        _write_variable_outputs(result, var, out_roots)
        all_assignments.append(result["assignments"])
        print(f"  wrote outputs to {len(out_roots)} method folder(s)")

    if all_assignments:
        combined = pd.concat(all_assignments, ignore_index=True)
        for root in out_roots:
            combined.to_parquet(root / "all_regime_assignments.parquet", index=False)
        print(f"\nCombined assignments written to all method folders")

    print("\nDone.")


if __name__ == "__main__":
    main()
