#!/usr/bin/env python3
"""
Assign every timestamp to a DEV-fitted cluster model.

Fit stays on DEV (identify_regimes.py with DEV dates).
This script only applies scaler + PCA + GMM/KMeans to all months.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(THIS_DIR))

from paths import get_clusters_dir
from clustering_core import build_feature_matrix
from identify_regimes import (
    _discover_value_columns,
    _load_aggregated,
    _load_metadata,
    _time_col_name,
)

_spec = _ilu.spec_from_file_location(
    "thesis_time_splits", CODE_DIR / "shared" / "splits" / "splits.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_time_splits = _mod.load_time_splits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["IDW", "BSS", "RFSI"])
    p.add_argument("--resolution", default="monthly")
    p.add_argument("--cluster-method", default="gmm")
    p.add_argument("--variables", nargs="+", default=["temperature", "precipitation"])
    return p.parse_args()


def main():
    args = parse_args()
    spec = load_time_splits()
    dev_start, dev_end = spec["windows"]["dev"]
    tag = f"{dev_start.date()}_{dev_end.date()}".replace("-", "")
    model_root = get_clusters_dir(args.method) / args.resolution / args.cluster_method / tag
    out_root = get_clusters_dir(args.method) / args.resolution / args.cluster_method
    out_root.mkdir(parents=True, exist_ok=True)

    time_col = _time_col_name(args.resolution)
    meta = _load_metadata()
    data = _load_aggregated(args.method, args.resolution, "all", None, None)
    var_to_col = _discover_value_columns(data, args.variables)

    temp_col = next((c for c in ("temp_mean", "temperature") if c in data.columns), None)
    print(f"DEV model dir: {model_root}")
    print(f"Write assignments: {out_root}")

    for var, value_col in var_to_col.items():
        pack_path = model_root / f"{var}_cluster_model.joblib"
        if not pack_path.exists():
            print(f"[SKIP] no model {pack_path}")
            continue
        pack = joblib.load(pack_path)
        feat_df = build_feature_matrix(
            data=data, meta=meta, var=var, time_col=time_col,
            value_col=value_col, resolution=args.resolution, temp_col=temp_col,
        )
        cols = pack["feature_columns"]
        missing = [c for c in cols if c not in feat_df.columns]
        if missing:
            raise RuntimeError(f"{var}: model features missing {missing}")
        X = feat_df[cols].to_numpy(dtype=float)
        Xs = pack["scaler"].transform(X)
        if pack["pca"] is not None:
            Xs = pack["pca"].transform(Xs)
        labels = pack["model"].predict(Xs)
        assign = pd.DataFrame({
            "timestamp": feat_df.index.to_numpy(),
            "variable": var,
            "cluster_id": labels.astype(int),
            "split": _mod.label_times(
                pd.DatetimeIndex(pd.to_datetime(feat_df.index, utc=True))
                .tz_convert("UTC").tz_localize(None)
            ).to_numpy(),
        })
        assign.to_parquet(out_root / f"{var}_assignments.parquet", index=False)
        assign.to_parquet(model_root / f"{var}_assignments_all.parquet", index=False)
        print(f"  {var}: {len(assign)} timestamps → {out_root / f'{var}_assignments.parquet'}")
        print(assign.groupby("split")["cluster_id"].nunique().to_string())


if __name__ == "__main__":
    main()
