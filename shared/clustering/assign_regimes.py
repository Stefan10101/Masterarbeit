#!/usr/bin/env python3
"""
Assign every timestamp to a cluster model fitted on the tune quarters.

identify_regimes.py fits scaler + PCA + GMM/KMeans on the tune quarters
(excluding held-out stations) and writes the model to the undated
clusters/{resolution}/{cluster_method}/ folder.

This script only applies that model to the full period and labels each
timestamp with its quarter split (train / tune / test / outside) from
shared/splits/quarter_splits.yaml.
"""

from __future__ import annotations

import argparse
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
from shared.splits.evaluation_sets import load_quarter_splits, quarter_of


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["IDW", "BSS", "RFSI"])
    p.add_argument("--resolution", default="monthly")
    p.add_argument("--cluster-method", default="gmm")
    p.add_argument("--variables", nargs="+", default=["temperature", "precipitation"])
    return p.parse_args()


def _split_label(ts, splits: dict) -> str:
    q = quarter_of(ts)
    for name, qs in splits.items():
        if q in qs:
            return name
    return "outside"


def main():
    args = parse_args()
    root = get_clusters_dir(args.method) / args.resolution / args.cluster_method
    root.mkdir(parents=True, exist_ok=True)
    splits = load_quarter_splits()

    time_col = _time_col_name(args.resolution)
    meta = _load_metadata()
    data = _load_aggregated(args.method, args.resolution, "all", None, None)
    var_to_col = _discover_value_columns(data, args.variables)

    temp_col = next((c for c in ("temp_mean", "temperature") if c in data.columns), None)
    print(f"Model / assignment dir: {root}")

    for var, value_col in var_to_col.items():
        pack_path = root / f"{var}_cluster_model.joblib"
        if not pack_path.exists():
            print(f"[SKIP] no model {pack_path}")
            print(f"       Run identify_regimes.py first (writes the undated folder).")
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

        timestamps = pd.to_datetime(feat_df.index, utc=True)
        assign = pd.DataFrame({
            "timestamp": timestamps.to_numpy(),
            "variable": var,
            "cluster_id": labels.astype(int),
            "split": [_split_label(t, splits) for t in timestamps],
        })
        out_path = root / f"{var}_assignments.parquet"
        assign.to_parquet(out_path, index=False)
        print(f"  {var}: {len(assign)} timestamps → {out_path}")
        print(assign.groupby("split")["cluster_id"].nunique().to_string())


if __name__ == "__main__":
    main()
