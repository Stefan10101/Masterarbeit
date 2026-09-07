#!/usr/bin/env python3
"""
Nested leave-location-out for pooled RFSI.

For each station S:
  fit one forest on all other stations (all selected times)
  predict S at every time it is observed, using neighbours that are not S

Each row is tagged train/dev/test from shared/splits/time_splits.yaml
so metrics can be filtered without a second LOO run.

This is the scientific number. The parquet written by produce_rfsi_maps.py
is a frozen-model check, not LLOCV.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from paths import (
    get_aggregated_data_path,
    get_domain_stations_path,
    get_nested_llocv_path,
)
from rfsi_core import RFSI, build_covariates, neighbor_width
from rfsi_optimizer import compute_metrics

import importlib.util as _ilu
_splits_path = Path(__file__).resolve().parents[1] / "shared" / "splits" / "splits.py"
_spec = _ilu.spec_from_file_location("thesis_time_splits", _splits_path)
_splits_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_splits_mod)
label_times = _splits_mod.label_times

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"


def as_naive_utc(values) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    return idx.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def time_col_name(cfg, time_res):
    try:
        return cfg["aggregation"][time_res]["time_col"]
    except Exception:
        return {
            "weekly": "year_week",
            "daily": "date",
            "monthly": "year_month",
            "half_hourly": "timestamp",
        }.get(time_res, "time")


def prepare_covariates(df, encoder=None, fit_encoder=False, use_elev=True, use_lc=True):
    X = build_covariates(
        elev=df["elev"].to_numpy() if use_elev and "elev" in df.columns else None,
        clc_code=df["clc_code"].to_numpy() if use_lc and "clc_code" in df.columns else None,
        use_elev=use_elev and "elev" in df.columns,
        use_lc=use_lc and "clc_code" in df.columns,
    )
    return X, None


def load_panel(cfg, var):
    time_res = cfg["time_resolution"]
    start = pd.Timestamp(cfg["start_date"]).tz_localize(None)
    end = pd.Timestamp(cfg["end_date"]).tz_localize(None)
    df = pd.read_parquet(get_aggregated_data_path("RFSI", time_res))
    col = time_col_name(cfg, time_res)
    if time_res == "weekly":
        raw = pd.to_datetime(df[col] + "-1", format="%Y-W%W-%w", utc=True)
    elif time_res == "monthly":
        raw = pd.to_datetime(df[col].astype(str) + "-01", utc=True)
    else:
        raw = pd.to_datetime(df[col], utc=True)
    df["time"] = as_naive_utc(raw)
    df = df[(df["time"] >= start) & (df["time"] <= end)]

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
    valid["split"] = label_times(valid["time"]).to_numpy()
    return valid


def run_variable(cfg, var, fit_splits, score_splits, max_stations, checkpoint_every):
    rfsi_cfg = cfg.get("rfsi", {})
    n_obs = int(rfsi_cfg.get("n_obs", 10))
    rf_fixed = dict(rfsi_cfg.get("rf_fixed", {"n_estimators": 250, "random_state": 22}))
    rf_params = {
        "n_estimators": int(rf_fixed.get("n_estimators", 250)),
        "max_depth": rf_fixed.get("max_depth"),
        "min_samples_leaf": int(rf_fixed.get("min_samples_leaf", 5)),
        "max_features": rf_fixed.get("max_features", "sqrt"),
        "random_state": int(rf_fixed.get("random_state", 22)),
        "n_jobs": -1,
    }
    rf_params = {k: v for k, v in rf_params.items() if v is not None}
    use_elev = bool(rfsi_cfg.get("use_elevation", True))
    use_lc = bool(rfsi_cfg.get("use_landcover", True))
    min_stations = int(cfg.get("min_stations_per_field", 10))

    valid = load_panel(cfg, var)
    if fit_splits != {"all"}:
        fit_mask = valid["split"].isin(fit_splits)
    else:
        fit_mask = valid["split"].isin(("train", "dev", "test"))
    if score_splits != {"all"}:
        score_mask = valid["split"].isin(score_splits)
    else:
        score_mask = valid["split"].isin(("train", "dev", "test"))

    fit_df = valid[fit_mask].copy()
    score_df = valid[score_mask].copy()
    if use_lc and "clc_code" not in fit_df.columns:
        raise RuntimeError("use_landcover=true but stations have no clc_code")

    names = sorted(score_df["station_name"].unique())
    if max_stations:
        names = names[: int(max_stations)]

    print(f"\n{'=' * 60}\nVARIABLE {var}")
    print(f"  fit rows {len(fit_df):,}  score rows {len(score_df):,}")
    print(f"  stations to leave out: {len(names)}")
    print(f"  n_obs={n_obs}  trees={rf_params['n_estimators']}")

    _, encoder = prepare_covariates(
        fit_df, fit_encoder=True, use_elev=use_elev, use_lc=use_lc
    )
    out_path = get_nested_llocv_path("RFSI", var, cfg["time_resolution"], "full")
    records = []
    t0 = time.perf_counter()

    for i, name in enumerate(tqdm(names, desc=var), start=1):
        train = fit_df[fit_df["station_name"] != name]
        hold = score_df[score_df["station_name"] == name]
        if hold.empty or train["station_name"].nunique() < min_stations:
            continue
        X_train, _ = prepare_covariates(
            train, encoder=encoder, use_elev=use_elev, use_lc=use_lc
        )
        model = RFSI(n_obs=n_obs, rf_params=rf_params)
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

        donors = valid[valid["station_name"] != name]
        for t, hold_t in hold.groupby("time", sort=False):
            # Neighbours are same-timestamp other stations (any year).
            # The forest itself was fit on TRAIN times only.
            others = donors[donors["time"] == t]
            if len(others) < max(min_stations, n_obs):
                continue
            X_hold, _ = prepare_covariates(
                hold_t, encoder=encoder, use_elev=use_elev, use_lc=use_lc
            )
            pred = model.predict_field(
                others[["x", "y"]].to_numpy(),
                others[var].to_numpy(),
                hold_t[["x", "y"]].to_numpy(),
                X_cov_pred=X_hold,
            )
            for row, yhat in zip(hold_t.itertuples(index=False), pred):
                records.append({
                    "time": row.time,
                    "split": row.split,
                    "station_name": name,
                    "variable": var,
                    "observed": float(getattr(row, var)),
                    "predicted": float(yhat),
                    "n_neighbors_avail": int(len(others)),
                })

        if checkpoint_every and i % checkpoint_every == 0 and records:
            pd.DataFrame(records).to_parquet(out_path, index=False)

    elapsed = time.perf_counter() - t0
    pred_df = pd.DataFrame(records)
    if pred_df.empty:
        raise RuntimeError(f"No LLOCV predictions for {var}")
    pred_df.to_parquet(out_path, index=False)

    print(f"  wrote {len(pred_df):,} rows → {out_path}")
    print(f"  wall {elapsed/60:.1f} min  ({elapsed/max(len(names),1):.1f} s/station)")
    for split_name, part in pred_df.groupby("split"):
        met = compute_metrics(part["observed"].to_numpy(), part["predicted"].to_numpy())
        print(
            f"  {split_name:7} n={len(part):6d}  "
            f"RMSE={met['rmse']:.3f}  MAE={met['mae']:.3f}  "
            f"NSE={met['nse']:.3f}  KGE={met['kge']:.3f}  CCC={met['ccc']:.3f}"
        )
    met_all = compute_metrics(pred_df["observed"].to_numpy(), pred_df["predicted"].to_numpy())
    print(
        f"  {'all':7} n={len(pred_df):6d}  "
        f"RMSE={met_all['rmse']:.3f}  MAE={met_all['mae']:.3f}  "
        f"NSE={met_all['nse']:.3f}  KGE={met_all['kge']:.3f}  CCC={met_all['ccc']:.3f}"
    )
    return pred_df


def parse_args():
    p = argparse.ArgumentParser(description="Nested station LLOCV for pooled RFSI")
    p.add_argument(
        "--variables",
        nargs="*",
        default=None,
        help="Default: rfsi.variables_to_process in config.yaml",
    )
    p.add_argument(
        "--fit",
        default="train",
        help="Comma list: train,dev,test or all. Times used to build each fold forest.",
    )
    p.add_argument(
        "--score",
        default="dev,test",
        help="Comma list: train,dev,test or all. Times written to the parquet.",
    )
    p.add_argument("--max-stations", type=int, default=0, help="Smoke test: first N stations")
    p.add_argument("--checkpoint-every", type=int, default=25)
    return p.parse_args()


def parse_split_arg(text: str) -> set[str]:
    text = text.strip().lower()
    if text == "all":
        return {"all"}
    names = {s.strip() for s in text.split(",") if s.strip()}
    bad = names - {"train", "dev", "test"}
    if bad:
        raise ValueError(f"unknown splits {bad}")
    return names


def main():
    args = parse_args()
    cfg = load_config()
    wanted = args.variables or cfg.get("rfsi", {}).get("variables_to_process") or [
        "temp_mean", "precip_sum",
    ]
    fit_splits = parse_split_arg(args.fit)
    score_splits = parse_split_arg(args.score)
    print("=" * 72)
    print("RFSI nested station LLOCV (pooled forest, one fold = one station)")
    print(f"  time_resolution={cfg['time_resolution']}")
    print(f"  fit={sorted(fit_splits)}  score={sorted(score_splits)}")
    print("=" * 72)
    for var in wanted:
        run_variable(
            cfg, var, fit_splits, score_splits,
            max_stations=args.max_stations,
            checkpoint_every=args.checkpoint_every,
        )
    print("\nNested LLOCV finished.")


if __name__ == "__main__":
    main()
