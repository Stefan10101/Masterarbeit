#!/usr/bin/env python3
"""
Station-fold LLOCV for Kriging.

Default: 5 station folds. Fit trend + variogram on --fit times of the
other stations. Score held-out stations on --score times using
same-timestamp donors.

--mode leave_one matches RFSI nested LLOCV.
--trend none runs the OK control.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_kriging_tuned_params_path, get_nested_llocv_path
from kriging_core import KrigingConfig, KrigingInterpolator
from kriging_data import load_config, load_panel, print_split_metrics


def parse_split_arg(text: str) -> set[str]:
    text = text.strip().lower()
    if text == "all":
        return {"all"}
    names = {s.strip() for s in text.split(",") if s.strip()}
    bad = names - {"train", "dev", "test"}
    if bad:
        raise ValueError(f"unknown splits {bad}")
    return names


def cfg_to_kriging(cfg, overrides: dict | None = None) -> KrigingConfig:
    k = dict(cfg.get("kriging", {}))
    t = overrides or {}
    return KrigingConfig(
        k=int(t.get("k", k.get("k", 16))),
        k_indicator=int(t.get("k_indicator", k.get("k_indicator", k.get("k", 16)))),
        family=str(t.get("family", k.get("family", "exponential"))),
        theta_deg=float(t.get("theta_deg", k.get("theta_deg", 0.0))),
        aniso_ratio=float(t.get("aniso_ratio", k.get("aniso_ratio", 1.0))),
        alpha_z=float(t.get("alpha_z", k.get("alpha_z", 0.0))),
        interactions=str(t.get("interactions", k.get("interactions", "both"))),
        trend=str(t.get("trend", k.get("trend", "linear"))),
        min_clc_count=int(k.get("min_clc_count", 15)),
        min_stations=int(cfg.get("min_stations_per_field", 10)),
        min_pairs=int(k.get("min_pairs", 200)),
        n_lags=int(k.get("n_lags", 12)),
        seed=int(k.get("seed", 22)),
        predict_tile=int(k.get("predict_tile", 8000)),
    )


def load_tuned(var: str, time_res: str) -> dict:
    path = get_kriging_tuned_params_path(var, time_res)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def station_folds(names: list[str], n_folds: int, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    names = list(names)
    rng.shuffle(names)
    folds = [[] for _ in range(n_folds)]
    for i, n in enumerate(names):
        folds[i % n_folds].append(n)
    return folds


def run_variable(cfg, var, fit_splits, score_splits, mode, n_folds, max_stations, trend_override):
    panel = load_panel(cfg, var)
    if fit_splits != {"all"}:
        fit_df = panel[panel["split"].isin(fit_splits)]
    else:
        fit_df = panel[panel["split"].isin(("train", "dev", "test"))]
    if score_splits != {"all"}:
        score_df = panel[panel["split"].isin(score_splits)]
    else:
        score_df = panel[panel["split"].isin(("train", "dev", "test"))]

    names = sorted(score_df["station_name"].unique())
    if max_stations:
        names = names[: int(max_stations)]
    if mode == "leave_one":
        folds = [[n] for n in names]
    else:
        folds = station_folds(names, n_folds, int(cfg.get("kriging", {}).get("seed", 22)))

    tuned = load_tuned(var, cfg["time_resolution"])
    if trend_override:
        tuned = {**tuned, "trend": trend_override}
    kcfg = cfg_to_kriging(cfg, tuned)
    tag = "ok" if kcfg.trend == "none" else "rk"
    out_path = get_nested_llocv_path("Kriging", f"{var}_{tag}", cfg["time_resolution"], "full")
    records = []
    t0 = time.perf_counter()
    print(f"\n{'=' * 60}\nVARIABLE {var}  trend={kcfg.trend}")
    print(f"  fit rows {len(fit_df):,}  score rows {len(score_df):,}")
    print(f"  folds={len(folds)}  k={kcfg.k}  k_ind={kcfg.k_indicator}  "
          f"family={kcfg.family}  az={kcfg.alpha_z}  r={kcfg.aniso_ratio}  "
          f"int={kcfg.interactions}")

    for fi, hold_names in enumerate(folds, start=1):
        hold_set = set(hold_names)
        train = fit_df[~fit_df["station_name"].isin(hold_set)]
        hold = score_df[score_df["station_name"].isin(hold_set)]
        if hold.empty or train["station_name"].nunique() < kcfg.min_stations:
            continue
        model = KrigingInterpolator(kcfg)
        model.fit(train, var, cfg["time_resolution"], cfg)
        for t, hold_t in hold.groupby("time", sort=False):
            others = panel[(panel["time"] == t) & (~panel["station_name"].isin(hold_set))]
            if len(others) < kcfg.min_stations:
                continue
            pred = model.predict_frame(others, hold_t, var)
            for row, yhat in zip(hold_t.itertuples(index=False), pred):
                records.append({
                    "time": row.time,
                    "split": row.split,
                    "station_name": row.station_name,
                    "variable": var,
                    "observed": float(getattr(row, var)),
                    "predicted": float(yhat),
                    "n_neighbors_avail": int(len(others)),
                    "fold": fi,
                    "trend": kcfg.trend,
                })
        print(f"  fold {fi}/{len(folds)}  hold={len(hold_set)}  rows={len(records)}")

    pred_df = pd.DataFrame(records)
    if pred_df.empty:
        raise RuntimeError(f"No LLOCV predictions for {var}")
    pred_df.to_parquet(out_path, index=False)
    print(f"  wrote {len(pred_df):,} rows → {out_path}")
    print(f"  wall {(time.perf_counter()-t0)/60:.1f} min")
    print_split_metrics(pred_df)
    return pred_df


def parse_args():
    p = argparse.ArgumentParser(description="Station-fold LLOCV for Kriging")
    p.add_argument("--variables", nargs="*", default=None)
    p.add_argument("--fit", default="train")
    p.add_argument("--score", default="all")
    p.add_argument("--mode", choices=["folds", "leave_one"], default="folds")
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--max-stations", type=int, default=0)
    p.add_argument("--trend", choices=["linear", "none"], default=None,
                   help="override config trend (none = OK control)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    wanted = args.variables or cfg.get("kriging", {}).get("variables_to_process") or [
        "temp_mean", "precip_sum",
    ]
    n_folds = args.folds or int(cfg.get("kriging", {}).get("n_folds", 5))
    print("=" * 72)
    print("Kriging station LLOCV")
    print(f"  time_resolution={cfg['time_resolution']}")
    print(f"  mode={args.mode}  folds={n_folds}")
    print(f"  fit={args.fit}  score={args.score}")
    print("=" * 72)
    for var in wanted:
        run_variable(
            cfg, var,
            parse_split_arg(args.fit),
            parse_split_arg(args.score),
            args.mode, n_folds, args.max_stations, args.trend,
        )
    print("\nKriging LLOCV finished.")


if __name__ == "__main__":
    main()
