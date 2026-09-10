#!/usr/bin/env python3
"""
Station-fold LLOCV for RGI.

Default: 5 station folds. Each fold trains on the other stations using
--fit times (default train) and scores the held-out stations on --score
times (default all). Rows tagged train/dev/test.

--mode leave_one matches RFSI nested LLOCV (one model per station).
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_nested_llocv_path, get_rgi_tuned_params_path
from shared.time_res import add_time_res_arg, apply_time_res
from rgi_core import RGI, RGIConfig
from rgi_data import attach_extras, extra_cols_for_var, load_config, load_panel, print_split_metrics
import yaml


def parse_split_arg(text: str) -> set[str]:
    text = text.strip().lower()
    if text == "all":
        return {"all"}
    names = {s.strip() for s in text.split(",") if s.strip()}
    bad = names - {"train", "dev", "test"}
    if bad:
        raise ValueError(f"unknown splits {bad}")
    return names


def cfg_to_rgi(cfg, overrides: dict | None = None) -> RGIConfig:
    r = dict(cfg.get("rgi", {}))
    tuned = overrides or {}
    return RGIConfig(
        k=int(tuned.get("k", r.get("k", 10))),
        n_layers=int(tuned.get("n_layers", r.get("n_layers", 3))),
        alpha=float(tuned.get("alpha", r.get("alpha", 0.2))),
        alpha_z=float(tuned.get("alpha_z", r.get("alpha_z", 10.0))),
        hidden_dim=int(r.get("hidden_dim", 64)),
        dropout=float(r.get("dropout", 0.1)),
        clc_embed_dim=int(r.get("clc_embed_dim", 8)),
        epochs=int(r.get("epochs", 80)),
        patience=int(r.get("patience", 12)),
        lr=float(r.get("lr", 1e-3)),
        weight_decay=float(r.get("weight_decay", 1e-4)),
        train_mask_frac=float(r.get("train_mask_frac", 0.25)),
        seed=int(r.get("seed", 22)),
        min_stations=int(cfg.get("min_stations_per_field", 10)),
        device=str(r.get("device", "auto")),
        use_amp=bool(r.get("use_amp", True)),
        two_step=bool(tuned.get("two_step", r.get("two_step", True))),
        tau_wet=float(r.get("tau_wet", 0.5)),
        extra_cols=tuple(tuned.get("extra_cols") or r.get("extra_cols") or ()),
        var_name=str(tuned.get("variable") or ""),
    )


def load_tuned(var: str, time_res: str) -> dict:
    path = get_rgi_tuned_params_path(var, time_res)
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


def run_variable(cfg, var, fit_splits, score_splits, mode, n_folds, max_stations):
    panel = attach_extras(load_panel(cfg, var), cfg, var)
    if cfg.get("_quick_months"):
        from Kriging.kriging_data import subset_times
        panel = subset_times(panel, cfg["_quick_months"])
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
        folds = station_folds(names, n_folds, int(cfg.get("rgi", {}).get("seed", 22)))

    tuned = load_tuned(var, cfg["time_resolution"])
    tuned["extra_cols"] = extra_cols_for_var(var, cfg)
    tuned["variable"] = var
    rcfg = cfg_to_rgi(cfg, tuned)
    rcfg.two_step = bool(cfg.get("rgi", {}).get("two_step", True)) and (
        var.lower().startswith("precip") or var.lower().startswith("snow")
    )
    out_path = get_nested_llocv_path("RGI", var, cfg["time_resolution"], "full")
    records = []
    t0 = time.perf_counter()
    print(f"\n{'=' * 60}\nVARIABLE {var}")
    print(f"  fit rows {len(fit_df):,}  score rows {len(score_df):,}")
    print(f"  folds={len(folds)}  k={rcfg.k}  layers={rcfg.n_layers}  "
          f"alpha={rcfg.alpha}  alpha_z={rcfg.alpha_z}")

    for fi, hold_names in enumerate(folds, start=1):
        hold_set = set(hold_names)
        train = fit_df[~fit_df["station_name"].isin(hold_set)]
        hold = score_df[score_df["station_name"].isin(hold_set)]
        if hold.empty or train["station_name"].nunique() < rcfg.min_stations:
            continue
        val = train[train["split"] == "dev"] if "dev" in train["split"].values else None
        model = RGI(rcfg)
        model.fit(train, var, val_df=val if val is not None and len(val) else None)
        # Same-timestamp senders, excluding the held-out stations.
        for t, hold_t in hold.groupby("time", sort=False):
            others = panel[(panel["time"] == t) & (~panel["station_name"].isin(hold_set))]
            if len(others) < rcfg.min_stations:
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
    p = argparse.ArgumentParser(description="Station-fold LLOCV for RGI")
    p.add_argument("--variables", nargs="*", default=None)
    p.add_argument("--fit", default="train")
    p.add_argument("--score", default="all")
    p.add_argument("--mode", choices=["folds", "leave_one"], default="folds")
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--max-stations", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--months", default=None)
    add_time_res_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    apply_time_res(cfg, args)
    wanted = args.variables or cfg.get("rgi", {}).get("variables_to_process") or [
        "temp_mean", "precip_sum",
    ]
    n_folds = args.folds or int(cfg.get("rgi", {}).get("n_folds", 5))
    if args.quick:
        n_folds = args.folds or 2
        cfg.setdefault("rgi", {})
        cfg["rgi"]["epochs"] = min(int(cfg["rgi"].get("epochs", 80)), 15)
        cfg["rgi"]["patience"] = min(int(cfg["rgi"].get("patience", 12)), 4)
        if args.months or True:
            from Kriging.kriging_data import subset_times
            # applied inside run after load — store on cfg
            cfg["_quick_months"] = args.months or "seasonal4"
    from rgi_core import describe_device, resolve_device
    print("=" * 72)
    print("RGI station LLOCV")
    print(f"  device {describe_device(resolve_device(cfg.get('rgi', {}).get('device', 'auto')))}")
    print(f"  time_resolution={cfg['time_resolution']}")
    print(f"  mode={args.mode}  folds={n_folds}")
    print(f"  fit={args.fit}  score={args.score}")
    print("=" * 72)
    for var in wanted:
        run_variable(
            cfg, var,
            parse_split_arg(args.fit),
            parse_split_arg(args.score),
            args.mode, n_folds, args.max_stations,
        )
    print("\nRGI LLOCV finished.")


if __name__ == "__main__":
    main()
