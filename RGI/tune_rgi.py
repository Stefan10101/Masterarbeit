#!/usr/bin/env python3
"""
DEV search for RGI hyperparameters.

Phase 1: k × n_layers with alpha=0.2 and a scaled alpha_z.
Phase 2: alpha × alpha_z at the winning k, n_layers.
Phase 3: expand alpha and alpha_z past the phase-2 corner, then
         re-search k at the new (alpha, alpha_z). Depth stays at
         the phase-1 winner (deeper did not help).

Score: 5-fold station CV, fit on TRAIN times, RMSE on held-out stations
at DEV times.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import itertools
import sys
import time

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_rgi_tuned_params_path
from rgi_core import RGI, RGIConfig
from rgi_data import compute_metrics, load_config, load_panel
from llocv_rgi import cfg_to_rgi, station_folds


def score_combo(panel, var, rcfg: RGIConfig, n_folds: int) -> float:
    train = panel[panel["split"] == "train"]
    dev = panel[panel["split"] == "dev"]
    names = sorted(dev["station_name"].unique())
    folds = station_folds(names, n_folds, rcfg.seed)
    obs, pred = [], []
    for hold_names in folds:
        hold_set = set(hold_names)
        tr = train[~train["station_name"].isin(hold_set)]
        ho = dev[dev["station_name"].isin(hold_set)]
        if ho.empty or tr["station_name"].nunique() < rcfg.min_stations:
            continue
        model = RGI(rcfg)
        model.fit(tr, var, val_df=None, log=None)
        # Neighbours are contemporaneous other stations (DEV year ≠ TRAIN year).
        for t, ho_t in ho.groupby("time", sort=False):
            others = panel[(panel["time"] == t) & (~panel["station_name"].isin(hold_set))]
            if len(others) < rcfg.min_stations:
                continue
            yhat = model.predict_frame(others, ho_t, var)
            obs.append(ho_t[var].to_numpy())
            pred.append(yhat)
    if not obs:
        return np.inf
    met = compute_metrics(np.concatenate(obs), np.concatenate(pred))
    return float(met["rmse"])


def _load_tuned(var, time_res) -> dict:
    path = get_rgi_tuned_params_path(var, time_res)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_tuned(var, time_res, payload: dict) -> Path:
    path = get_rgi_tuned_params_path(var, time_res)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"  wrote {path}")
    return path


def run_variable(cfg, var, phase: str):
    panel = load_panel(cfg, var)
    rgi_cfg = cfg.get("rgi", {})
    search = rgi_cfg.get("search", {})
    expand = rgi_cfg.get("search_expand", {})
    n_folds = int(rgi_cfg.get("n_folds", 5))
    base = cfg_to_rgi(cfg)
    existing = _load_tuned(var, cfg["time_resolution"])

    if phase in ("1", "all"):
        combos = list(itertools.product(
            search.get("k", [base.k]),
            search.get("n_layers", [base.n_layers]),
        ))
        print(f"\n{var} phase 1  {len(combos)} combos  (k × layers)")
        best = (np.inf, {"k": base.k, "n_layers": base.n_layers})
        for k, L in combos:
            rcfg = cfg_to_rgi(cfg, {"k": k, "n_layers": L, "alpha": 0.2, "alpha_z": 10.0})
            t0 = time.perf_counter()
            rmse = score_combo(panel, var, rcfg, n_folds)
            print(f"  k={k:2d} L={L}  DEV RMSE={rmse:.4f}  ({time.perf_counter()-t0:.0f}s)")
            if rmse < best[0]:
                best = (rmse, {"k": int(k), "n_layers": int(L)})
        phase1 = best
    else:
        phase1 = (existing.get("dev_rmse", np.nan), {
            "k": int(existing.get("k", base.k)),
            "n_layers": int(existing.get("n_layers", base.n_layers)),
            "alpha": float(existing.get("alpha", base.alpha)),
            "alpha_z": float(existing.get("alpha_z", base.alpha_z)),
        })

    if phase in ("2", "all"):
        combos = list(itertools.product(
            search.get("alpha", [base.alpha]),
            search.get("alpha_z", [base.alpha_z]),
        ))
        print(f"\n{var} phase 2  {len(combos)} combos  (alpha × alpha_z)  "
              f"k={phase1[1]['k']} L={phase1[1]['n_layers']}")
        best = (np.inf, dict(phase1[1]))
        for a, az in combos:
            over = {**phase1[1], "alpha": float(a), "alpha_z": float(az)}
            rcfg = cfg_to_rgi(cfg, over)
            t0 = time.perf_counter()
            rmse = score_combo(panel, var, rcfg, n_folds)
            print(f"  alpha={a:.2f} az={az:4.0f}  DEV RMSE={rmse:.4f}  "
                  f"({time.perf_counter()-t0:.0f}s)")
            if rmse < best[0]:
                best = (rmse, over)
    else:
        best = phase1

    if phase == "3":
        locked = {
            "k": int(existing.get("k", best[1].get("k", base.k))),
            "n_layers": int(existing.get("n_layers", best[1].get("n_layers", 2))),
            "alpha": float(existing.get("alpha", best[1].get("alpha", base.alpha))),
            "alpha_z": float(existing.get("alpha_z", best[1].get("alpha_z", base.alpha_z))),
        }
        a_grid = expand.get("alpha", [0.3, 0.4, 0.5, 0.6])
        az_grid = expand.get("alpha_z", [50.0, 100.0, 150.0, 200.0])
        combos = list(itertools.product(a_grid, az_grid))
        print(f"\n{var} phase 3a  {len(combos)} combos  (alpha × alpha_z beyond corner)  "
              f"k={locked['k']} L={locked['n_layers']}")
        best = (np.inf, dict(locked))
        for a, az in combos:
            over = {**locked, "alpha": float(a), "alpha_z": float(az)}
            rcfg = cfg_to_rgi(cfg, over)
            t0 = time.perf_counter()
            rmse = score_combo(panel, var, rcfg, n_folds)
            print(f"  alpha={a:.2f} az={az:4.0f}  DEV RMSE={rmse:.4f}  "
                  f"({time.perf_counter()-t0:.0f}s)")
            if rmse < best[0]:
                best = (rmse, over)
        k_grid = expand.get("k", [5, 10, 15, 20])
        print(f"\n{var} phase 3b  {len(k_grid)} combos  (re-search k)  "
              f"alpha={best[1]['alpha']:.2f} az={best[1]['alpha_z']:.0f} L={best[1]['n_layers']}")
        locked_al = dict(best[1])
        for k in k_grid:
            over = {**locked_al, "k": int(k)}
            rcfg = cfg_to_rgi(cfg, over)
            t0 = time.perf_counter()
            rmse = score_combo(panel, var, rcfg, n_folds)
            print(f"  k={k:2d}  DEV RMSE={rmse:.4f}  ({time.perf_counter()-t0:.0f}s)")
            if rmse < best[0]:
                best = (rmse, over)

    out = {
        **best[1],
        "dev_rmse": None if not np.isfinite(best[0]) else float(best[0]),
        "variable": var,
        "time_resolution": cfg["time_resolution"],
        "n_folds": n_folds,
    }
    _write_tuned(var, cfg["time_resolution"], out)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="DEV search for RGI")
    p.add_argument("--variables", nargs="*", default=None)
    p.add_argument("--phase", choices=["1", "2", "3", "all"], default="all")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    wanted = args.variables or cfg.get("rgi", {}).get("variables_to_process") or [
        "temp_mean",
    ]
    from rgi_core import describe_device, resolve_device
    print("=" * 72)
    print("RGI DEV search")
    print(f"  phase={args.phase}  time_resolution={cfg['time_resolution']}")
    print(f"  device {describe_device(resolve_device(cfg.get('rgi', {}).get('device', 'auto')))}")
    print("=" * 72)
    for var in wanted:
        run_variable(cfg, var, args.phase)
    print("\nRGI tune finished.")


if __name__ == "__main__":
    main()
