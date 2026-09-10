#!/usr/bin/env python3
"""
Staged DEV search for Kriging.

Phase 1: k (then k_indicator for precip)
Phase 2: variogram family
Phase 3: theta × aniso_ratio × alpha_z
Phase 4: interaction set
Phase 5: expand alpha_z past the phase-3 corner, then re-search k

Score: 5-fold station CV. Fit on TRAIN times of in-fold stations,
RMSE on held-out stations at DEV times. Same-timestamp donors.
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

from paths import get_kriging_tuned_params_path
from shared.time_res import add_time_res_arg, apply_time_res
from kriging_core import KrigingConfig, KrigingInterpolator
from kriging_data import (
    attach_pack_terrain,
    attach_temperature,
    compute_block,
    compute_metrics,
    is_precip,
    load_config,
    load_panel,
    pack_from_master,
    subset_times,
)
from llocv_kriging import cfg_to_kriging, station_folds


def score_combo(panel, var, kcfg: KrigingConfig, n_folds: int, cfg, pack=None) -> float:
    train = panel[panel["split"] == "train"]
    dev = panel[panel["split"] == "dev"]
    names = sorted(dev["station_name"].unique())
    folds = station_folds(names, n_folds, kcfg.seed)
    obs, pred = [], []
    for hold_names in folds:
        hold_set = set(hold_names)
        tr = train[~train["station_name"].isin(hold_set)]
        ho = dev[dev["station_name"].isin(hold_set)]
        if ho.empty or tr["station_name"].nunique() < kcfg.min_stations:
            continue
        model = KrigingInterpolator(kcfg)
        model.pack = pack
        model.fit(tr, var, cfg["time_resolution"], cfg)
        for t, ho_t in ho.groupby("time", sort=False):
            others = panel[(panel["time"] == t) & (~panel["station_name"].isin(hold_set))]
            if len(others) < kcfg.min_stations:
                continue
            yhat = model.predict_frame(others, ho_t, var)
            obs.append(ho_t[var].to_numpy())
            pred.append(yhat)
    if not obs:
        return np.inf
    met = compute_metrics(np.concatenate(obs), np.concatenate(pred))
    return float(met["rmse"])


def _load_tuned(var, time_res) -> dict:
    path = get_kriging_tuned_params_path(var, time_res)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_tuned(var, time_res, payload: dict) -> Path:
    path = get_kriging_tuned_params_path(var, time_res)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"  wrote {path}")
    return path


def _keys_from(best_over: dict, base: KrigingConfig) -> dict:
    return {
        "k": int(best_over.get("k", base.k)),
        "k_indicator": int(best_over.get("k_indicator", base.k_indicator)),
        "family": str(best_over.get("family", base.family)),
        "theta_deg": float(best_over.get("theta_deg", base.theta_deg)),
        "aniso_ratio": float(best_over.get("aniso_ratio", base.aniso_ratio)),
        "alpha_z": float(best_over.get("alpha_z", base.alpha_z)),
        "interactions": str(best_over.get("interactions", base.interactions)),
        "trend": str(best_over.get("trend", base.trend)),
    }


def _eval_grid(panel, var, cfg, n_folds, locked: dict, varying: list[dict], label: str, pack=None):
    print(f"\n{var} {label}  {len(varying)} combos")
    best = (np.inf, dict(locked))
    for extra in varying:
        over = {**locked, **extra}
        kcfg = cfg_to_kriging(cfg, over)
        t0 = time.perf_counter()
        rmse = score_combo(panel, var, kcfg, n_folds, cfg, pack=pack)
        pretty = " ".join(f"{k}={v}" for k, v in extra.items())
        print(f"  {pretty}  DEV RMSE={rmse:.4f}  ({time.perf_counter()-t0:.0f}s)")
        if rmse < best[0]:
            best = (rmse, over)
    return best


def run_variable(cfg, var, phase: str, months=None, n_folds=None, pack=None, rh_t_mode=None):
    panel = load_panel(cfg, var)
    if months:
        panel = subset_times(panel, months)
    if str(var).lower().startswith("rh") and (rh_t_mode or cfg.get("kriging", {}).get("rh_t_mode", "none")) in ("predicted", "observed"):
        panel = attach_temperature(panel, cfg)
    panel = attach_pack_terrain(panel, pack)
    kblock = cfg.get("kriging", {})
    search = kblock.get("search", {})
    n_folds = int(n_folds or kblock.get("n_folds", 5))
    base = cfg_to_kriging(cfg)
    existing = _load_tuned(var, cfg["time_resolution"])
    locked = _keys_from(existing, base) if existing else _keys_from({}, base)
    # phase 0 freeze: linear main+interactions from config, isotropic exponential
    if not existing and phase in ("1", "all"):
        locked.update({
            "family": "exponential",
            "theta_deg": 0.0,
            "aniso_ratio": 1.0,
            "alpha_z": 0.0,
            "interactions": "both",
            "trend": "linear",
        })
    best = (existing.get("dev_rmse", np.inf), locked)

    if phase in ("1", "all"):
        grid = [{"k": int(k), "k_indicator": int(k)} for k in search.get("k", [base.k])]
        best = _eval_grid(panel, var, cfg, n_folds, locked, grid, pack=pack, label= "phase 1a  k")
        locked = dict(best[1])
        if is_precip(var):
            grid = [{"k_indicator": int(k)} for k in search.get("k", [base.k])]
            best = _eval_grid(panel, var, cfg, n_folds, locked, grid, pack=pack, label= "phase 1b  k_indicator")
            locked = dict(best[1])

    if phase in ("2", "all"):
        grid = [{"family": str(f)} for f in search.get("family", [base.family])]
        best = _eval_grid(panel, var, cfg, n_folds, locked, grid, pack=pack, label= "phase 2  family")
        locked = dict(best[1])

    if phase in ("3", "all"):
        grid = [
            {"theta_deg": float(th), "aniso_ratio": float(r), "alpha_z": float(az)}
            for th, r, az in itertools.product(
                search.get("theta_deg", [0.0]),
                search.get("aniso_ratio", [1.0]),
                search.get("alpha_z", [0.0]),
            )
        ]
        best = _eval_grid(panel, var, cfg, n_folds, locked, grid, pack=pack, label= "phase 3  metric")
        locked = dict(best[1])

    if phase in ("4", "all"):
        grid = [{"interactions": str(s)} for s in search.get("interactions", [base.interactions])]
        best = _eval_grid(panel, var, cfg, n_folds, locked, grid, pack=pack, label= "phase 4  interactions")
        locked = dict(best[1])

    if phase in ("5", "all"):
        expand = kblock.get("search_expand", {})
        az_grid = [float(v) for v in expand.get("alpha_z", search.get("alpha_z", [locked["alpha_z"]]))]
        if float(locked["alpha_z"]) not in az_grid:
            az_grid.append(float(locked["alpha_z"]))
        az_grid = sorted(set(az_grid))
        grid = [{"alpha_z": az} for az in az_grid]
        best = _eval_grid(panel, var, cfg, n_folds, locked, grid, pack=pack, label= "phase 5a  alpha_z expand")
        locked = dict(best[1])
        k_grid = [int(v) for v in expand.get("k", search.get("k", [locked["k"]]))]
        grid = [{"k": k, "k_indicator": k} for k in k_grid]
        best = _eval_grid(panel, var, cfg, n_folds, locked, grid, pack=pack, label= "phase 5b  k re-search")
        locked = dict(best[1])
        if is_precip(var):
            grid = [{"k_indicator": k} for k in k_grid]
            best = _eval_grid(panel, var, cfg, n_folds, locked, grid, pack=pack, label= "phase 5c  k_indicator")
            locked = dict(best[1])

    out = {
        **locked,
        "dev_rmse": None if not np.isfinite(best[0]) else float(best[0]),
        "variable": var,
        "time_resolution": cfg["time_resolution"],
        "n_folds": n_folds,
        "phase": phase,
        "rh_t_mode": rh_t_mode or kblock.get("rh_t_mode", "none"),
        "tune_months": months or "all",
    }
    _write_tuned(var, cfg["time_resolution"], out)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Staged DEV search for Kriging")
    p.add_argument("--variables", nargs="*", default=None)
    p.add_argument("--phase", choices=["1", "2", "3", "4", "5", "all"], default=None)
    p.add_argument("--months", default=None)
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--rh-t-mode", default=None, choices=["none", "predicted", "observed"])
    add_time_res_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    apply_time_res(cfg, args)
    wanted = args.variables or cfg.get("kriging", {}).get("variables_to_process") or [
        "temp_mean",
    ]
    comp = compute_block(cfg)
    phase = args.phase or ("1" if args.quick else comp.get("tune_phase", "1"))
    months = args.months or ("seasonal4" if args.quick else comp.get("tune_months", "seasonal4"))
    n_folds = args.folds or (2 if args.quick else int(comp.get("tune_folds", 5)))
    pack = pack_from_master(cfg, int(cfg.get("kriging", {}).get("n_regions", 6)))
    print("=" * 72)
    print("Kriging DEV search")
    print(f"  phase={phase}  months={months}  folds={n_folds}  pack={'yes' if pack else 'no'}")
    print(f"  time_resolution={cfg['time_resolution']}")
    print("=" * 72)
    for var in wanted:
        run_variable(cfg, var, phase, months=months, n_folds=n_folds, pack=pack, rh_t_mode=args.rh_t_mode)
    print("\nKriging tune finished.")


if __name__ == "__main__":
    main()
