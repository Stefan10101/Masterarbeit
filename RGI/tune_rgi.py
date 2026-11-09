#!/usr/bin/env python3
"""
RGI hyperparameter search.

Phase 1: k × n_layers with alpha=0.2 and a scaled alpha_z.
Phase 2: alpha × alpha_z at the winning k, n_layers.
Phase 3: expand alpha and alpha_z past the phase-2 corner, then
         re-search k at the new (alpha, alpha_z).

Protocol (pooled): fit on full pool × train quarters;
score fixed tune-targets on tune quarters (donors exclude them only at prediction).
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
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res, variables_from_cfg
from shared.splits.evaluation_sets import apply_evaluation_protocol
from rgi_core import RGI, RGIConfig
from rgi_data import attach_extras, compute_metrics, extra_cols_for_var, load_config, load_panel, subset_times
from llocv_rgi import cfg_to_rgi, station_folds


def score_combo(panel, var, rcfg: RGIConfig, n_folds: int) -> float:
    """Fit on full pool × train; score fixed tune-targets on tune quarters.
    Donors exclude tune-targets only at prediction time."""
    if "is_tune_target" not in panel.columns or not panel["is_tune_target"].any():
        return np.inf
    train = panel[(~panel["is_heldout"]) & (panel["split"] == "train")]
    hold = panel[(~panel["is_heldout"]) & panel["is_tune_target"]
                 & (panel["split"] == "tune")]
    if hold.empty or train["station_name"].nunique() < rcfg.min_stations:
        return np.inf
    model = RGI(rcfg)
    model.fit(train, var, val_df=None, log=None)
    obs, pred = [], []
    for t, ho_t in hold.groupby("time", sort=False):
        # Temporary donors: exclude the tune-targets
        others = panel[(panel["time"] == t) & (~panel["is_heldout"])
                       & (~panel["is_tune_target"])]
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
    # Fresh trial log per variable (was previously accumulating across variables).
    cfg["_trials"] = []
    panel = attach_extras(load_panel(cfg, var), cfg, var)
    tagged, _, _, _ = apply_evaluation_protocol(panel, variable=var)
    # Full pool on train+tune; held-out dropped. Fixed tune-targets used for scoring.
    panel = tagged[(~tagged["is_heldout"]) & (tagged["split"].isin(["train", "tune"]))].copy()
    if cfg.get("_quick_months"):
        panel = subset_times(panel, cfg["_quick_months"])
    if cfg.get("_max_times") and panel["time"].nunique() > cfg["_max_times"]:
        rng = np.random.default_rng(int(cfg.get("rgi", {}).get("seed", 22)))
        ts = panel["time"].drop_duplicates().to_numpy()
        keep = rng.choice(ts, size=cfg["_max_times"], replace=False)
        panel = panel[panel["time"].isin(keep)].copy()
    panel, _hours = apply_hour_cut(panel, cfg["time_resolution"], cfg.get("_hours"))
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
            wall = time.perf_counter() - t0
            print(f"  k={k:2d} L={L}  DEV RMSE={rmse:.4f}  ({wall:.0f}s)")
            cfg["_trials"].append({
                "phase": 1,
                "k": int(k),
                "n_layers": int(L),
                "alpha": 0.2,
                "alpha_z": 10.0,
                "rmse": float(rmse),
                "wall_s": round(wall, 1),
            })
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
            wall = time.perf_counter() - t0
            print(f"  alpha={a:.2f} az={az:4.0f}  DEV RMSE={rmse:.4f}  "
                  f"({wall:.0f}s)")
            cfg["_trials"].append({
                "phase": 2,
                "k": int(over["k"]),
                "n_layers": int(over["n_layers"]),
                "alpha": float(a),
                "alpha_z": float(az),
                "rmse": float(rmse),
                "wall_s": round(wall, 1),
            })
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
    if cfg.get("_trials"):
        import pandas as pd
        sp = get_rgi_tuned_params_path(var, cfg["time_resolution"]).with_name(f"{var}_search.csv")
        pd.DataFrame(cfg["_trials"]).to_csv(sp, index=False)
        print(f"  wrote {sp}  ({len(cfg['_trials'])} trials)")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="DEV search for RGI")
    p.add_argument("--variables", nargs="*", default=None)
    p.add_argument("--phase", choices=["1", "2", "3", "all"], default="all")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--months", default=None)
    p.add_argument("--max-times", type=int, default=400,
                   help="cap timestamps for a proof run")
    add_hours_arg(p)
    add_time_res_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    apply_time_res(cfg, args)
    cfg["_hours"] = args.hours
    cfg["_max_times"] = args.max_times
    cfg["_trials"] = []
    wanted = variables_from_cfg(cfg, args.variables)
    from rgi_core import describe_device, resolve_device
    print("=" * 72)
    print("RGI hyperparameter search")
    print(f"  phase={args.phase}  time_resolution={cfg['time_resolution']}")
    print(f"  protocol: fit train × pool, score fixed tune-targets × tune")
    print(f"  device {describe_device(resolve_device(cfg.get('rgi', {}).get('device', 'auto')))}")
    print("=" * 72)
    if args.quick:
        cfg.setdefault("rgi", {})
        cfg["rgi"]["epochs"] = min(int(cfg["rgi"].get("epochs", 80)), 15)
        cfg["rgi"]["patience"] = min(int(cfg["rgi"].get("patience", 12)), 4)
        cfg["rgi"]["n_folds"] = args.folds or 2
        cfg["rgi"]["search"] = {
            "k": [int(cfg["rgi"].get("k", 10))],
            "n_layers": [int(cfg["rgi"].get("n_layers", 3))],
            "alpha": [float(cfg["rgi"].get("alpha", 0.2))],
            "alpha_z": [float(cfg["rgi"].get("alpha_z", 10.0))],
        }
        cfg["_quick_months"] = args.months or "seasonal4"
        args.phase = "1"
    for var in wanted:
        run_variable(cfg, var, args.phase)
    print("\nRGI tune finished.")


if __name__ == "__main__":
    main()
