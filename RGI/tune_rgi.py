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
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from rgi_data import hyper_source
from paths import get_rgi_tuned_params_path
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res, variables_from_cfg, cap_timestamps
from shared.splits.evaluation_sets import (
    apply_evaluation_protocol, load_early_stop_times, early_stop_kind,
)
from rgi_core import RGI, RGIConfig
from rgi_data import attach_extras, compute_metrics, extra_cols_for_var, load_config, load_panel, subset_times
from llocv_rgi import cfg_to_rgi, station_folds


def score_combo(panel, var, rcfg: RGIConfig, n_folds: int,
                champion_rmse: float | None = None) -> dict:
    """Fit on full pool × train; score fixed tune-targets on tune quarters.
    Donors exclude tune-targets only at prediction time.

    champion_rmse: after 40 % of epochs, abort if the original-unit probe
    is still > 20 % above the champion and flat over the last 10 probes.
    Returns {rmse, early_stopped, stop_reason, score_at_stop, n_units_scored}.
    """
    def _ret(rmse, stopped=False, reason="full", at=None, n=0):
        return {
            "rmse": float(rmse) if np.isfinite(rmse) else np.inf,
            "early_stopped": bool(stopped),
            "stop_reason": reason,
            "score_at_stop": float(at) if at is not None and np.isfinite(at) else np.nan,
            "n_units_scored": int(n),
        }
    if "is_tune_target" not in panel.columns or not panel["is_tune_target"].any():
        return _ret(np.inf, reason="no_targets")
    train = panel[(~panel["is_heldout"]) & (panel["split"] == "train")]
    hold = panel[(~panel["is_heldout"]) & panel["is_tune_target"]
                 & (panel["split"] == "tune")]
    if hold.empty or train["station_name"].nunique() < rcfg.min_stations:
        return _ret(np.inf, reason="insufficient")

    def _probe(model) -> float:
        """Original-unit RMSE of the current net on the hold set."""
        was_training = model.net.training
        model.net.eval()
        obs, pred = [], []
        try:
            for t, ho_t in hold.groupby("time", sort=False):
                others = panel[(panel["time"] == t) & (~panel["is_heldout"])
                               & (~panel["is_tune_target"])]
                if len(others) < rcfg.min_stations:
                    continue
                yhat = model.predict_frame(others, ho_t, var)
                obs.append(ho_t[var].to_numpy())
                pred.append(np.asarray(yhat, dtype=float))
        finally:
            model.net.train(was_training)
        if not obs:
            return np.inf
        return float(compute_metrics(np.concatenate(obs), np.concatenate(pred))["rmse"])

    model = RGI(rcfg)
    probe_hist: list[float] = []
    min_epoch = int(0.40 * rcfg.epochs)

    def on_epoch(m, epoch, _va):
        if champion_rmse is None or epoch < min_epoch:
            return False
        rmse = _probe(m)
        probe_hist.append(rmse)
        if (len(probe_hist) >= 10
                and rmse > champion_rmse * 1.20
                and probe_hist[-1] >= probe_hist[-10]):
            return True
        return False

    info = model.fit(train, var, val_df=None, log=None, on_epoch=on_epoch)
    if info.get("early_stopped"):
        at = probe_hist[-1] if probe_hist else np.inf
        return _ret(at, stopped=True, reason="stalled_margin",
                    at=at, n=info.get("epochs", 0))
    full = _probe(model)
    return _ret(full, n=info.get("epochs", rcfg.epochs), at=full)


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
    src = hyper_source(cfg, var)
    if src != var:
        parent = get_rgi_tuned_params_path(src, cfg["time_resolution"])
        child = get_rgi_tuned_params_path(var, cfg["time_resolution"])
        if parent.exists():
            child.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(parent, child)
            print(f"[{var}] inherited hypers from {src}")
        else:
            print(f"[{var}] parent {src} has no tuned file yet — skipping")
        return
    # Fresh trial log per variable (was previously accumulating across variables).
    cfg["_trials"] = []
    panel = attach_extras(load_panel(cfg, var), cfg, var)
    tagged, _, _, _ = apply_evaluation_protocol(panel, variable=var)
    # Full pool on train+tune; held-out dropped. Fixed tune-targets used for scoring.
    panel = tagged[(~tagged["is_heldout"]) & (tagged["split"].isin(["train", "tune"]))].copy()
    if cfg.get("_quick_months"):
        panel = subset_times(panel, cfg["_quick_months"])
    panel, _hours = apply_hour_cut(panel, cfg["time_resolution"], cfg.get("_hours"))
    # Early-stop protocol: score only the eval set (max 52). Training stays full.
    time_res = cfg["time_resolution"]
    kind = early_stop_kind("RGI")
    eval_raw = load_early_stop_times(time_res, kind, which="eval")
    if eval_raw:
        pt = pd.to_datetime(panel["time"], utc=True)
        want = set(str(w) for w in eval_raw)
        if time_res == "monthly":
            key = pt.dt.strftime("%Y-%m")
        elif time_res == "weekly":
            key = pt.dt.strftime("%Y-W%W")
        elif time_res == "daily":
            key = pt.dt.strftime("%Y-%m-%d")
        else:
            key = pt.dt.strftime("%Y-%m-%dT%H:%M:%S")
        # keep all train rows; restrict tune rows to the eval set
        is_tune = panel["split"] == "tune"
        keep = (~is_tune) | key.isin(want)
        if keep.any():
            panel = panel.loc[keep].copy()
            print(f"  early-stop eval tune times="
                  f"{panel.loc[panel['split']=='tune','time'].nunique()}", flush=True)
    elif cfg.get("_max_times"):
        panel = cap_timestamps(panel, cfg["_max_times"], time_res,
                               seed=int(cfg.get("rgi", {}).get("seed", 22)))
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
        champion = None
        for k, L in combos:
            rcfg = cfg_to_rgi(cfg, {"k": k, "n_layers": L, "alpha": 0.2, "alpha_z": 10.0})
            t0 = time.perf_counter()
            res = score_combo(panel, var, rcfg, n_folds, champion_rmse=champion)
            wall = time.perf_counter() - t0
            flag = " STOP" if res["early_stopped"] else ""
            print(f"  k={k:2d} L={L}  DEV RMSE={res['rmse']:.4f}  ({wall:.0f}s){flag}")
            cfg["_trials"].append({
                "phase": 1,
                "k": int(k),
                "n_layers": int(L),
                "alpha": 0.2,
                "alpha_z": 10.0,
                **res,
                "wall_s": round(wall, 1),
            })
            if not res["early_stopped"] and res["rmse"] < best[0]:
                best = (res["rmse"], {"k": int(k), "n_layers": int(L)})
                champion = res["rmse"]
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
        champion = None
        for a, az in combos:
            over = {**phase1[1], "alpha": float(a), "alpha_z": float(az)}
            rcfg = cfg_to_rgi(cfg, over)
            t0 = time.perf_counter()
            res = score_combo(panel, var, rcfg, n_folds, champion_rmse=champion)
            wall = time.perf_counter() - t0
            flag = " STOP" if res["early_stopped"] else ""
            print(f"  alpha={a:.2f} az={az:4.0f}  DEV RMSE={res['rmse']:.4f}  "
                  f"({wall:.0f}s){flag}")
            cfg["_trials"].append({
                "phase": 2,
                "k": int(over["k"]),
                "n_layers": int(over["n_layers"]),
                "alpha": float(a),
                "alpha_z": float(az),
                **res,
                "wall_s": round(wall, 1),
            })
            if not res["early_stopped"] and res["rmse"] < best[0]:
                best = (res["rmse"], over)
                champion = res["rmse"]
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
        az_grid = expand.get("alpha_z", [50.0, 200.0, 500.0])
        combos = list(itertools.product(a_grid, az_grid))
        print(f"\n{var} phase 3a  {len(combos)} combos  (alpha × alpha_z beyond corner)  "
              f"k={locked['k']} L={locked['n_layers']}")
        best = (np.inf, dict(locked))
        champion = None
        for a, az in combos:
            over = {**locked, "alpha": float(a), "alpha_z": float(az)}
            rcfg = cfg_to_rgi(cfg, over)
            t0 = time.perf_counter()
            res = score_combo(panel, var, rcfg, n_folds, champion_rmse=champion)
            flag = " STOP" if res["early_stopped"] else ""
            print(f"  alpha={a:.2f} az={az:4.0f}  DEV RMSE={res['rmse']:.4f}  "
                  f"({time.perf_counter()-t0:.0f}s){flag}")
            if not res["early_stopped"] and res["rmse"] < best[0]:
                best = (res["rmse"], over)
                champion = res["rmse"]
        k_grid = expand.get("k", [8, 16, 32, 64])
        print(f"\n{var} phase 3b  {len(k_grid)} combos  (re-search k)  "
              f"alpha={best[1]['alpha']:.2f} az={best[1]['alpha_z']:.0f} L={best[1]['n_layers']}")
        locked_al = dict(best[1])
        champion = best[0] if np.isfinite(best[0]) else None
        for k in k_grid:
            over = {**locked_al, "k": int(k)}
            rcfg = cfg_to_rgi(cfg, over)
            t0 = time.perf_counter()
            res = score_combo(panel, var, rcfg, n_folds, champion_rmse=champion)
            flag = " STOP" if res["early_stopped"] else ""
            print(f"  k={k:2d}  DEV RMSE={res['rmse']:.4f}  ({time.perf_counter()-t0:.0f}s){flag}")
            if not res["early_stopped"] and res["rmse"] < best[0]:
                best = (res["rmse"], over)
                champion = res["rmse"]

    out = {
        **best[1],
        "dev_rmse": None if not np.isfinite(best[0]) else float(best[0]),
        "variable": var,
        "time_resolution": cfg["time_resolution"],
        "n_folds": n_folds,
    }
    _write_tuned(var, cfg["time_resolution"], out)
    if cfg.get("_trials"):
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
