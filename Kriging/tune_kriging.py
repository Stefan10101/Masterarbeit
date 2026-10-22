#!/usr/bin/env python3
"""
Staged DEV search for Kriging.

More than 3 interacting hyperparameters, so the search stays phased:
  1 k, 2 family, 3 metric (theta, aniso, alpha_z), 4 interactions, 5 refine.

Score: pool stations at tune quarters. A seeded 20% of the pool is the
tune target; the rest are donors. The official held-out set is never used.
Fit on pool x train quarters.
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
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res
from kriging_core import KrigingConfig, KrigingInterpolator
from shared.splits.evaluation_sets import prepare_for_tuning
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


def _tune_split(panel, seed=22, frac=0.2):
    """Seeded subset of pool stations used as the tune score targets."""
    pool = panel.loc[~panel["is_heldout"], "station_name"].drop_duplicates()
    rng = np.random.default_rng(seed)
    names = pool.to_numpy()
    rng.shuffle(names)
    n = max(1, int(round(len(names) * frac)))
    return set(names[:n])


def score_combo(panel, var, kcfg: KrigingConfig, n_folds: int, cfg, pack=None) -> float:
    """Pool stations x tune quarters. Donors are the other pool stations."""
    targets = _tune_split(panel, seed=int(cfg.get("kriging", {}).get("seed", 22)))
    train = panel[(~panel["is_heldout"]) & (panel["split"] == "train")]
    score = panel[(~panel["is_heldout"]) & (panel["split"] == "tune")
                  & (panel["station_name"].isin(targets))]
    if score.empty or train["station_name"].nunique() < kcfg.min_stations:
        return np.inf
    model = KrigingInterpolator(kcfg)
    model.pack = pack
    model.fit(train, var, cfg["time_resolution"], cfg)
    obs, pred = [], []
    for t, sc_t in score.groupby("time", sort=False):
        donors = panel[(panel["time"] == t) & (~panel["is_heldout"])
                       & (~panel["station_name"].isin(targets))]
        if len(donors) < kcfg.min_stations:
            continue
        yhat = model.predict_frame(donors, sc_t, var)
        obs.append(sc_t[var].to_numpy())
        pred.append(np.asarray(yhat, dtype=float))
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
    rows = []
    for extra in varying:
        over = {**locked, **extra}
        kcfg = cfg_to_kriging(cfg, over)
        t0 = time.perf_counter()
        rmse = score_combo(panel, var, kcfg, n_folds, cfg, pack=pack)
        sec = time.perf_counter() - t0
        pretty = " ".join(f"{k}={v}" for k, v in extra.items())
        print(f"  {pretty}  DEV RMSE={rmse:.4f}  ({sec:.0f}s)")
        rows.append({**extra, "rmse": rmse, "seconds": round(sec, 1), "phase": label})
        if rmse < best[0]:
            best = (rmse, over)
    return best, rows


def run_variable(cfg, var, phase: str, months=None, n_folds=None, pack=None,
                 rh_t_mode=None, max_times=400):
    panel = load_panel(cfg, var)
    # Keep held-out stations: they are the tune targets. Drop test quarters.
    from shared.splits.evaluation_sets import apply_evaluation_protocol
    tagged, _, _, _ = apply_evaluation_protocol(panel)
    panel = tagged[tagged["split"].isin(["train", "tune"])].copy()
    args_max = max_times
    if months:
        panel = subset_times(panel, months)
    if args_max and panel["time"].nunique() > args_max:
        rng = np.random.default_rng(int(cfg.get("kriging", {}).get("seed", 22)))
        keep = []
        for sp in ("train", "tune"):
            ts = panel.loc[panel["split"] == sp, "time"].drop_duplicates().to_numpy()
            n = max(1, int(round(args_max * len(ts) / max(panel["time"].nunique(), 1))))
            keep.extend(rng.choice(ts, size=min(n, len(ts)), replace=False).tolist())
        panel = panel[panel["time"].isin(keep)].copy()
    panel, _hours = apply_hour_cut(panel, cfg["time_resolution"], cfg.get("_hours"))
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
    search_rows = []

    def _take(label, grid):
        nonlocal best
        best, rows = _eval_grid(panel, var, cfg, n_folds, locked, grid, pack=pack, label=label)
        search_rows.extend(rows)
        return dict(best[1])

    if phase in ("1", "all"):
        grid = [{"k": int(k), "k_indicator": int(k)} for k in search.get("k", [base.k])]
        locked = _take("phase 1a  k", grid)
        if is_precip(var) and cfg["time_resolution"] != "monthly":
            grid = [{"k_indicator": int(k)} for k in search.get("k", [base.k])]
            locked = _take("phase 1b  k_indicator", grid)
        elif is_precip(var):
            print(f"{var} phase 1b skipped: monthly fields are almost always wet, "
                  f"k_indicator does not change the score", flush=True)

    if phase in ("2", "all"):
        grid = [{"family": str(f)} for f in search.get("family", [base.family])]
        locked = _take("phase 2  family", grid)

    if phase in ("3", "all"):
        grid = [
            {"theta_deg": float(th), "aniso_ratio": float(r), "alpha_z": float(az)}
            for th, r, az in itertools.product(
                search.get("theta_deg", [0.0]),
                search.get("aniso_ratio", [1.0]),
                search.get("alpha_z", [0.0]),
            )
        ]
        locked = _take("phase 3  metric", grid)

    if phase in ("4", "all"):
        grid = [{"interactions": str(s)} for s in search.get("interactions", [base.interactions])]
        locked = _take("phase 4  interactions", grid)

    if phase in ("5", "all"):
        expand = kblock.get("search_expand", {})
        az_grid = [float(v) for v in expand.get("alpha_z", search.get("alpha_z", [locked["alpha_z"]]))]
        if float(locked["alpha_z"]) not in az_grid:
            az_grid.append(float(locked["alpha_z"]))
        az_grid = sorted(set(az_grid))
        locked = _take("phase 5a  alpha_z expand", [{"alpha_z": az} for az in az_grid])
        k_grid = [int(v) for v in expand.get("k", search.get("k", [locked["k"]]))]
        locked = _take("phase 5b  k re-search", [{"k": k, "k_indicator": k} for k in k_grid])
        if is_precip(var):
            locked = _take("phase 5c  k_indicator", [{"k_indicator": k} for k in k_grid])

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
    if search_rows:
        import pandas as pd
        sp = get_kriging_tuned_params_path(var, cfg["time_resolution"]).with_name(
            f"{var}_search.csv")
        pd.DataFrame(search_rows).to_csv(sp, index=False)
        print(f"  wrote {sp}  ({len(search_rows)} trials)")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Staged DEV search for Kriging")
    p.add_argument("--variables", nargs="*", default=None)
    p.add_argument("--phase", choices=["1", "2", "3", "4", "5", "all"], default=None)
    p.add_argument("--months", default=None)
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--max-times", type=int, default=400,
                   help="cap tune timestamps, sampled within the tune split")
    p.add_argument("--rh-t-mode", default=None, choices=["none", "predicted", "observed"])
    add_hours_arg(p)
    add_time_res_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    apply_time_res(cfg, args)
    cfg["_hours"] = args.hours
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
        run_variable(cfg, var, phase, months=months, n_folds=n_folds, pack=pack,
                     rh_t_mode=args.rh_t_mode, max_times=args.max_times)
    print("\nKriging tune finished.")


if __name__ == "__main__":
    main()
