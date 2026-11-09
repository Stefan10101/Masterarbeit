#!/usr/bin/env python3
"""
Staged search for Kriging.

More than 3 interacting hyperparameters, so the search stays phased:
  1 k, 2 family, 3 metric (theta, aniso, alpha_z), 4 interactions, 5 refine.

Protocol (per-timestamp): train + tune quarters, score on the fixed
tune-targets (donors exclude those targets during scoring).
The official held-out set is never used.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_kriging_tuned_params_path
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res, variables_from_cfg
from kriging_core import KrigingConfig, KrigingInterpolator
from shared.splits.evaluation_sets import apply_evaluation_protocol, load_tune_targets
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


def score_combo(panel, var, kcfg: KrigingConfig, n_folds: int, cfg, pack=None, jobs: int = 1) -> float:
    """Train+tune quarters. Score fixed tune-targets; donors exclude them."""
    if "is_tune_target" not in panel.columns or not panel["is_tune_target"].any():
        print("    early inf (no tune-targets tagged)")
        return np.inf
    score = panel[(~panel["is_heldout"]) & panel["is_tune_target"]
                  & (panel["split"].isin(["train", "tune"]))]
    train = panel[(~panel["is_heldout"]) & (~panel["is_tune_target"])
                  & (panel["split"].isin(["train", "tune"]))]
    if score.empty or train["station_name"].nunique() < kcfg.min_stations:
        reason = "score empty" if score.empty else f"train stations {train['station_name'].nunique()} < {kcfg.min_stations}"
        print(f"    early inf ({reason})")
        return np.inf
    model = KrigingInterpolator(kcfg)
    model.pack = pack
    # Global variogram from the donor stations on train+tune
    model.fit(train, var, cfg["time_resolution"], cfg)
    groups = list(score.groupby("time", sort=False))

    def one(item):
        t, sc_t = item
        donors = panel[(panel["time"] == t) & (~panel["is_heldout"])
                       & (~panel["is_tune_target"])]
        if len(donors) < kcfg.min_stations:
            return None
        yhat = model.predict_frame(donors, sc_t, var)
        return sc_t[var].to_numpy(), np.asarray(yhat, dtype=float)

    if jobs == 1:
        parts = [one(g) for g in groups]
    else:
        parts = Parallel(n_jobs=jobs, backend="threading")(delayed(one)(g) for g in groups)
    parts = [p for p in parts if p is not None]
    if not parts:
        return np.inf
    obs = np.concatenate([p[0] for p in parts])
    pred = np.concatenate([p[1] for p in parts])
    met = compute_metrics(obs, pred)
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


def _eval_grid(panel, var, cfg, n_folds, locked: dict, varying: list[dict], label: str, pack=None, jobs: int = 1):
    print(f"\n{var} {label}  {len(varying)} combos  jobs={jobs}")
    rows = []

    def one(extra):
        over = {**locked, **extra}
        kcfg = cfg_to_kriging(cfg, over)
        t0 = time.perf_counter()
        rmse = score_combo(panel, var, kcfg, n_folds, cfg, pack=pack, jobs=1)
        sec = time.perf_counter() - t0
        return extra, rmse, sec

    if jobs == 1 or len(varying) == 1:
        results = [one(extra) for extra in varying]
    else:
        # Parallel over combos; each combo scores its times sequentially
        # so the process/thread pool is not nested.
        results = Parallel(n_jobs=min(jobs, len(varying)), backend="threading")(
            delayed(one)(extra) for extra in varying
        )
    best = (np.inf, dict(locked))
    for extra, rmse, sec in results:
        pretty = " ".join(f"{k}={v}" for k, v in extra.items())
        print(f"  {pretty}  DEV RMSE={rmse:.4f}  ({sec:.0f}s)")
        rows.append({**extra, "rmse": rmse, "seconds": round(sec, 1), "phase": label})
        if rmse < best[0]:
            best = (rmse, {**locked, **extra})
    return best, rows


def run_variable(cfg, var, phase: str, months=None, n_folds=None, pack=None,
                 rh_t_mode=None, max_times=400, jobs: int = 1):
    panel = load_panel(cfg, var)
    # Per-timestamp: train + tune, drop held-out. Fixed tune-targets used for scoring.
    tagged, _, _, _ = apply_evaluation_protocol(panel, variable=var)
    panel = tagged[(~tagged["is_heldout"]) & (tagged["split"].isin(["train", "tune"]))].copy()
    args_max = max_times
    if months:
        panel = subset_times(panel, months)
    if args_max and panel["time"].nunique() > args_max:
        rng = np.random.default_rng(int(cfg.get("kriging", {}).get("seed", 22)))
        keep = []
        for sp in ("train", "tune"):
            ts = panel.loc[panel["split"] == sp, "time"].drop_duplicates()
            if ts.empty:
                continue
            n = max(1, int(round(args_max * len(ts) / max(panel["time"].nunique(), 1))))
            chosen = ts.sample(n=min(n, len(ts)), random_state=rng).tolist()
            keep.extend(chosen)
        if keep:
            panel = panel[panel["time"].isin(pd.to_datetime(keep))].copy()
    panel, _hours = apply_hour_cut(panel, cfg["time_resolution"], cfg.get("_hours"))
    if str(var).lower().startswith("rh") and (rh_t_mode or cfg.get("kriging", {}).get("rh_t_mode", "none")) in ("predicted", "observed"):
        panel = attach_temperature(panel, cfg)
    panel = attach_pack_terrain(panel, pack)
    # Diagnostics: why the score may be empty for some resolutions
    print(f"  {var} panel after prep: rows={len(panel)}  "
          f"times={panel['time'].nunique() if not panel.empty else 0}  "
          f"stations={panel['station_name'].nunique() if not panel.empty else 0}")
    if not panel.empty:
        vc = panel.groupby("split")["time"].nunique()
        print(f"    times per split: {vc.to_dict()}")
        print(f"    tune-targets: {panel['is_tune_target'].sum()}  "
              f"donor stations: {panel.loc[~panel['is_tune_target'], 'station_name'].nunique()}")
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
        best, rows = _eval_grid(panel, var, cfg, n_folds, locked, grid, pack=pack, label=label, jobs=jobs)
        search_rows.extend(rows)
        return dict(best[1])

    if phase in ("1", "all"):
        grid = [{"k": int(k), "k_indicator": int(k)} for k in search.get("k", [base.k])]
        locked = _take("phase 1a  k", grid)
        # Snow stays continuous. Phase 1b used to pass only k_indicator, which
        # stored k as NaN and spent ~47 min a trial without changing the score.
        if str(var) == "precip_sum" and cfg["time_resolution"] != "monthly":
            if locked.get("k") is None:
                raise RuntimeError("phase 1b needs the k locked in phase 1a")
            grid = [
                {"k": int(locked["k"]), "k_indicator": int(k)}
                for k in search.get("k", [base.k])
            ]
            locked = _take("phase 1b  k_indicator", grid)
        elif is_precip(var):
            print(f"{var} phase 1b skipped: indicator neighbourhood is only "
                  f"searched for precip_sum", flush=True)

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
        if str(var) == "precip_sum":
            if locked.get("k") is None:
                raise RuntimeError("phase 5c needs the k locked by the earlier phases")
            locked = _take(
                "phase 5c  k_indicator",
                [{"k": int(locked["k"]), "k_indicator": int(k)} for k in k_grid],
            )

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
    p.add_argument("--quick", action="store_true",
                   help="force phase 1, seasonal4 months, 2 folds")
    p.add_argument("--jobs", type=int, default=None,
                   help="parallel combos within a phase (default: half the logical cores)")
    add_hours_arg(p)
    add_time_res_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    apply_time_res(cfg, args)
    cfg["_hours"] = args.hours
    wanted = variables_from_cfg(cfg, args.variables)
    comp = compute_block(cfg)
    phase = args.phase or ("1" if args.quick else comp.get("tune_phase", "all"))
    months = args.months or ("seasonal4" if args.quick else comp.get("tune_months", "seasonal4"))
    n_folds = args.folds or (2 if args.quick else int(comp.get("tune_folds", 5)))
    jobs = int(args.jobs) if args.jobs is not None else max(1, (os.cpu_count() or 1) // 2)
    pack = pack_from_master(cfg, int(cfg.get("kriging", {}).get("n_regions", 6)))
    print("=" * 72)
    print("Kriging DEV search")
    print(f"  phase={phase}  months={months}  folds={n_folds}  jobs={jobs}  pack={'yes' if pack else 'no'}")
    print(f"  time_resolution={cfg['time_resolution']}")
    print("=" * 72)
    for var in wanted:
        run_variable(cfg, var, phase, months=months, n_folds=n_folds, pack=pack,
                     rh_t_mode=args.rh_t_mode, max_times=args.max_times, jobs=jobs)
    print("\nKriging tune finished.")


if __name__ == "__main__":
    main()
