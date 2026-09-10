#!/usr/bin/env python3
"""DEV search for Frei.

Default is coordinate descent on a seasonal-4 month cut so the full
n_regions × shrink × k × across_w grid does not run first.
--mode grid restores the product search. --quick forces 2 folds + seasonal4 + coord.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import itertools
import os
import sys

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_frei_tuned_params_path
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res, cap_jobs
from frei_data import (
    attach_temperature,
    compute_block,
    compute_metrics,
    load_frei_config,
    load_panel,
    precip_trace,
    subset_times,
)
from llocv_frei import cfg_to_frei, pack_from_master, run_llocv

_PANEL = None
_CFG = None
_PACKS = None
_N_FOLDS = 5
_VAR = None


def _init(panel, cfg, packs, n_folds, var):
    global _PANEL, _CFG, _PACKS, _N_FOLDS, _VAR
    _PANEL = panel
    _CFG = cfg
    _PACKS = packs
    _N_FOLDS = n_folds
    _VAR = var


def _eval_cell(nr, sh, k, aw):
    fcfg = cfg_to_frei(_CFG, {"n_regions": nr, "shrink": sh, "k": k, "across_w": aw})
    fcfg.select_metric = False
    pred = run_llocv(_PANEL, _VAR, fcfg, _N_FOLDS, {"train"}, {"dev"}, _PACKS.get(int(nr)))
    met = compute_metrics(pred["observed"], pred["predicted"])
    nfin = int(np.isfinite(pred["predicted"]).sum()) if len(pred) else 0
    return {
        "n_regions": int(nr),
        "shrink": float(sh),
        "k": int(k),
        "across_w": float(aw),
        "rmse": float(met["rmse"]),
        "n": int(len(pred)),
        "finite": nfin,
    }


def _run_cells(cells, jobs):
    if jobs == 1:
        return [_eval_cell(*c) for c in cells]
    results = []
    with ProcessPoolExecutor(
        max_workers=jobs,
        initializer=_init,
        initargs=(_PANEL, _CFG, _PACKS, _N_FOLDS, _VAR),
    ) as ex:
        futs = {ex.submit(_eval_cell, *c): c for c in cells}
        for fut in as_completed(futs):
            results.append(fut.result())
    return results


def _print_rows(var, results, best):
    results = sorted(results, key=lambda r: (r["n_regions"], r["shrink"], r["k"], r["across_w"]))
    for row in results:
        print(
            f"{var} nreg={row['n_regions']} shrink={row['shrink']} "
            f"k={row['k']} across={row['across_w']} "
            f"RMSE={row['rmse']:.3f} n={row['n']} finite={row['finite']}",
            flush=True,
        )
        if best is None or (np.isfinite(row["rmse"]) and row["rmse"] < best["rmse"]):
            best = {k: row[k] for k in ("n_regions", "shrink", "k", "across_w", "rmse")}
    return best


def _coord_search(nregs, shrinks, ks, aws, jobs, var, defaults):
    locked = dict(defaults)
    best = None
    seen = set()
    builders = [
        ("n_regions", lambda: [(nr, locked["shrink"], locked["k"], locked["across_w"]) for nr in nregs]),
        ("shrink", lambda: [(locked["n_regions"], sh, locked["k"], locked["across_w"]) for sh in shrinks]),
        ("k", lambda: [(locked["n_regions"], locked["shrink"], k, locked["across_w"]) for k in ks]),
        ("across_w", lambda: [(locked["n_regions"], locked["shrink"], locked["k"], aw) for aw in aws]),
    ]
    for name, builder in builders:
        cells = [c for c in builder() if c not in seen]
        if not cells:
            continue
        print(f"{var} coord phase {name} cells={len(cells)} locked={locked}", flush=True)
        rows = _run_cells(cells, jobs)
        for c in cells:
            seen.add(c)
        best = _print_rows(var, rows, best)
        locked["n_regions"] = best["n_regions"]
        locked["shrink"] = best["shrink"]
        locked["k"] = best["k"]
        locked["across_w"] = best["across_w"]
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default=None)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--mode", choices=["coord", "grid"], default=None)
    p.add_argument("--months", default=None, help="all | seasonal4 | YYYY-MM,YYYY-MM")
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--rh-t-mode", default=None, choices=["none", "predicted", "observed"])
    add_hours_arg(p)
    add_time_res_arg(p)
    args = p.parse_args()
    cfg = load_frei_config()
    time_res = apply_time_res(cfg, args)
    block = cfg["frei"]
    comp = compute_block(cfg)
    variables = [args.variable] if args.variable else block.get("variables_to_process", ["temp_mean"])
    search = block.get("search", {})
    n_folds = args.folds or (2 if args.quick else int(comp.get("tune_folds", block.get("n_folds", 5))))
    months = args.months or ("seasonal4" if args.quick else comp.get("tune_months", "seasonal4"))
    mode = args.mode or ("coord" if args.quick else comp.get("tune_mode", "coord"))
    jobs = max(1, int(args.jobs))

    for var in variables:
        panel = load_panel(cfg, var)
        rh_mode = args.rh_t_mode or block.get("rh_t_mode", "none")
        if var.lower().startswith("rh") and rh_mode in ("predicted", "observed"):
            panel = attach_temperature(panel, cfg)
        panel = subset_times(panel, months)
        panel, hours = apply_hour_cut(panel, time_res, args.hours)
        jobs = cap_jobs(jobs, len(panel))
        print(
            f"{var} rows={len(panel)} times={panel['time'].nunique()} "
            f"splits={panel.groupby('split').size().to_dict()} "
            f"mode={mode} months={months} hours={hours} folds={n_folds} jobs={jobs}",
            flush=True,
        )
        nregs = list(search.get("n_regions", [block.get("n_regions", 6)]))
        shrinks = list(search.get("shrink", [block.get("shrink", 0.3)]))
        ks = list(search.get("k", [block.get("k", 16)]))
        aws = list(search.get("across_w", [block.get("across_w", 4.0)]))
        packs = {}
        for nr in nregs:
            print(f"building pack n_regions={nr}", flush=True)
            packs[int(nr)] = pack_from_master(cfg, int(nr))
        _init(panel, cfg, packs, n_folds, var)
        defaults = {
            "n_regions": int(block.get("n_regions", nregs[0])),
            "shrink": float(block.get("shrink", shrinks[0])),
            "k": int(block.get("k", ks[0])),
            "across_w": float(block.get("across_w", aws[0])),
        }
        if mode == "grid":
            cells = list(itertools.product(nregs, shrinks, ks, aws))
            print(f"{var} grid cells={len(cells)}", flush=True)
            results = _run_cells(cells, jobs)
            best = _print_rows(var, results, None)
        else:
            best = _coord_search(nregs, shrinks, ks, aws, jobs, var, defaults)
        if best is None:
            print(f"{var} no finite DEV result", flush=True)
            continue
        best["trace"] = precip_trace(cfg, time_res, var)
        best["rh_t_mode"] = rh_mode if var.lower().startswith("rh") else "none"
        best["tune_months"] = months
        best["tune_mode"] = mode
        best["tune_folds"] = n_folds
        path = get_frei_tuned_params_path(var, time_res)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(best, f)
        print(f"wrote {path} best={best}", flush=True)


if __name__ == "__main__":
    main()
