#!/usr/bin/env python3
"""DEV search for GAM formulas. Parallel over cells."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import itertools
import os
import sys
import time
import traceback

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_gam_tuned_params_path
from gam_data import (
    attach_pack_terrain,
    attach_temperature,
    compute_block,
    compute_metrics,
    load_gam_config,
    load_panel,
    pack_from_master,
    subset_times,
)
from gam_core import two_step_var
from llocv_gam import cfg_to_gam, run_llocv

_PANEL = None
_CFG = None
_N_FOLDS = 5
_VAR = None


def _init(panel, cfg, n_folds, var):
    global _PANEL, _CFG, _N_FOLDS, _VAR
    _PANEL = panel
    _CFG = cfg
    _N_FOLDS = n_folds
    _VAR = var


def _eval_cell(form, ns, tau):
    tag = f"{_VAR} form={form} ns={ns} tau={tau}"
    print(f"START {tag}", flush=True)
    t0 = time.time()
    try:
        gcfg = cfg_to_gam(_CFG, {"formula": form, "n_splines": ns, "tau_wet": tau})
        pred = run_llocv(_PANEL, _VAR, gcfg, _N_FOLDS, {"train"}, {"dev"}, progress=tag)
        met = compute_metrics(pred["observed"], pred["predicted"])
        nfin = int(np.isfinite(pred["predicted"]).sum()) if len(pred) else 0
        row = {
            "formula": form,
            "n_splines": int(ns),
            "tau_wet": float(tau),
            "rmse": float(met["rmse"]),
            "n": int(len(pred)),
            "finite": nfin,
            "sec": float(time.time() - t0),
        }
        print(
            f"DONE  {tag} RMSE={row['rmse']:.3f} n={row['n']} finite={row['finite']} {row['sec']:.0f}s",
            flush=True,
        )
        return row
    except Exception as exc:
        print(f"CRASH {tag} {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default=None)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--months", default=None)
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--rh-t-mode", default=None, choices=["none", "predicted", "observed"])
    args = p.parse_args()
    cfg = load_gam_config()
    time_res = cfg["time_resolution"]
    block = cfg["gam"]
    comp = compute_block(cfg)
    variables = [args.variable] if args.variable else block.get("variables_to_process", ["temp_mean"])
    search = block.get("search", {})
    n_folds = args.folds or (2 if args.quick else int(comp.get("tune_folds", block.get("n_folds", 5))))
    months = args.months or ("seasonal4" if args.quick else comp.get("tune_months", "seasonal4"))
    jobs = max(1, int(args.jobs))
    pack = pack_from_master(cfg, int(block.get("n_regions", 6)))

    for var in variables:
        panel = load_panel(cfg, var)
        rh_mode = args.rh_t_mode or block.get("rh_t_mode", "none")
        if str(var).lower().startswith("rh") and rh_mode in ("predicted", "observed"):
            panel = attach_temperature(panel, cfg)
        panel = attach_pack_terrain(panel, pack)
        panel = subset_times(panel, months)
        print(var, "rows", len(panel), "times", panel["time"].nunique(),
              "splits", panel.groupby("split").size().to_dict(),
              "months", months, "folds", n_folds, "jobs", jobs, flush=True)
        if args.quick:
            formulas = [block.get("formula", "te_xy_s_elev")]
            ns_list = [block.get("n_splines", 10)]
            taus = [float(block.get("tau_wet", 0.5))]
        else:
            formulas = list(search.get("formula", [block.get("formula", "te_xy_s_elev")]))
            ns_list = list(search.get("n_splines", [block.get("n_splines", 10)]))
            if two_step_var(var):
                taus = list(search.get("tau_wet", [block.get("tau_wet", 0.5)]))
            else:
                taus = [float(block.get("tau_wet", 0.5))]
        cells = list(itertools.product(formulas, ns_list, taus))
        n_dev_t = int(panel.loc[panel["split"] == "dev", "time"].nunique())
        n_sta = int(panel["station_name"].nunique())
        print(
            f"{var} cells={len(cells)} stations={n_sta} dev_times={n_dev_t} "
            f"folds={n_folds}  (each cell ≈ {n_folds}×{n_dev_t} GAM fits)",
            flush=True,
        )
        for i, c in enumerate(cells, start=1):
            print(f"  queued {i}/{len(cells)} form={c[0]} ns={c[1]} tau={c[2]}", flush=True)
        if jobs == 1:
            _init(panel, cfg, n_folds, var)
            results = [_eval_cell(*c) for c in cells]
        else:
            results = []
            print(
                f"workers={jobs}  (worker fold lines may appear late on Windows; "
                f"DONE lines print as soon as a cell finishes)",
                flush=True,
            )
            with ProcessPoolExecutor(
                max_workers=jobs,
                initializer=_init,
                initargs=(panel, cfg, n_folds, var),
            ) as ex:
                futs = {ex.submit(_eval_cell, *c): c for c in cells}
                n_left = len(futs)
                for fut in as_completed(futs):
                    cell = futs[fut]
                    n_left -= 1
                    try:
                        row = fut.result()
                    except Exception as exc:
                        print(
                            f"CELL FAILED form={cell[0]} ns={cell[1]} tau={cell[2]} "
                            f"{type(exc).__name__}: {exc}  remaining={n_left}",
                            flush=True,
                        )
                        raise
                    results.append(row)
                    print(
                        f"PARENT got form={row['formula']} ns={row['n_splines']} "
                        f"RMSE={row['rmse']:.3f} {row.get('sec', 0):.0f}s  remaining={n_left}",
                        flush=True,
                    )
        results.sort(key=lambda r: (r["formula"], r["n_splines"], r["tau_wet"]))
        best = None
        for row in results:
            print(
                f"{var} form={row['formula']} ns={row['n_splines']} tau={row['tau_wet']} "
                f"RMSE={row['rmse']:.3f} n={row['n']} finite={row['finite']}",
                flush=True,
            )
            if best is None or row["rmse"] < best["rmse"]:
                best = {k: row[k] for k in ("formula", "n_splines", "tau_wet", "rmse")}
                best["rh_t_mode"] = rh_mode
                best["tune_months"] = months
        path = get_gam_tuned_params_path(var, time_res)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(best, f)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
