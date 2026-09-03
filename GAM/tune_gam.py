#!/usr/bin/env python3
"""DEV search for GAM formulas. Parallel over cells."""

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

from paths import get_gam_tuned_params_path
from gam_data import compute_metrics, load_gam_config, load_panel
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
    gcfg = cfg_to_gam(_CFG, {"formula": form, "n_splines": ns, "tau_wet": tau})
    pred = run_llocv(_PANEL, _VAR, gcfg, _N_FOLDS, {"train"}, {"dev"})
    met = compute_metrics(pred["observed"], pred["predicted"])
    nfin = int(np.isfinite(pred["predicted"]).sum()) if len(pred) else 0
    return {
        "formula": form,
        "n_splines": int(ns),
        "tau_wet": float(tau),
        "rmse": float(met["rmse"]),
        "n": int(len(pred)),
        "finite": nfin,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default=None)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = p.parse_args()
    cfg = load_gam_config()
    time_res = cfg["time_resolution"]
    block = cfg["gam"]
    variables = [args.variable] if args.variable else block.get("variables_to_process", ["temp_mean"])
    search = block.get("search", {})
    n_folds = int(block.get("n_folds", 5))
    jobs = max(1, int(args.jobs))

    for var in variables:
        panel = load_panel(cfg, var)
        print(var, "rows", len(panel), "splits", panel.groupby("split").size().to_dict(),
              "jobs", jobs, flush=True)
        formulas = list(search.get("formula", [block.get("formula", "te_xy_s_elev")]))
        ns_list = list(search.get("n_splines", [block.get("n_splines", 10)]))
        if two_step_var(var):
            taus = list(search.get("tau_wet", [block.get("tau_wet", 0.5)]))
        else:
            taus = [float(block.get("tau_wet", 0.5))]
        cells = list(itertools.product(formulas, ns_list, taus))
        print(f"{var} cells={len(cells)}", flush=True)
        if jobs == 1:
            _init(panel, cfg, n_folds, var)
            results = [_eval_cell(*c) for c in cells]
        else:
            results = []
            with ProcessPoolExecutor(
                max_workers=jobs,
                initializer=_init,
                initargs=(panel, cfg, n_folds, var),
            ) as ex:
                futs = {ex.submit(_eval_cell, *c): c for c in cells}
                for fut in as_completed(futs):
                    results.append(fut.result())
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
        path = get_gam_tuned_params_path(var, time_res)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(best, f)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
