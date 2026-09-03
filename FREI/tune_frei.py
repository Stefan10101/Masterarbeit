#!/usr/bin/env python3
"""DEV search for Frei: n_regions, shrink, k, across_w. Parallel over cells."""

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
from frei_data import compute_metrics, load_frei_config, load_panel
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default=None)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = p.parse_args()
    cfg = load_frei_config()
    time_res = cfg["time_resolution"]
    block = cfg["frei"]
    variables = [args.variable] if args.variable else block.get("variables_to_process", ["temp_mean"])
    search = block.get("search", {})
    n_folds = int(block.get("n_folds", 5))
    jobs = max(1, int(args.jobs))

    for var in variables:
        panel = load_panel(cfg, var)
        print(var, "rows", len(panel), "splits", panel.groupby("split").size().to_dict(),
              "jobs", jobs, flush=True)
        nregs = list(search.get("n_regions", [block.get("n_regions", 6)]))
        packs = {}
        for nr in nregs:
            print(f"building pack n_regions={nr}", flush=True)
            packs[int(nr)] = pack_from_master(cfg, int(nr))
        cells = list(itertools.product(
            nregs,
            search.get("shrink", [block.get("shrink", 0.3)]),
            search.get("k", [block.get("k", 16)]),
            search.get("across_w", [block.get("across_w", 4.0)]),
        ))
        print(f"{var} cells={len(cells)}", flush=True)
        if jobs == 1:
            _init(panel, cfg, packs, n_folds, var)
            results = [_eval_cell(*c) for c in cells]
        else:
            results = []
            with ProcessPoolExecutor(
                max_workers=jobs,
                initializer=_init,
                initargs=(panel, cfg, packs, n_folds, var),
            ) as ex:
                futs = {ex.submit(_eval_cell, *c): c for c in cells}
                for fut in as_completed(futs):
                    results.append(fut.result())
        results.sort(key=lambda r: (r["n_regions"], r["shrink"], r["k"], r["across_w"]))
        best = None
        for row in results:
            print(
                f"{var} nreg={row['n_regions']} shrink={row['shrink']} "
                f"k={row['k']} across={row['across_w']} "
                f"RMSE={row['rmse']:.3f} n={row['n']} finite={row['finite']}",
                flush=True,
            )
            if best is None or row["rmse"] < best["rmse"]:
                best = {k: row[k] for k in ("n_regions", "shrink", "k", "across_w", "rmse")}
        path = get_frei_tuned_params_path(var, time_res)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(best, f)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
