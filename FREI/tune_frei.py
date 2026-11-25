#!/usr/bin/env python3
"""
Frei hyperparameter search.

Protocol (per-timestamp): train + tune quarters, score on the fixed
tune-targets (donors exclude those targets during scoring).
Default is coordinate descent. --mode grid restores the product search.
--quick forces 2 folds + seasonal4 + coord.
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

from frei_data import hyper_source
from paths import get_frei_tuned_params_path
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res, cap_jobs, variables_from_cfg, cap_timestamps
from shared.splits.evaluation_sets import (
    apply_evaluation_protocol, load_tune_targets,
    load_early_stop_times, early_stop_kind,
)
from frei_data import (
    load_idw_residual_params,
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
PREFIX_MARGIN = 0.20


def _match_times(panel_times: pd.Series, wanted: list, time_res: str) -> set:
    """Panel timestamps matching the yaml early-stop list.
    Returns values in the panel's own dtype so isin matches.
    """
    if not wanted:
        return set()
    import pandas as pd
    pt = pd.to_datetime(panel_times, utc=True)
    want = set(str(w) for w in wanted)
    if time_res == "monthly":
        key = pt.dt.strftime("%Y-%m")
    elif time_res == "weekly":
        key = pt.dt.strftime("%Y-W%W")
    elif time_res == "daily":
        key = pt.dt.strftime("%Y-%m-%d")
    else:
        key = pt.dt.strftime("%Y-%m-%dT%H:%M:%S")
    mask = key.isin(want).to_numpy()
    return set(panel_times[mask].unique())



def _init(panel, cfg, packs, n_folds, var):
    global _PANEL, _CFG, _PACKS, _N_FOLDS, _VAR
    _PANEL = panel
    _CFG = cfg
    _PACKS = packs
    _N_FOLDS = n_folds
    _VAR = var


def _eval_cell(nsec, sh, aw, panel=None):
    # Residual p/Fz/k come from IDW. n_regions is fixed at 11.
    # panel overrides the global so the caller can pass a prefix or eval subset.
    idw = getattr(_eval_cell, "idw", {"power": 2.0, "Fz": 0.25, "k": 16})
    nr = 11
    fcfg = cfg_to_frei(_CFG, {
        "n_regions": nr, "n_sections": nsec, "shrink": sh, "across_w": aw,
        "power": idw["power"], "Fz": idw["Fz"], "k": idw["k"],
    })
    fcfg.select_metric = False
    use = panel if panel is not None else _PANEL
    pred = run_llocv(use, _VAR, fcfg, _N_FOLDS, {"train"}, {"train", "tune"}, _PACKS.get(nr))
    met = compute_metrics(pred["observed"], pred["predicted"])
    nfin = int(np.isfinite(pred["predicted"]).sum()) if len(pred) else 0
    return {
        "n_regions": nr,
        "n_sections": int(nsec),
        "shrink": float(sh),
        "across_w": float(aw),
        "power": float(idw["power"]),
        "Fz": float(idw["Fz"]),
        "k": int(idw["k"]),
        "rmse": float(met["rmse"]),
        "n": int(len(pred)),
        "finite": nfin,
    }


def _score_cells(cells, jobs, panel):
    """Score cells on a given panel. Sequential if jobs==1, else process pool."""
    if not cells:
        return []
    if jobs == 1 or panel is not _PANEL:
        # filtered panel must be passed explicitly; workers only see the global
        return [_eval_cell(*c, panel=panel) for c in cells]
    return _run_cells(cells, jobs)


def _early_stop_search(cells, jobs, var, eval_panel, prefix_panel):
    """Score cells with the prefix/eval early-stop protocol.

    First cell is always scored on the full eval set (becomes champion).
    Later cells are probed on the prefix; aborted if > 20 % worse than
    the champion's prefix score. Survivors are scored on the eval set.
    """
    results = []
    champion = None          # full eval rmse
    champion_prefix = None   # prefix rmse of the champion
    for i, cell in enumerate(cells):
        if champion is None:
            full = _eval_cell(*cell, panel=eval_panel)
            full["early_stopped"] = False
            full["stop_reason"] = "full"
            full["score_at_stop"] = full["rmse"]
            full["n_units_scored"] = full["n"]
            results.append(full)
            champion = full["rmse"]
            # champion's prefix score for later comparisons
            pref = _eval_cell(*cell, panel=prefix_panel)
            champion_prefix = pref["rmse"]
            print(f"  {var} champion nsec={full['n_sections']} shrink={full['shrink']} "
                  f"across={full['across_w']} rmse={champion:.3f}", flush=True)
            continue
        pref = _eval_cell(*cell, panel=prefix_panel)
        if (np.isfinite(pref["rmse"]) and np.isfinite(champion_prefix)
                and pref["rmse"] > champion_prefix * (1.0 + PREFIX_MARGIN)):
            pref["early_stopped"] = True
            pref["stop_reason"] = "prefix_margin"
            pref["score_at_stop"] = pref["rmse"]
            pref["n_units_scored"] = pref["n"]
            results.append(pref)
            print(f"  {var} early-stop nsec={pref['n_sections']} shrink={pref['shrink']} "
                  f"across={pref['across_w']} prefix={pref['rmse']:.3f} "
                  f"> {champion_prefix:.3f}", flush=True)
            continue
        full = _eval_cell(*cell, panel=eval_panel)
        full["early_stopped"] = False
        full["stop_reason"] = "full"
        full["score_at_stop"] = full["rmse"]
        full["n_units_scored"] = full["n"]
        results.append(full)
        if np.isfinite(full["rmse"]) and full["rmse"] < champion:
            champion = full["rmse"]
            champion_prefix = pref["rmse"]
            print(f"  {var} new champion nsec={full['n_sections']} shrink={full['shrink']} "
                  f"across={full['across_w']} rmse={champion:.3f}", flush=True)
    return results


def _print_rows(var, results, best):
    results = sorted(results, key=lambda r: (r["n_sections"], r["shrink"], r["across_w"]))
    for row in results:
        print(
            f"{var} nsec={row['n_sections']} shrink={row['shrink']} "
            f"across={row['across_w']} "
            f"RMSE={row['rmse']:.3f} n={row['n']} finite={row['finite']}",
            flush=True,
        )
        if best is None or (np.isfinite(row["rmse"]) and row["rmse"] < best["rmse"]):
            best = {k: row[k] for k in ("n_regions", "n_sections", "shrink", "across_w", "power", "Fz", "k", "rmse")}
    return best


def _coord_search(nsecs, shrinks, aws, jobs, var, defaults, eval_panel, prefix_panel):
    locked = dict(defaults)
    best = None
    seen = set()
    all_rows = []
    builders = [
        ("n_sections", lambda: [(ns, locked["shrink"], locked["across_w"]) for ns in nsecs]),
        ("shrink", lambda: [(locked["n_sections"], sh, locked["across_w"]) for sh in shrinks]),
        ("across_w", lambda: [(locked["n_sections"], locked["shrink"], aw) for aw in aws]),
    ]
    for name, builder in builders:
        cells = [c for c in builder() if c not in seen]
        if not cells:
            continue
        print(f"{var} coord phase {name} cells={len(cells)} locked={locked}", flush=True)
        rows = _early_stop_search(cells, jobs, var, eval_panel, prefix_panel)
        for c in cells:
            seen.add(c)
        all_rows.extend(rows)
        finished = [r for r in rows if not r.get("early_stopped")]
        best = _print_rows(var, finished, best)
        if best is not None:
            locked["n_sections"] = best["n_sections"]
            locked["shrink"] = best["shrink"]
            locked["across_w"] = best["across_w"]
    return best, all_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default=None, help="single variable (prefer --variables)")
    p.add_argument("--variables", nargs="+", default=None, help="list of variables")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--mode", choices=["coord", "grid"], default=None)
    p.add_argument("--months", default=None, help="all | seasonal4 | YYYY-MM,YYYY-MM")
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--quick", action="store_true",
                   help="2 folds + seasonal4 + coord descent")
    p.add_argument("--max-times", type=int, default=400,
                   help="cap timestamps, sampled within each split")
    p.add_argument("--rh-t-mode", default=None, choices=["none", "predicted", "observed"])
    add_hours_arg(p)
    add_time_res_arg(p)
    args = p.parse_args()
    cfg = load_frei_config()
    time_res = apply_time_res(cfg, args)
    block = cfg["frei"]
    comp = compute_block(cfg)
    if args.variables:
        variables = args.variables
    elif args.variable:
        variables = [args.variable]
    else:
        variables = variables_from_cfg(cfg)
    search = block.get("search", {})
    n_folds = args.folds or (2 if args.quick else int(comp.get("tune_folds", block.get("n_folds", 5))))
    months = args.months or ("seasonal4" if args.quick else comp.get("tune_months", "seasonal4"))
    mode = args.mode or ("coord" if args.quick else comp.get("tune_mode", "coord"))
    jobs = max(1, int(args.jobs))

    for var in variables:
        src = hyper_source(cfg, var)
        if src != var:
            parent = get_frei_tuned_params_path(src, time_res)
            child = get_frei_tuned_params_path(var, time_res)
            if parent.exists():
                child.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(parent, child)
                print(f"[{var}] inherited hypers from {src}")
            else:
                print(f"[{var}] parent {src} has no tuned file yet — skipping")
            continue
        panel = load_panel(cfg, var)
        from shared.splits.evaluation_sets import apply_evaluation_protocol
        tagged, _, _, _ = apply_evaluation_protocol(panel, variable=var)
        # Pool stations only for the search. Drop the official held-out set.
        panel = tagged[(~tagged["is_heldout"]) & (tagged["split"].isin(["train", "tune"]))].copy()
        panel = subset_times(panel, months)
        panel, hours = apply_hour_cut(panel, time_res, args.hours)
        # Early-stop protocol: score only the eval set (max 52). Prefix (20) probes.
        kind = early_stop_kind("Frei")
        eval_raw = load_early_stop_times(time_res, kind, which="eval")
        prefix_raw = load_early_stop_times(time_res, kind, which="prefix")
        eval_set = _match_times(panel["time"], eval_raw, time_res)
        prefix_set = _match_times(panel["time"], prefix_raw, time_res)
        if eval_set:
            eval_panel = panel[panel["time"].isin(eval_set)].copy()
            prefix_panel = panel[panel["time"].isin(prefix_set)].copy()
            print(f"  early-stop eval={eval_panel['time'].nunique()} "
                  f"prefix={prefix_panel['time'].nunique()}", flush=True)
        else:
            if args.max_times:
                panel = cap_timestamps(panel, args.max_times, time_res, seed=int(block.get("seed", 22)))
            eval_panel = panel
            prefix_panel = panel
            print("  no early-stop yaml; scoring full window", flush=True)
        jobs = cap_jobs(jobs, len(eval_panel))
        print(
            f"{var} rows={len(eval_panel)} times={eval_panel['time'].nunique()} "
            f"splits={eval_panel.groupby('split').size().to_dict()} "
            f"tune-targets={int(eval_panel['is_tune_target'].sum())} "
            f"mode={mode} months={months} hours={hours} folds={n_folds} jobs={jobs}",
            flush=True,
        )
        nsecs = list(search.get("n_sections", [block.get("n_sections", 3)]))
        shrinks = list(search.get("shrink", [block.get("shrink", 0.3)]))
        aws = list(search.get("across_w", [block.get("across_w", 4.0)]))
        idw = load_idw_residual_params(time_res, var)
        _eval_cell.idw = idw
        nr = int(block.get("n_regions", 11))
        print(f"building pack n_regions={nr}", flush=True)
        packs = {nr: pack_from_master(cfg, nr)}
        _init(eval_panel, cfg, packs, n_folds, var)
        defaults = {
            "n_sections": int(block.get("n_sections", nsecs[0])),
            "shrink": float(block.get("shrink", shrinks[0])),
            "across_w": float(block.get("across_w", aws[0])),
        }
        if mode == "grid":
            cells = list(itertools.product(nsecs, shrinks, aws))
            print(f"{var} grid cells={len(cells)}", flush=True)
            results = _early_stop_search(cells, jobs, var, eval_panel, prefix_panel)
            finished = [r for r in results if not r.get("early_stopped")]
            best = _print_rows(var, finished, None)
            all_rows = results
        else:
            best, all_rows = _coord_search(
                nsecs, shrinks, aws, jobs, var, defaults, eval_panel, prefix_panel,
            )
        if best is None:
            print(f"{var} no finite DEV result", flush=True)
            continue
        best["trace"] = precip_trace(cfg, time_res, var)
        best.update(idw)
        best["tune_months"] = months
        best["tune_mode"] = mode
        best["tune_folds"] = n_folds
        path = get_frei_tuned_params_path(var, time_res)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(best, f)
        if all_rows:
            import pandas as pd
            sp = path.with_name(f"{var}_search.csv")
            pd.DataFrame(all_rows).to_csv(sp, index=False)
            print(f"wrote {sp}  ({len(all_rows)} trials)", flush=True)
        print(f"wrote {path} best={best}", flush=True)


if __name__ == "__main__":
    main()
