#!/usr/bin/env python3
"""Station-fold LLOCV for Frei."""

from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_frei_tuned_params_path, get_master_grid_path, get_nested_llocv_path
from shared.splits.evaluation_sets import apply_evaluation_protocol
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res
from frei_core import FreiConfig, FreiInterpolator, two_step_var
from frei_data import (
    attach_temperature,
    data_sources,
    load_frei_config,
    load_panel,
    precip_trace,
    print_split_metrics,
    subset_times,
)


def parse_split_arg(text: str) -> set[str]:
    text = text.strip().lower()
    if text == "all":
        return {"all"}
    return {s.strip() for s in text.split(",") if s.strip()}


def station_folds(names, n_folds, seed):
    rng = np.random.default_rng(seed)
    names = list(names)
    rng.shuffle(names)
    folds = [[] for _ in range(n_folds)]
    for i, n in enumerate(names):
        folds[i % n_folds].append(n)
    return folds


def _covar_arrays(df, keys):
    """Extract enabled covariate columns (config key -> cache column)."""
    colmap = {
        "elev": "elev", "smooth_elev": "smooth_elev", "slope": "slope",
        "northness": "northness", "tpi": "tpi", "clc": "clc_level2",
        "dist_to_coast": "dist_to_coast", "signed_dist_to_ridge": "signed_dist_to_ridge",
    }
    out = []
    for k in keys or []:
        col = colmap.get(k, k)
        if col == "elev":
            continue  # already the main elev term
        if col in df.columns:
            out.append(df[col].to_numpy(dtype=float))
    return out or None


def cfg_to_frei(cfg, overrides=None) -> FreiConfig:
    f = dict(cfg.get("frei", {}))
    o = overrides or {}
    return FreiConfig(
        n_regions=int(o.get("n_regions", f.get("n_regions", 6))),
        shrink=float(o.get("shrink", f.get("shrink", 0.3))),
        blend_km=float(f.get("blend_km", 15.0)),
        summit_tpi=float(f.get("summit_tpi", 80.0)),
        coldpool_tpi=float(f.get("coldpool_tpi", -60.0)),
        accum_pct=float(f.get("accum_pct", 96.0)),
        k=int(o.get("k", f.get("k", 16))),
        power=float(o.get("power", f.get("power", 2.0))),
        across_w=float(o.get("across_w", f.get("across_w", 4.0))),
        select_metric=bool(o.get("select_metric", f.get("select_metric", True))),
        lam_z=float(o.get("lam_z", f.get("lam_z", 150.0))),
        two_step=bool(f.get("two_step", True)),
        tau_wet=float(o.get("tau_wet", f.get("tau_wet", 0.5))),
        trace=float(o.get("trace", f.get("trace", 0.1))),
        min_stations=int(cfg.get("min_stations_per_field", 10)),
        predict_tile=int(f.get("predict_tile", 20000)),
        seed=int(f.get("seed", 22)),
        across_w_grid=tuple(f.get("search", {}).get("across_w", [2.0, 4.0, 8.0])),
        rh_t_mode=str(o.get("rh_t_mode", f.get("rh_t_mode", "none"))),
        snow_terrain=bool(f.get("snow_terrain", True)),
        wind_elev_bg=bool(f.get("wind_elev_bg", True)),
        enabled_covariates=tuple(
            c for c, on in (cfg.get("covariates") or {}).items() if on
        ),
    )


def load_tuned(var, time_res):
    path = get_frei_tuned_params_path(var, time_res)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _base_cache_path(res: int) -> Path:
    from paths import get_watershed_base_path
    return get_watershed_base_path(res)


def pack_from_master(cfg, n_regions: int = 6):
    """Watershed pack. DEM-derived fields are cached; only the n_regions merge is redone."""
    try:
        import xarray as xr
    except ImportError:
        return None
    from shared.watersheds import (
        WatershedConfig, d8_flow, flow_accum, label_basins, merge_basins,
        tpi, flags_from_tpi, valley_unit, region_distance,
    )
    _, _, grid_method = data_sources(cfg)
    res = int(cfg.get("resolutions_to_process", [1000])[0])
    path = get_master_grid_path(grid_method, res)
    if not path.exists():
        return None
    g = xr.open_dataset(path)
    if "elev" not in g:
        return None
    f = cfg.get("frei", {})
    wcfg = WatershedConfig(
        n_regions=int(n_regions),
        accum_pct=float(f.get("accum_pct", 96.0)),
        summit_tpi=float(f.get("summit_tpi", 80.0)),
        coldpool_tpi=float(f.get("coldpool_tpi", -60.0)),
    )
    cache = _base_cache_path(res)
    if cache.exists():
        print(f"    loading cached watershed base {cache.name}", flush=True)
        z = np.load(cache)
        raw = z["raw"]
        tpi_map = z["tpi"]
        summit = z["summit"]
        cold = z["coldpool"]
        uy = z["uy"]
        ux = z["ux"]
        xs = z["xs"]
        ys = z["ys"]
        dx = float(z["dx"])
    else:
        dem = g["elev"].values.astype(np.float64)
        xs = g["x"].values.astype(np.float64)
        ys = g["y"].values.astype(np.float64)
        dx = abs(float(xs[1] - xs[0])) if len(xs) > 1 else 1000.0
        print(f"    watershed base: grid {dem.shape} (cached after this run)", flush=True)
        print("    D8 flow …", flush=True)
        direc = d8_flow(dem)
        print("    accumulation …", flush=True)
        acc = flow_accum(direc)
        print("    basins …", flush=True)
        raw = label_basins(direc, acc, wcfg.accum_pct, wcfg.min_basin_cells)
        print("    TPI / valley axis …", flush=True)
        radius = max(int(wcfg.tpi_radius_m / max(dx, 1.0)), 1)
        tpi_map = tpi(dem, radius)
        summit, cold = flags_from_tpi(tpi_map, wcfg.summit_tpi, wcfg.coldpool_tpi)
        uy, ux = valley_unit(direc)
        np.savez_compressed(
            cache, raw=raw, tpi=tpi_map, summit=summit, coldpool=cold,
            uy=uy, ux=ux, xs=xs, ys=ys, dx=np.array(dx),
        )
        print(f"    cached {cache}", flush=True)
    print(f"    merge to n_regions={n_regions}", flush=True)
    regions = merge_basins(raw, g["elev"].values.astype(np.float64), xs, ys, int(n_regions))
    return {
        "regions": regions,
        "region_dist": region_distance(regions),
        "tpi": tpi_map,
        "summit": summit,
        "coldpool": cold,
        "uy": uy,
        "ux": ux,
        "xs": np.asarray(xs, dtype=np.float64),
        "ys": np.asarray(ys, dtype=np.float64),
        "dx": float(dx),
        "n_regions": int(max(int(regions.max()), 1)),
        "cfg": wcfg,
    }


def run_llocv(panel, var, fcfg: FreiConfig, n_folds, fit_splits, score_splits, pack=None, jobs: int = 1):
    """
    Protocol-aware LLOCV.
    - Fitting always excludes held-out stations.
    - If score_splits contains 'test' (or 'all'): score held-out x test (final numbers).
    - Otherwise (tuning): score pool stations on the requested splits,
      with a station fold so targets are not also donors.
    """
    panel, held_out, pool, qsplits = apply_evaluation_protocol(panel)
    # Protocol uses 'tune'; older callers still say 'dev'.
    def _canon(names):
        return {"tune" if n == "dev" else n for n in names}
    fit_ok = {"train"} if fit_splits == {"all"} else _canon(set(fit_splits))
    fit_df = panel[(~panel["is_heldout"]) & (panel["split"].isin(fit_ok))]

    score_ok = _canon(set(score_splits))
    # Only the final test uses the held-out stations. Tune scores the pool.
    if score_splits == {"all"} or "test" in score_ok:
        score_df = panel[(panel["is_heldout"]) & (panel["split"].isin(score_ok))]
    else:
        score_df = panel[(~panel["is_heldout"]) & (panel["split"].isin(score_ok))]

    if score_df.empty or fit_df["station_name"].nunique() < fcfg.min_stations:
        return pd.DataFrame(columns=["station_name", "time", "split", "observed", "predicted"])

    model = FreiInterpolator(fcfg, pack=pack)
    rows = []
    first_err = None
    # 'test' is the final number and uses held-out stations.
    # 'tune' scores pool stations with a station fold.
    heldout_score = score_splits == {"all"} or "test" in score_ok

    def _score_group(don, q):
        if don["station_name"].nunique() < fcfg.min_stations or q.empty:
            return None
        hat, _ = model.predict_timestamp(
            don["x"].to_numpy(), don["y"].to_numpy(), don["elev"].to_numpy(),
            don[var].to_numpy(),
            q["x"].to_numpy(), q["y"].to_numpy(), q["elev"].to_numpy(),
            var=var,
            extra_s=_covar_arrays(don, fcfg.enabled_covariates),
            extra_q=_covar_arrays(q, fcfg.enabled_covariates),
        )
        if hat is None:
            return None
        part = q[["station_name", "time", "split"]].copy()
        part["observed"] = q[var].to_numpy()
        part["predicted"] = np.asarray(hat, dtype=float)
        return part

    if heldout_score:
        # Re-select targets: held-out stations on the requested quarters.
        # score_df may still be the pool (old 'dev' path); override it here.
        targets = panel[(panel["is_heldout"]) & (panel["split"].isin(score_ok))]
        groups = [(t, q, panel[(~panel["is_heldout"]) & (panel["time"] == t)])
                  for t, q in targets.groupby("time", sort=False)]
    else:
        # Pool stations on the tune quarters. Score one station fold at a time
        # so the targets are not also the donors (that gave RMSE = 0) and the
        # donors are not empty (that gave n = 0).
        names = score_df["station_name"].unique()
        folds = station_folds(names, max(int(n_folds), 2), seed=22)
        groups = []
        for fold in folds:
            fset = set(fold)
            targets = score_df[score_df["station_name"].isin(fset)]
            for t, q in targets.groupby("time", sort=False):
                don = panel[(~panel["is_heldout"]) & (panel["time"] == t)
                            & (~panel["station_name"].isin(fset))]
                groups.append((t, q, don))

    if jobs == 1:
        results = [_score_group(don, q) for _t, q, don in tqdm(groups, desc=f"  {var}", leave=False)]
    else:
        results = Parallel(n_jobs=jobs, backend="threading")(
            delayed(_score_group)(don, q) for _t, q, don in tqdm(groups, desc=f"  {var}", leave=False)
        )
    for part in results:
        if part is not None:
            rows.append(part)
    if not rows:
        if first_err is not None:
            print(f"  [WARN] Frei score produced no rows: {type(first_err).__name__}: {first_err}", flush=True)
        return pd.DataFrame(columns=["station_name", "time", "split", "observed", "predicted"])
    return pd.concat(rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default=None)
    p.add_argument("--fit", default="train")
    p.add_argument("--score", default="dev,test")
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--months", default=None, help="all | seasonal4 | YYYY-MM,YYYY-MM")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--jobs", type=int, default=None,
                   help="parallel score groups (default: half the logical cores)")
    p.add_argument("--rh-t-mode", default=None, choices=["none", "predicted", "observed"])
    add_hours_arg(p)
    add_time_res_arg(p)
    args = p.parse_args()
    cfg = load_frei_config()
    time_res = apply_time_res(cfg, args)
    variables = [args.variable] if args.variable else cfg["frei"].get("variables_to_process", ["temp_mean"])
    n_folds = args.folds or int(cfg["frei"].get("n_folds", 5))
    months = args.months
    if args.quick:
        n_folds = args.folds or 2
        months = months or "seasonal4"
    for var in variables:
        tuned = load_tuned(var, time_res)
        tuned["trace"] = precip_trace(cfg, time_res, var)
        if args.rh_t_mode:
            tuned["rh_t_mode"] = args.rh_t_mode
        fcfg = cfg_to_frei(cfg, tuned)
        if args.quick:
            fcfg.select_metric = False
        pack = pack_from_master(cfg, fcfg.n_regions)
        panel = load_panel(cfg, var)
        if var.lower().startswith("rh") and fcfg.rh_t_mode in ("predicted", "observed"):
            panel = attach_temperature(panel, cfg)
        if months:
            panel = subset_times(panel, months)
        panel, hours = apply_hour_cut(panel, time_res, args.hours)
        jobs = int(args.jobs) if args.jobs is not None else max(1, (os.cpu_count() or 1) // 2)
        pred = run_llocv(
            panel, var, fcfg, n_folds,
            parse_split_arg(args.fit), parse_split_arg(args.score),
            pack, jobs=jobs,
        )
        out = get_nested_llocv_path("Frei", var, time_res, domain="full")
        pred.to_parquet(out, index=False)
        print(f"{var} folds={n_folds} months={months or 'all'} hours={hours} rh_t={fcfg.rh_t_mode} -> {out}")
        print_split_metrics(pred)


if __name__ == "__main__":
    main()
