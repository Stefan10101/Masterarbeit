#!/usr/bin/env python3
"""
RGI production maps.

Fit one model per variable on TRAIN+DEV station times (all stations).
Predict every timestamp on the master grid. Query cells never send values.
Station parquet from this script is a frozen-model check, not nested LLOCV.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from shared.map_io import save_map_with_partials
from paths import (
    get_interpolated_map_path,
    get_master_grid_path,
    get_rgi_model_path,
    get_rgi_tuned_params_path,
)
from rgi_core import RGI
from rgi_data import (
    attach_extras,
    data_sources,
    extra_cols_for_var,
    load_config,
    load_panel,
    subset_times,
)
from llocv_rgi import cfg_to_rgi
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res, variables_from_cfg

try:
    import xarray as xr
except ImportError:
    xr = None


def load_tuned(var: str, time_res: str) -> dict:
    path = get_rgi_tuned_params_path(var, time_res)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def grid_clc(grid) -> np.ndarray:
    for name in ("clc_code", "clc", "landcover", "CLC"):
        if name in grid:
            return np.nan_to_num(grid[name].values.astype(np.float32), nan=0.0)
    return None


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--variables", nargs="*", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--jobs", type=int, default=None,
                   help="parallel timestamps (default: 1 on CUDA, half cores on CPU)")
    p.add_argument("--months", default=None)
    add_hours_arg(p)
    add_time_res_arg(p)
    args = p.parse_args()
    cfg = load_config()
    apply_time_res(cfg, args)
    if args.quick:
        cfg.setdefault("rgi", {})
        cfg["rgi"]["epochs"] = min(int(cfg["rgi"].get("epochs", 80)), 15)
        cfg["rgi"]["patience"] = min(int(cfg["rgi"].get("patience", 12)), 4)
        cfg["_quick_months"] = args.months or "seasonal4"
    if xr is None:
        raise RuntimeError("xarray is required for produce_rgi_maps.py")

    domain = cfg["domain"]["preset"]
    time_res = cfg["time_resolution"]
    start = pd.Timestamp(cfg["start_date"])
    end = pd.Timestamp(cfg["end_date"])
    resolutions = cfg.get("resolutions_to_process", [1000])
    rgi_cfg = cfg.get("rgi", {})
    variables = variables_from_cfg(cfg, args.variables)
    save_model = bool(rgi_cfg.get("save_model", True))
    tile = int(rgi_cfg.get("predict_tile", 120000))
    min_stations = int(cfg.get("min_stations_per_field", 10))
    _, _, grid_method = data_sources(cfg)

    from rgi_core import describe_device, resolve_device
    print("=" * 72)
    print(f"RGI production | {domain} | {time_res}")
    print(f"Period: {start.date()} → {end.date()}")
    print(f"  device {describe_device(resolve_device(rgi_cfg.get('device', 'auto')))}")
    print("=" * 72)

    for var in variables:
        panel = attach_extras(load_panel(cfg, var), cfg, var)
        panel, _hours = apply_hour_cut(panel, time_res, args.hours)
        # Fit on full train+dev. --months / --quick only select prediction times.
        fit = panel[panel["split"].isin(("train", "dev"))]
        if fit.empty:
            print(f"  [SKIP] {var}: no train/dev rows")
            continue
        # --months always restricts prediction. --quick defaults to seasonal4.
        pred = panel
        month_spec = args.months or cfg.get("_quick_months")
        if month_spec:
            pred = subset_times(panel, month_spec)
        tuned = load_tuned(var, time_res)
        tuned["extra_cols"] = extra_cols_for_var(var, cfg)
        tuned["variable"] = var
        rcfg = cfg_to_rgi(cfg, tuned)
        rcfg.two_step = bool(rgi_cfg.get("two_step", True)) and (
            str(var).lower().startswith("precip") or str(var).lower().startswith("snow")
        )
        print(f"\n>>> {var}  k={rcfg.k} L={rcfg.n_layers} "
              f"alpha={rcfg.alpha} az={rcfg.alpha_z}")
        print(f"  fitting on {len(fit):,} train+dev rows")
        val = fit[fit["split"] == "dev"]
        model = RGI(rcfg)
        info = model.fit(fit, var, val_df=val if len(val) else None)
        print(f"  fit done  val_mse={info['best_val_mse']:.4f}  epochs={info['epochs']}")
        if save_model:
            path = get_rgi_model_path(var, time_res)
            model.save(path)
            print(f"  model → {path}")

        for res in resolutions:
            grid_path = get_master_grid_path(grid_method, res)
            grid = xr.open_dataset(grid_path)
            ny, nx = grid.y.size, grid.x.size
            xx, yy = np.meshgrid(grid.x.values, grid.y.values)
            elev = grid["elev"].values if "elev" in grid else np.full((ny, nx), np.nan)
            clc = grid_clc(grid)
            flat_x = xx.ravel()
            flat_y = yy.ravel()
            # COP30 / master-grid nodata is a finite sentinel (often << 0), not NaN.
            elev_f = elev.astype(np.float32)
            valid_cell = np.isfinite(elev_f.ravel()) & (elev_f.ravel() > -100.0)
            flat_e = np.where(valid_cell, elev_f.ravel(), 0.0).astype(np.float32)
            flat_c = (
                np.zeros(flat_x.size, dtype=np.int32)
                if clc is None
                else np.nan_to_num(clc.ravel(), nan=0.0).astype(np.int32)
            )

            out_file = get_interpolated_map_path(
                "RGI", domain, var, res,
                start_date=str(start.date()),
                end_date=str(end.date()),
                time_resolution=time_res,
            )
            if out_file.exists() and not cfg.get("overwrite_maps", True):
                print(f"  skip existing {out_file.name}")
                grid.close()
                continue

            times = sorted(pred["time"].unique())
            # Concurrent large-graph forwards on a single consumer GPU freeze the
            # driver. Force sequential when the model is on CUDA unless the user
            # explicitly overrides with --jobs.
            on_cuda = (
                torch is not None
                and hasattr(model, "device")
                and getattr(model.device, "type", None) == "cuda"
            )
            if args.jobs is not None:
                n_jobs = max(1, int(args.jobs))
            elif on_cuda:
                n_jobs = 1
            else:
                n_jobs = max(1, (os.cpu_count() or 1) // 2)
            if on_cuda and n_jobs > 1:
                print(f"  warning: --jobs={n_jobs} on CUDA may freeze the GPU")
            # Cap tile size on CUDA so a single forward (esp. k=32, L=5) stays
            # inside consumer VRAM while still saturating the GPU. Config value
            # is still respected on CPU.
            tile_use = tile
            if on_cuda and tile_use > 96000:
                tile_use = 96000
                print(f"  CUDA tile capped at {tile_use} (config was {tile})")

            def one(t):
                df_t = pred[pred["time"] == t]
                if len(df_t) < min_stations:
                    return None
                pred_flat = np.full(flat_x.size, np.nan, dtype=np.float32)
                idx = np.flatnonzero(valid_cell)
                n_tiles = 0
                for s in range(0, len(idx), tile_use):
                    sl = idx[s:s + tile_use]
                    qry = pd.DataFrame({
                        "x": flat_x[sl],
                        "y": flat_y[sl],
                        "elev": flat_e[sl],
                        "clc_code": flat_c[sl],
                        var: np.zeros(len(sl), dtype=np.float32),
                    })
                    pred_flat[sl] = model.predict_frame(df_t, qry, var)
                    n_tiles += 1
                    # Periodic cleanup only — every-tile sync + empty_cache
                    # leaves the GPU idle. Every 4 tiles is enough to bound VRAM.
                    if on_cuda and (n_tiles % 4 == 0):
                        torch.cuda.empty_cache()
                if on_cuda:
                    torch.cuda.empty_cache()
                return t, pred_flat.reshape(ny, nx)

            if n_jobs == 1:
                results = [one(t) for t in tqdm(times, desc=f"  {var} {res}m")]
            else:
                results = Parallel(n_jobs=n_jobs, backend="threading")(
                    delayed(one)(t) for t in tqdm(times, desc=f"  {var} {res}m")
                )
            results = [r for r in results if r is not None]
            if not results:
                print(f"  {var}: no fields")
                grid.close()
                continue
            kept, stacks = zip(*results)
            data_3d = np.stack(stacks).astype(np.float32)
            n_kept = len(kept)
            # data_3d already stacked to n_kept
            time_coord = pd.DatetimeIndex(pd.to_datetime(kept, utc=True)).tz_convert(
                "UTC"
            ).tz_localize(None).to_numpy(dtype="datetime64[ns]")
            ds = xr.Dataset(
                {var: (("time", "y", "x"), data_3d)},
                coords={"time": time_coord, "y": grid.y.values, "x": grid.x.values},
            )
            if "elev" in grid:
                ds["elev"] = (("y", "x"), grid.elev.values.astype(np.float32))
            ds["k"] = ("time", np.full(n_kept, rcfg.k, dtype=np.int16))
            ds.attrs.update({
                "title": f"{var} - RGI - {time_res} - {domain}",
                "domain": domain,
                "resolution_m": int(res),
                "method": "RGI",
                "method_full": "Residual Graph Interpolator (initial residual, spatial-only)",
                "time_resolution": time_res,
                "variable": var,
                "k": rcfg.k,
                "n_layers": rcfg.n_layers,
                "alpha": rcfg.alpha,
                "alpha_z": rcfg.alpha_z,
                "start_date": str(start.date()),
                "end_date": str(end.date()),
                "crs": "EPSG:31287",
                "created": datetime.now().isoformat(),
            })
            out_file.parent.mkdir(parents=True, exist_ok=True)
            save_map_with_partials(ds, out_file, time_res)
            print(f"  saved {n_kept} timesteps → {out_file}")
            grid.close()
            del data_3d, ds
            gc.collect()

    print("\nRGI production finished.")


if __name__ == "__main__":
    main()
