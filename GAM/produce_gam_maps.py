#!/usr/bin/env python3
"""GAM production maps on the master grid."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time
import traceback

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_interpolated_map_path, get_master_grid_path
from gam_core import GAMInterpolator
from gam_data import (
    attach_pack_terrain,
    attach_temperature,
    clc_group,
    data_sources,
    load_gam_config,
    load_panel,
    pack_from_master,
    precip_trace,
    subset_splits,
    subset_times,
    subset_years,
)
from llocv_gam import cfg_to_gam, load_tuned
from shared.terrain import aspect_trig, slope_aspect_from_dem

try:
    import xarray as xr
except ImportError:
    xr = None


def grid_terrain(elev, gx, gy):
    dx = float(gx[1] - gx[0]) if len(gx) > 1 else 1.0
    dy = float(gy[1] - gy[0]) if len(gy) > 1 else 1.0
    slope, aspect = slope_aspect_from_dem(elev, abs(dx), abs(dy))
    sinasp, cosasp = aspect_trig(aspect)
    return slope, sinasp, cosasp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default=None)
    p.add_argument("--years", default=None)
    p.add_argument("--months", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--rh-t-mode", default=None, choices=["none", "predicted", "observed"])
    return p.parse_args()


def main():
    args = parse_args()
    print("produce_gam_maps start", flush=True)
    if xr is None:
        raise RuntimeError("xarray required")
    cfg = load_gam_config()
    domain = cfg["domain"]["preset"]
    time_res = cfg["time_resolution"]
    _, _, grid_method = data_sources(cfg)
    variables = [args.variable] if args.variable else (cfg["gam"].get("variables_to_process") or ["temp_mean"])
    months = args.months or ("seasonal4" if args.quick else None)
    splits = args.split or ("test" if args.quick else "all")
    pack = pack_from_master(cfg, int(cfg.get("gam", {}).get("n_regions", 6)))
    min_stations = int(cfg.get("min_stations_per_field", 10))
    print(
        f"domain={domain} time_res={time_res} vars={variables} "
        f"res={cfg.get('resolutions_to_process', [1000])} grid_from={grid_method}",
        flush=True,
    )

    for res in cfg.get("resolutions_to_process", [1000]):
        gpath = get_master_grid_path(grid_method, res)
        print(f"open grid {gpath} exists={gpath.exists()}", flush=True)
        grid = xr.open_dataset(gpath)
        x = grid["x"].values.astype(np.float64)
        y = grid["y"].values.astype(np.float64)
        z = grid["elev"].values.astype(np.float64) if "elev" in grid else np.zeros((y.size, x.size))
        xx, yy = np.meshgrid(x, y)
        zz = z if z.ndim == 2 else np.zeros_like(xx)
        print(f"grid ny={y.size} nx={x.size} ncell={xx.size} terrain…", flush=True)
        slope, sinasp, cosasp = grid_terrain(zz, x, y)
        clc = None
        for name in ("clc_code", "clc", "landcover"):
            if name in grid:
                clc = clc_group(np.nan_to_num(grid[name].values, nan=0).astype(np.int32)).ravel()
                break
        xq, yq, zq = xx.ravel(), yy.ravel(), zz.ravel()
        sl_q, sa_q, ca_q = slope.ravel(), sinasp.ravel(), cosasp.ravel()
        print(f"grid ready clc={'yes' if clc is not None else 'no'}", flush=True)

        for var in variables:
            t_var = time.time()
            print(f"{var} load panel…", flush=True)
            panel = load_panel(cfg, var)
            tuned = load_tuned(var, time_res)
            tuned["trace"] = precip_trace(cfg, time_res, var)
            if args.rh_t_mode:
                tuned["rh_t_mode"] = args.rh_t_mode
            gcfg = cfg_to_gam(cfg, tuned)
            if str(var).lower().startswith("rh") and gcfg.rh_t_mode in ("predicted", "observed"):
                panel = attach_temperature(panel, cfg)
                if gcfg.rh_t_mode == "observed":
                    print("produce: rh_t_mode=observed has no grid T; using predicted", flush=True)
                    gcfg.rh_t_mode = "predicted"
            panel = attach_pack_terrain(panel, pack)
            panel = subset_splits(panel, splits)
            if months:
                panel = subset_times(panel, months)
            panel = subset_years(panel, args.years)
            n_t = int(panel["time"].nunique())
            print(
                f"{var} rows={len(panel)} times={n_t} "
                f"form={gcfg.formula} ns={gcfg.n_splines} tau={gcfg.tau_wet} "
                f"tuned={bool(tuned)}",
                flush=True,
            )
            fields, used = [], []
            n_skip = 0
            times = list(panel.groupby("time"))
            for i, (ts, sl) in enumerate(times, start=1):
                if sl["station_name"].nunique() < min_stations:
                    n_skip += 1
                    continue
                t0 = time.time()
                model = GAMInterpolator(gcfg)
                tmean = sl["temp_mean"].to_numpy() if "temp_mean" in sl.columns else None
                tmean_q = None
                if tmean is not None and gcfg.rh_t_mode == "predicted":
                    tmod = GAMInterpolator(gcfg)
                    tmean_q = tmod.predict_timestamp(
                        sl["x"].to_numpy(), sl["y"].to_numpy(), sl["elev"].to_numpy(),
                        tmean, xq, yq, zq,
                        clc=sl["clc_group"].to_numpy(), clc_q=clc,
                        slope=sl["slope"].to_numpy() if "slope" in sl.columns else None,
                        slope_q=sl_q,
                        sinasp=sl["sinasp"].to_numpy() if "sinasp" in sl.columns else None,
                        sinasp_q=sa_q,
                        cosasp=sl["cosasp"].to_numpy() if "cosasp" in sl.columns else None,
                        cosasp_q=ca_q,
                        var="temp_mean",
                    )
                region = sl["region"].to_numpy() if "region" in sl.columns else None
                region_q = None
                if pack is not None and "regions" in pack:
                    from shared.watersheds import sample_region
                    region_q = sample_region(pack["regions"], pack["xs"], pack["ys"], xq, yq)
                hat = model.predict_timestamp(
                    sl["x"].to_numpy(), sl["y"].to_numpy(), sl["elev"].to_numpy(),
                    sl[var].to_numpy(), xq, yq, zq,
                    clc=sl["clc_group"].to_numpy(), clc_q=clc,
                    slope=sl["slope"].to_numpy() if "slope" in sl.columns else None,
                    slope_q=sl_q,
                    sinasp=sl["sinasp"].to_numpy() if "sinasp" in sl.columns else None,
                    sinasp_q=sa_q,
                    cosasp=sl["cosasp"].to_numpy() if "cosasp" in sl.columns else None,
                    cosasp_q=ca_q,
                    var=var,
                    trace=gcfg.trace,
                    tmean=tmean, tmean_q=tmean_q,
                    region=region, region_q=region_q,
                )
                fields.append(hat.reshape(yy.shape).astype(np.float32))
                used.append(np.datetime64(ts, "ns"))
                be = getattr(model, "_backend", "?")
                print(
                    f"{var} {i}/{len(times)} {ts} stations={sl['station_name'].nunique()} "
                    f"{time.time() - t0:.1f}s backend={be}",
                    flush=True,
                )
            if not fields:
                print(f"{var} no fields written skip={n_skip}", flush=True)
                continue
            da = xr.DataArray(np.stack(fields), dims=("time", "y", "x"),
                              coords={"time": np.array(used), "y": y, "x": x}, name=var)
            out = get_interpolated_map_path("GAM", domain, var, res, cfg["start_date"], cfg["end_date"], time_res)
            print(f"{var} write {out}", flush=True)
            xr.Dataset({var: da}).to_netcdf(out)
            print(f"wrote {out} fields={len(fields)} skip={n_skip} {time.time() - t_var:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
