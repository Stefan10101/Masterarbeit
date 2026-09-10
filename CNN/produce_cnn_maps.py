#!/usr/bin/env python3
"""CNN production maps. Trains if no frozen weights exist."""

from __future__ import annotations

from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_cnn_model_path, get_interpolated_map_path, get_master_grid_path
from cnn_core import CNNInterpolator, cyclic_time, grid_terrain, idw_raster, station_to_raster, two_step_var
from cnn_data import clc_group, data_sources, load_cnn_config, load_panel, precip_trace
from train_cnn import cfg_to_cnn, grid_index, is_subdaily, load_tuned, train_one
from shared.time_res import add_time_res_arg, apply_time_res

try:
    import xarray as xr
except ImportError:
    xr = None


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--months", default=None)
    add_time_res_arg(p)
    args = p.parse_args()
    print("produce_cnn_maps start", flush=True)
    cfg = load_cnn_config()
    apply_time_res(cfg, args)
    if args.quick:
        cfg.setdefault("cnn", {})
        cfg["cnn"]["epochs"] = min(int(cfg["cnn"].get("epochs", 40)), 8)
        cfg["cnn"]["patience"] = min(int(cfg["cnn"].get("patience", 8)), 3)
    if xr is None:
        raise RuntimeError("xarray required")
    domain = cfg["domain"]["preset"]
    time_res = cfg["time_resolution"]
    _, _, grid_method = data_sources(cfg)
    variables = [args.variable] if args.variable else (cfg["cnn"].get("variables_to_process") or ["temp_mean"])
    min_stations = int(cfg.get("min_stations_per_field", 10))
    print(f"domain={domain} time_res={time_res} vars={variables}", flush=True)

    for res in cfg.get("resolutions_to_process", [1000]):
        gpath = get_master_grid_path(grid_method, res)
        print(f"open grid {gpath} exists={gpath.exists()}", flush=True)
        grid = xr.open_dataset(gpath)
        gx = grid["x"].values.astype(np.float64)
        gy = grid["y"].values.astype(np.float64)
        elev = grid["elev"].values.astype(np.float64) if "elev" in grid else np.zeros((gy.size, gx.size))
        clc = np.zeros_like(elev)
        for name in ("clc_code", "clc", "landcover"):
            if name in grid:
                clc = clc_group(np.nan_to_num(grid[name].values, nan=0).astype(np.int32)).astype(np.float64)
                break
        slope, sinasp, cosasp = grid_terrain(elev, gx, gy)
        ny, nx = elev.shape

        for var in variables:
            t_var = time.time()
            print(f"{var} load panel…", flush=True)
            panel = load_panel(cfg, var)
            if args.quick or args.months:
                from Kriging.kriging_data import subset_times
                panel = subset_times(panel, args.months or "seasonal4")
            ccfg = cfg_to_cnn(cfg, load_tuned(var, time_res))
            wpath = get_cnn_model_path(var, time_res)
            print(f"{var} weights={wpath} exists={wpath.exists()}", flush=True)
            if wpath.exists():
                try:
                    blob = torch.load(wpath, map_location="cpu", weights_only=False)
                except TypeError:
                    blob = torch.load(wpath, map_location="cpu")
                in_ch = int(blob.get("in_ch", 9))
                out_ch = int(blob.get("out_ch", 1))
                model = CNNInterpolator(ccfg, in_ch=in_ch, out_ch=out_ch)
                model.load_state_dict(blob)
                model.net.to(model.device)
                print(
                    f"{var} loaded n_levels={model.cfg.n_levels} base={model.cfg.base_ch} "
                    f"in={model.in_ch} out={model.out_ch} device={model.device}",
                    flush=True,
                )
            else:
                print(f"{var} no weights, training…", flush=True)
                model, _ = train_one(
                    panel, var, ccfg, elev, clc, gx, gy,
                    slope, sinasp, cosasp, time_res=time_res,
                    trace=precip_trace(cfg, time_res, var),
                )
                torch.save(model.state_dict(), wpath)
            fields, used = [], []
            subdaily = is_subdaily(panel, time_res)
            times = list(panel.groupby("time"))
            for i, (ts, sl) in enumerate(times, start=1):
                if sl["station_name"].nunique() < min_stations:
                    continue
                t0 = time.time()
                rows, cols = grid_index(sl["x"].to_numpy(), sl["y"].to_numpy(), gx, gy)
                val = sl[var].to_numpy()
                field, mask = station_to_raster(rows, cols, val, ny, nx)
                base = idw_raster(rows, cols, val, ny, nx, k=ccfg.idw_k) if model.cfg.residual_idw else None
                inp = np.where(mask > 0, field - base, 0.0).astype(np.float32) if base is not None else field
                tch = cyclic_time(ts, subdaily) if model.cfg.cyclic_time else None
                hat = model.predict_raster(
                    inp, mask, elev, clc,
                    slope=slope, sinasp=sinasp, cosasp=cosasp,
                    time_ch=tch,
                    two_step=model.cfg.two_step and two_step_var(var),
                    base=base,
                    var=var,
                )
                fields.append(hat.astype(np.float32))
                used.append(np.datetime64(ts, "ns"))
                if i == 1 or i % 6 == 0 or i == len(times):
                    print(f"{var} {i}/{len(times)} {ts} {time.time() - t0:.1f}s", flush=True)
            if not fields:
                print(f"{var} no fields", flush=True)
                continue
            da = xr.DataArray(np.stack(fields), dims=("time", "y", "x"),
                              coords={"time": np.array(used), "y": gy, "x": gx}, name=var)
            out = get_interpolated_map_path("CNN", domain, var, res, cfg["start_date"], cfg["end_date"], time_res)
            print(f"{var} write {out}", flush=True)
            xr.Dataset({var: da}).to_netcdf(out)
            print(f"wrote {out} fields={len(fields)} {time.time() - t_var:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
