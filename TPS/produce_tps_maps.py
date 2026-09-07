#!/usr/bin/env python3
"""TPS / E-OBS production maps on the master grid."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_interpolated_map_path, get_master_grid_path, get_tps_tuned_params_path
from tps_core import TPSInterpolator, attach_watershed
from tps_data import data_sources, load_panel, load_tps_config, precip_trace, uses_two_step
from llocv_tps import cfg_to_tps, load_tuned, month_key, pack_from_master, predict_rows

try:
    import xarray as xr
except ImportError:
    xr = None


def main():
    cfg = load_tps_config()
    if xr is None:
        raise RuntimeError("xarray required")
    domain = cfg["domain"]["preset"]
    time_res = cfg["time_resolution"]
    start = str(cfg["start_date"]).replace("-", "")
    end = str(cfg["end_date"]).replace("-", "")
    _, _, grid_method = data_sources(cfg)
    variables = cfg["tps"].get("variables_to_process") or ["temp_mean"]
    min_stations = int(cfg.get("min_stations_per_field", 10))

    for res in cfg.get("resolutions_to_process", [1000]):
        grid = xr.open_dataset(get_master_grid_path(grid_method, res))
        x = grid["x"].values.astype(np.float64)
        y = grid["y"].values.astype(np.float64)
        if "elev" in grid:
            z = grid["elev"].values.astype(np.float64)
        else:
            z = np.zeros_like(x)
        xx, yy = np.meshgrid(x, y)
        if z.ndim == 2:
            zz = z
        else:
            zz = np.meshgrid(x, y)[0] * 0.0
        xq, yq, zq = xx.ravel(), yy.ravel(), zz.ravel()

        for var in variables:
            panel = load_panel(cfg, var)
            tuned = load_tuned(var, time_res)
            tuned["trace"] = precip_trace(cfg, time_res, var)
            tcfg = cfg_to_tps(cfg, tuned)
            model = TPSInterpolator(tcfg)
            attach_watershed(
                model,
                pack_from_master(cfg, tcfg.n_regions),
                panel["x"].to_numpy(), panel["y"].to_numpy(), panel["elev"].to_numpy(),
            )
            times = sorted(panel["time"].unique())
            fields = []
            used = []
            for ts in times:
                sl = panel[panel["time"] == ts]
                if sl["station_name"].nunique() < min_stations:
                    continue
                # fake query frame so predict_rows can reuse E-OBS monthly logic
                q = pd.DataFrame({"x": xq, "y": yq, "elev": zq, "time": ts})
                hat = predict_rows(model, sl, q, var, monthly_panel=panel, time_res=time_res)
                fields.append(hat.reshape(yy.shape).astype(np.float32))
                used.append(np.datetime64(ts, "ns"))
            if not fields:
                print(f"skip {var}: no timestamps")
                continue
            da = xr.DataArray(
                np.stack(fields),
                dims=("time", "y", "x"),
                coords={"time": np.array(used), "y": y, "x": x},
                name=var,
            )
            ds = xr.Dataset({var: da})
            out = get_interpolated_map_path("TPS", domain, var, res, cfg["start_date"], cfg["end_date"], time_res)
            ds.to_netcdf(out)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
