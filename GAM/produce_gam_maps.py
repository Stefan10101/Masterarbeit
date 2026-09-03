#!/usr/bin/env python3
"""GAM production maps on the master grid."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_interpolated_map_path, get_master_grid_path
from gam_core import GAMInterpolator
from gam_data import clc_group, data_sources, load_gam_config, load_panel
from llocv_gam import cfg_to_gam, load_tuned

try:
    import xarray as xr
except ImportError:
    xr = None


def main():
    cfg = load_gam_config()
    if xr is None:
        raise RuntimeError("xarray required")
    domain = cfg["domain"]["preset"]
    time_res = cfg["time_resolution"]
    _, _, grid_method = data_sources(cfg)
    variables = cfg["gam"].get("variables_to_process") or ["temp_mean"]
    min_stations = int(cfg.get("min_stations_per_field", 10))

    for res in cfg.get("resolutions_to_process", [1000]):
        grid = xr.open_dataset(get_master_grid_path(grid_method, res))
        x = grid["x"].values.astype(np.float64)
        y = grid["y"].values.astype(np.float64)
        z = grid["elev"].values.astype(np.float64) if "elev" in grid else np.zeros((y.size, x.size))
        xx, yy = np.meshgrid(x, y)
        zz = z if z.ndim == 2 else np.zeros_like(xx)
        clc = None
        for name in ("clc_code", "clc", "landcover"):
            if name in grid:
                clc = clc_group(np.nan_to_num(grid[name].values, nan=0).astype(np.int32)).ravel()
                break
        xq, yq, zq = xx.ravel(), yy.ravel(), zz.ravel()

        for var in variables:
            panel = load_panel(cfg, var)
            gcfg = cfg_to_gam(cfg, load_tuned(var, time_res))
            fields, used = [], []
            for ts, sl in panel.groupby("time"):
                if sl["station_name"].nunique() < min_stations:
                    continue
                model = GAMInterpolator(gcfg)
                hat = model.predict_timestamp(
                    sl["x"].to_numpy(), sl["y"].to_numpy(), sl["elev"].to_numpy(),
                    sl[var].to_numpy(), xq, yq, zq,
                    clc=sl["clc_group"].to_numpy(), clc_q=clc,
                    slope=sl["slope"].to_numpy() if "slope" in sl.columns else None,
                    sinasp=sl["sinasp"].to_numpy() if "sinasp" in sl.columns else None,
                    cosasp=sl["cosasp"].to_numpy() if "cosasp" in sl.columns else None,
                    var=var,
                )
                fields.append(hat.reshape(yy.shape).astype(np.float32))
                used.append(np.datetime64(ts, "ns"))
            if not fields:
                continue
            da = xr.DataArray(np.stack(fields), dims=("time", "y", "x"),
                              coords={"time": np.array(used), "y": y, "x": x}, name=var)
            out = get_interpolated_map_path("GAM", domain, var, res, cfg["start_date"], cfg["end_date"], time_res)
            xr.Dataset({var: da}).to_netcdf(out)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
