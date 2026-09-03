#!/usr/bin/env python3
"""Frei production maps on the master grid."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_interpolated_map_path, get_master_grid_path
from frei_core import FreiInterpolator
from frei_data import data_sources, load_frei_config, load_panel
from llocv_frei import cfg_to_frei, load_tuned, pack_from_master

try:
    import xarray as xr
except ImportError:
    xr = None


def main():
    cfg = load_frei_config()
    if xr is None:
        raise RuntimeError("xarray required")
    domain = cfg["domain"]["preset"]
    time_res = cfg["time_resolution"]
    _, _, grid_method = data_sources(cfg)
    variables = cfg["frei"].get("variables_to_process") or ["temp_mean"]
    min_stations = int(cfg.get("min_stations_per_field", 10))

    for res in cfg.get("resolutions_to_process", [1000]):
        grid = xr.open_dataset(get_master_grid_path(grid_method, res))
        x = grid["x"].values.astype(np.float64)
        y = grid["y"].values.astype(np.float64)
        z = grid["elev"].values.astype(np.float64) if "elev" in grid else np.zeros((y.size, x.size))
        xx, yy = np.meshgrid(x, y)
        zz = z if z.ndim == 2 else np.zeros_like(xx)
        xq, yq, zq = xx.ravel(), yy.ravel(), zz.ravel()

        for var in variables:
            panel = load_panel(cfg, var)
            fcfg = cfg_to_frei(cfg, load_tuned(var, time_res))
            pack = pack_from_master(cfg, fcfg.n_regions)
            model = FreiInterpolator(fcfg, pack=pack)
            use_profile = not str(var).lower().startswith("precip")
            fields, used = [], []
            for ts, sl in panel.groupby("time"):
                if sl["station_name"].nunique() < min_stations:
                    continue
                hat, _ = model.predict_timestamp(
                    sl["x"].to_numpy(), sl["y"].to_numpy(), sl["elev"].to_numpy(),
                    sl[var].to_numpy(), xq, yq, zq, use_profile=use_profile,
                )
                fields.append(hat.reshape(yy.shape).astype(np.float32))
                used.append(np.datetime64(ts, "ns"))
            if not fields:
                continue
            da = xr.DataArray(np.stack(fields), dims=("time", "y", "x"),
                              coords={"time": np.array(used), "y": y, "x": x}, name=var)
            out = get_interpolated_map_path("Frei", domain, var, res, cfg["start_date"], cfg["end_date"], time_res)
            xr.Dataset({var: da}).to_netcdf(out)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
