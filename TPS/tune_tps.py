#!/usr/bin/env python3
"""DEV search for TPS: λ, αz, protocol."""

from __future__ import annotations

from pathlib import Path
import argparse
import itertools
import sys

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_tps_tuned_params_path
from tps_core import TPSConfig
from tps_data import (
    attach_pack_terrain,
    attach_temperature,
    compute_block,
    compute_metrics,
    load_panel,
    load_tps_config,
    pack_from_master,
    subset_times,
)
from llocv_tps import cfg_to_tps, pack_from_master, run_llocv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default=None)
    p.add_argument("--months", default=None)
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--rh-t-mode", default=None, choices=["none", "predicted", "observed"])
    args = p.parse_args()
    cfg = load_tps_config()
    time_res = cfg["time_resolution"]
    block = cfg["tps"]
    variables = [args.variable] if args.variable else block.get("variables_to_process", ["temp_mean"])
    search = block.get("search", {})
    n_folds = args.folds or (2 if args.quick else int(block.get("n_folds", 5)))
    months = args.months or ("seasonal4" if args.quick else compute_block(cfg).get("tune_months", "seasonal4"))
    for var in variables:
        panel = load_panel(cfg, var)
        rh_mode = args.rh_t_mode or block.get("rh_t_mode", "none")
        if str(var).lower().startswith("rh") and rh_mode in ("predicted", "observed"):
            panel = attach_temperature(panel, cfg)
        pack = pack_from_master(cfg, int(block.get("n_regions", 6)))
        panel = attach_pack_terrain(panel, pack)
        panel = subset_times(panel, months)
        if pack is None:
            print(f"{var}: no DEM pack, watershed mode will equal global")
        else:
            print(f"{var}: watershed pack n_regions={pack['n_regions']}")
        best = None
        if args.quick:
            protocols = ["tps"]
            kernels = [block.get("kernel", "3d")]
            lams = [block.get("lam", 1.0)]
            azs = [block.get("alpha_z", 100.0)]
            modes = ["global"]
        else:
            protocols = list(search.get("protocol", [block.get("protocol", "eobs")]))
            if time_res == "monthly":
                protocols = ["tps"]
            kernels = list(search.get("kernel", [block.get("kernel", "3d")]))
            lams = list(search.get("lam", [block.get("lam", 1.0)]))
            azs = list(search.get("alpha_z", [block.get("alpha_z", 100.0)]))
            modes = list(search.get("alpha_z_mode", [block.get("alpha_z_mode", "global")]))
        for proto, kern, lam, az, mode in itertools.product(protocols, kernels, lams, azs, modes):
            tcfg = cfg_to_tps(cfg, {
                "protocol": proto, "kernel": kern, "lam": lam,
                "alpha_z": az, "alpha_z_mode": mode,
                "rh_t_mode": rh_mode,
            })
            this_pack = pack if mode == "watershed" else None
            pred = run_llocv(panel, var, tcfg, n_folds, {"train"}, {"dev"}, this_pack, time_res)
            met = compute_metrics(pred["observed"], pred["predicted"])
            print(f"{var} proto={proto} kern={kern} lam={lam} az={az} mode={mode} RMSE={met['rmse']:.3f} n={len(pred)}")
            if best is None or met["rmse"] < best["rmse"]:
                best = {
                    "protocol": proto, "kernel": kern, "lam": float(lam),
                    "alpha_z": float(az), "alpha_z_mode": mode,
                    "rmse": float(met["rmse"]),
                }
        path = get_tps_tuned_params_path(var, time_res)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(best, f)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
