#!/usr/bin/env python3
"""
Held-out scoring for IDW.

Donors = pool stations, targets = held-out stations,
scored only on test quarters. Uses the global (p, Fz, k) from tune_idw.py.

Writes the predictions parquet and prints RMSE/MAE/NSE/KGE overall and by quarter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.neighbors import KDTree

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from idw_core import interpolate_idw
from idw_data import load_config, load_global_params, load_panel, print_metrics
from paths import get_llocv_path
from shared.splits.evaluation_sets import (
    apply_evaluation_protocol,
    load_holdout,
    load_quarter_splits,
    quarter_of,
)
from shared.time_res import add_time_res_arg, apply_time_res


TRACE = {
    "precip_sum": {"half_hourly": 0.05, "daily": 0.1, "weekly": 0.7, "monthly": 1.0},
    "snow_mean": {"half_hourly": 0.5, "daily": 0.5, "weekly": 1.0, "monthly": 1.0},
}


def score_field(donors: pd.DataFrame, targets: pd.DataFrame, params: dict,
                var: str = "", trace: float = 1.0) -> pd.DataFrame:
    if len(donors) < 5 or len(targets) < 1:
        return pd.DataFrame()
    k = min(int(params["k"]), len(donors))
    tree = KDTree(donors[["x", "y"]].to_numpy())
    pred = interpolate_idw(
        station_values=donors["value"].to_numpy(),
        station_coords=donors[["x", "y"]].to_numpy(),
        station_elev=donors["elev"].to_numpy(),
        target_coords=targets[["x", "y"]].to_numpy(),
        target_elev=targets["elev"].to_numpy(),
        tree=tree,
        p=float(params["p"]), Fz=float(params["Fz"]), k=k,
        var=var, trace=trace,
    )
    out = targets[["station_name", "x", "y", "time"]].copy()
    out["observed"] = targets["value"].to_numpy()
    out["predicted"] = pred
    out["quarter"] = out["time"].map(quarter_of)
    out["split"] = "test"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variables", nargs="+", default=None)
    p.add_argument("--months", default=None)
    add_time_res_arg(p)
    args = p.parse_args()

    cfg = load_config()
    time_res = apply_time_res(cfg, args)
    domain = cfg.get("domain", {}).get("preset", "full")
    variables = args.variables or ["temp_mean", "precip_sum"]
    held_out, _pool = load_holdout()
    held_set = set(held_out)
    test_quarters = set(load_quarter_splits()["test"])

    print("=" * 72)
    print(f"IDW held-out scoring | {time_res} | test quarters={sorted(test_quarters)}")
    print("=" * 72)

    for var in variables:
        panel = load_panel(cfg, var)
        tagged, _, _, _ = apply_evaluation_protocol(panel)
        if args.months and args.months != "all":
            from idw_data import subset_times
            tagged = subset_times(tagged, args.months)

        params = load_global_params(time_res, var)
        times = sorted(tagged.loc[tagged["quarter"].isin(test_quarters), "time"].unique())
        print(f"\n----- {var}  {len(times)} test timestamps -----")

        parts = []
        for t in times:
            field = tagged[tagged["time"] == t]
            donors = field[~field["station_name"].isin(held_set)]
            targets = field[field["station_name"].isin(held_set)]
            scored = score_field(donors, targets, params, var=var,
                                 trace=TRACE.get(var, {}).get(time_res, 1.0))
            if len(scored):
                parts.append(scored)

        if not parts:
            print(f"  [WARN] no scored rows for {var}")
            continue

        pred = pd.concat(parts, ignore_index=True)
        out = get_llocv_path(
            "IDW", domain, var, 1000,
            start_date=str(cfg["start_date"]),
            end_date=str(cfg["end_date"]),
            time_resolution=time_res,
        )
        pred.to_parquet(out, index=False)
        print(f"  wrote {out.name}  ({len(pred)} rows)")
        print_metrics(pred, by="quarter")

    print("\nDone.")


if __name__ == "__main__":
    main()
