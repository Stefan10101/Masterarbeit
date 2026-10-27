#!/usr/bin/env python3
"""Synthetic experiments for GAM. Ablation via extras, multi-field, baselines, gate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "GAM"))
sys.path.insert(0, str(CODE / "shared"))

from synthetic import generate_synthetic_panel
from synthetic_score import (
    score_recovery, print_score, assert_no_leak,
    mean_baseline, idw_baseline, print_baselines,
    gate_diagnostic, print_gate, aggregate, print_ablation,
)
from gam_core import GAMConfig, GAMInterpolator

VARS = ["temp_mean", "precip_sum", "snow_mean", "wind_mean", "rh_mean"]


def n_for(var: str) -> int:
    return 200 if var.startswith(("precip", "snow")) else 120


def run_one(var: str):
    panel = generate_synthetic_panel(var, n_stations=n_for(var), n_fields=20, seed=23)
    held = set(panel.loc[panel["is_heldout"], "station_name"].unique())
    test_times = panel.loc[panel["split"] == "test", "time"].unique()
    gcfg = GAMConfig(formula="s_xy_s_elev", n_splines=8, min_stations=8,
                     use_clc=False, use_terrain=True)
    model = GAMInterpolator(gcfg)
    mets = []
    # wind: TPI goes in the slope slot; snow: slope + northness (sinasp)
    if var.startswith("wind"):
        covar, s_col, a_col = "tpi", "tpi", None
    elif var.startswith("snow"):
        covar, s_col, a_col = "slope", "slope", "northness"
    else:
        covar, s_col, a_col = "none", None, None
    mets_drop = []
    for i, t0 in enumerate(test_times):
        field = panel[panel["time"] == t0]
        donors = field[~field["is_heldout"]]
        queries = field[field["is_heldout"]]
        assert_no_leak(donors, held)
        if len(queries) < 5:
            continue
        def _pred(use_extra):
            sl = donors[s_col].to_numpy() if (use_extra and s_col) else None
            slq = queries[s_col].to_numpy() if (use_extra and s_col) else None
            sa = donors[a_col].to_numpy() if (use_extra and a_col) else None
            saq = queries[a_col].to_numpy() if (use_extra and a_col) else None
            return model.predict_timestamp(
                donors["x"].to_numpy(), donors["y"].to_numpy(), donors["elev"].to_numpy(),
                donors[var].to_numpy(),
                queries["x"].to_numpy(), queries["y"].to_numpy(), queries["elev"].to_numpy(),
                var=var, slope=sl, slope_q=slq, sinasp=sa, sinasp_q=saq,
            )
        hat = _pred(True)
        mets.append(score_recovery(queries[var], hat, var))
        if covar != "none":
            hat_d = _pred(False)
            mets_drop.append(score_recovery(queries[var], hat_d, var, allow_constant=True))
        if i == 0:
            print_baselines(var, queries[var].to_numpy(),
                            mean_baseline(donors, queries, var),
                            idw_baseline(donors, queries, var), hat)
            if var.startswith(("precip", "snow")):
                print_gate("GAM", var, gate_diagnostic(queries[var], hat, var))
    agg = aggregate(mets)
    print_score("GAM", var, agg, len(held), tag=f"{agg.get('n_fields', 0)}fld")
    if mets_drop:
        print_ablation(var, agg, aggregate(mets_drop), covar)


def main():
    for v in VARS:
        run_one(v)


if __name__ == "__main__":
    main()
