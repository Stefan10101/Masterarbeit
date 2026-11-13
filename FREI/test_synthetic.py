#!/usr/bin/env python3
"""Synthetic experiments for Frei. Ablation on extras, multi-field, baselines, gate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "FREI"))
sys.path.insert(0, str(CODE / "shared"))

from synthetic import generate_synthetic_panel
from synthetic_score import (
    score_recovery, print_score, assert_no_leak,
    mean_baseline, idw_baseline, print_baselines,
    gate_diagnostic, print_gate, aggregate, print_ablation,
)
from frei_core import FreiConfig, FreiInterpolator

VARS = ["temp_mean", "precip_sum", "snow_mean", "wind_mean", "rh_mean"]


def n_for(var: str) -> int:
    return 220 if var.startswith(("precip", "snow")) else 130


def predict(donors, queries, var):
    fcfg = FreiConfig(n_regions=1, n_sections=3, shrink=0.2, k=12, across_w=2.0,
                      Fz=0.3, select_metric=False, min_stations=8, two_step=True)
    model = FreiInterpolator(fcfg)
    hat, _ = model.predict_timestamp(
        donors["x"].to_numpy(), donors["y"].to_numpy(), donors["elev"].to_numpy(),
        donors[var].to_numpy(),
        queries["x"].to_numpy(), queries["y"].to_numpy(), queries["elev"].to_numpy(),
        var=var,
    )
    return hat


def run_one(var: str):
    panel = generate_synthetic_panel(var, n_stations=n_for(var), n_fields=20, seed=22)
    held = set(panel.loc[panel["is_heldout"], "station_name"].unique())
    test_times = panel.loc[panel["split"] == "test", "time"].unique()
    mets, mets_drop = [], []
    covar = "tpi" if var.startswith("wind") else ("slope" if var.startswith("snow") else "none")
    for i, t0 in enumerate(test_times):
        field = panel[panel["time"] == t0]
        donors = field[~field["is_heldout"]]
        queries = field[field["is_heldout"]]
        assert_no_leak(donors, held)
        if len(queries) < 5:
            continue
        hat = predict(donors, queries, var)
        mets.append(score_recovery(queries[var], hat, var))
        if i == 0:
            print_baselines(var, queries[var].to_numpy(),
                            mean_baseline(donors, queries, var),
                            idw_baseline(donors, queries, var), hat)
            if var.startswith(("precip", "snow")):
                print_gate("Frei", var, gate_diagnostic(queries[var], hat, var))
    agg = aggregate(mets)
    print_score("Frei", var, agg, len(held), tag=f"{agg.get('n_fields', 0)}fld")
    if mets_drop:
        print_ablation(var, agg, aggregate(mets_drop), covar)


def main():
    for v in VARS:
        run_one(v)


if __name__ == "__main__":
    main()
