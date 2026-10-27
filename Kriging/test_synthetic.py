#!/usr/bin/env python3
"""Synthetic experiments for Kriging. Ablation, multi-field, baselines, gate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "Kriging"))
sys.path.insert(0, str(CODE / "shared"))

from synthetic import generate_synthetic_panel
from synthetic_score import (
    score_recovery, print_score, assert_no_leak,
    mean_baseline, idw_baseline, print_baselines,
    gate_diagnostic, print_gate, aggregate, print_ablation,
)
from kriging_core import KrigingInterpolator, KrigingConfig

VARS = ["temp_mean", "precip_sum", "snow_mean", "wind_mean", "rh_mean"]


def n_for(var: str) -> int:
    return 200 if var.startswith(("precip", "snow")) else 120


def covars(var, drop=False):
    """Ablate the optional extras only. Elev stays in the main trend."""
    if drop:
        return ()
    if var.startswith("snow"):
        return ("slope", "northness")
    if var.startswith("wind"):
        return ("tpi",)
    return ()


def predict(donors, queries, var, drop=False):
    cfg = KrigingConfig(
        k=20, family="exponential", alpha_z=40.0, trend="linear",
        enabled_covariates=covars(var, drop), min_stations=10,
    )
    model = KrigingInterpolator(cfg=cfg)
    model.fit(donors, var=var, time_res="daily", cfg_yaml={})
    return model.predict_frame(donors, queries, var=var)


def run_one(var: str):
    panel = generate_synthetic_panel(var, n_stations=n_for(var), n_fields=20, seed=24)
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
        hat = predict(donors, queries, var, drop=False)
        mets.append(score_recovery(queries[var], hat, var, allow_constant=True))
        if covar != "none":
            hat_d = predict(donors, queries, var, drop=True)
            mets_drop.append(score_recovery(queries[var], hat_d, var, allow_constant=True))
        if i == 0:
            print_baselines(var, queries[var].to_numpy(),
                            mean_baseline(donors, queries, var),
                            idw_baseline(donors, queries, var), hat)
            if var.startswith(("precip", "snow")):
                print_gate("Kriging", var, gate_diagnostic(queries[var], hat, var))
    agg = aggregate(mets)
    print_score("Kriging", var, agg, len(held), tag=f"{agg.get('n_fields', 0)}fld")
    if mets_drop:
        print_ablation(var, agg, aggregate(mets_drop), covar)


def main():
    for v in VARS:
        run_one(v)


if __name__ == "__main__":
    main()
