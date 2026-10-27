#!/usr/bin/env python3
"""Synthetic experiments for RFSI. Pooled fit, ablation, multi-field, baselines, gate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "RFSI"))
sys.path.insert(0, str(CODE / "shared"))

from synthetic import generate_synthetic_panel
from synthetic_score import (
    score_recovery, print_score, assert_no_leak,
    mean_baseline, idw_baseline, print_baselines,
    gate_diagnostic, print_gate, aggregate, print_ablation,
)
from rfsi_core import RFSI, build_covariates, extras_for_var

VARS = ["temp_mean", "precip_sum", "snow_mean", "wind_mean", "rh_mean"]


def n_for(var: str) -> int:
    return 180 if var.startswith(("precip", "snow")) else 110


def make_model(var):
    return RFSI(
        n_obs=10,
        rf_params={
            "n_estimators": 80, "max_depth": 14, "min_samples_leaf": 2,
            "max_features": "sqrt", "random_state": 42, "n_jobs": 1,
        },
        two_step=var.startswith(("precip", "snow")),
        var_name=var,
    )


def cov_matrix(df, var, drop=False):
    extras = None if drop else extras_for_var(df, var)
    return build_covariates(
        elev=None if drop else df["elev"].to_numpy(),
        clc_code=None if drop else df["clc_code"].to_numpy(),
        extras=extras,
    )


def run_one(var: str):
    panel = generate_synthetic_panel(var, n_stations=n_for(var), n_fields=20, seed=25)
    held = set(panel.loc[panel["is_heldout"], "station_name"].unique())
    train = panel[(~panel["is_heldout"]) & (panel["split"] == "train")]
    assert_no_leak(train, held)
    model = make_model(var)
    model.fit_pooled(
        train["time"].to_numpy(), train[["x", "y"]].to_numpy(), train[var].to_numpy(),
        X_cov=cov_matrix(train, var, drop=False), min_stations=10,
    )
    model_d = make_model(var)
    model_d.fit_pooled(
        train["time"].to_numpy(), train[["x", "y"]].to_numpy(), train[var].to_numpy(),
        X_cov=cov_matrix(train, var, drop=True), min_stations=10,
    )
    test_times = panel.loc[panel["split"] == "test", "time"].unique()
    mets, mets_drop = [], []
    covar = "elev+clc"
    for i, t0 in enumerate(test_times):
        field = panel[panel["time"] == t0]
        donors = field[~field["is_heldout"]]
        queries = field[field["is_heldout"]]
        if len(queries) < 5:
            continue
        hat = model.predict_field(
            donors[["x", "y"]].to_numpy(), donors[var].to_numpy(),
            queries[["x", "y"]].to_numpy(),
            X_cov_pred=cov_matrix(queries, var, drop=False),
        )
        mets.append(score_recovery(queries[var], hat, var))
        hat_d = model_d.predict_field(
            donors[["x", "y"]].to_numpy(), donors[var].to_numpy(),
            queries[["x", "y"]].to_numpy(),
            X_cov_pred=cov_matrix(queries, var, drop=True),
        )
        mets_drop.append(score_recovery(queries[var], hat_d, var, allow_constant=True))
        if i == 0:
            print_baselines(var, queries[var].to_numpy(),
                            mean_baseline(donors, queries, var),
                            idw_baseline(donors, queries, var), hat)
            if var.startswith(("precip", "snow")):
                print_gate("RFSI", var, gate_diagnostic(queries[var], hat, var))
    agg = aggregate(mets)
    print_score("RFSI", var, agg, len(held), tag=f"{agg.get('n_fields', 0)}fld")
    print_ablation(var, agg, aggregate(mets_drop), covar)


def main():
    for v in VARS:
        run_one(v)


if __name__ == "__main__":
    main()
