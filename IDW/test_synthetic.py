#!/usr/bin/env python3
"""Synthetic experiments for IDW. Larger panel, multi-field, baselines, gate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import KDTree

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "IDW"))
sys.path.insert(0, str(CODE / "shared"))

from synthetic import generate_synthetic_panel
from synthetic_score import (
    score_recovery, print_score, assert_no_leak,
    mean_baseline, idw_baseline, print_baselines,
    gate_diagnostic, print_gate, aggregate,
)
from idw_core import interpolate_idw

VARS = ["temp_mean", "precip_sum", "snow_mean", "wind_mean", "rh_mean"]


def n_for(var: str) -> int:
    return 240 if var.startswith(("precip", "snow")) else 140


def predict_field(donors, queries, var):
    tree = KDTree(donors[["x", "y"]].to_numpy())
    return interpolate_idw(
        donors["value"].to_numpy(), donors[["x", "y"]].to_numpy(), donors["elev"].to_numpy(),
        queries[["x", "y"]].to_numpy(), queries["elev"].to_numpy(), tree,
        p=2.0, Fz=0.5, k=16, var=var,
    )


def run_one(var: str):
    panel = generate_synthetic_panel(var, n_stations=n_for(var), n_fields=20, seed=21)
    held = set(panel.loc[panel["is_heldout"], "station_name"].unique())
    test_times = panel.loc[panel["split"] == "test", "time"].unique()
    mets = []
    for t0 in test_times:
        field = panel[panel["time"] == t0]
        donors = field[~field["is_heldout"]]
        queries = field[field["is_heldout"]]
        assert_no_leak(donors, held)
        if len(queries) < 5 or len(donors) < 10:
            continue
        hat = predict_field(donors, queries, var)
        mets.append(score_recovery(queries["value"], hat, var))
        if t0 == test_times[0]:
            print_baselines(
                var, queries["value"].to_numpy(),
                mean_baseline(donors, queries, "value"),
                idw_baseline(donors, queries, "value"),
                hat,
            )
            if var.startswith(("precip", "snow")):
                print_gate("IDW", var, gate_diagnostic(queries["value"], hat, var))
    agg = aggregate(mets)
    print_score("IDW", var, agg, len(held), tag=f"{agg.get('n_fields', 0)}fld")


def main():
    for v in VARS:
        run_one(v)


if __name__ == "__main__":
    main()
