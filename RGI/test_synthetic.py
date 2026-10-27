#!/usr/bin/env python3
"""Synthetic experiments for RGI. Longer train, multi-field, baselines, gate, curve."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "RGI"))
sys.path.insert(0, str(CODE / "shared"))

from synthetic import generate_synthetic_panel
from synthetic_score import (
    score_recovery, print_score, assert_no_leak,
    mean_baseline, idw_baseline, print_baselines,
    gate_diagnostic, print_gate, aggregate, print_loss_curve,
)
from rgi_core import RGI, RGIConfig

VARS = ["temp_mean", "precip_sum", "snow_mean", "wind_mean", "rh_mean"]


def n_for(var: str) -> int:
    return 160 if var.startswith(("precip", "snow")) else 100


def extra_cols(var: str):
    if var.startswith("snow"):
        return ("slope", "northness")
    if var.startswith("wind"):
        return ("tpi",)
    return ()


def run_one(var: str):
    panel = generate_synthetic_panel(var, n_stations=n_for(var), n_fields=20, seed=26)
    held = set(panel.loc[panel["is_heldout"], "station_name"].unique())
    cfg = RGIConfig(
        k=10, n_layers=2, alpha=0.3, alpha_z=30.0,
        epochs=30, patience=8, extra_cols=extra_cols(var),
        two_step=var.startswith(("precip", "snow")),
    )
    model = RGI(cfg)
    train = panel[(~panel["is_heldout"]) & (panel["split"] == "train")]
    assert_no_leak(train, held)
    history = []
    def _log(msg=""):
        # fit logs "early stop epoch N  val_mse=X" — capture is inside fit
        pass
    info = model.fit(train, var=var, log=_log)
    # history not exposed; print epoch count from info if present
    if isinstance(info, dict) and "history" in info:
        history = [h.get("train", h.get("val", 0)) for h in info["history"]]
        print_loss_curve("RGI", history)

    test_times = panel.loc[panel["split"] == "test", "time"].unique()
    mets = []
    for i, t0 in enumerate(test_times):
        field = panel[panel["time"] == t0]
        donors = field[~field["is_heldout"]]
        queries = field[field["is_heldout"]]
        if len(queries) < 5:
            continue
        hat = model.predict_stations_at_time(field, var=var, hold_names=held)
        hat = np.asarray(hat, dtype=float)
        obs = queries[var].to_numpy()[: len(hat)]
        mets.append(score_recovery(obs, hat[: len(obs)], var, allow_constant=True))
        if i == 0:
            print_baselines(var, obs,
                            mean_baseline(donors, queries, var)[: len(obs)],
                            idw_baseline(donors, queries, var)[: len(obs)],
                            hat[: len(obs)])
            if var.startswith(("precip", "snow")):
                print_gate("RGI", var, gate_diagnostic(obs, hat[: len(obs)], var))
    agg = aggregate(mets)
    print_score("RGI", var, agg, len(held), tag=f"{agg.get('n_fields', 0)}fld")


def main():
    for v in VARS:
        run_one(v)


if __name__ == "__main__":
    main()
