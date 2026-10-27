#!/usr/bin/env python3
"""Synthetic experiments for SMACNP. Real model, 25 epochs, loss curve, multi-field."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "SMACNP"))
sys.path.insert(0, str(CODE / "shared"))

from synthetic import generate_synthetic_panel
from synthetic_score import (
    score_recovery, print_score, assert_no_leak,
    mean_baseline, idw_baseline, print_baselines,
    gate_diagnostic, print_gate, aggregate, print_loss_curve,
)
from smacnp_core import SMACNP, gaussian_nll, two_step_loss

VARS = ["temp_mean", "precip_sum", "snow_mean", "wind_mean", "rh_mean"]
FEATS = ["elev", "smooth_elev", "tpi", "dist_to_coast", "signed_dist_to_ridge"]
EPOCHS = 25


def n_for(var: str) -> int:
    return 140 if var.startswith(("precip", "snow")) else 80


def tensors(df, feat_cols, mu=None, sd=None):
    xy = df[["x", "y"]].to_numpy() / 1000.0
    attr = df[feat_cols].to_numpy()
    if mu is None:
        mu, sd = attr.mean(0), np.maximum(attr.std(0), 1.0)
    attr = (attr - mu) / sd
    x = np.hstack([xy, attr]).astype(np.float32)
    y = df["value"].to_numpy().astype(np.float32).reshape(-1, 1)
    return x, y, mu, sd


def run_one(var: str):
    panel = generate_synthetic_panel(var, n_stations=n_for(var), n_fields=16, seed=27)
    held = set(panel.loc[panel["is_heldout"], "station_name"].unique())
    two_step = var.startswith(("precip", "snow"))
    model = SMACNP(
        input_dim=2 + len(FEATS), spatial_dim=2, attr_dim=len(FEATS),
        y_dim=1, two_step=two_step,
        r_dim=32, w_dim=64, v_dim=64, hidden_size=96,
        num_layers_mlp=2, num_heads=4, k_localized=10,
        dropout_rate=0.0, use_positional_encoding=False,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train = panel[(~panel["is_heldout"]) & (panel["split"] == "train")]
    assert_no_leak(train, held)
    history = []
    for ep in range(EPOCHS):
        losses = []
        for _, g in train.groupby("time"):
            if len(g) < 16:
                continue
            x, y, _, _ = tensors(g, FEATS)
            n = len(g)
            n_ctx = max(10, int(0.7 * n))
            idx = np.random.default_rng(ep).permutation(n)
            ctx, tgt = idx[:n_ctx], idx[n_ctx:]
            x_ctx = torch.from_numpy(x[ctx]).unsqueeze(0)
            y_ctx = torch.from_numpy(y[ctx]).unsqueeze(0)
            x_tgt = torch.from_numpy(x[tgt]).unsqueeze(0)
            y_tgt = torch.from_numpy(y[tgt]).unsqueeze(0)
            opt.zero_grad()
            pred = model(x_ctx, y_ctx, x_tgt)
            loss = two_step_loss(pred, y_tgt) if two_step else gaussian_nll(pred["mu"], pred["variance"], y_tgt)
            if torch.isfinite(loss):
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)) if losses else np.nan)
    print_loss_curve("SMACNP", history)

    model.eval()
    test_times = panel.loc[panel["split"] == "test", "time"].unique()
    mets = []
    for i, t0 in enumerate(test_times):
        field = panel[panel["time"] == t0]
        donors = field[~field["is_heldout"]]
        queries = field[field["is_heldout"]]
        if len(queries) < 5:
            continue
        xd, yd, mu, sd = tensors(donors, FEATS)
        xq, _, _, _ = tensors(queries, FEATS, mu, sd)
        with torch.no_grad():
            pred = model(
                torch.from_numpy(xd).unsqueeze(0),
                torch.from_numpy(yd).unsqueeze(0),
                torch.from_numpy(xq).unsqueeze(0),
            )
            if two_step:
                p = torch.sigmoid(pred["occurrence_logits"]).numpy()[0, :, 0]
                amt = pred["amount_mu"].numpy()[0, :, 0]
                hat = np.where(p >= 0.5, np.maximum(amt, 0.0), 0.0)
            else:
                hat = pred["mu"].numpy()[0, :, 0]
                p = None
        mets.append(score_recovery(queries["value"], hat, var, allow_constant=True))
        if i == 0:
            print_baselines(var, queries["value"].to_numpy(),
                            mean_baseline(donors, queries, "value"),
                            idw_baseline(donors, queries, "value"), hat)
            if two_step:
                print_gate("SMACNP", var, gate_diagnostic(queries["value"], hat, var, p_wet=p))
    agg = aggregate(mets)
    print_score("SMACNP", var, agg, len(held), tag=f"{agg.get('n_fields', 0)}fld")


def main():
    for v in VARS:
        run_one(v)


if __name__ == "__main__":
    main()
