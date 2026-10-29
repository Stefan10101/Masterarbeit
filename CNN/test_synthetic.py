#!/usr/bin/env python3
"""Synthetic experiments for CNN. Real U-Net, 20 epochs, loss curve, multi-field."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "CNN"))
sys.path.insert(0, str(CODE / "shared"))

from synthetic import generate_synthetic_panel
from synthetic_score import (
    score_recovery, print_score, assert_no_leak,
    mean_baseline, idw_baseline, print_baselines,
    gate_diagnostic, print_gate, aggregate, print_loss_curve,
)
from cnn_core import CNNInterpolator, CNNConfig

VARS = ["temp_mean", "precip_sum", "snow_mean", "wind_mean", "rh_mean"]
EPOCHS = 40
SIDE = 64


def n_for(var: str) -> int:
    return 160 if var.startswith(("precip", "snow")) else 90


def rasterize(df, side=SIDE, span=100_000.0):
    value = np.zeros((side, side), dtype=np.float32)
    mask = np.zeros((side, side), dtype=np.float32)
    elev = np.zeros((side, side), dtype=np.float32)
    clc = np.zeros((side, side), dtype=np.int32)
    counts = np.zeros((side, side), dtype=np.float32)
    ix = np.clip((df["x"].to_numpy() / span * (side - 1)).astype(int), 0, side - 1)
    iy = np.clip((df["y"].to_numpy() / span * (side - 1)).astype(int), 0, side - 1)
    for i, (r, c) in enumerate(zip(iy, ix)):
        value[r, c] += float(df["value"].iloc[i])
        elev[r, c] += float(df["elev"].iloc[i])
        clc[r, c] = int(df["clc_level2"].iloc[i])
        counts[r, c] += 1.0
        mask[r, c] = 1.0
    counts = np.maximum(counts, 1.0)
    return value / counts, mask, elev / counts, clc


def run_one(var: str):
    panel = generate_synthetic_panel(var, n_stations=n_for(var), n_fields=12, seed=28)
    held = set(panel.loc[panel["is_heldout"], "station_name"].unique())
    log_amount = var.startswith(("precip", "snow"))
    residual = not log_amount
    two_step = log_amount
    out_ch = 2 if two_step else 1
    cfg = CNNConfig(n_levels=3, base_ch=8, use_terrain=False, cyclic_time=False, device="cpu", two_step=two_step)
    model = CNNInterpolator(cfg, in_ch=20, out_ch=out_ch)
    torch.nn.init.normal_(model.net.out.weight, std=0.01)
    raw = panel["value"].to_numpy(dtype=np.float64)
    if log_amount:
        raw = np.log1p(np.maximum(raw, 0.0))
        model.y_mean = float(raw.mean())
        model.y_std = float(max(raw.std(), 1e-3))
    elif residual:
        # Residual target is near zero; let the net predict it directly.
        model.y_mean = 0.0
        model.y_std = 1.0
    else:
        model.y_mean = float(raw.mean())
        model.y_std = float(max(raw.std(), 1e-3))
    model.z_mean = float(panel["elev"].mean())
    model.z_std = float(max(panel["elev"].std(), 1.0))
    model.log_amount = log_amount
    opt = torch.optim.Adam(model.net.parameters(), lr=1e-3)
    train = panel[(~panel["is_heldout"]) & (panel["split"] == "train")]
    assert_no_leak(train, held)
    history = []
    for _ in range(EPOCHS):
        losses = []
        for _, g in train.groupby("time"):
            val, mask, elev, clc = rasterize(g)
            # Residual baseline: plain IDW on the same raster (exclude-self not critical here)
            base = None
            if residual:
                base = idw_on_raster(g, SIDE)
            tgt_raw = val.copy()
            if log_amount:
                tgt_raw = np.log1p(np.maximum(tgt_raw, 0.0))
            elif base is not None:
                tgt_raw = tgt_raw - base
            tgt = (tgt_raw - model.y_mean) / model.y_std
            ch = model.pack_channels(val, mask, elev, clc)
            if log_amount:
                ch = ch.copy()
                ch[0] = np.where(mask > 0, (np.log1p(np.maximum(val, 0.0)) - model.y_mean) / model.y_std, 0.0)
            x = torch.from_numpy(ch).unsqueeze(0)
            m = torch.from_numpy(mask).unsqueeze(0)
            opt.zero_grad()
            hat = model.net(x)
            if two_step:
                # head 0: wet logit, head 1: standardized log-amount
                wet = (val > 0.1).astype(np.float32)
                amt = np.log1p(np.maximum(val, 0.0))
                amt = (amt - model.y_mean) / model.y_std
                y_wet = torch.from_numpy(wet)
                y_amt = torch.from_numpy(amt)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(hat[0, 0], y_wet)
                mse = ((hat[0, 1] - y_amt) ** 2 * m[0] * y_wet).sum() / (m[0] * y_wet).sum().clamp(min=1)
                loss = bce + mse
            else:
                y = torch.from_numpy(tgt.astype(np.float32)).unsqueeze(0)
                loss = ((hat[0, 0] - y[0]) ** 2 * m[0]).sum() / m.sum().clamp(min=1)
            if torch.isfinite(loss):
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)) if losses else np.nan)
    print_loss_curve("CNN", history)

    test_times = panel.loc[panel["split"] == "test", "time"].unique()
    mets = []
    for i, t0 in enumerate(test_times):
        field = panel[panel["time"] == t0]
        donors = field[~field["is_heldout"]]
        queries = field[field["is_heldout"]]
        if len(queries) < 4:
            continue
        val, mask, elev, clc = rasterize(donors)
        base = idw_on_raster(donors, SIDE) if residual else None
        pred_ras = model.predict_raster(val, mask, elev, clc, var=var, base=base, two_step=two_step)
        ix = np.clip((queries["x"].to_numpy() / 100_000 * (SIDE - 1)).astype(int), 0, SIDE - 1)
        iy = np.clip((queries["y"].to_numpy() / 100_000 * (SIDE - 1)).astype(int), 0, SIDE - 1)
        hat = pred_ras[iy, ix]
        mets.append(score_recovery(queries["value"], hat, var, allow_constant=True))
        if i == 0:
            print_baselines(var, queries["value"].to_numpy(),
                            mean_baseline(donors, queries, "value"),
                            idw_baseline(donors, queries, "value"), hat)
            if var.startswith(("precip", "snow")):
                print_gate("CNN", var, gate_diagnostic(queries["value"], hat, var))
    agg = aggregate(mets)
    print_score("CNN", var, agg, len(held), tag=f"{agg.get('n_fields', 0)}fld")


def idw_on_raster(df, side, span=100_000.0, k=8):
    """Plain IDW (p=2) rasterized onto the same grid the CNN sees."""
    from synthetic_score import idw_baseline
    # Build a dense query grid and scatter the IDW values
    ys = np.linspace(0, span, side)
    xs = np.linspace(0, span, side)
    xx, yy = np.meshgrid(xs, ys)
    q = pd.DataFrame({"x": xx.ravel(), "y": yy.ravel()})
    # idw_baseline expects a DataFrame with the value column named as the var;
    # we only have "value" here.
    donors = df.copy()
    if "value" not in donors.columns:
        donors["value"] = donors.iloc[:, -1]
    hat = idw_baseline(donors, q, "value", k=k)
    return hat.reshape(side, side).astype(np.float32)


def main():
    for v in VARS:
        run_one(v)


if __name__ == "__main__":
    main()
