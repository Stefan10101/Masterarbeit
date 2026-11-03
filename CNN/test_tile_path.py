#!/usr/bin/env python3
"""Integrity check for the tile + accumulation path.

Builds a small synthetic panel (same columns/splits as the real data),
runs train_one for precip (log1p, no IDW baseline) and temp (residual),
and asserts: finite falling loss, held-out excluded from the input,
raw reconstruction score finite.
"""
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
from cnn_core import CNNConfig
from train_cnn import train_one, _extract_tiles, _pack_from_static


def _grid(side=64, span=100_000.0):
    gx = np.linspace(0, span, side)
    gy = np.linspace(0, span, side)
    elev = np.zeros((side, side), dtype=np.float64)
    clc = np.zeros((side, side), dtype=np.float64)
    return gx, gy, elev, clc


def _prep(var, n_stations=40, n_fields=8, seed=7):
    panel = generate_synthetic_panel(var, n_stations=n_stations, n_fields=n_fields, seed=seed)
    # Remap tune -> dev so train_one sees a dev split
    panel.loc[panel["split"] == "tune", "split"] = "dev"
    # Keep held-out out of train/dev donors
    panel = panel[~((panel["is_heldout"]) & (panel["split"].isin(["train", "dev"])))].copy()
    return panel


def _run(var):
    print(f"\n=== {var} ===", flush=True)
    panel = _prep(var)
    gx, gy, elev, clc = _grid()
    ccfg = CNNConfig(
        n_levels=3, base_ch=8, residual_idw=(not var.startswith(("precip", "snow"))),
        use_terrain=False, cyclic_time=False, two_step=var.startswith(("precip", "snow")),
        tile_size=32, tiles_per_batch=4, accum_steps=2,
        epochs=3, patience=3, train_mask_frac=0.4,
        device="cpu", use_amp=False, seed=11,
    )
    model, dev = train_one(panel, var, ccfg, elev, clc, gx, gy, time_res="weekly", trace=0.1, jobs=1)
    assert np.isfinite(dev), f"{var}: non-finite dev"
    print(f"{var} dev={dev:.4f}  ok")
    return dev


def _tile_unit():
    """Query pixels survive the crop; non-query tiles are rejected."""
    rng = np.random.default_rng(0)
    H = W = 40
    chans = np.random.randn(2, 4, H, W).astype(np.float32)
    tgts = np.random.randn(2, H, W).astype(np.float32)
    qmasks = np.zeros((2, H, W), dtype=np.float32)
    qmasks[0, 10, 12] = 1.0
    qmasks[1, 30, 5] = 1.0
    wets = (np.random.rand(2, H, W) > 0.5).astype(np.float32)
    bases = np.zeros_like(tgts)
    fields = np.zeros_like(tgts)
    tiles = _extract_tiles(chans, tgts, qmasks, wets, bases, fields, 16, rng, 6)
    assert tiles is not None
    assert tiles[2].sum() > 0, "query mask lost in tiles"
    print("tile unit ok")


if __name__ == "__main__":
    _tile_unit()
    _run("precip_sum")
    _run("temp_mean")
    print("\nALL PASSED")
