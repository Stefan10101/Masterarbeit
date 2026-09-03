#!/usr/bin/env python3
"""Fit the pooled U-Net on TRAIN timestamps, early-stop on DEV stations."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_cnn_model_path, get_cnn_tuned_params_path, get_master_grid_path
from cnn_core import (
    CNNConfig,
    CNNInterpolator,
    cyclic_time,
    idw_raster,
    n_in_channels,
    station_to_raster,
    two_step_var,
)
from cnn_data import data_sources, load_cnn_config, load_panel

try:
    import xarray as xr
except ImportError:
    xr = None


def cfg_to_cnn(cfg, overrides=None) -> CNNConfig:
    c = dict(cfg.get("cnn", {}))
    o = overrides or {}
    return CNNConfig(
        n_levels=int(o.get("n_levels", c.get("n_levels", 4))),
        base_ch=int(o.get("base_ch", c.get("base_ch", 32))),
        residual_idw=bool(o.get("residual_idw", c.get("residual_idw", True))),
        idw_k=int(c.get("idw_k", 8)),
        use_terrain=bool(c.get("use_terrain", True)),
        cyclic_time=bool(c.get("cyclic_time", True)),
        two_step=bool(c.get("two_step", True)),
        tau_wet=float(o.get("tau_wet", c.get("tau_wet", 0.5))),
        train_mask_frac=float(o.get("train_mask_frac", c.get("train_mask_frac", 0.25))),
        epochs=int(c.get("epochs", 40)),
        patience=int(c.get("patience", 8)),
        lr=float(c.get("lr", 1e-3)),
        weight_decay=float(c.get("weight_decay", 1e-4)),
        batch_times=int(c.get("batch_times", 4)),
        seed=int(c.get("seed", 22)),
        device=str(c.get("device", "auto")),
        use_amp=bool(c.get("use_amp", True)),
        min_stations=int(cfg.get("min_stations_per_field", 10)),
    )


def load_tuned(var, time_res):
    path = get_cnn_tuned_params_path(var, time_res)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def grid_index(x, y, gx, gy):
    dx = float(gx[1] - gx[0]) if len(gx) > 1 else 1.0
    dy = float(gy[1] - gy[0]) if len(gy) > 1 else 1.0
    cols = np.clip(np.round((x - gx[0]) / dx).astype(int), 0, len(gx) - 1)
    rows = np.clip(np.round((y - gy[0]) / dy).astype(int), 0, len(gy) - 1)
    return rows, cols


def build_frames(panel, var, gx, gy, elev, clc, ccfg: CNNConfig, split, subdaily=False):
    sl = panel[panel["split"] == split]
    frames = []
    ny, nx = elev.shape
    for ts, part in sl.groupby("time"):
        if part["station_name"].nunique() < ccfg.min_stations:
            continue
        rows, cols = grid_index(part["x"].to_numpy(), part["y"].to_numpy(), gx, gy)
        val = part[var].to_numpy()
        field, mask = station_to_raster(rows, cols, val, ny, nx)
        base = idw_raster(rows, cols, val, ny, nx, k=ccfg.idw_k) if ccfg.residual_idw else np.zeros_like(field)
        resid = np.where(mask > 0, field - base, 0.0)
        frames.append({
            "value": resid.astype(np.float32) if ccfg.residual_idw else field,
            "mask": mask, "base": base, "target": resid if ccfg.residual_idw else field,
            "rows": rows, "cols": cols, "obs": val, "time": ts,
            "time_ch": cyclic_time(ts, subdaily) if ccfg.cyclic_time else None,
        })
    return frames


def pack_batch(frames, idx, elev, clc, model: CNNInterpolator, rng, mask_frac, train: bool):
    chans = []
    tgts = []
    msks = []
    for i in idx:
        fr = frames[i]
        mask = fr["mask"].copy()
        value = fr["value"].copy()
        if train and mask_frac > 0:
            r = fr["rows"]
            c = fr["cols"]
            n = len(r)
            drop = rng.random(n) < mask_frac
            if drop.any() and (~drop).sum() >= 4:
                value[r[drop], c[drop]] = 0.0
                mask[r[drop], c[drop]] = 0.0
        chans.append(model.pack_channels(
            value, mask, elev, clc, time_ch=fr.get("time_ch"),
        ))
        tgt = (fr["target"] - model.y_mean) / model.y_std
        # loss mask = original station pixels
        tgts.append(tgt.astype(np.float32))
        msks.append(fr["mask"].astype(np.float32))
    x = torch.from_numpy(np.stack(chans).astype(np.float32)).to(model.device)
    y = torch.from_numpy(np.stack(tgts)).to(model.device)
    m = torch.from_numpy(np.stack(msks)).to(model.device)
    return x, y, m


def train_one(panel, var, ccfg: CNNConfig, elev, clc, gx, gy):
    subdaily = bool(panel["time"].dt.hour.nunique() > 1) if hasattr(panel["time"].dt, "hour") else False
    out_ch = 2 if (ccfg.two_step and two_step_var(var)) else 1
    in_ch = n_in_channels(ccfg, subdaily)
    model = CNNInterpolator(ccfg, in_ch=in_ch, out_ch=out_ch)
    train_fr = build_frames(panel, var, gx, gy, elev, clc, ccfg, "train", subdaily)
    dev_fr = build_frames(panel, var, gx, gy, elev, clc, ccfg, "dev", subdaily)
    if not train_fr:
        raise RuntimeError(f"no train frames for {var}")
    vals = np.concatenate([
        f["target"][f["mask"] > 0] if ccfg.residual_idw else f["obs"]
        for f in train_fr
    ])
    model.y_mean = float(vals.mean())
    model.y_std = float(vals.std() + 1e-6)
    model.z_mean = float(np.nanmean(elev))
    model.z_std = float(np.nanstd(elev) + 1.0)

    opt = torch.optim.AdamW(model.net.parameters(), lr=ccfg.lr, weight_decay=ccfg.weight_decay)
    rng = np.random.default_rng(ccfg.seed)
    best_state, best_loss, wait = None, np.inf, 0
    n = len(train_fr)
    for epoch in range(ccfg.epochs):
        model.net.train()
        order = rng.permutation(n)
        total = 0.0
        seen = 0
        for i0 in range(0, n, ccfg.batch_times):
            idx = order[i0:i0 + ccfg.batch_times]
            x, y, m = pack_batch(train_fr, idx, elev, clc, model, rng, ccfg.train_mask_frac, True)
            opt.zero_grad()
            hat = model.net(x)
            if hat.shape[1] == 2:
                logit, amt = hat[:, 0], hat[:, 1]
                wet = (y > (0.0 - model.y_mean) / model.y_std).float()
                bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, wet, reduction="none")
                mse = (amt - y) ** 2
                loss = ((bce + mse) * m).sum() / m.sum().clamp_min(1.0)
            else:
                pred = hat[:, 0] if hat.dim() == 4 else hat
                loss = ((pred - y) ** 2 * m).sum() / m.sum().clamp_min(1.0)
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(m.sum())
            seen += int(m.sum())
        # DEV
        model.net.eval()
        dloss, dseen = 0.0, 0
        with torch.no_grad():
            for i0 in range(0, len(dev_fr), ccfg.batch_times):
                idx = list(range(i0, min(i0 + ccfg.batch_times, len(dev_fr))))
                x, y, m = pack_batch(dev_fr, idx, elev, clc, model, rng, 0.0, False)
                hat = model.net(x)
                pred = hat[:, 0] if hat.dim() == 4 else hat
                dloss += float(((pred - y) ** 2 * m).sum())
                dseen += int(m.sum())
        dev = dloss / max(dseen, 1)
        print(f"epoch {epoch:03d} train={total/max(seen,1):.4f} dev={dev:.4f}")
        if dev < best_loss:
            best_loss = dev
            best_state = {k: v.detach().cpu().clone() for k, v in model.net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= ccfg.patience:
                break
    if best_state is not None:
        model.net.load_state_dict(best_state)
    return model, best_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default=None)
    args = p.parse_args()
    cfg = load_cnn_config()
    if xr is None:
        raise RuntimeError("xarray required")
    time_res = cfg["time_resolution"]
    _, _, grid_method = data_sources(cfg)
    res = int(cfg.get("resolutions_to_process", [1000])[0])
    grid = xr.open_dataset(get_master_grid_path(grid_method, res))
    gx = grid["x"].values.astype(np.float64)
    gy = grid["y"].values.astype(np.float64)
    elev = grid["elev"].values.astype(np.float64) if "elev" in grid else np.zeros((gy.size, gx.size))
    clc = np.zeros_like(elev)
    for name in ("clc_code", "clc", "landcover"):
        if name in grid:
            from cnn_data import clc_group
            clc = clc_group(np.nan_to_num(grid[name].values, nan=0).astype(np.int32)).astype(np.float64)
            break
    variables = [args.variable] if args.variable else cfg["cnn"].get("variables_to_process", ["temp_mean"])
    for var in variables:
        panel = load_panel(cfg, var)
        ccfg = cfg_to_cnn(cfg, load_tuned(var, time_res))
        model, dev = train_one(panel, var, ccfg, elev, clc, gx, gy)
        path = get_cnn_model_path(var, time_res)
        torch.save(model.state_dict(), path)
        print(f"wrote {path} dev_loss={dev:.4f}")


if __name__ == "__main__":
    main()
