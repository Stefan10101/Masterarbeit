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
    grid_terrain,
    idw_raster,
    n_in_channels,
    station_to_raster,
    two_step_var,
)
from cnn_data import data_sources, load_cnn_config, load_panel, precip_trace

try:
    import xarray as xr
except ImportError:
    xr = None


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


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


def is_subdaily(panel, time_res: str | None = None) -> bool:
    if time_res == "half_hourly":
        return True
    if time_res in ("daily", "weekly", "monthly", "seasonal"):
        return False
    t = panel["time"]
    if hasattr(t, "dt"):
        try:
            return int(t.dt.hour.nunique()) > 1
        except Exception:
            return False
    return False


def build_frames(panel, var, gx, gy, elev, clc, ccfg: CNNConfig, split, subdaily=False, trace=0.1):
    sl = panel[panel["split"] == split]
    frames = []
    ny, nx = elev.shape
    two = ccfg.two_step and two_step_var(var)
    for ts, part in sl.groupby("time"):
        if part["station_name"].nunique() < ccfg.min_stations:
            continue
        rows, cols = grid_index(part["x"].to_numpy(), part["y"].to_numpy(), gx, gy)
        val = part[var].to_numpy()
        field, mask = station_to_raster(rows, cols, val, ny, nx)
        base = idw_raster(rows, cols, val, ny, nx, k=ccfg.idw_k) if ccfg.residual_idw else np.zeros_like(field)
        resid = np.where(mask > 0, field - base, 0.0)
        wet = ((field > trace) & (mask > 0)).astype(np.float32)
        frames.append({
            "value": resid.astype(np.float32) if ccfg.residual_idw else field,
            "mask": mask,
            "base": base,
            "field": field.astype(np.float32),
            "target": resid if ccfg.residual_idw else field,
            "wet": wet,
            "rows": rows,
            "cols": cols,
            "obs": val,
            "time": ts,
            "time_ch": cyclic_time(ts, subdaily) if ccfg.cyclic_time else None,
            "two_step": two,
        })
    return frames


def pack_batch(frames, idx, elev, clc, model: CNNInterpolator, rng, mask_frac, train: bool,
               slope=None, sinasp=None, cosasp=None):
    """Hide a fraction of station pixels in the input. Loss uses those pixels.

    Using the full station mask as the loss (old code) lets the net copy the
    visible residual channel and ignore interpolation.
    """
    chans, tgts, qmasks, wets, bases, fields = [], [], [], [], [], []
    for i in idx:
        fr = frames[i]
        mask_in = fr["mask"].copy()
        value = fr["value"].copy()
        qmask = np.zeros_like(fr["mask"], dtype=np.float32)
        r = fr["rows"]
        c = fr["cols"]
        if mask_frac > 0 and len(r) >= 5:
            drop = rng.random(len(r)) < mask_frac
            if drop.any() and (~drop).sum() >= 4:
                value[r[drop], c[drop]] = 0.0
                mask_in[r[drop], c[drop]] = 0.0
                qmask[r[drop], c[drop]] = 1.0
        if qmask.sum() < 4:
            qmask = fr["mask"].astype(np.float32)
        chans.append(model.pack_channels(
            value, mask_in, elev, clc,
            slope=slope, sinasp=sinasp, cosasp=cosasp,
            time_ch=fr.get("time_ch"),
        ))
        tgt = (fr["target"] - model.y_mean) / model.y_std
        tgts.append(tgt.astype(np.float32))
        qmasks.append(qmask.astype(np.float32))
        wets.append(fr.get("wet", np.zeros_like(fr["mask"])).astype(np.float32))
        bases.append(np.asarray(fr["base"], dtype=np.float32))
        fields.append(np.asarray(fr["field"], dtype=np.float32))
    x = torch.from_numpy(np.stack(chans).astype(np.float32)).to(model.device)
    y = torch.from_numpy(np.stack(tgts)).to(model.device)
    m = torch.from_numpy(np.stack(qmasks)).to(model.device)
    w = torch.from_numpy(np.stack(wets)).to(model.device)
    b = torch.from_numpy(np.stack(bases)).to(model.device)
    f = torch.from_numpy(np.stack(fields)).to(model.device)
    return x, y, m, w, b, f


def train_one(panel, var, ccfg: CNNConfig, elev, clc, gx, gy, slope=None, sinasp=None, cosasp=None,
              time_res: str | None = None, trace: float = 0.1):
    subdaily = is_subdaily(panel, time_res)
    out_ch = 2 if (ccfg.two_step and two_step_var(var)) else 1
    in_ch = n_in_channels(ccfg, subdaily)
    model = CNNInterpolator(ccfg, in_ch=in_ch, out_ch=out_ch)
    if slope is None and ccfg.use_terrain:
        slope, sinasp, cosasp = grid_terrain(elev, gx, gy)
    train_fr = build_frames(panel, var, gx, gy, elev, clc, ccfg, "train", subdaily, trace)
    dev_fr = build_frames(panel, var, gx, gy, elev, clc, ccfg, "dev", subdaily, trace)
    if not train_fr:
        raise RuntimeError(f"no train frames for {var}")
    vals = np.concatenate([f["target"][f["mask"] > 0] for f in train_fr])
    model.y_mean = float(vals.mean())
    model.y_std = float(vals.std() + 1e-6)
    model.z_mean = float(np.nanmean(elev))
    model.z_std = float(np.nanstd(elev) + 1.0)

    opt = torch.optim.AdamW(model.net.parameters(), lr=ccfg.lr, weight_decay=ccfg.weight_decay)
    use_amp = bool(ccfg.use_amp) and model.device.type == "cuda"
    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    rng = np.random.default_rng(ccfg.seed)
    best_state, best_loss, wait = None, np.inf, 0
    n = len(train_fr)
    two = out_ch == 2
    for epoch in range(ccfg.epochs):
        model.net.train()
        order = rng.permutation(n)
        total = 0.0
        seen = 0
        for i0 in range(0, n, ccfg.batch_times):
            idx = order[i0:i0 + ccfg.batch_times]
            x, y, m, wet, base_t, field_t = pack_batch(
                train_fr, idx, elev, clc, model, rng, ccfg.train_mask_frac, True,
                slope, sinasp, cosasp,
            )
            opt.zero_grad(set_to_none=True)
            if use_amp:
                try:
                    ctx = torch.amp.autocast("cuda", enabled=True)
                except TypeError:
                    ctx = torch.cuda.amp.autocast(enabled=True)
            else:
                ctx = _nullcontext()
            with ctx:
                hat = model.net(x)
                if two:
                    logit, amt = hat[:, 0], hat[:, 1]
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, wet, reduction="none")
                    mse = (amt - y) ** 2
                    amt_m = m * wet
                    loss_bce = (bce * m).sum() / m.sum().clamp_min(1.0)
                    loss_mse = (mse * amt_m).sum() / amt_m.sum().clamp_min(1.0)
                    loss = loss_bce + loss_mse
                else:
                    pred = hat[:, 0] if hat.dim() == 4 else hat
                    loss = ((pred - y) ** 2 * m).sum() / m.sum().clamp_min(1.0)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            total += float(loss.detach()) * int(m.sum())
            seen += int(m.sum())
        model.net.eval()
        dloss, dseen = 0.0, 0
        with torch.no_grad():
            for i0 in range(0, len(dev_fr), ccfg.batch_times):
                idx = list(range(i0, min(i0 + ccfg.batch_times, len(dev_fr))))
                x, y, m, wet, base_t, field_t = pack_batch(
                    dev_fr, idx, elev, clc, model, rng, ccfg.train_mask_frac, False,
                    slope, sinasp, cosasp,
                )
                hat = model.net(x)
                if two:
                    p = torch.sigmoid(hat[:, 0])
                    amt = hat[:, 1] * model.y_std + model.y_mean
                    if ccfg.residual_idw:
                        amt = amt + base_t
                    recon = torch.where(p >= ccfg.tau_wet, amt, torch.zeros_like(amt))
                    dloss += float(((recon - field_t) ** 2 * m).sum())
                else:
                    pred = hat[:, 0] if hat.dim() == 4 else hat
                    recon = pred * model.y_std + model.y_mean
                    if ccfg.residual_idw:
                        recon = recon + base_t
                    dloss += float(((recon - field_t) ** 2 * m).sum())
                dseen += int(m.sum())
        if not dev_fr:
            dev = total / max(seen, 1)
        else:
            dev = dloss / max(dseen, 1)
        print(
            f"epoch {epoch:03d} train_mse={total/max(seen,1):.4f} "
            f"dev_mse={dev:.4f} dev_rmse={np.sqrt(max(dev, 0.0)):.3f}",
            flush=True,
        )
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
    slope, sinasp, cosasp = grid_terrain(elev, gx, gy)
    variables = [args.variable] if args.variable else cfg["cnn"].get("variables_to_process", ["temp_mean"])
    for var in variables:
        panel = load_panel(cfg, var)
        ccfg = cfg_to_cnn(cfg, load_tuned(var, time_res))
        model, dev = train_one(
            panel, var, ccfg, elev, clc, gx, gy,
            slope, sinasp, cosasp, time_res=time_res,
            trace=precip_trace(cfg, time_res, var),
        )
        path = get_cnn_model_path(var, time_res)
        torch.save(model.state_dict(), path)
        print(f"wrote {path} dev_loss={dev:.4f}")


if __name__ == "__main__":
    main()
