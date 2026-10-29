#!/usr/bin/env python3
"""
Pooled raster U-Net. Target is residual to raster IDW.

Channels:
  value, mask, elev, slope, sinasp, cosasp, CLC, sin_doy, cos_doy [, sin_hour, cos_hour]
Precip/snow: two output heads (wet-dry logit, amount). Others: one head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


@dataclass
class CNNConfig:
    n_levels: int = 4
    base_ch: int = 32
    residual_idw: bool = True
    idw_k: int = 40
    idw_p: float = 1.5
    idw_fz: float = 1.0
    use_terrain: bool = True
    cyclic_time: bool = True
    two_step: bool = True
    tau_wet: float = 0.5
    train_mask_frac: float = 0.25
    epochs: int = 40
    patience: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_times: int = 4
    tile_size: int = 64
    tiles_per_batch: int = 64
    seed: int = 22
    device: str = "auto"
    use_amp: bool = True
    min_stations: int = 10
    rh_t_mode: str = "none"
    gate_d0_km: float = 35.0


def two_step_var(var: str) -> bool:
    v = var.lower()
    return v.startswith("precip") or v.startswith("snow")


def clip_var(var: str, pred) -> np.ndarray:
    v = str(var).lower()
    out = np.asarray(pred, dtype=np.float64)
    if v.startswith("precip") or v.startswith("snow") or v.startswith("wind"):
        return np.maximum(out, 0.0)
    if v.startswith("rh"):
        return np.clip(out, 0.0, 100.0)
    return out


def n_in_channels(cfg: CNNConfig, subdaily: bool, extra_ch: int = 0) -> int:
    n = 3  # value, mask, elev
    if cfg.use_terrain:
        n += 3  # slope, sinasp, cosasp
    n += 16  # CLC level-2 one-hot (0=missing + 15 classes)
    n += 1   # station-proximity weight
    if cfg.cyclic_time:
        n += 2
        if subdaily:
            n += 2
    return n + int(extra_ch)


def station_weight(mask, dx_m: float = 1000.0, dy_m: float = 1000.0, d0_km: float = 35.0) -> np.ndarray:
    """Soft gate from distance to the nearest station pixel. 1 at stations, ~0 far away."""
    from scipy.ndimage import distance_transform_edt
    m = np.asarray(mask) > 0
    if not m.any():
        return np.zeros(m.shape, dtype=np.float32)
    # sampling=(row spacing, col spacing) in metres
    d_m = distance_transform_edt(~m, sampling=(abs(dy_m), abs(dx_m)))
    d0 = max(float(d0_km), 1.0) * 1000.0
    return np.exp(-d_m / d0).astype(np.float32)


def pick_device(name: str):
    if torch is None:
        raise RuntimeError("PyTorch is required for CNN")
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        g = 8 if c_out >= 8 else 1
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(g, c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(g, c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def _infer_unet(sd):
    """n_levels / base / in_ch from a SmallUNet state_dict."""
    downs = []
    for k in sd:
        if k.startswith("down.") and k.endswith(".net.0.weight"):
            downs.append(int(k.split(".")[1]))
    n_levels = (max(downs) + 1) if downs else 4
    w0 = sd.get("down.0.net.0.weight")
    if w0 is None:
        return n_levels, 32, 9
    return n_levels, int(w0.shape[0]), int(w0.shape[1])


class SmallUNet(nn.Module):
    def __init__(self, in_ch: int, base: int = 32, n_levels: int = 4, out_ch: int = 1):
        super().__init__()
        n_levels = int(np.clip(n_levels, 3, 5))
        chs = [base * (2 ** i) for i in range(n_levels)]
        self.down = nn.ModuleList()
        cprev = in_ch
        for c in chs:
            self.down.append(ConvBlock(cprev, c))
            cprev = c
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList()
        for i in range(n_levels - 1, 0, -1):
            self.up.append(ConvBlock(chs[i] + chs[i - 1], chs[i - 1]))
        self.out = nn.Conv2d(chs[0], out_ch, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        skips = []
        h = x
        for i, blk in enumerate(self.down):
            h = blk(h)
            skips.append(h)
            if i < len(self.down) - 1:
                h = self.pool(h)
        for i, blk in enumerate(self.up):
            skip = skips[-(i + 2)]
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = blk(torch.cat([h, skip], dim=1))
        return self.out(h)



def idw_fz_raster(sx, sy, sz, sv, gx, gy, elev, p=1.5, Fz=1.0, k=40, exclude_self=True):
    """Same distance as the IDW method: w^2 = (1-Fz)*w_xy^2 + Fz*w_z^2, w = d^(-p).

    exclude_self drops a station from its own cell so the residual is not identically 0.
    """
    from scipy.spatial import cKDTree

    sx = np.asarray(sx, dtype=np.float64)
    sy = np.asarray(sy, dtype=np.float64)
    sz = np.asarray(sz, dtype=np.float64)
    sv = np.asarray(sv, dtype=np.float64)
    elev = np.asarray(elev, dtype=np.float64)
    ny, nx = elev.shape
    yy, xx = np.meshgrid(np.asarray(gy, dtype=np.float64), np.asarray(gx, dtype=np.float64), indexing="ij")
    txy = np.column_stack([xx.ravel(), yy.ravel()])
    tz = elev.ravel()
    tree = cKDTree(np.column_stack([sx, sy]))
    kk = int(min(max(k, 1), len(sv)))
    qk = min(kk + 1, len(sv)) if exclude_self else kk
    # A station sits in a cell, not on its centre. Half a cell (~707 m at 1 km)
    # is the geographic equivalent of the old exact pixel match.
    dx = float(gx[1] - gx[0]) if len(gx) > 1 else 1000.0
    dy = float(gy[1] - gy[0]) if len(gy) > 1 else 1000.0
    self_tol = 0.5 * (abs(dx) ** 2 + abs(dy) ** 2) ** 0.5
    out = np.empty(txy.shape[0], dtype=np.float32)
    chunk = 120_000
    for s in range(0, txy.shape[0], chunk):
        e = min(s + chunk, txy.shape[0])
        d, ix = tree.query(txy[s:e], k=qk)
        d = np.atleast_2d(np.asarray(d, dtype=np.float64))
        ix = np.atleast_2d(np.asarray(ix))
        if exclude_self and d.shape[1] > 1:
            use_d, use_i = d[:, :kk].copy(), ix[:, :kk].copy()
            hit = d[:, 0] <= self_tol
            if hit.any():
                use_d[hit] = d[hit, 1:kk + 1]
                use_i[hit] = ix[hit, 1:kk + 1]
            d, ix = use_d, use_i
        d = np.maximum(d, 1e-6)
        dz = np.maximum(np.abs(sz[ix] - tz[s:e, None]), 1e-6)
        w_xy = d ** (-p)
        w_z = dz ** (-p)
        w = (1.0 - Fz) * w_xy ** 2 + Fz * w_z ** 2
        w /= w.sum(axis=1, keepdims=True)
        out[s:e] = (w * sv[ix]).sum(axis=1).astype(np.float32)
    return out.reshape(ny, nx)



def idw_fz_points(sx, sy, sz, sv, tx, ty, tz, p=1.5, Fz=1.0, k=40, exclude_self=False):
    """Point IDW at the target coordinates, same weights as the IDW method.

    exclude_self is for targets that are the source stations: drop the nearest
    neighbour, which is the station itself.
    """
    from scipy.spatial import cKDTree

    sx = np.asarray(sx, dtype=np.float64)
    sy = np.asarray(sy, dtype=np.float64)
    sz = np.asarray(sz, dtype=np.float64)
    sv = np.asarray(sv, dtype=np.float64)
    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)
    tz = np.asarray(tz, dtype=np.float64)
    if len(tx) == 0 or len(sx) == 0:
        return np.zeros(len(tx), dtype=np.float32)
    tree = cKDTree(np.column_stack([sx, sy]))
    kk = int(min(max(k, 1), len(sv) - (1 if exclude_self and len(sv) > 1 else 0)))
    qk = min(kk + 1, len(sv)) if exclude_self else kk
    d, ix = tree.query(np.column_stack([tx, ty]), k=max(qk, 1))
    d = np.atleast_2d(np.asarray(d, dtype=np.float64))
    ix = np.atleast_2d(np.asarray(ix))
    if exclude_self and d.shape[1] > 1:
        d, ix = d[:, 1:kk + 1], ix[:, 1:kk + 1]
    d = np.maximum(d, 1e-6)
    dz = np.maximum(np.abs(sz[ix] - tz[:, None]), 1e-6)
    w_xy = d ** (-p)
    w_z = dz ** (-p)
    w = (1.0 - Fz) * w_xy ** 2 + Fz * w_z ** 2
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return (w * sv[ix]).sum(axis=1).astype(np.float32)


def sample_at_points(field, x, y, gx, gy):
    """Bilinear sample of a raster at exact station coordinates."""
    field = np.asarray(field, dtype=np.float64)
    gx = np.asarray(gx, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)
    dx = float(gx[1] - gx[0]) if len(gx) > 1 else 1.0
    dy = float(gy[1] - gy[0]) if len(gy) > 1 else 1.0
    fc = np.clip((np.asarray(x) - gx[0]) / dx, 0, len(gx) - 1.001)
    fr = np.clip((np.asarray(y) - gy[0]) / dy, 0, len(gy) - 1.001)
    r0 = np.floor(fr).astype(int)
    c0 = np.floor(fc).astype(int)
    r1 = np.minimum(r0 + 1, field.shape[0] - 1)
    c1 = np.minimum(c0 + 1, field.shape[1] - 1)
    wr, wc = fr - r0, fc - c0
    return (
        field[r0, c0] * (1 - wr) * (1 - wc)
        + field[r0, c1] * (1 - wr) * wc
        + field[r1, c0] * wr * (1 - wc)
        + field[r1, c1] * wr * wc
    ).astype(np.float32)


def idw_raster(rows, cols, values, ny, nx, k: int = 8, exclude_self: bool = True) -> np.ndarray:
    """IDW in pixel space. exclude_self drops the 0-distance hit so station
    cells get a real neighbour baseline (otherwise residual is identically 0).
    """
    from scipy.spatial import cKDTree

    values = np.asarray(values, dtype=np.float64)
    pts = np.column_stack([rows.astype(np.float64), cols.astype(np.float64)])
    yy, xx = np.mgrid[0:ny, 0:nx]
    grid = np.column_stack([yy.ravel(), xx.ravel()])
    tree = cKDTree(pts)
    kk = min(max(int(k), 1), len(values))
    qk = min(kk + 1, len(values)) if exclude_self else kk
    d, ix = tree.query(grid, k=qk)
    if qk == 1:
        d = np.asarray(d, dtype=np.float64)[:, None]
        ix = np.asarray(ix)[:, None]
    else:
        d = np.asarray(d, dtype=np.float64)
        ix = np.asarray(ix)
    if exclude_self and d.shape[1] > 1:
        out_d = d[:, :kk].copy()
        out_ix = ix[:, :kk].copy()
        hit = d[:, 0] <= 1e-12
        if hit.any():
            sl_d = d[hit, 1:]
            sl_i = ix[hit, 1:]
            if sl_d.shape[1] < kk:
                sl_d = np.pad(sl_d, ((0, 0), (0, kk - sl_d.shape[1])), mode="edge")
                sl_i = np.pad(sl_i, ((0, 0), (0, kk - sl_i.shape[1])), mode="edge")
            out_d[hit] = sl_d[:, :kk]
            out_ix[hit] = sl_i[:, :kk]
        d, ix = out_d, out_ix
    d = np.maximum(d, 1e-6)
    w = d ** (-2.0)
    w /= w.sum(axis=1, keepdims=True)
    out = (w * values[ix]).sum(axis=1)
    return out.reshape(ny, nx).astype(np.float32)


def grid_terrain(elev, gx, gy):
    """Slope / aspect rasters from the master-grid DEM."""
    from shared.terrain import aspect_trig, slope_aspect_from_dem

    dx = float(gx[1] - gx[0]) if len(gx) > 1 else 1.0
    dy = float(gy[1] - gy[0]) if len(gy) > 1 else 1.0
    slope, aspect = slope_aspect_from_dem(elev, abs(dx), abs(dy))
    sinasp, cosasp = aspect_trig(aspect)
    return (
        np.asarray(slope, dtype=np.float32),
        np.asarray(sinasp, dtype=np.float32),
        np.asarray(cosasp, dtype=np.float32),
    )


def station_to_raster(rows, cols, values, ny, nx):
    field = np.zeros((ny, nx), dtype=np.float32)
    mask = np.zeros((ny, nx), dtype=np.float32)
    count = np.zeros((ny, nx), dtype=np.float32)
    r = np.clip(rows, 0, ny - 1)
    c = np.clip(cols, 0, nx - 1)
    np.add.at(field, (r, c), values.astype(np.float32))
    np.add.at(count, (r, c), 1.0)
    good = count > 0
    field[good] /= count[good]
    mask[good] = 1.0
    return field, mask


def cyclic_time(ts, subdaily: bool):
    """Day-of-year, plus hour at sub-daily. Times are UTC. No np.datetime64 tz warning."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    doy = int(t.dayofyear)
    ang = 2 * np.pi * doy / 365.25
    out = [np.sin(ang), np.cos(ang)]
    if subdaily:
        hour = t.hour + t.minute / 60.0
        ha = 2 * np.pi * hour / 24.0
        out += [np.sin(ha), np.cos(ha)]
    return out


class CNNInterpolator:
    def __init__(self, cfg: CNNConfig, in_ch: int = 9, out_ch: int = 1):
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        self.net = SmallUNet(in_ch=in_ch, base=cfg.base_ch, n_levels=cfg.n_levels, out_ch=out_ch).to(self.device)
        self.y_mean = 0.0
        self.y_std = 1.0
        self.z_mean = 0.0
        self.z_std = 1.0
        self.log_amount = False
        self.in_ch = in_ch
        self.out_ch = out_ch

    def state_dict(self):
        return {
            "net": self.net.state_dict(),
            "y_mean": self.y_mean, "y_std": self.y_std,
            "z_mean": self.z_mean, "z_std": self.z_std,
            "log_amount": bool(self.log_amount),
            "amt_clip": float(getattr(self, "_amt_clip", 1e3)),
            "in_ch": self.in_ch, "out_ch": self.out_ch,
            "cfg": self.cfg.__dict__,
        }

    def load_state_dict(self, blob):
        saved = blob.get("cfg") or {}
        if isinstance(saved, dict):
            for k, v in saved.items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)
        net_sd = blob["net"]
        n_levels, base_ch, in_ch = _infer_unet(net_sd)
        self.in_ch = int(blob.get("in_ch", in_ch))
        self.out_ch = int(blob.get("out_ch", self.out_ch))
        if "down.0.net.0.weight" in net_sd:
            self.in_ch = int(net_sd["down.0.net.0.weight"].shape[1])
        if "out.weight" in net_sd:
            self.out_ch = int(net_sd["out.weight"].shape[0])
        self.cfg.n_levels = n_levels
        self.cfg.base_ch = base_ch
        self.net = SmallUNet(self.in_ch, base_ch, n_levels, self.out_ch).to(self.device)
        self.net.load_state_dict(net_sd)
        self.y_mean = blob["y_mean"]
        self.y_std = blob["y_std"]
        self.z_mean = blob["z_mean"]
        self.z_std = blob["z_std"]
        self.log_amount = bool(blob.get("log_amount", False))
        self._amt_clip = float(blob.get("amt_clip", 1e3))

    def pack_channels(self, value, mask, elev, clc, slope=None, sinasp=None, cosasp=None,
                      time_ch=None, extras=None, dx_m: float = 1000.0, dy_m: float = 1000.0):
        v = np.where(mask > 0, (value - self.y_mean) / self.y_std, 0.0)
        elev = np.asarray(elev, dtype=np.float64)
        bad = ~np.isfinite(elev)
        if bad.any():
            elev = elev.copy()
            elev[bad] = self.z_mean
        z = (elev - self.z_mean) / max(self.z_std, 1.0)
        chans = [v, mask, z]
        if self.cfg.use_terrain:
            ny, nx = value.shape
            sl = np.zeros((ny, nx), dtype=np.float32) if slope is None else np.asarray(slope, dtype=np.float32)
            sl = np.where(np.isfinite(sl), sl, 0.0) / 45.0
            sa = np.zeros((ny, nx), dtype=np.float32) if sinasp is None else np.asarray(sinasp, dtype=np.float32)
            ca = np.ones((ny, nx), dtype=np.float32) if cosasp is None else np.asarray(cosasp, dtype=np.float32)
            chans.append(sl)
            chans.append(np.where(np.isfinite(sa), sa, 0.0))
            chans.append(np.where(np.isfinite(ca), ca, 0.0))
        # proximity weight: 1 at stations, decays with distance. Explicit sparsity signal.
        d0 = float(getattr(self.cfg, "gate_d0_km", 35.0))
        chans.append(station_weight(mask, dx_m, dy_m, d0))
        # CLC level-2 one-hot (16 channels). Best for a conv net: each class
        # gets its own filter instead of a meaningless ordinal scale.
        from shared.clc import level2_index, N_LEVEL2
        idx = level2_index(np.asarray(clc).reshape(-1)).reshape(value.shape)
        for k in range(N_LEVEL2):
            chans.append((idx == k).astype(np.float32))
        if time_ch:
            for t in time_ch:
                chans.append(np.full(value.shape, float(t), dtype=np.float32))
        if extras:
            for ex in extras:
                chans.append(np.asarray(ex, dtype=np.float32))
        packed = np.stack(chans, axis=0).astype(np.float32)
        if packed.shape[0] != int(self.in_ch):
            raise RuntimeError(
                f"CNN packed {packed.shape[0]} channels but net expects {self.in_ch}. "
                f"use_terrain={self.cfg.use_terrain} cyclic_time={self.cfg.cyclic_time}"
            )
        return packed

    def predict_raster(self, value, mask, elev, clc, slope=None, sinasp=None, cosasp=None,
                       time_ch=None, two_step: bool = False, base=None, extras=None,
                       var: str = "", donor_wet: float | None = None) -> np.ndarray:
        self.net.eval()
        ch = self.pack_channels(value, mask, elev, clc, slope, sinasp, cosasp, time_ch, extras=extras)
        x = torch.from_numpy(ch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            hat = self.net(x).cpu().numpy()[0]
        if two_step and hat.ndim == 3 and hat.shape[0] >= 2:
            p = 1.0 / (1.0 + np.exp(-hat[0]))
            amt = hat[1] * self.y_std + self.y_mean
            if self.log_amount:
                amt = np.expm1(amt)
            elif base is not None:
                amt = amt + base
            amt = np.maximum(amt, 0.0)
            cap = float(getattr(self, "_amt_clip", 1e3))
            amt = np.minimum(amt, cap)
            gate = p >= self.cfg.tau_wet
            # Rank-gate fallback: if the indicator never clears tau but wet
            # donors exist, open the highest-p fraction matching the donor rate.
            if not gate.any() and donor_wet is not None and donor_wet > 0 and p.size:
                k = max(1, int(np.ceil(donor_wet * p.size)))
                thr = np.partition(p.ravel(), -k)[-k]
                gate = p >= thr
            out = np.where(gate, amt, 0.0)
            w = station_weight(mask, d0_km=float(getattr(self.cfg, "gate_d0_km", 35.0)))
            if base is not None:
                out = w * out + (1.0 - w) * base
            return clip_var(var, out).astype(np.float32)
        raw = hat[0] if hat.ndim == 3 else hat
        out = raw * self.y_std + self.y_mean
        if self.log_amount:
            out = np.expm1(out)
            cap = float(getattr(self, "_amt_clip", 1e3))
            out = np.minimum(np.maximum(out, 0.0), cap)
        w = station_weight(mask, d0_km=float(getattr(self.cfg, "gate_d0_km", 35.0)))
        if base is not None:
            out = w * out + (1.0 - w) * base
        return clip_var(var, out).astype(np.float32)
