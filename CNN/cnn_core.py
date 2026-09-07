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
    idw_k: int = 8
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
    seed: int = 22
    device: str = "auto"
    use_amp: bool = True
    min_stations: int = 10


def two_step_var(var: str) -> bool:
    v = var.lower()
    return v.startswith("precip") or v.startswith("snow")


def n_in_channels(cfg: CNNConfig, subdaily: bool) -> int:
    n = 3  # value, mask, elev
    if cfg.use_terrain:
        n += 3  # slope, sinasp, cosasp
    n += 1  # clc
    if cfg.cyclic_time:
        n += 2
        if subdaily:
            n += 2
    return n


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
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
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


def idw_raster(rows, cols, values, ny, nx, k: int = 8) -> np.ndarray:
    from scipy.spatial import cKDTree

    values = np.asarray(values, dtype=np.float64)
    pts = np.column_stack([rows.astype(np.float64), cols.astype(np.float64)])
    yy, xx = np.mgrid[0:ny, 0:nx]
    grid = np.column_stack([yy.ravel(), xx.ravel()])
    tree = cKDTree(pts)
    kk = min(max(int(k), 1), len(values))
    d, ix = tree.query(grid, k=kk)
    if kk == 1:
        d = np.asarray(d, dtype=np.float64)[:, None]
        ix = np.asarray(ix)[:, None]
    else:
        d = np.asarray(d, dtype=np.float64)
        ix = np.asarray(ix)
    hit = d[:, 0] <= 1e-12
    d = np.maximum(d, 1e-6)
    w = d ** (-2.0)
    w[hit] = 0.0
    w[hit, 0] = 1.0
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
    t = np.datetime64(ts, "ns").astype("datetime64[D]")
    # day of year
    start = np.datetime64(str(t)[:4] + "-01-01")
    doy = int((t - start) / np.timedelta64(1, "D")) + 1
    ang = 2 * np.pi * doy / 365.25
    out = [np.sin(ang), np.cos(ang)]
    if subdaily:
        hour = (np.datetime64(ts, "ns") - np.datetime64(ts, "D")) / np.timedelta64(1, "h")
        ha = 2 * np.pi * float(hour) / 24.0
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
        self.in_ch = in_ch
        self.out_ch = out_ch

    def state_dict(self):
        return {
            "net": self.net.state_dict(),
            "y_mean": self.y_mean, "y_std": self.y_std,
            "z_mean": self.z_mean, "z_std": self.z_std,
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

    def pack_channels(self, value, mask, elev, clc, slope=None, sinasp=None, cosasp=None, time_ch=None):
        v = np.where(mask > 0, (value - self.y_mean) / self.y_std, 0.0)
        z = (elev - self.z_mean) / max(self.z_std, 1.0)
        chans = [v, mask, z]
        if self.cfg.use_terrain:
            ny, nx = value.shape
            chans.append(np.zeros((ny, nx)) if slope is None else slope / 45.0)
            chans.append(np.zeros((ny, nx)) if sinasp is None else sinasp)
            chans.append(np.ones((ny, nx)) if cosasp is None else cosasp)
        chans.append(clc / 5.0)
        if time_ch:
            for t in time_ch:
                chans.append(np.full(value.shape, float(t), dtype=np.float32))
        packed = np.stack(chans, axis=0).astype(np.float32)
        if packed.shape[0] != int(self.in_ch):
            raise RuntimeError(
                f"CNN packed {packed.shape[0]} channels but net expects {self.in_ch}. "
                f"use_terrain={self.cfg.use_terrain} cyclic_time={self.cfg.cyclic_time}"
            )
        return packed

    def predict_raster(self, value, mask, elev, clc, slope=None, sinasp=None, cosasp=None,
                       time_ch=None, two_step: bool = False, base=None) -> np.ndarray:
        self.net.eval()
        ch = self.pack_channels(value, mask, elev, clc, slope, sinasp, cosasp, time_ch)
        x = torch.from_numpy(ch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            hat = self.net(x).cpu().numpy()[0]
        if two_step and hat.ndim == 3 and hat.shape[0] >= 2:
            p = 1.0 / (1.0 + np.exp(-hat[0]))
            amt = hat[1] * self.y_std + self.y_mean
            if base is not None:
                amt = amt + base
            amt = np.maximum(amt, 0.0)
            return np.where(p >= self.cfg.tau_wet, amt, 0.0).astype(np.float32)
        raw = hat[0] if hat.ndim == 3 else hat
        out = raw * self.y_std + self.y_mean
        if base is not None:
            out = out + base
        return out.astype(np.float32)
