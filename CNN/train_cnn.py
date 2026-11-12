#!/usr/bin/env python3
"""Fit the pooled U-Net on TRAIN timestamps, early-stop on DEV stations."""

from __future__ import annotations

from pathlib import Path
import argparse
import time
import sys
from concurrent.futures import ThreadPoolExecutor
import queue

import numpy as np
import pandas as pd
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import get_cnn_model_path, get_cnn_tuned_params_path, get_master_grid_path
from cnn_core import (
    idw_fz_raster, idw_fz_points,
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
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res, variables_from_cfg
from shared.splits.evaluation_sets import prepare_for_tuning

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
        tile_size=int(c.get("tile_size", 128)),
        tiles_per_batch=int(c.get("tiles_per_batch", 32)),
        accum_steps=int(c.get("accum_steps", 8)),
        seed=int(c.get("seed", 22)),
        device=str(c.get("device", "auto")),
        use_amp=bool(c.get("use_amp", True)),
        min_stations=int(cfg.get("min_stations_per_field", 10)),
        rh_t_mode=str(c.get("rh_t_mode", "none")),
        gate_d0_km=float(c.get("gate_d0_km", 35.0)),
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



def _idw_baseline(var, time_res):
    """Tuned IDW (p, Fz, k). Tries the Methods path, then the old CODE path."""
    import yaml
    from paths import get_idw_tuned_params_path, get_method_output_dir
    res = time_res or "monthly"
    from paths import get_method_output_dir
    out = get_method_output_dir("IDW")
    candidates = [
        out / "global_params" / res / f"{var}_params.yaml",
        get_idw_tuned_params_path(var, res),
        out / "cluster_params" / res / "idw" / f"{var}_idw_params.yaml",
    ]
    for path in candidates:
        if path.exists():
            d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            print(f"  baseline IDW from {path}", flush=True)
            return float(d.get("p", 1.5)), float(d.get("Fz", 1.0)), int(d.get("k", 40))
    print(f"  [WARN] no IDW yaml for {var} {res}, fallback p=1.5 Fz=1.0 k=40", flush=True)
    print(f"         looked in {candidates[0]}", flush=True)
    return 1.5, 1.0, 40



def _station_fingerprint(names) -> str:
    """Order-sensitive. A different row order must not reuse a cached pts array."""
    import hashlib
    s = "|".join(map(str, names))
    return hashlib.md5(s.encode()).hexdigest()[:10]


def _idw_cache_path(var, time_res, ts, p, fz, k, target_names, exclude_self: bool, space: str = "raw"):
    """Cache key includes the target station set, exclude_self and the value space.

    pts is per-target, so a previous run with a different station count
    (or a different hold-out) must not be reused. Old _pool.npz files are ignored.
    space='log' is IDW of log1p and must not reuse a raw-mm cache.
    """
    from paths import get_method_output_dir
    key = pd.Timestamp(ts)
    if key.tzinfo is not None:
        key = key.tz_convert("UTC").tz_localize(None)
    d = get_method_output_dir("CNN") / "idw_cache" / (time_res or "monthly") / var
    d.mkdir(parents=True, exist_ok=True)
    flag = "ex" if exclude_self else "in"
    fp = _station_fingerprint(target_names)
    return d / f"{key:%Y%m%dT%H%M}_p{p}_fz{fz}_k{k}_{flag}_{space}_{fp}.npz"


def _load_idw(path):
    if not path.exists():
        return None
    z = np.load(path)
    return z["base"], z["pts"]


def _save_idw(path, base, pts):
    np.savez_compressed(path, base=np.asarray(base, dtype=np.float32), pts=np.asarray(pts, dtype=np.float32))


def build_frames(panel, var, gx, gy, elev, clc, ccfg: CNNConfig, split, subdaily=False, trace=0.1,
                 donors=None, time_res=None, jobs=1, log_space: bool = False):
    """donors: pool stations used as the IDW source. Dev frames must not IDW from
    the held-out stations themselves, or the baseline is ~60 sparse neighbours.

    log_space: IDW is of log1p(value). pts and sv are then in log space, so the
    amount head can be trained on the residual to that baseline.
    """
    sl = panel[panel["split"] == split]
    donor_src = donors if donors is not None else panel
    frames = []
    ny, nx = elev.shape
    two = ccfg.two_step and two_step_var(var)
    groups = list(sl.groupby("time"))
    space = "log" if log_space else "raw"

    from concurrent.futures import ThreadPoolExecutor

    def _vals(series):
        v = series.to_numpy(dtype=np.float64)
        return np.log1p(np.maximum(v, 0.0)) if log_space else v

    def _ensure(item):
        i, ts, part = item
        tkey = pd.Timestamp(ts)
        if tkey.tzinfo is not None:
            tkey = tkey.tz_convert("UTC").tz_localize(None)
        dtimes = pd.to_datetime(donor_src["time"])
        if getattr(dtimes.dt, "tz", None) is not None:
            dtimes = dtimes.dt.tz_convert("UTC").dt.tz_localize(None)
        don = donor_src.loc[dtimes == tkey]
        # Score frames must not IDW from the targets themselves.
        if donors is not None:
            don = don[~don["station_name"].isin(part["station_name"])]
        if don.empty or len(don) < 4 or not ccfg.residual_idw:
            return "skip"
        same = donors is None
        cache = _idw_cache_path(var, time_res, tkey, ccfg.idw_p, ccfg.idw_fz, ccfg.idw_k,
                                part["station_name"].to_numpy(), same, space)
        if cache.exists():
            try:
                base, pts = _load_idw(cache)
                if len(pts) == len(part):
                    return "cached"
            except Exception:
                pass
        dx_, dy_, dz_ = don["x"].to_numpy(), don["y"].to_numpy(), don["elev"].to_numpy()
        dv = _vals(don[var])
        base = idw_fz_raster(dx_, dy_, dz_, dv, gx, gy, elev, p=ccfg.idw_p, Fz=ccfg.idw_fz, k=ccfg.idw_k, exclude_self=same)
        pts = idw_fz_points(dx_, dy_, dz_, dv, part["x"].to_numpy(), part["y"].to_numpy(), part["elev"].to_numpy(),
                            p=ccfg.idw_p, Fz=ccfg.idw_fz, k=ccfg.idw_k, exclude_self=same)
        _save_idw(cache, base, pts)
        return "built"

    n_jobs = max(1, int(jobs))
    items = [(i, ts, part) for i, (ts, part) in enumerate(groups, start=1)]
    print(f"    {split}: {len(items)} timestamps, Fz IDW jobs={n_jobs}", flush=True)
    if n_jobs > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            for done, tag in enumerate(ex.map(_ensure, items), start=1):
                if done % 25 == 0 or done == len(items):
                    print(f"    idw {done}/{len(items)} {tag}", flush=True)
    else:
        for it in items:
            _ensure(it)

    for i, (ts, part) in enumerate(groups, start=1):
        if part["station_name"].nunique() < ccfg.min_stations:
            print(f"    [{i}/{len(groups)}] {ts} skipped", flush=True)
            continue
        n_st = part["station_name"].nunique()
        rows, cols = grid_index(part["x"].to_numpy(), part["y"].to_numpy(), gx, gy)
        val = part[var].to_numpy()
        field, mask = station_to_raster(rows, cols, val, ny, nx)
        tkey = pd.Timestamp(ts)
        if tkey.tzinfo is not None:
            tkey = tkey.tz_convert("UTC").tz_localize(None)
        dtimes = pd.to_datetime(donor_src["time"])
        if getattr(dtimes.dt, "tz", None) is not None:
            dtimes = dtimes.dt.tz_convert("UTC").dt.tz_localize(None)
        don = donor_src.loc[dtimes == tkey]
        # For score frames the targets must not be donors
        if donors is not None:
            don = don[~don["station_name"].isin(part["station_name"])]
        if don.empty or len(don) < 4:
            print(f"    [{i}/{len(groups)}] {ts} no donors (n={len(don)})", flush=True)
            continue
        n_don = don["station_name"].nunique()
        extra = f" donors={n_don}" if n_don != n_st else ""
        dx_, dy_, dz_ = don["x"].to_numpy(), don["y"].to_numpy(), don["elev"].to_numpy()
        dv = _vals(don[var])
        sx, sy, sz = part["x"].to_numpy(), part["y"].to_numpy(), part["elev"].to_numpy()
        # exclude_self only when the targets are among the donors (train path)
        same = donors is None
        cache = _idw_cache_path(var, time_res, tkey, ccfg.idw_p, ccfg.idw_fz, ccfg.idw_k,
                                part["station_name"].to_numpy(), same, space)
        hit = _load_idw(cache) if ccfg.residual_idw else None
        if hit is not None and len(hit[1]) == len(rows):
            base, pts = hit
            base = base.copy()
            base[rows, cols] = pts
            tag = "cached"
        elif ccfg.residual_idw:
            base = idw_fz_raster(
                dx_, dy_, dz_, dv, gx, gy, elev,
                p=ccfg.idw_p, Fz=ccfg.idw_fz, k=ccfg.idw_k, exclude_self=same,
            )
            pts = idw_fz_points(
                dx_, dy_, dz_, dv, sx, sy, sz,
                p=ccfg.idw_p, Fz=ccfg.idw_fz, k=ccfg.idw_k, exclude_self=same,
            )
            _save_idw(cache, base, pts)
            base = base.copy()
            base[rows, cols] = pts
            tag = "built"
        else:
            base = np.zeros_like(field)
            tag = "none"
        print(f"    [{i}/{len(groups)}] {ts}  stations={n_st}{extra} {tag}", flush=True)
        resid = np.where(mask > 0, field - base, 0.0)
        wet = ((field > trace) & (mask > 0)).astype(np.float32)
        frames.append({
            "rows": np.asarray(rows, dtype=np.int32),
            "cols": np.asarray(cols, dtype=np.int32),
            "obs": np.asarray(val, dtype=np.float32),
            "pts": np.asarray(base[rows, cols], dtype=np.float32),
            "sx": sx.astype(np.float32),
            "sy": sy.astype(np.float32),
            "sz": sz.astype(np.float32),
            "sv": (np.log1p(np.maximum(val, 0.0)) if log_space else val).astype(np.float32),
            "tmean": part["temp_mean"].to_numpy(dtype=np.float32) if "temp_mean" in part.columns else None,
            "wet": (np.asarray(val) > trace).astype(np.float32),
            "time": ts,
            "time_ch": cyclic_time(ts, subdaily) if ccfg.cyclic_time else None,
            "two_step": two,
        })
    return frames



def _crop(arr, r, c, tile, shape):
    """tile x tile window centered on (r, c), zero-padded at the border."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(shape)
    half = tile // 2
    ny, nx = arr.shape
    out = np.zeros((tile, tile), dtype=np.float32)
    r0, c0 = int(r) - half, int(c) - half
    rs, cs = max(r0, 0), max(c0, 0)
    re, ce = min(r0 + tile, ny), min(c0 + tile, nx)
    out[rs - r0:rs - r0 + (re - rs), cs - c0:cs - c0 + (ce - cs)] = arr[rs:re, cs:ce]
    return out


def pack_tiles(frames, elev, clc, model, rng, n_tiles, tile, train,
               slope=None, sinasp=None, cosasp=None):
    """Local windows centered on stations. Center pixel is the only target.

    The center is zeroed in the input so the net cannot read the observation.
    IDW on the frame is already exclude-self, so the residual target is real.
    Returns numpy arrays; the caller moves them to the GPU.
    """
    half = tile // 2
    shape = elev.shape
    chans, tgts, qmasks, wets = [], [], [], []
    n_fr = len(frames)
    for _ in range(n_tiles):
        fr = frames[int(rng.integers(0, n_fr))]
        r = np.asarray(fr["rows"])
        c = np.asarray(fr["cols"])
        si = int(rng.integers(0, len(r)))
        rr, cc = int(r[si]), int(c[si])
        field = np.asarray(fr["field"], dtype=np.float32)
        base = np.asarray(fr["base"], dtype=np.float32)
        value = np.asarray(fr["value"], dtype=np.float32)
        mask = np.asarray(fr["mask"], dtype=np.float32)
        vc = _crop(value, rr, cc, tile, shape)
        mc = _crop(mask, rr, cc, tile, shape)
        vc[half, half] = 0.0
        mc[half, half] = 0.0
        chans.append(model.pack_channels(
            vc, mc, _crop(elev, rr, cc, tile, shape), _crop(clc, rr, cc, tile, shape),
            slope=None if slope is None else _crop(slope, rr, cc, tile, shape),
            sinasp=None if sinasp is None else _crop(sinasp, rr, cc, tile, shape),
            cosasp=None if cosasp is None else _crop(cosasp, rr, cc, tile, shape),
            time_ch=fr.get("time_ch"),
        ))
        tgt = np.zeros((tile, tile), dtype=np.float32)
        tgt[half, half] = (field[rr, cc] - base[rr, cc] - model.y_mean) / model.y_std
        qm = np.zeros((tile, tile), dtype=np.float32)
        qm[half, half] = 1.0
        wet = np.zeros((tile, tile), dtype=np.float32)
        wet[half, half] = float(fr["wet"][rr, cc]) if "wet" in fr else 0.0
        tgts.append(tgt)
        qmasks.append(qm)
        wets.append(wet)
    return (np.stack(chans), np.stack(tgts), np.stack(qmasks), np.stack(wets))


def _to_dev(arrs, device):
    """Pin memory on CUDA so the non-blocking transfer can overlap with CPU work."""
    out = []
    use_pin = device.type == "cuda"
    for a in arrs:
        t = torch.from_numpy(np.ascontiguousarray(a))
        if use_pin:
            t = t.pin_memory()
        out.append(t.to(device, non_blocking=True))
    return tuple(out)



def _attach_static(frames, elev, clc, model, slope, sinasp, cosasp, extra=None):
    """One shared terrain/CLC grid. Time is a constant channel, filled per frame.

    extra: one raster or a list of rasters appended after the base channels.
    For RH the last extra is a zero placeholder for the per-frame tmean channel.
    """
    z = np.zeros_like(elev, dtype=np.float32)
    n_time = len(frames[0].get("time_ch") or [])
    if extra is None:
        extras = None
    elif isinstance(extra, (list, tuple)):
        extras = list(extra) or None
    else:
        extras = [extra]
    shared = model.pack_channels(
        z, z, elev, clc, slope=slope, sinasp=sinasp, cosasp=cosasp,
        time_ch=([0.0] * n_time) or None,
        extras=extras,
    )
    model._shared_static = shared
    model._n_time = n_time
    model._has_extra = bool(extras)
    # Channels before the cyclic time block (value, mask, z, terrain, prox, CLC).
    # Must match the order in CNNInterpolator.pack_channels.
    n_before_time = 3
    if model.cfg.use_terrain:
        n_before_time += 3
    n_before_time += 1 + 16  # proximity + CLC one-hot
    model._time_start = n_before_time
    model._n_cov_extras = 0 if extras is None else len(extras)


def _pack_from_static(frames, idx, model, rng, mask_frac):
    """Build one batch from the shared grid and the sparse station values.

    When stations are dropped (train_mask_frac), the residual baseline is
    recomputed from the kept stations only so the target residual is real.
    """
    from cnn_core import idw_fz_points
    ny, nx = model._shared_static.shape[-2:]
    chans, tgts, qmasks, wets, bases, fields = [], [], [], [], [], []
    for i in idx:
        fr = frames[i]
        r = np.asarray(fr["rows"]); c = np.asarray(fr["cols"])
        obs = np.asarray(fr["obs"], dtype=np.float32)
        pts = np.asarray(fr["pts"], dtype=np.float32)
        drop = np.zeros(len(r), dtype=bool)
        if mask_frac > 0 and len(r) >= 5:
            cand = rng.random(len(r)) < mask_frac
            if cand.any() and (~cand).sum() >= 4:
                drop = cand
        keep = ~drop
        # Rebuild residual baseline from kept stations when we drop any.
        # sv is log1p when log_residual, raw otherwise.
        if model.cfg.residual_idw and drop.any() and keep.sum() >= 4:
            sx = fr.get("sx", np.zeros(0))
            sy = fr.get("sy", np.zeros(0))
            sz = fr.get("sz", np.zeros(0))
            sv = fr.get("sv", np.zeros(0))
            if len(sx) == len(r):
                base_pts = idw_fz_points(
                    sx[keep], sy[keep], sz[keep], sv[keep],
                    sx[drop], sy[drop], sz[drop],
                    p=model.cfg.idw_p, Fz=model.cfg.idw_fz, k=model.cfg.idw_k,
                    exclude_self=False,
                )
                pts = pts.copy()
                pts[drop] = base_pts
        ch = model._shared_static.copy()
        tc = fr.get("time_ch") or []
        # Time channels sit at a fixed offset (before extras / elev-bin).
        t0 = int(getattr(model, "_time_start", 0))
        for j, tv in enumerate(tc):
            if t0 + j < ch.shape[0]:
                ch[t0 + j] = float(tv)
        log_amt = bool(getattr(model, "log_amount", False))
        log_res = bool(getattr(model, "log_residual", False))
        value = np.zeros((ny, nx), dtype=np.float32)
        mask_in = np.zeros((ny, nx), dtype=np.float32)
        if log_res:
            value[r[keep], c[keep]] = np.log1p(np.maximum(obs[keep], 0.0))
        elif log_amt:
            value[r[keep], c[keep]] = np.log1p(np.maximum(obs[keep], 0.0))
        else:
            value[r[keep], c[keep]] = obs[keep] - pts[keep]
        mask_in[r[keep], c[keep]] = 1.0
        in_mean = float(getattr(model, "in_mean", model.y_mean))
        in_std = float(getattr(model, "in_std", model.y_std))
        ch[0] = np.where(mask_in > 0, (value - in_mean) / max(in_std, 1e-6), 0.0)
        ch[1] = mask_in
        # RH tmean: last extra channel (placeholder was appended in _attach_static).
        if getattr(model, "_is_rh", False) and fr.get("tmean") is not None:
            tm = np.asarray(fr["tmean"], dtype=np.float32)
            tfield = np.zeros((ny, nx), dtype=np.float32)
            tfield[r[keep], c[keep]] = tm[keep]
            # tmean is the final extra; elev-bin (two-step only) is never present for RH.
            ch[-1] = tfield / 20.0
        qmask = np.zeros((ny, nx), dtype=np.float32)
        if drop.any():
            qmask[r[drop], c[drop]] = 1.0
        else:
            qmask[r, c] = 1.0
        field = np.zeros((ny, nx), dtype=np.float32)
        base = np.zeros((ny, nx), dtype=np.float32)
        wet = np.zeros((ny, nx), dtype=np.float32)
        field[r, c] = obs
        base[r, c] = pts if log_res else (0.0 if log_amt else pts)
        wet[r, c] = np.asarray(fr["wet"], dtype=np.float32)
        if log_res:
            tgt_raw = np.zeros((ny, nx), dtype=np.float32)
            tgt_raw[r, c] = np.log1p(np.maximum(obs, 0.0)) - pts
        elif log_amt:
            tgt_raw = np.zeros((ny, nx), dtype=np.float32)
            tgt_raw[r, c] = np.log1p(np.maximum(obs, 0.0))
        else:
            tgt_raw = field - base
        chans.append(ch)
        tgts.append(((tgt_raw - model.y_mean) / model.y_std).astype(np.float32))
        qmasks.append(qmask)
        wets.append(wet)
        bases.append(base)
        fields.append(field)
    return tuple(np.stack(a).astype(np.float32) for a in (chans, tgts, qmasks, wets, bases, fields))



def _crop_centered(arr, r, c, tile, H, W):
    """Zero-padded tile centered on (r, c). Works for (C,H,W) or (H,W)."""
    half = tile // 2
    r0, c0 = int(r) - half, int(c) - half
    rs, cs = max(r0, 0), max(c0, 0)
    re, ce = min(r0 + tile, H), min(c0 + tile, W)
    or0, oc0 = rs - r0, cs - c0
    if arr.ndim == 3:
        out = np.zeros((arr.shape[0], tile, tile), dtype=np.float32)
        out[:, or0:or0 + (re - rs), oc0:oc0 + (ce - cs)] = arr[:, rs:re, cs:ce]
    else:
        out = np.zeros((tile, tile), dtype=np.float32)
        out[or0:or0 + (re - rs), oc0:oc0 + (ce - cs)] = arr[rs:re, cs:ce]
    return out


def _extract_tiles(chans, tgts, qmasks, wets, bases, fields, tile, rng, n_tiles, require_query=True):
    """Crop tiles from already-correct full-grid tensors.

    Sampling prefers query pixels so the loss is never empty.
    Returns the same 6-tuple layout as _pack_from_static, or None.
    """
    B, _, H, W = chans.shape
    out = [[] for _ in range(6)]
    srcs = (chans, tgts, qmasks, wets, bases, fields)
    attempts = 0
    max_attempts = max(n_tiles * 10, 20)
    while len(out[0]) < n_tiles and attempts < max_attempts:
        attempts += 1
        bi = int(rng.integers(0, B))
        q = qmasks[bi] > 0
        if require_query and not q.any():
            continue
        if q.any():
            ys, xs = np.where(q)
            si = int(rng.integers(0, len(ys)))
            r, c = int(ys[si]), int(xs[si])
        else:
            r = int(rng.integers(0, H))
            c = int(rng.integers(0, W))
        crops = [_crop_centered(a[bi], r, c, tile, H, W) for a in srcs]
        if require_query and not (crops[2] > 0).any():
            continue
        for bucket, crop in zip(out, crops):
            bucket.append(crop)
    if not out[0]:
        return None
    return tuple(np.stack(a).astype(np.float32) for a in out)



def _tiles_direct(frames, fi, model, rng, mask_frac, tile, n_tiles):
    """Build query-centered tiles without materialising the full grid.

    The old path copied a ~572 MiB raster per field just to crop 16 tiles.
    Here we crop the shared static once per tile and stamp only the stations
    that fall inside that window. Residual IDW is still on the points.
    """
    from cnn_core import idw_fz_points
    fr = frames[int(fi)]
    shared = model._shared_static
    C, H, W = shared.shape
    r = np.asarray(fr["rows"], dtype=np.int32)
    c = np.asarray(fr["cols"], dtype=np.int32)
    obs = np.asarray(fr["obs"], dtype=np.float32)
    pts = np.asarray(fr["pts"], dtype=np.float32)
    n_st = len(r)
    drop = np.zeros(n_st, dtype=bool)
    if mask_frac > 0 and n_st >= 5:
        cand = rng.random(n_st) < mask_frac
        if cand.any() and (~cand).sum() >= 4:
            drop = cand
    keep = ~drop
    log_amt = bool(getattr(model, "log_amount", False))
    log_res = bool(getattr(model, "log_residual", False))
    if model.cfg.residual_idw and drop.any() and keep.sum() >= 4:
        sx = fr.get("sx", np.zeros(0))
        sy = fr.get("sy", np.zeros(0))
        sz = fr.get("sz", np.zeros(0))
        sv = fr.get("sv", np.zeros(0))
        if len(sx) == n_st:
            base_pts = idw_fz_points(
                sx[keep], sy[keep], sz[keep], sv[keep],
                sx[drop], sy[drop], sz[drop],
                p=model.cfg.idw_p, Fz=model.cfg.idw_fz, k=model.cfg.idw_k,
                exclude_self=False,
            )
            pts = pts.copy()
            pts[drop] = base_pts
    # Query centers: dropped stations if any, else all stations.
    q_idx = np.flatnonzero(drop) if drop.any() else np.arange(n_st)
    if q_idx.size == 0:
        return None
    half = tile // 2
    t0 = int(getattr(model, "_time_start", 0))
    tc = fr.get("time_ch") or []
    is_rh = bool(getattr(model, "_is_rh", False)) and fr.get("tmean") is not None
    tm = np.asarray(fr["tmean"], dtype=np.float32) if is_rh else None
    wet_all = np.asarray(fr["wet"], dtype=np.float32)
    y_mean = float(model.y_mean)
    y_std = float(model.y_std)
    in_mean = float(getattr(model, "in_mean", y_mean))
    in_std = float(getattr(model, "in_std", y_std))
    chans, tgts, qmasks, wets, bases, fields = [], [], [], [], [], []
    for _ in range(n_tiles):
        si = int(q_idx[int(rng.integers(0, q_idx.size))])
        cr, cc = int(r[si]), int(c[si])
        # Crop window in full-grid coords.
        r0, c0 = cr - half, cc - half
        rs, cs = max(r0, 0), max(c0, 0)
        re, ce = min(r0 + tile, H), min(c0 + tile, W)
        or0, oc0 = rs - r0, cs - c0
        ch = np.zeros((C, tile, tile), dtype=np.float32)
        ch[:, or0:or0 + (re - rs), oc0:oc0 + (ce - cs)] = shared[:, rs:re, cs:ce]
        for j, tv in enumerate(tc):
            if t0 + j < C:
                ch[t0 + j] = float(tv)
        # Stations inside this tile.
        tr = r - r0
        tc_ = c - c0
        inside = (tr >= 0) & (tr < tile) & (tc_ >= 0) & (tc_ < tile)
        keep_in = inside & keep
        # Value / mask from kept stations.
        if keep_in.any():
            rr, cc_ = tr[keep_in], tc_[keep_in]
            if log_res or log_amt:
                val = np.log1p(np.maximum(obs[keep_in], 0.0))
            else:
                val = obs[keep_in] - pts[keep_in]
            ch[0, rr, cc_] = (val - in_mean) / max(in_std, 1e-6)
            ch[1, rr, cc_] = 1.0
            if is_rh:
                ch[-1, rr, cc_] = tm[keep_in] / 20.0
        # Targets and masks on all stations in the tile.
        qmask = np.zeros((tile, tile), dtype=np.float32)
        wet = np.zeros((tile, tile), dtype=np.float32)
        base = np.zeros((tile, tile), dtype=np.float32)
        field = np.zeros((tile, tile), dtype=np.float32)
        tgt = np.zeros((tile, tile), dtype=np.float32)
        if inside.any():
            rr, cc_ = tr[inside], tc_[inside]
            field[rr, cc_] = obs[inside]
            if log_res or not log_amt:
                base[rr, cc_] = pts[inside]
            wet[rr, cc_] = wet_all[inside]
            if drop.any():
                q_in = inside & drop
                if q_in.any():
                    qmask[tr[q_in], tc_[q_in]] = 1.0
            else:
                qmask[rr, cc_] = 1.0
            if log_res:
                raw = np.log1p(np.maximum(obs[inside], 0.0)) - pts[inside]
            elif log_amt:
                raw = np.log1p(np.maximum(obs[inside], 0.0))
            else:
                raw = field[rr, cc_] - base[rr, cc_]
            tgt[rr, cc_] = (raw - y_mean) / y_std
        if not (qmask > 0).any():
            continue
        chans.append(ch)
        tgts.append(tgt)
        qmasks.append(qmask)
        wets.append(wet)
        bases.append(base)
        fields.append(field)
    if not chans:
        return None
    return tuple(np.stack(a).astype(np.float32) for a in (chans, tgts, qmasks, wets, bases, fields))


def train_one(panel, var, ccfg: CNNConfig, elev, clc, gx, gy, slope=None, sinasp=None, cosasp=None,
              time_res: str | None = None, trace: float = 0.1, jobs: int = 1,
              cfg: dict | None = None):
    subdaily = is_subdaily(panel, time_res)
    use_two = bool(ccfg.two_step) and two_step_var(var)
    if use_two:
        obs = panel.loc[panel["split"] == "train", var].to_numpy()
        wet = float(np.mean(obs > trace)) if len(obs) else 0.0
        # Precip/snow keep the indicator even when one class dominates.
        # Weekly precip is ~95% wet; without a gate the amount head cannot
        # emit zeros and the score collapses.
        print(f"  two_step on: wet={wet:.3f}", flush=True)
    ccfg.two_step = use_two
    log_amount = two_step_var(var)
    if log_amount:
        # Amount head is a residual to IDW in log1p. Input stays the raw log
        # field so the indicator can see occurrence. a=b=0 of the residual is IDW.
        ccfg.residual_idw = True
        print("  precip path: residual to IDW in log1p", flush=True)
    out_ch = 2 if use_two else 1
    from cnn_core import load_extra_covariates
    if cfg is not None:
        cov_extras, n_cov = load_extra_covariates(
            cfg, int(cfg.get("resolutions_to_process", [1000])[0])
        )
    else:
        cov_extras, n_cov = [], 0
    extra = n_cov
    if str(var).lower().startswith("rh"):
        extra += 1  # tmean (per-variable, not a terrain covariate)
    if use_two:
        extra += 1  # smooth 250 m elev-bin index (not free one-hot)
    in_ch = n_in_channels(ccfg, subdaily, extra_ch=extra)
    model = CNNInterpolator(ccfg, in_ch=in_ch, out_ch=out_ch)
    # torch.compile disabled: on Windows (no Triton) it stalls at the first
    # forward with near-zero CPU/GPU use. Prefetch + larger batch + AMP
    # are the speed path that works here.
    if slope is None and ccfg.use_terrain:
        slope, sinasp, cosasp = grid_terrain(elev, gx, gy)
    p_idw, fz, k_idw = _idw_baseline(var, time_res)
    ccfg.idw_p, ccfg.idw_fz, ccfg.idw_k = p_idw, fz, k_idw
    print(f"  baseline IDW p={p_idw} Fz={fz} k={k_idw}", flush=True)
    train_fr = build_frames(panel, var, gx, gy, elev, clc, ccfg, "train", subdaily, trace, time_res=time_res, jobs=jobs, log_space=log_amount)
    # Donors are same-timestamp stations. Train and dev are different months,
    # so the donor pool must be the full (held-out-free) panel, not train times only.
    dev_fr = build_frames(panel, var, gx, gy, elev, clc, ccfg, "dev", subdaily, trace, donors=panel, time_res=time_res, jobs=jobs, log_space=log_amount)
    if not train_fr:
        raise RuntimeError(f"no train frames for {var}")
    print(f"  frames train={len(train_fr)}  dev={len(dev_fr)}", flush=True)
    if not dev_fr:
        print("  WARNING: no dev frames; dev loss will mirror train", flush=True)
    if dev_fr and not log_amount:
        err = np.concatenate([f["obs"] - f["pts"] for f in dev_fr])
        print(f"  dev baseline IDW RMSE={np.sqrt(np.mean(err**2)):.2f}  n={len(err)}", flush=True)
    if dev_fr and log_amount:
        err = np.concatenate([
            f["obs"] - np.expm1(np.asarray(f["pts"])) for f in dev_fr
        ])
        print(f"  dev baseline IDW-log RMSE={np.sqrt(np.mean(err**2)):.2f}  n={len(err)}", flush=True)
    vals = np.concatenate([f["obs"] for f in train_fr]).astype(np.float64)
    if log_amount:
        wet_vals = vals[vals > trace]
        model._amt_clip = float(np.quantile(wet_vals, 0.99) * 1.5) if len(wet_vals) else 1e3
        logv = np.log1p(np.maximum(vals, 0.0))
        base = np.concatenate([np.asarray(f["pts"]) for f in train_fr])
        resid = logv - base
        model.y_mean = float(resid.mean())
        model.y_std = float(max(resid.std(), 1e-3))
        model.in_mean = float(logv.mean())
        model.in_std = float(max(logv.std(), 1e-3))
        model.log_residual = True
    else:
        model.y_mean = float(vals.mean())
        model.y_std = float(vals.std() + 1e-6)
        model.in_mean = model.y_mean
        model.in_std = model.y_std
    model.log_amount = log_amount
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
    two = out_ch == 2
    n = len(train_fr)
    tile = int(getattr(ccfg, "tile_size", 128) or 128)
    # Larger effective batch: more tiles per GPU step and more accumulation.
    # RTX 5070 (12 GB) can take 16-32 tiles of 128 comfortably with AMP.
    tiles_per = max(1, int(getattr(ccfg, "tiles_per_batch", 32) or 32))
    accum = max(1, int(getattr(ccfg, "accum_steps", 8) or 8))
    # Tile path: build one field on CPU, crop query-centered tiles, accumulate.
    # Avoids full-grid backward (cuDNN) and epoch-wide materialization (UI freeze).
    print(f"  tile path  tile={tile}  tiles/batch={tiles_per}  accum={accum}", flush=True)
    # Keep cuDNN autotuned for the fixed tile size.
    if model.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    # Enabled core covariates as extra channels (same list produce will use).
    # RH tmean is a per-frame channel; supply a zero placeholder so the static
    # pack has the correct channel count. It is overwritten below.
    static_extras = list(cov_extras) if cov_extras else []
    is_rh = str(var).lower().startswith("rh")
    if is_rh:
        static_extras.append(np.zeros_like(elev, dtype=np.float32))
    _attach_static(train_fr + dev_fr, elev, clc, model, slope, sinasp, cosasp,
                   extra=static_extras or None)
    model._is_rh = is_rh

    def _loss_from(hat, y, m, wet):
        if two:
            logit, amt = hat[:, 0], hat[:, 1]
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, wet, reduction="none")
            mse = (amt - y) ** 2
            amt_m = m * wet
            loss_bce = (bce * m).sum() / m.sum().clamp_min(1.0)
            loss_mse = (mse * amt_m).sum() / amt_m.sum().clamp_min(1.0)
            return loss_bce + loss_mse, int(m.sum())
        pred = hat[:, 0] if hat.dim() == 4 else hat
        return ((pred - y) ** 2 * m).sum() / m.sum().clamp_min(1.0), int(m.sum())

    def _amp_ctx():
        if not use_amp:
            return _nullcontext()
        try:
            return torch.amp.autocast("cuda", enabled=True)
        except TypeError:
            return torch.cuda.amp.autocast(enabled=True)

    def _prepare_field(fi, local_rng):
        """CPU work for one field: tile-direct pack (no full-grid copy).
        local_rng is private to the calling thread.
        """
        return _tiles_direct(train_fr, int(fi), model, local_rng, ccfg.train_mask_frac, tile, tiles_per)

    for epoch in range(ccfg.epochs):
        t_ep = time.perf_counter()
        order = rng.permutation(n)
        model.net.train()
        total = 0.0
        seen = 0
        opt.zero_grad(set_to_none=True)
        micro = 0
        # Prefetch: prepare the next field's tiles on a background thread
        # while the GPU processes the current one. Queue depth 2.
        prep_q = queue.Queue(maxsize=2)
        def _worker(order_list, q, seed):
            local_rng = np.random.default_rng(seed)
            for fi in order_list:
                tiles = _prepare_field(fi, local_rng)
                q.put((fi, tiles))
            q.put(None)  # sentinel
        with ThreadPoolExecutor(max_workers=1) as prep:
            fut = prep.submit(_worker, order, prep_q, int(rng.integers(0, 2**31)))
            bi = 0
            while True:
                item = prep_q.get()
                if item is None:
                    break
                fi, tiles = item
                bi += 1
                if tiles is None:
                    continue
                x, y, m, wet, base_t, field_t = _to_dev(tiles, model.device)
                with _amp_ctx():
                    hat = model.net(x)
                    loss, n_q = _loss_from(hat, y, m, wet)
                # Scale so accum steps keep the same effective step size.
                scaled = loss / float(accum)
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                total += float(loss.detach()) * n_q
                seen += n_q
                micro += 1
                if micro % accum == 0:
                    if scaler is not None:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                if bi == 1 or bi % 50 == 0 or bi == n:
                    print(f"    epoch {epoch:03d}  field {bi}/{n}  "
                          f"train_mse={total / max(seen, 1):.4f}", flush=True)
            fut.result()  # surface any worker exception
        if micro % accum != 0:
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
        model.net.eval()
        dloss, dseen = 0.0, 0
        with torch.no_grad():
            for fi in range(len(dev_fr)):
                tiles = _tiles_direct(dev_fr, fi, model, rng, 0.0, tile, tiles_per)
                if tiles is None:
                    continue
                x, y, m, wet, base_t, field_t = _to_dev(tiles, model.device)
                hat = model.net(x)
                if two:
                    pw = torch.sigmoid(hat[:, 0])
                    amt = hat[:, 1] * model.y_std + model.y_mean
                    if getattr(model, "log_residual", False):
                        cap = float(getattr(model, "_amt_clip", 1e3))
                        amt = torch.clamp(amt + base_t, max=float(np.log1p(cap)))
                        amt = torch.expm1(amt)
                        recon = torch.where(pw >= ccfg.tau_wet, amt, torch.zeros_like(amt))
                    elif model.log_amount:
                        amt = torch.expm1(amt)
                        recon = amt
                    elif ccfg.residual_idw:
                        recon = amt + base_t
                    else:
                        recon = amt
                else:
                    pred = hat[:, 0] if hat.dim() == 4 else hat
                    recon = pred * model.y_std + model.y_mean
                    if model.log_amount:
                        recon = torch.expm1(recon)
                    elif ccfg.residual_idw:
                        recon = recon + base_t
                dloss += float((((recon - field_t) ** 2) * m).sum())
                dseen += int(m.sum())
        if not dev_fr or dseen == 0:
            dev = total / max(seen, 1)
        else:
            dev = dloss / max(dseen, 1)
        print(
            f"epoch {epoch:03d} train_mse={total/max(seen,1):.4f} "
            f"dev_mse={dev:.4f} dev_rmse_mm={np.sqrt(max(dev, 0.0)):.3f} "
            f"{time.perf_counter()-t_ep:.1f}s",
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
    p.add_argument("--quick", action="store_true")
    add_hours_arg(p)
    add_time_res_arg(p)
    args = p.parse_args()
    cfg = load_cnn_config()
    apply_time_res(cfg, args)
    if args.quick:
        cfg.setdefault("cnn", {})
        cfg["cnn"]["epochs"] = min(int(cfg["cnn"].get("epochs", 40)), 8)
        cfg["cnn"]["patience"] = min(int(cfg["cnn"].get("patience", 8)), 3)
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
    variables = [args.variable] if args.variable else variables_from_cfg(cfg)
    for var in variables:
        panel = load_panel(cfg, var)
        panel = prepare_for_tuning(panel, variable=var)
        # Station hold-out on the tune (now 'dev') timestamps so donors exist.
        # Without this every pool station is a target and build_frames drops the frame.
        if "dev" in panel["split"].values:
            rng = np.random.default_rng(int(cfg.get("cnn", {}).get("seed", 22)))
            pool_names = panel.loc[panel["split"] == "dev", "station_name"].drop_duplicates().to_numpy()
            pool_names = np.array(pool_names, copy=True)
            rng.shuffle(pool_names)
            n_tgt = max(1, int(round(len(pool_names) * 0.2)))
            targets = set(pool_names[:n_tgt])
            is_dev = panel["split"] == "dev"
            panel.loc[is_dev & ~panel["station_name"].isin(targets), "split"] = "pool"
        panel, _hours = apply_hour_cut(panel, time_res, args.hours)
        ccfg = cfg_to_cnn(cfg, load_tuned(var, time_res))
        model, dev = train_one(
            panel, var, ccfg, elev, clc, gx, gy,
            slope, sinasp, cosasp, time_res=time_res,
            trace=precip_trace(cfg, time_res, var),
            cfg=cfg,
        )
        path = get_cnn_model_path(var, time_res)
        torch.save(model.state_dict(), path)
        print(f"wrote {path} dev_loss={dev:.4f}")


if __name__ == "__main__":
    main()
