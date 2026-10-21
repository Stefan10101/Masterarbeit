#!/usr/bin/env python3
"""Fit the pooled U-Net on TRAIN timestamps, early-stop on DEV stations."""

from __future__ import annotations

from pathlib import Path
import argparse
import time
import sys

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
from shared.time_res import add_hours_arg, add_time_res_arg, apply_hour_cut, apply_time_res
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
        tile_size=int(c.get("tile_size", 64)),
        tiles_per_batch=int(c.get("tiles_per_batch", 64)),
        seed=int(c.get("seed", 22)),
        device=str(c.get("device", "auto")),
        use_amp=bool(c.get("use_amp", True)),
        min_stations=int(cfg.get("min_stations_per_field", 10)),
        rh_t_mode=str(c.get("rh_t_mode", "none")),
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



def _idw_cache_path(var, time_res, ts, p, fz, k):
    from paths import get_method_output_dir
    key = pd.Timestamp(ts)
    if key.tzinfo is not None:
        key = key.tz_convert("UTC").tz_localize(None)
    d = get_method_output_dir("CNN") / "idw_cache" / (time_res or "monthly") / var
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key:%Y%m%dT%H%M}_p{p}_fz{fz}_k{k}.npz"


def _load_idw(path):
    if not path.exists():
        return None
    z = np.load(path)
    return z["base"], z["pts"]


def _save_idw(path, base, pts):
    np.savez_compressed(path, base=np.asarray(base, dtype=np.float32), pts=np.asarray(pts, dtype=np.float32))


def build_frames(panel, var, gx, gy, elev, clc, ccfg: CNNConfig, split, subdaily=False, trace=0.1,
                 donors=None, time_res=None, jobs=1):
    """donors: pool stations used as the IDW source. Dev frames must not IDW from
    the held-out stations themselves, or the baseline is ~60 sparse neighbours."""
    sl = panel[panel["split"] == split]
    donor_src = donors if donors is not None else panel
    frames = []
    ny, nx = elev.shape
    two = ccfg.two_step and two_step_var(var)
    groups = list(sl.groupby("time"))

    from concurrent.futures import ThreadPoolExecutor

    def _ensure(item):
        i, ts, part = item
        tkey = pd.Timestamp(ts)
        if tkey.tzinfo is not None:
            tkey = tkey.tz_convert("UTC").tz_localize(None)
        dtimes = pd.to_datetime(donor_src["time"])
        if getattr(dtimes.dt, "tz", None) is not None:
            dtimes = dtimes.dt.tz_convert("UTC").dt.tz_localize(None)
        don = donor_src.loc[dtimes == tkey]
        if don.empty or len(don) < 4 or not ccfg.residual_idw:
            return "skip"
        cache = _idw_cache_path(var, time_res, tkey, ccfg.idw_p, ccfg.idw_fz, ccfg.idw_k)
        if cache.exists():
            return "cached"
        dx_, dy_, dz_, dv = don["x"].to_numpy(), don["y"].to_numpy(), don["elev"].to_numpy(), don[var].to_numpy()
        same = donors is None
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
        if don.empty or len(don) < 4:
            print(f"    [{i}/{len(groups)}] {ts} no donors (n={len(don)})", flush=True)
            continue
        n_don = don["station_name"].nunique()
        extra = f" donors={n_don}" if n_don != n_st else ""
        dx_, dy_, dz_, dv = don["x"].to_numpy(), don["y"].to_numpy(), don["elev"].to_numpy(), don[var].to_numpy()
        sx, sy, sz = part["x"].to_numpy(), part["y"].to_numpy(), part["elev"].to_numpy()
        same = donors is None
        cache = _idw_cache_path(var, time_res, tkey, ccfg.idw_p, ccfg.idw_fz, ccfg.idw_k)
        hit = _load_idw(cache) if ccfg.residual_idw else None
        if hit is not None:
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
    return tuple(torch.from_numpy(a).to(device, non_blocking=True) for a in arrs)



def _attach_static(frames, elev, clc, model, slope, sinasp, cosasp):
    """One shared terrain/CLC grid. Time is a constant channel, filled per frame.

    A daily run is ~1500 frames. Storing the 24-channel grid on each one is
    ~110 GB. The grid does not change with the day.
    """
    z = np.zeros_like(elev, dtype=np.float32)
    n_time = len(frames[0].get("time_ch") or [])
    shared = model.pack_channels(
        z, z, elev, clc, slope=slope, sinasp=sinasp, cosasp=cosasp,
        time_ch=([0.0] * n_time) or None,
    )
    model._shared_static = shared
    model._n_time = n_time


def _pack_from_static(frames, idx, model, rng, mask_frac):
    """Build one batch from the shared grid and the sparse station values.

    Nothing full-grid is kept on the frame. The batch is the only dense array,
    and it is discarded after the step.
    """
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
        ch = model._shared_static.copy()
        tc = fr.get("time_ch") or []
        for j, tv in enumerate(tc):
            ch[-(len(tc) - j)] = float(tv)
        value = np.zeros((ny, nx), dtype=np.float32)
        mask_in = np.zeros((ny, nx), dtype=np.float32)
        value[r[keep], c[keep]] = obs[keep] - pts[keep]
        mask_in[r[keep], c[keep]] = 1.0
        ch[0] = np.where(mask_in > 0, (value - model.y_mean) / model.y_std, 0.0)
        ch[1] = mask_in
        qmask = np.zeros((ny, nx), dtype=np.float32)
        if drop.any():
            qmask[r[drop], c[drop]] = 1.0
        else:
            qmask[r, c] = 1.0
        field = np.zeros((ny, nx), dtype=np.float32)
        base = np.zeros((ny, nx), dtype=np.float32)
        wet = np.zeros((ny, nx), dtype=np.float32)
        field[r, c] = obs
        base[r, c] = pts
        wet[r, c] = np.asarray(fr["wet"], dtype=np.float32)
        chans.append(ch)
        tgts.append(((field - base - model.y_mean) / model.y_std).astype(np.float32))
        qmasks.append(qmask)
        wets.append(wet)
        bases.append(base)
        fields.append(field)
    return tuple(np.stack(a).astype(np.float32) for a in (chans, tgts, qmasks, wets, bases, fields))


def pack_batch(frames, idx, elev, clc, model: CNNInterpolator, rng, mask_frac, train: bool,
               slope=None, sinasp=None, cosasp=None, extras_fn=None):
    """Hide a fraction of stations, rebuild IDW from the rest, lose on hidden pixels.

    Self-inclusive IDW made station residuals identically 0 and the net learned
    nothing. Donor-only IDW is the actual interpolation residual.
    """
    chans, tgts, qmasks, wets, bases, fields = [], [], [], [], [], []
    for i in idx:
        fr = frames[i]
        r = np.asarray(fr["rows"])
        c = np.asarray(fr["cols"])
        obs = np.asarray(fr["obs"], dtype=np.float64)
        field = np.asarray(fr["field"], dtype=np.float32)
        ny, nx = field.shape
        drop = np.zeros(len(r), dtype=bool)
        if mask_frac > 0 and len(r) >= 5:
            cand = rng.random(len(r)) < mask_frac
            if cand.any() and (~cand).sum() >= 4:
                drop = cand
        keep = ~drop
        if keep.sum() < 4:
            keep = np.ones(len(r), dtype=bool)
            drop = ~keep
        # Full-grid IDW is cached on the frame. Rebuilding it per batch is the
        # CPU stall that leaves the GPU idle.
        base = fr.get("base")
        if base is None:
            base = idw_raster(r[keep], c[keep], obs[keep], ny, nx, k=model.cfg.idw_k, exclude_self=True)
        value = np.zeros_like(field)
        mask_in = np.zeros_like(field)
        value[r[keep], c[keep]] = field[r[keep], c[keep]] - base[r[keep], c[keep]]
        mask_in[r[keep], c[keep]] = 1.0
        qmask = np.zeros_like(field)
        if drop.any():
            qmask[r[drop], c[drop]] = 1.0
        else:
            qmask[r, c] = 1.0
        extras = extras_fn(fr) if extras_fn is not None else None
        chans.append(model.pack_channels(
            value, mask_in, elev, clc,
            slope=slope, sinasp=sinasp, cosasp=cosasp,
            time_ch=fr.get("time_ch"), extras=extras,
        ))
        tgt = ((field - base) - model.y_mean) / model.y_std
        tgts.append(tgt.astype(np.float32))
        qmasks.append(qmask.astype(np.float32))
        wet = np.zeros_like(fr["field"])
        wet[r, c] = np.asarray(fr.get("wet", 0.0), dtype=np.float32)
        wets.append(wet)
        bases.append(np.asarray(base, dtype=np.float32))
        fields.append(field)
    return (
        np.stack(chans).astype(np.float32),
        np.stack(tgts).astype(np.float32),
        np.stack(qmasks).astype(np.float32),
        np.stack(wets).astype(np.float32),
        np.stack(bases).astype(np.float32),
        np.stack(fields).astype(np.float32),
    )


def train_one(panel, var, ccfg: CNNConfig, elev, clc, gx, gy, slope=None, sinasp=None, cosasp=None,
              time_res: str | None = None, trace: float = 0.1, jobs: int = 1):
    subdaily = is_subdaily(panel, time_res)
    use_two = bool(ccfg.two_step) and two_step_var(var)
    if use_two:
        obs = panel.loc[panel["split"] == "train", var].to_numpy()
        wet = float(np.mean(obs > trace)) if len(obs) else 0.0
        # One class 80% of the time: the indicator cannot move the score.
        if wet >= 0.80 or wet <= 0.20:
            use_two = False
            print(f"  two_step off: wet={wet:.3f} (indicator is constant)", flush=True)
        else:
            print(f"  two_step on: wet={wet:.3f}", flush=True)
    ccfg.two_step = use_two
    out_ch = 2 if use_two else 1
    in_ch = n_in_channels(ccfg, subdaily)
    model = CNNInterpolator(ccfg, in_ch=in_ch, out_ch=out_ch)
    if slope is None and ccfg.use_terrain:
        slope, sinasp, cosasp = grid_terrain(elev, gx, gy)
    p_idw, fz, k_idw = _idw_baseline(var, time_res)
    ccfg.idw_p, ccfg.idw_fz, ccfg.idw_k = p_idw, fz, k_idw
    print(f"  baseline IDW p={p_idw} Fz={fz} k={k_idw}", flush=True)
    train_fr = build_frames(panel, var, gx, gy, elev, clc, ccfg, "train", subdaily, trace, time_res=time_res, jobs=jobs)
    pool = panel[panel["split"].isin(["train", "pool"])]
    dev_fr = build_frames(panel, var, gx, gy, elev, clc, ccfg, "dev", subdaily, trace, donors=pool, time_res=time_res, jobs=jobs)
    if not train_fr:
        raise RuntimeError(f"no train frames for {var}")
    if dev_fr:
        err = np.concatenate([f["obs"] - f["pts"] for f in dev_fr])
        print(f"  dev baseline IDW RMSE={np.sqrt(np.mean(err**2)):.2f}  n={len(err)}", flush=True)
    vals = np.concatenate([f["obs"] - f["pts"] for f in train_fr])
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
    two = out_ch == 2
    n = len(train_fr)
    b = max(1, int(ccfg.batch_times))
    order = rng.permutation(n)
    batches = [order[i:i + b] for i in range(0, n, b)]
    # Monthly has ~48 images, so the mask must change or the net memorizes them.
    # Daily and finer have enough fields that one mask is fine, and prebuilding
    # it keeps the GPU busy.
    remask = time_res in (None, "monthly", "weekly", "seasonal")
    _attach_static(train_fr + dev_fr, elev, clc, model, slope, sinasp, cosasp)
    dev_idx = [list(range(i, min(i + b, len(dev_fr)))) for i in range(0, len(dev_fr), b)]
    dev_batches = None  # packed inside the eval loop
    train_batches = None
    fut = None
    pool = None
    if remask:
        print(f"  full grid  batch={b}  static once, new mask each epoch", flush=True)
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=1)

        def _epoch_batches(seed):
            rr = np.random.default_rng(seed)
            return [_pack_from_static(train_fr, idx, model, rr, ccfg.train_mask_frac) for idx in batches]

        fut = pool.submit(_epoch_batches, int(rng.integers(1e9)))
    else:
        print(f"  full grid  batch={b}  pack one batch at a time ({len(batches)} / epoch)", flush=True)
    for epoch in range(ccfg.epochs):
        t_ep = time.perf_counter()
        if remask:
            train_batches = fut.result()
            fut = pool.submit(_epoch_batches, int(rng.integers(1e9)))
            epoch_batches = train_batches
        else:
            epoch_batches = (_pack_from_static(train_fr, idx, model, rng, ccfg.train_mask_frac) for idx in batches)
        model.net.train()
        total = 0.0
        seen = 0
        n_b = len(batches)
        for bi, arrs in enumerate(epoch_batches, start=1):
            x, y, m, wet, base_t, field_t = _to_dev(arrs, model.device)
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
            if bi == 1 or bi % 50 == 0 or bi == n_b:
                print(f"    epoch {epoch:03d}  batch {bi}/{n_b}  "
                      f"train_mse={total / max(seen, 1):.4f}", flush=True)
        model.net.eval()
        dloss, dseen = 0.0, 0
        with torch.no_grad():
            for arrs in (_pack_from_static(dev_fr, idx, model, rng, 0.0) for idx in dev_idx):
                x, y, m, wet, base_t, field_t = _to_dev(arrs, model.device)
                hat = model.net(x)
                if two:
                    p = torch.sigmoid(hat[:, 0])
                    amt = hat[:, 1] * model.y_std + model.y_mean
                    if ccfg.residual_idw:
                        amt = amt + base_t
                    recon = torch.where(p + 1e-4 >= ccfg.tau_wet, amt, torch.zeros_like(amt))
                else:
                    pred = hat[:, 0] if hat.dim() == 4 else hat
                    recon = pred * model.y_std + model.y_mean
                    if ccfg.residual_idw:
                        recon = recon + base_t
                dloss += float((((recon - field_t) ** 2) * m).sum())
                dseen += int(m.sum())
        if not dev_fr:
            dev = total / max(seen, 1)
        else:
            dev = dloss / max(dseen, 1)
        print(
            f"epoch {epoch:03d} train_mse={total/max(seen,1):.4f} "
            f"dev_mse={dev:.4f} dev_rmse={np.sqrt(max(dev, 0.0)):.3f} "
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
    if pool is not None:
        pool.shutdown(wait=False)
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
    variables = [args.variable] if args.variable else cfg["cnn"].get("variables_to_process", ["temp_mean"])
    for var in variables:
        panel = load_panel(cfg, var)
        panel = prepare_for_tuning(panel)
        panel, _hours = apply_hour_cut(panel, time_res, args.hours)
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
