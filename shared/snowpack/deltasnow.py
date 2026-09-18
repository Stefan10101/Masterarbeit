#!/usr/bin/env python3
"""ΔSNOW (Winkler, Schellander, Gruber 2021; nixmass::swe.delta.snow).

Original Alpine calibration (dyn_rho_max=FALSE). HS in metres, SWE in mm.
Port of R/swe.delta.snow.R in nixmass — layer create / overburden /
dry compaction / scaleH / drenchH. Dynamic rho.max branch is not used.
"""

from __future__ import annotations

import numpy as np

# nixmass defaults, original model (Winkler et al. 2021, 14 Alpine sites)
ALPINE_PARAMS = {
    "rho_max": 401.2588,       # kg m-3
    "rho_null": 81.19417,      # kg m-3
    "c_ov": 0.0005104722,      # -
    "k_ov": 0.37856737,        # -
    "k": 0.02993175,           # m3 kg-1
    "tau": 0.02362476,         # m
    "eta_null": 8523356.0,     # Pa s
    "timestep_h": 24.0,
}

G = 9.81
PREC = 1e-10


def _dry_compact(h, swe, age, rho_max, k, ts, eta_null):
    """Viscous compaction of existing layers. SWE unchanged."""
    n = len(h)
    if n == 0:
        return h, swe, age
    swe_hat = np.cumsum(swe[::-1])[::-1]
    h_out = np.zeros(n, dtype=float)
    age_out = np.zeros(n, dtype=float)
    for i in range(n):
        if h[i] <= PREC:
            continue
        denom = 1.0 + (swe_hat[i] * G * ts) / eta_null * np.exp(-k * swe[i] / h[i])
        h_dd = h[i] / denom
        if swe[i] / max(h_dd, PREC) > rho_max:
            h_dd = swe[i] / rho_max
        h_out[i] = h_dd
        age_out[i] = age[i] + 1.0
    return h_out, swe.copy(), age_out


def _drench(h, swe, age, hobs, rho_max):
    """HS drop larger than tau: densify top-down, runoff if already at rho_max."""
    n = len(h)
    if n == 0:
        return h, swe, age
    h = h.copy()
    swe = swe.copy()
    for i in range(n - 1, -1, -1):
        others = float(np.sum(h) - h[i])
        h_at_max = swe[i] / rho_max if rho_max > 0 else 0.0
        if others + h_at_max - hobs >= PREC:
            h[i] = h_at_max
        else:
            h[i] = h_at_max + abs(others + h_at_max - hobs)
            break
    dens = np.divide(swe, h, out=np.zeros_like(swe), where=h > PREC)
    if np.all(rho_max - dens <= PREC) and np.sum(h) > PREC:
        scale = hobs / np.sum(h)
        runoff = (np.sum(h) - hobs) * rho_max
        if runoff < 0:
            runoff = 0.0
        h = h * scale
        swe = swe * scale
    return h, swe, age


def _scale(h_yest, swe_yest, age_today, hobs_yest, hobs_today, rho_max, k, ts):
    """|deltaH| <= tau: re-compact with layerwise eta so modeled HS = observed."""
    n = len(h_yest)
    if n == 0 or hobs_yest <= PREC:
        return h_yest.copy(), swe_yest.copy(), age_today.copy()
    swe = swe_yest.copy()
    swe_hat = np.cumsum(swe[::-1])[::-1]
    eta_cor = np.zeros(n, dtype=float)
    for i in range(n):
        if h_yest[i] <= PREC:
            continue
        rho_d = swe[i] / h_yest[i]
        x = ts * G * swe_hat[i] * np.exp(-k * rho_d)
        p = h_yest[i] / hobs_yest
        den = h_yest[i] - hobs_today * p
        if abs(den) < PREC:
            eta_cor[i] = 0.0
        else:
            eta_cor[i] = hobs_today * x * p / den
    h_dd = np.zeros(n, dtype=float)
    for i in range(n):
        if h_yest[i] <= PREC or eta_cor[i] <= PREC:
            h_dd[i] = 0.0 if h_yest[i] <= PREC else h_yest[i]
            continue
        h_dd[i] = h_yest[i] / (
            1.0 + (swe_hat[i] * G * ts) / eta_cor[i] * np.exp(-k * swe[i] / h_yest[i])
        )
    idx_max = np.where(np.divide(swe, h_dd, out=np.zeros(n), where=h_dd > PREC) - rho_max > PREC)[0]
    if len(idx_max):
        swe_excess = swe[idx_max] - h_dd[idx_max] * rho_max
        swe[idx_max] = swe[idx_max] - swe_excess
        leftover = float(np.sum(swe_excess))
        others = [i for i in range(n) if i not in set(idx_max)]
        j = len(others) - 1
        while leftover > PREC and j >= 0:
            i = others[j]
            room = h_dd[i] * rho_max - swe[i]
            take = min(max(room, 0.0), leftover)
            swe[i] += take
            leftover -= take
            j -= 1
    return h_dd, swe, age_today.copy()


def deltasnow_hs(hs_m: np.ndarray, params: dict | None = None) -> np.ndarray:
    """Daily HS [m] -> SWE [mm]. Series should start at 0; NA not allowed."""
    p = dict(ALPINE_PARAMS)
    if params:
        p.update(params)
    hs = np.asarray(hs_m, dtype=float)
    n = len(hs)
    swe_out = np.zeros(n, dtype=float)
    if n == 0:
        return swe_out
    hs = np.where(np.isfinite(hs), np.maximum(hs, 0.0), 0.0)

    rho_max = float(p["rho_max"])
    rho_null = float(p["rho_null"])
    c_ov = float(p["c_ov"])
    k_ov = float(p["k_ov"])
    k = float(p["k"])
    tau = float(p["tau"])
    eta_null = float(p["eta_null"])
    ts = float(p["timestep_h"]) * 3600.0

    h = np.zeros(0, dtype=float)
    swe = np.zeros(0, dtype=float)
    age = np.zeros(0, dtype=float)
    h_yest = h.copy()
    swe_yest = swe.copy()
    h_tom = h.copy()
    swe_tom = swe.copy()
    age_tom = age.copy()
    H = 0.0

    for t in range(n):
        h_yest, swe_yest = h.copy(), swe.copy()
        if len(h_tom):
            h, swe, age = h_tom.copy(), swe_tom.copy(), age_tom.copy()
        H = float(np.sum(h)) if len(h) else 0.0
        hobs = float(hs[t])
        prev = float(hs[t - 1]) if t else 0.0

        if hobs <= PREC:
            h = np.zeros(0, dtype=float)
            swe = np.zeros(0, dtype=float)
            age = np.zeros(0, dtype=float)
            h_tom = h.copy()
            swe_tom = swe.copy()
            age_tom = age.copy()
            swe_out[t] = 0.0
            continue

        if prev <= PREC:
            h = np.array([hobs], dtype=float)
            swe = np.array([rho_null * hobs], dtype=float)
            age = np.array([1.0], dtype=float)
            H = hobs
        else:
            delta_h = hobs - H
            if delta_h > tau:
                rho = np.divide(swe, h, out=np.zeros_like(swe), where=h > PREC)
                sigma0 = delta_h * rho_null * G
                den = np.maximum(rho_max - rho, PREC)
                eps = c_ov * sigma0 * np.exp(-k_ov * rho / den)
                eps = np.clip(eps, 0.0, 0.95)
                h = (1.0 - eps) * h
                age = age + 1.0
                H = float(np.sum(h))
                new_h = max(hobs - H, 0.0)
                h = np.append(h, new_h)
                swe = np.append(swe, rho_null * new_h)
                age = np.append(age, 1.0)
            elif delta_h >= -tau:
                h, swe, age = _scale(
                    h_yest, swe_yest, age, prev, hobs, rho_max, k, ts
                )
            else:
                h, swe, age = _drench(h, swe, age, hobs, rho_max)

        swe_out[t] = float(np.sum(swe))
        h_tom, swe_tom, age_tom = _dry_compact(h, swe, age, rho_max, k, ts, eta_null)

    return swe_out


def deltasnow_daily_cm(hs_cm: np.ndarray, gap_days: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Daily HS [cm] -> (SWE mm, bulk density kg m-3).

    Linear-fills gaps of ``gap_days`` or less. Longer gaps split the run
    so ΔSNOW restarts after the break. Density is NA where HS < 1 cm.
    """
    hs_cm = np.asarray(hs_cm, dtype=float)
    n = len(hs_cm)
    swe = np.full(n, np.nan)
    dens = np.full(n, np.nan)
    if n == 0:
        return swe, dens

    hs_m = hs_cm / 100.0
    valid = np.isfinite(hs_m)
    if not valid.any():
        return swe, dens

    filled = hs_m.copy()
    idx = np.arange(n)
    if (~valid).any() and valid.sum() >= 2:
        filled[~valid] = np.interp(
            idx[~valid], idx[valid], hs_m[valid], left=np.nan, right=np.nan
        )
        # drop interpolated points that sit in a hole wider than gap_days
        hole = ~valid
        i = 0
        while i < n:
            if not hole[i]:
                i += 1
                continue
            j = i
            while j < n and hole[j]:
                j += 1
            if (j - i) > gap_days:
                filled[i:j] = np.nan
            i = j

    run = np.isfinite(filled)
    i = 0
    while i < n:
        if not run[i]:
            i += 1
            continue
        j = i
        while j < n and run[j]:
            j += 1
        seg = filled[i:j].copy()
        if seg[0] > PREC:
            seg = np.concatenate([[0.0], seg])
            out = deltasnow_hs(seg)[1:]
        else:
            out = deltasnow_hs(seg)
        swe[i:j] = out
        hs_obs_cm = filled[i:j] * 100.0
        dens[i:j] = np.divide(
            swe[i:j] * 100.0,
            hs_obs_cm,
            out=np.full(j - i, np.nan),
            where=hs_obs_cm >= 1.0,
        )
        i = j

    swe = np.where(np.isfinite(hs_cm) & (hs_cm <= PREC * 100), 0.0, swe)
    dens = np.where(np.isfinite(hs_cm) & (hs_cm < 1.0), np.nan, dens)
    return swe, dens
