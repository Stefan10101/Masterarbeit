#!/usr/bin/env python3
"""
Frei / SPARTACUS interpolator.

Profile: 3-piece lower | join | upper, per DEM watershed.
Summits stay in the profile fit. Cold-pool flags are dropped from it.
Basin profiles shrink toward a domain-wide profile and blend only
on divide buffers.

Residuals: IDW with valley-axis cost distance from shared.watersheds.
Precip/snow: two-step (no temperature profile), p_wet >= tau.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


@dataclass
class FreiConfig:
    n_regions: int = 6
    shrink: float = 0.3
    blend_km: float = 15.0
    summit_tpi: float = 80.0
    coldpool_tpi: float = -60.0
    accum_pct: float = 96.0
    k: int = 16
    power: float = 2.0
    across_w: float = 4.0
    select_metric: bool = True
    lam_z: float = 150.0
    two_step: bool = True
    tau_wet: float = 0.5
    trace: float = 0.1
    min_stations: int = 10
    predict_tile: int = 20000
    seed: int = 22
    across_w_grid: tuple = (2.0, 4.0, 8.0)


def two_step_var(var: str) -> bool:
    v = var.lower()
    return v.startswith("precip") or v.startswith("snow")


def _linear_tz(z, t):
    A = np.column_stack([np.ones_like(z), z])
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    return coef, A @ coef


def _profile(z, t0, z0, g_lo, z1, z2, g_hi):
    z = np.asarray(z, dtype=np.float64)
    lo, hi = (z1, z2) if z1 <= z2 else (z2, z1)
    t_lo = t0 + g_lo * (lo - z0)
    t_hi = t_lo + g_hi * (hi - lo)
    t = np.empty_like(z)
    low, high = z <= lo, z >= hi
    mid = ~low & ~high
    t[low] = t0 + g_lo * (z[low] - z0)
    t[high] = t_hi + g_hi * (z[high] - hi)
    if mid.any() and hi > lo:
        t[mid] = t_lo + (z[mid] - lo) / (hi - lo) * (t_hi - t_lo)
    elif mid.any():
        t[mid] = t0 + g_lo * (z[mid] - z0)
    return t


def fit_profile(z, t):
    z = np.asarray(z, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    coef, _ = _linear_tz(z, t)
    z0 = float(np.median(z))
    t0 = float(coef[0] + coef[1] * z0)
    q1, q2 = np.quantile(z, [0.35, 0.65])
    x0 = np.array([t0, z0, coef[1], q1, q2, coef[1]])
    if len(t) < 6:
        return x0

    def fun(p):
        return _profile(z, *p) - t

    lo = np.array([-80.0, float(z.min()) - 200.0, -0.02, float(z.min()), float(z.min()), -0.015])
    hi = np.array([40.0, float(z.max()) + 200.0, 0.005, float(z.max()), float(z.max()), 0.005])
    try:
        return least_squares(fun, x0, bounds=(lo, hi), loss="soft_l1", f_scale=1.5, max_nfev=250).x
    except Exception:
        return x0


def eval_profile(z, params):
    return _profile(z, *params)


def _mix(params_list, shrink, global_p):
    out = []
    s = float(np.clip(shrink, 0.0, 1.0))
    for p in params_list:
        out.append((1.0 - s) * p + s * global_p)
    return out


class FreiInterpolator:
    def __init__(self, cfg: FreiConfig, pack: dict | None = None):
        self.cfg = cfg
        self.pack = pack

    def _regions(self, x, y):
        if self.pack is None:
            return np.ones(len(x), dtype=int)
        from shared.watersheds import sample_region
        return sample_region(self.pack["regions"], self.pack["xs"], self.pack["ys"], x, y)

    def _flags(self, x, y):
        if self.pack is None:
            n = len(x)
            return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        from shared.watersheds import sample_region
        summit = sample_region(self.pack["summit"].astype(int), self.pack["xs"], self.pack["ys"], x, y) > 0
        cold = sample_region(self.pack["coldpool"].astype(int), self.pack["xs"], self.pack["ys"], x, y) > 0
        return summit, cold

    def _weights(self, x, y):
        n = len(np.asarray(x))
        n_r = int(self.pack["n_regions"]) if self.pack else 1
        if self.pack is None or "region_dist" not in self.pack:
            w = np.zeros((n, n_r))
            if self.pack is None:
                w[:, 0] = 1.0
                return w, np.ones(n, dtype=int)
            reg = self._regions(x, y)
            w[np.arange(n), np.clip(reg - 1, 0, n_r - 1)] = 1.0
            return w, reg
        from shared.watersheds import sample_blend_weights
        dx = float(self.pack.get("dx", 1000.0))
        blend_cells = max(float(self.cfg.blend_km) * 1000.0 / max(dx, 1.0), 1.0)
        w = sample_blend_weights(
            self.pack["region_dist"], self.pack["xs"], self.pack["ys"],
            x, y, blend_cells,
        )
        if w.shape[1] != n_r:
            w2 = np.zeros((n, n_r))
            w2[:, : min(n_r, w.shape[1])] = w[:, : min(n_r, w.shape[1])]
            w = w2
        return w, self._regions(x, y)

    def _background(self, x, y, z, t):
        w, reg = self._weights(x, y)
        summit, cold = self._flags(x, y)
        n_r = w.shape[1]
        keep = summit | ~cold
        if keep.sum() < 5:
            keep = np.ones(len(t), dtype=bool)
        global_p = fit_profile(z[keep], t[keep])
        params = []
        for r in range(n_r):
            m = (reg == r + 1) & (~cold | summit)
            if m.sum() < 5:
                m = summit | (reg == r + 1)
            if m.sum() < 3:
                params.append(global_p.copy())
            else:
                params.append(fit_profile(z[m], t[m]))
        params = _mix(params, self.cfg.shrink, global_p)
        bg = np.zeros_like(t, dtype=np.float64)
        for r, p in enumerate(params):
            bg += w[:, r] * eval_profile(z, p)
        return bg, params, global_p

    def _bg_at(self, xq, yq, zq, params):
        w, _ = self._weights(xq, yq)
        out = np.zeros(len(xq), dtype=np.float64)
        for r, p in enumerate(params):
            out += w[:, r] * eval_profile(zq, p)
        return out

    def _horiz_m(self, x0, y0, x1, y1):
        if self.pack is None:
            return np.hypot(x1 - x0, y1 - y0)
        from shared.watersheds import cost_between
        xs, ys = self.pack["xs"], self.pack["ys"]
        dx = float(self.pack.get("dx", abs(xs[1] - xs[0]) if len(xs) > 1 else 1000.0))
        dy = float(ys[1] - ys[0]) if len(ys) > 1 else dx
        c0 = np.clip(((x0 - xs[0]) / dx), 0, len(xs) - 1)
        r0 = np.clip(((y0 - ys[0]) / dy), 0, len(ys) - 1)
        c1 = np.clip(((x1 - xs[0]) / dx), 0, len(xs) - 1)
        r1 = np.clip(((y1 - ys[0]) / dy), 0, len(ys) - 1)
        cells = cost_between(r0, c0, r1, c1, self.pack["uy"], self.pack["ux"], self.pack["tpi"], self.cfg.across_w)
        return cells * dx

    def _residual_idw(self, x, y, z, resid, xq, yq, zq, across_w, ix=None):
        old = self.cfg.across_w
        self.cfg.across_w = across_w
        k = min(int(self.cfg.k), len(resid))
        if ix is None:
            tree = cKDTree(np.column_stack([x, y]))
            _, ix = tree.query(np.column_stack([xq, yq]), k=k)
        if k == 1 or np.ndim(ix) == 1:
            ix = np.asarray(ix)[:, None]
        else:
            ix = np.asarray(ix)
        k = ix.shape[1]
        d = np.empty((len(xq), k), dtype=np.float64)
        for j in range(k):
            horiz = self._horiz_m(xq, yq, x[ix[:, j]], y[ix[:, j]])
            dz = zq - z[ix[:, j]]
            d[:, j] = np.sqrt(horiz * horiz + (self.cfg.lam_z * dz) ** 2)
        self.cfg.across_w = old
        hit = d[:, 0] <= 1e-9
        d = np.maximum(d, 1e-6)
        w = d ** (-float(self.cfg.power))
        w[hit] = 0.0
        w[hit, 0] = 1.0
        w /= w.sum(1, keepdims=True)
        return (w * resid[ix]).sum(1)

    def _pick_across(self, x, y, z, resid):
        if not self.cfg.select_metric or len(resid) < 8:
            return self.cfg.across_w
        k = min(int(self.cfg.k), len(resid) - 1)
        tree = cKDTree(np.column_stack([x, y]))
        _, ix = tree.query(np.column_stack([x, y]), k=k + 1)
        ix = np.atleast_2d(ix)[:, 1:]
        best = (np.inf, self.cfg.across_w)
        for aw in self.cfg.across_w_grid:
            hat = self._residual_idw(x, y, z, resid, x, y, z, aw, ix=ix)
            mse = float(np.mean((hat - resid) ** 2))
            if mse < best[0]:
                best = (mse, float(aw))
        return best[1]

    def _field(self, x, y, z, t, xq, yq, zq, use_profile: bool):
        if use_profile:
            bg_s, params, _ = self._background(x, y, z, t)
            resid = t - bg_s
            bg_q = self._bg_at(xq, yq, zq, params)
        else:
            resid = t - float(np.mean(t))
            bg_q = np.full(xq.size, float(np.mean(t)))
            params = None
        aw = self._pick_across(x, y, z, resid)
        n = xq.size
        out = np.empty(n)
        tile = self.cfg.predict_tile
        for i0 in range(0, n, tile):
            sl = slice(i0, i0 + tile)
            out[sl] = bg_q[sl] + self._residual_idw(x, y, z, resid, xq[sl], yq[sl], zq[sl], aw)
        return out, {"across_w": aw, "params": params}

    def predict_timestamp(self, x, y, elev, values, xq, yq, zq, use_profile: bool = True,
                          var: str = "temp_mean"):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        z = np.asarray(elev, float)
        t = np.asarray(values, float)
        xq = np.asarray(xq, float)
        yq = np.asarray(yq, float)
        zq = np.asarray(zq, float)
        if self.cfg.two_step and two_step_var(var):
            wet = (t > self.cfg.trace).astype(float)
            p, info = self._field(x, y, z, wet, xq, yq, zq, use_profile=False)
            p = np.clip(p, 0.0, 1.0)
            wet_m = t > self.cfg.trace
            if wet_m.sum() >= 5:
                amt, info2 = self._field(
                    x[wet_m], y[wet_m], z[wet_m], t[wet_m], xq, yq, zq, use_profile=False,
                )
                amt = np.maximum(amt, 0.0)
                info = {**info, "amount_across_w": info2.get("across_w")}
            else:
                amt = np.zeros(xq.size)
            return np.where(p >= self.cfg.tau_wet, amt, 0.0), info
        return self._field(x, y, z, t, xq, yq, zq, use_profile)
