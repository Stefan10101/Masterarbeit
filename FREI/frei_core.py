#!/usr/bin/env python3
"""
Frei / SPARTACUS interpolator (redesigned).

Background: generalized piecewise-linear elevation profile, fitted
independently per variable and per watershed region. Number of sections
(n_sections in 1..5) is a free hyperparameter. Only cold-pool stations
are dropped from the profile fit. Basin profiles shrink toward a
domain-wide profile and blend on divide buffers.

Residuals: modified IDW (separate horizontal / vertical weights via Fz)
using the valley-axis cost distance as the horizontal metric.
power, Fz and k are taken from the IDW-tuned params for the same
variable and timescale (not searched here).

Precip/snow: two-step. Occurrence is a residual field; amount uses the
piecewise-linear profile in log1p space on wet stations only.
No covariates. No RH temperature coupling.
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
    n_sections: int = 3
    shrink: float = 0.3
    blend_km: float = 15.0
    coldpool_tpi: float = -60.0
    accum_pct: float = 96.0
    k: int = 16
    power: float = 2.0
    Fz: float = 0.3
    across_w: float = 4.0
    select_metric: bool = True
    two_step: bool = True
    tau_wet: float = 0.5
    log_amount: bool = True
    trace: float = 0.1
    min_stations: int = 10
    predict_tile: int = 20000
    seed: int = 22
    across_w_grid: tuple = (2.0, 4.0, 8.0)


def two_step_var(var: str) -> bool:
    v = var.lower()
    return v.startswith("precip") or v.startswith("snow")


def uses_profile(var: str) -> bool:
    return True


def clip_var(var: str, pred: np.ndarray) -> np.ndarray:
    v = var.lower()
    out = np.asarray(pred, dtype=np.float64)
    if v.startswith("precip") or v.startswith("snow") or v.startswith("wind"):
        return np.maximum(out, 0.0)
    if v.startswith("rh"):
        return np.clip(out, 0.0, 100.0)
    return out


def _linear_tz(z, t):
    A = np.column_stack([np.ones_like(z), z])
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    return coef


def _eval_piecewise(z, t_ref, slopes, bps, z_ref):
    """Continuous piecewise-linear profile.

    t_ref: value at z_ref
    slopes: array of length n_sections
    bps: sorted breakpoints, length n_sections-1 (empty if n_sections==1)
    """
    z = np.asarray(z, dtype=np.float64)
    slopes = np.asarray(slopes, dtype=np.float64)
    n = len(slopes)
    if n == 1 or len(bps) == 0:
        return t_ref + slopes[0] * (z - z_ref)

    bps = np.sort(np.asarray(bps, dtype=np.float64))
    edges = np.concatenate([[-1e12], bps, [1e12]])
    # value at each edge by integrating from z_ref
    sec_ref = int(np.clip(np.searchsorted(bps, z_ref), 0, n - 1))
    edge_val = np.zeros(n + 1)
    # left edge of section sec_ref
    edge_val[sec_ref] = t_ref - slopes[sec_ref] * (z_ref - edges[sec_ref])
    # walk right
    for i in range(sec_ref, n):
        edge_val[i + 1] = edge_val[i] + slopes[i] * (edges[i + 1] - edges[i])
    # walk left
    for i in range(sec_ref - 1, -1, -1):
        edge_val[i] = edge_val[i + 1] - slopes[i] * (edges[i + 1] - edges[i])

    out = np.empty_like(z)
    for i in range(n):
        m = (z >= edges[i]) & (z < edges[i + 1])
        if m.any():
            out[m] = edge_val[i] + slopes[i] * (z[m] - edges[i])
    # last point if any
    if not np.isfinite(out).all():
        out[~np.isfinite(out)] = t_ref
    return out


def fit_profile(z, t, n_sections: int = 3):
    z = np.asarray(z, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n = int(np.clip(n_sections, 1, 5))
    if len(t) < max(6, n + 3):
        coef = _linear_tz(z, t)
        return {"n": 1, "t_ref": float(coef[0]), "slopes": np.array([coef[1]]),
                "bps": np.array([]), "z_ref": 0.0}

    z_ref = float(np.median(z))
    coef = _linear_tz(z, t)
    t_ref0 = float(coef[0] + coef[1] * z_ref)
    slopes0 = np.full(n, float(coef[1]))
    if n > 1:
        qs = np.linspace(0.25, 0.75, n - 1)
        bps0 = np.quantile(z, qs)
    else:
        bps0 = np.array([])

    x0 = np.concatenate([[t_ref0], slopes0, bps0])

    def fun(p):
        t_ref = p[0]
        slopes = p[1:1 + n]
        bps = p[1 + n:]
        return _eval_piecewise(z, t_ref, slopes, bps, z_ref) - t

    # bounds
    zmin, zmax = float(z.min()), float(z.max())
    lo = np.concatenate([[-80.0], np.full(n, -0.02), np.full(n - 1, zmin)])
    hi = np.concatenate([[40.0], np.full(n, 0.005), np.full(n - 1, zmax)])
    try:
        res = least_squares(fun, x0, bounds=(lo, hi), loss="soft_l1",
                            f_scale=1.5, max_nfev=300)
        p = res.x
    except Exception:
        p = x0

    return {
        "n": n,
        "t_ref": float(p[0]),
        "slopes": np.asarray(p[1:1 + n], dtype=np.float64),
        "bps": np.sort(np.asarray(p[1 + n:], dtype=np.float64)) if n > 1 else np.array([]),
        "z_ref": z_ref,
    }


def eval_profile(z, params):
    return _eval_piecewise(z, params["t_ref"], params["slopes"],
                           params["bps"], params["z_ref"])


def _mix(params_list, shrink, global_p):
    """Shrink regional parameters toward the global profile (same n)."""
    s = float(np.clip(shrink, 0.0, 1.0))
    out = []
    for p in params_list:
        if p["n"] != global_p["n"]:
            out.append(global_p)
            continue
        mixed = {
            "n": p["n"],
            "t_ref": (1.0 - s) * p["t_ref"] + s * global_p["t_ref"],
            "slopes": (1.0 - s) * p["slopes"] + s * global_p["slopes"],
            "bps": (1.0 - s) * p["bps"] + s * global_p["bps"] if p["n"] > 1 else np.array([]),
            "z_ref": p["z_ref"],
        }
        out.append(mixed)
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

    def _cold(self, x, y):
        if self.pack is None or "coldpool" not in self.pack:
            return np.zeros(len(x), dtype=bool)
        from shared.watersheds import sample_region
        return sample_region(self.pack["coldpool"].astype(int),
                             self.pack["xs"], self.pack["ys"], x, y) > 0

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
        cold = self._cold(x, y)
        n_r = w.shape[1]
        nsec = int(self.cfg.n_sections)
        keep = ~cold
        if keep.sum() < 5:
            keep = np.ones(len(t), dtype=bool)
        global_p = fit_profile(z[keep], t[keep], nsec)
        params = []
        for r in range(n_r):
            m = (reg == r + 1) & (~cold)
            if m.sum() < 5:
                m = (reg == r + 1)
            if m.sum() < 3:
                params.append(global_p)
            else:
                params.append(fit_profile(z[m], t[m], nsec))
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
        cells = cost_between(r0, c0, r1, c1, self.pack["uy"], self.pack["ux"],
                             self.pack["tpi"], self.cfg.across_w)
        return cells * dx

    def _residual_idw(self, x, y, z, resid, xq, yq, zq, across_w, ix=None):
        """Modified IDW: w = (1-Fz)*w_horiz + Fz*w_z, horiz = valley cost."""
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
        horiz = np.empty((len(xq), k), dtype=np.float64)
        dz = np.empty((len(xq), k), dtype=np.float64)
        for j in range(k):
            horiz[:, j] = self._horiz_m(xq, yq, x[ix[:, j]], y[ix[:, j]])
            dz[:, j] = np.abs(zq - z[ix[:, j]])
        self.cfg.across_w = old

        hit = horiz[:, 0] <= 1e-9
        horiz = np.maximum(horiz, 1e-6)
        dz = np.maximum(dz, 1e-6)
        p = float(self.cfg.power)
        Fz = float(np.clip(self.cfg.Fz, 0.0, 1.0))
        w_h = horiz ** (-p)
        w_z = dz ** (-p)
        w = (1.0 - Fz) * w_h + Fz * w_z
        w[hit] = 0.0
        w[hit, 0] = 1.0
        wsum = w.sum(1, keepdims=True)
        wsum = np.where(wsum > 0, wsum, 1.0)
        w /= wsum
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

    def _field(self, x, y, z, t, xq, yq, zq, use_profile: bool = True):
        info = {}
        if use_profile and len(t) >= 5:
            bg_s, params, _ = self._background(x, y, z, t)
            bg_q = self._bg_at(xq, yq, zq, params)
            info["params"] = params
            resid = t - bg_s
        else:
            bg_s = np.full_like(t, np.nanmean(t))
            bg_q = np.full(len(xq), np.nanmean(t))
            resid = t - bg_s
        aw = self._pick_across(x, y, z, resid)
        info["across_w"] = aw
        tile = int(self.cfg.predict_tile)
        out = np.empty(len(xq), dtype=np.float64)
        for i0 in range(0, len(xq), tile):
            sl = slice(i0, i0 + tile)
            out[sl] = bg_q[sl] + self._residual_idw(
                x, y, z, resid, xq[sl], yq[sl], zq[sl], aw
            )
        return out, info

    def predict_timestamp(self, x, y, elev, values, xq, yq, zq,
                          use_profile: bool = True, var: str = "temp_mean",
                          **_ignored):
        """Predict one timestamp. Extra kwargs (old extra_s, t_obs, ...) ignored."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        z = np.asarray(elev, float)
        t = np.asarray(values, float)
        xq = np.asarray(xq, float)
        yq = np.asarray(yq, float)
        zq = np.asarray(zq, float)

        if self.cfg.two_step and two_step_var(var):
            hat, info = self._two_step(x, y, z, t, xq, yq, zq, var)
            return clip_var(var, hat), info

        hat, info = self._field(x, y, z, t, xq, yq, zq, use_profile=use_profile)
        return clip_var(var, hat), info

    def _two_step(self, x, y, z, t, xq, yq, zq, var):
        # Occurrence: residual field on the wet indicator
        wet = (t > self.cfg.trace).astype(float)
        p, info = self._field(x, y, z, wet, xq, yq, zq, use_profile=False)
        p = np.clip(p, 0.0, 1.0)

        wet_m = t > self.cfg.trace
        if wet_m.sum() >= 5:
            raw = np.maximum(t[wet_m], 0.0)
            amt_in = np.log1p(raw) if self.cfg.log_amount else raw
            amt, info2 = self._field(
                x[wet_m], y[wet_m], z[wet_m], amt_in,
                xq, yq, zq, use_profile=True,
            )
            amt = np.maximum(np.expm1(amt), 0.0) if self.cfg.log_amount else np.maximum(amt, 0.0)
            info["amount_across_w"] = info2.get("across_w")
        else:
            amt = np.zeros(xq.size)
        return np.where(p >= self.cfg.tau_wet, amt, 0.0), info
