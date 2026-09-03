#!/usr/bin/env python3
"""
3D / partial thin-plate spline + E-OBS protocol.

protocol=eobs: monthly TPS background, anomaly step = KrigingInterpolator
(T difference; precip/snow indicator + amount).
kernel=3d: φ(r)=r on (x, y, αz z)
kernel=partial: φ(r)=r² log r on (x, y) + linear elev in the null space
alpha_z_mode=watershed: z is scaled by a per-basin αz (DEV phase 2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "Kriging"))


@dataclass
class TPSConfig:
    lam: float = 1.0
    alpha_z: float = 100.0
    alpha_z_mode: str = "global"     # global | watershed
    n_regions: int = 6
    region_alpha: dict = field(default_factory=dict)
    kernel: str = "3d"               # 3d | partial
    nugget: float = 1e-6
    protocol: str = "eobs"
    k: int = 32
    family: str = "exponential"
    k_indicator: int = 16
    tau_wet: float = 0.5
    min_stations: int = 10
    predict_tile: int = 20000
    seed: int = 22


def estimate_region_alpha(x, y, elev, regions, default_az: float) -> dict:
    """Per-basin αz from median |Δxy|/|Δz| of local station pairs."""
    from scipy.spatial import cKDTree

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(elev, float)
    regions = np.asarray(regions, int)
    out = {}
    for r in np.unique(regions):
        m = regions == r
        if m.sum() < 6:
            out[int(r)] = float(default_az)
            continue
        xy = np.column_stack([x[m], y[m]])
        zz = z[m]
        k = min(6, int(m.sum()))
        d, ix = cKDTree(xy).query(xy, k=k)
        d = np.atleast_2d(d)
        ix = np.atleast_2d(ix)
        ratios = []
        for i in range(len(zz)):
            for jpos in range(1, k):
                j = int(ix[i, jpos])
                dz = abs(zz[i] - zz[j])
                if dz > 25.0:
                    ratios.append(float(d[i, jpos]) / dz)
        out[int(r)] = float(np.median(ratios)) if ratios else float(default_az)
    return out


def attach_watershed(model: "TPSInterpolator", pack: dict | None, x, y, elev) -> None:
    """Bind DEM regions and fill region_alpha if missing."""
    if pack is None or model.cfg.alpha_z_mode != "watershed":
        model.regions_fn = None
        return
    from shared.watersheds import sample_region

    def _fn(xq, yq):
        return sample_region(pack["regions"], pack["xs"], pack["ys"], xq, yq)

    model.regions_fn = _fn
    if not model.cfg.region_alpha:
        regs = _fn(x, y)
        model.cfg.region_alpha = estimate_region_alpha(x, y, elev, regs, model.cfg.alpha_z)
        counts = {int(r): int((regs == r).sum()) for r in np.unique(regs)}
        print(f"  watershed αz={ {k: round(v, 1) for k, v in sorted(model.cfg.region_alpha.items())} } n={counts}")


def _scale_z(elev, regions, cfg: TPSConfig) -> np.ndarray:
    z = np.asarray(elev, dtype=np.float64)
    if cfg.alpha_z_mode != "watershed" or regions is None:
        return z * float(cfg.alpha_z)
    out = np.empty_like(z)
    for i, r in enumerate(np.asarray(regions, dtype=int)):
        out[i] = z[i] * float(cfg.region_alpha.get(int(r), cfg.alpha_z))
    return out


def _coords(x, y, elev, cfg: TPSConfig, regions=None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = _scale_z(elev, regions, cfg)
    if cfg.kernel == "partial":
        return np.column_stack([x, y])
    return np.column_stack([x, y, z])


def _phi(d: np.ndarray, kernel: str) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    if kernel == "partial":
        d = np.maximum(d, 1e-12)
        return d * d * np.log(d)
    return d


def fit_tps(coords, values, lam, nugget, extra_poly=None):
    n = coords.shape[0]
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    kernel = "partial" if extra_poly is not None else "3d"
    A = _phi(d, kernel)
    np.fill_diagonal(A, float(lam) + float(nugget))
    if extra_poly is None:
        P = np.column_stack([np.ones(n), coords])
    else:
        P = np.asarray(extra_poly, dtype=np.float64)
    m = P.shape[1]
    K = np.zeros((n + m, n + m))
    K[:n, :n] = A
    K[:n, n:] = P
    K[n:, :n] = P.T
    rhs = np.zeros(n + m)
    rhs[:n] = values
    try:
        sol = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
    return sol[:n], sol[n:], P.shape[1]


def eval_tps(coords_obs, w, c, coords_pred, kernel, tile, extra_pred=None):
    n = coords_pred.shape[0]
    out = np.empty(n)
    for i0 in range(0, n, tile):
        sl = coords_pred[i0:i0 + tile]
        d = np.sqrt(((sl[:, None, :] - coords_obs[None, :, :]) ** 2).sum(-1))
        if extra_pred is None:
            trend = c[0] + sl @ c[1:]
        else:
            trend = extra_pred[i0:i0 + tile] @ c
        out[i0:i0 + tile] = _phi(d, kernel) @ w + trend
    return out


def _kriging():
    try:
        from kriging_core import KrigingConfig, KrigingInterpolator
        return KrigingConfig, KrigingInterpolator
    except Exception:
        return None, None


class TPSInterpolator:
    def __init__(self, cfg: TPSConfig, regions_fn=None):
        self.cfg = cfg
        self.regions_fn = regions_fn  # callable(x,y) -> region ids

    def _regs(self, x, y):
        if self.regions_fn is None or self.cfg.alpha_z_mode != "watershed":
            return None
        return self.regions_fn(x, y)

    def _poly(self, x, y, elev, coords):
        if self.cfg.kernel != "partial":
            return None
        return np.column_stack([np.ones(len(x)), coords, np.asarray(elev, float)])

    def predict_field(self, x, y, elev, values, xq, yq, zq) -> np.ndarray:
        rs = self._regs(x, y)
        rq = self._regs(xq, yq)
        co = _coords(x, y, elev, self.cfg, rs)
        cq = _coords(xq, yq, zq, self.cfg, rq)
        extra = self._poly(x, y, elev, co)
        w, c, _ = fit_tps(co, np.asarray(values, float), self.cfg.lam, self.cfg.nugget, extra)
        extra_q = None if extra is None else self._poly(xq, yq, zq, cq)
        return eval_tps(co, w, c, cq, self.cfg.kernel, self.cfg.predict_tile, extra_q)

    def predict_timestamp(self, stn, val, elev, xq, yq, zq) -> np.ndarray:
        return self.predict_field(stn[:, 0], stn[:, 1], elev, val, xq, yq, zq)

    def _anom_krige(self, donors: dict, queries: dict, values, var: str, is_zero_inf: bool):
        KrigingConfig, KrigingInterpolator = _kriging()
        if KrigingInterpolator is None:
            return None
        import pandas as pd
        don = pd.DataFrame({
            "x": donors["x"], "y": donors["y"], "elev": donors["elev"],
            var: values, "time": donors["time"],
            "clc_group": donors.get("clc", np.zeros(len(values), dtype=int)),
            "cluster_id": 0, "station_name": np.arange(len(values)),
        })
        q = pd.DataFrame({
            "x": queries["x"], "y": queries["y"], "elev": queries["elev"],
            "clc_group": queries.get("clc", np.zeros(len(queries["x"]), dtype=int)),
            "cluster_id": 0,
        })
        kcfg = KrigingConfig(
            k=self.cfg.k, k_indicator=self.cfg.k_indicator,
            family=self.cfg.family, trend="none",
            alpha_z=float(self.cfg.alpha_z),
            min_stations=self.cfg.min_stations, seed=self.cfg.seed,
        )
        model = KrigingInterpolator(kcfg)
        try:
            model.fit(don, var, "daily", {"kriging": {"trace_mm": {"daily": 0.1}}})
            # force two-step from the variable name
            model.two_step = is_zero_inf
            if is_zero_inf:
                model.trace = 0.0  # anomalies already relative; presence handled outside
            return model.predict_frame(don, q, var)
        except Exception:
            return None

    def predict_eobs_timestamp(
        self, stn, val, elev, month_stn, month_val, month_elev,
        xq, yq, zq, is_precip: bool, clc_s=None, clc_q=None, time=None,
    ) -> np.ndarray:
        bg = self.predict_field(month_stn[:, 0], month_stn[:, 1], month_elev, month_val, xq, yq, zq)
        bg_s = self.predict_field(
            month_stn[:, 0], month_stn[:, 1], month_elev, month_val,
            stn[:, 0], stn[:, 1], elev,
        )
        if is_precip:
            wet = (np.asarray(val, float) > 0).astype(float)
            amount = np.asarray(val, float)
            # indicator + amount on the raw daily field, then scale by monthly TPS
            # Haylock: anomaly vs monthly. Indicator on wet-day, amount = daily/monthly.
            ratio = amount / np.maximum(bg_s, 1e-6)
            anom = ratio
        else:
            anom = np.asarray(val, float) - bg_s

        donors = {
            "x": stn[:, 0], "y": stn[:, 1], "elev": elev,
            "clc": np.zeros(len(val), dtype=int) if clc_s is None else clc_s,
            "time": np.datetime64("2000-01-01") if time is None else time,
        }
        queries = {
            "x": xq, "y": yq, "elev": zq,
            "clc": np.zeros(len(xq), dtype=int) if clc_q is None else clc_q,
        }
        hat = self._anom_krige(donors, queries, anom, "anom", is_precip)
        if hat is None:
            coords = _coords(stn[:, 0], stn[:, 1], elev, self.cfg, self._regs(stn[:, 0], stn[:, 1]))
            cq = _coords(xq, yq, zq, self.cfg, self._regs(xq, yq))
            tree = cKDTree(coords)
            k = min(self.cfg.k, len(anom))
            d, ix = tree.query(cq, k=k)
            d = np.maximum(np.atleast_2d(d), 1e-6)
            w = d ** (-2.0)
            w /= w.sum(1, keepdims=True)
            hat = (w * anom[np.atleast_2d(ix)]).sum(1)
        if is_precip:
            return np.maximum(bg, 0.0) * np.maximum(hat, 0.0)
        return bg + hat
