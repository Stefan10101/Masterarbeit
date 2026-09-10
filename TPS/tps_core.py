#!/usr/bin/env python3
"""
3D / partial thin-plate spline + E-OBS protocol.

protocol=tps: interpolate each timestamp (two-step for precip/snow).
protocol=eobs: monthly TPS background + anomaly kriging
  (T difference; precip/snow ratio on wet days). On monthly time_res
  eobs is identical to tps.
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
    two_step: bool = True
    trace: float = 0.1
    min_stations: int = 10
    predict_tile: int = 20000
    seed: int = 22
    rh_t_mode: str = "none"
    wind_watershed: bool = True
    snow_terrain: bool = True


def two_step_var(var: str) -> bool:
    v = var.lower()
    return v.startswith("precip") or v.startswith("snow")


def clip_var(var: str, pred: np.ndarray) -> np.ndarray:
    v = str(var).lower()
    out = np.asarray(pred, dtype=np.float64)
    if v.startswith("precip") or v.startswith("snow") or v.startswith("wind"):
        return np.maximum(out, 0.0)
    if v.startswith("rh"):
        return np.clip(out, 0.0, 100.0)
    return out


def estimate_region_alpha(x, y, elev, regions, default_az: float) -> dict:
    """Per-basin αz from median |Δxy|/|Δz| of local station pairs."""
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
        if k == 1:
            d = np.asarray(d, float)[:, None]
            ix = np.asarray(ix)[:, None]
        else:
            d = np.asarray(d, float)
            ix = np.asarray(ix)
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
        print(f"  watershed az={ {k: round(v, 1) for k, v in sorted(model.cfg.region_alpha.items())} } n={counts}")


def _scale_z(elev, regions, cfg: TPSConfig) -> np.ndarray:
    z = np.asarray(elev, dtype=np.float64)
    if cfg.alpha_z_mode != "watershed" or regions is None:
        return z * float(cfg.alpha_z)
    regs = np.asarray(regions, dtype=int)
    n = int(max(int(regs.max()) + 1, 1))
    lookup = np.full(n, float(cfg.alpha_z), dtype=np.float64)
    for r, v in cfg.region_alpha.items():
        ri = int(r)
        if 0 <= ri < n:
            lookup[ri] = float(v)
    return z * lookup[np.clip(regs, 0, n - 1)]


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


def fit_tps(coords, values, lam, nugget, extra_poly=None, kernel: str = "3d"):
    n = coords.shape[0]
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
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


def _idw(coords, values, query, k: int) -> np.ndarray:
    values = np.asarray(values, float)
    k = min(max(int(k), 1), len(values))
    tree = cKDTree(coords)
    d, ix = tree.query(query, k=k)
    if k == 1:
        d = np.asarray(d, float)[:, None]
        ix = np.asarray(ix)[:, None]
    else:
        d = np.asarray(d, float)
        ix = np.asarray(ix)
    hit = d[:, 0] <= 1e-12
    d = np.maximum(d, 1e-6)
    w = d ** (-2.0)
    w[hit] = 0.0
    w[hit, 0] = 1.0
    w /= w.sum(axis=1, keepdims=True)
    return (w * values[ix]).sum(axis=1)


def _kriging():
    try:
        from kriging_core import KrigingConfig, KrigingInterpolator
        return KrigingConfig, KrigingInterpolator
    except Exception:
        return None, None


class TPSInterpolator:
    def __init__(self, cfg: TPSConfig, regions_fn=None):
        self.cfg = cfg
        self.regions_fn = regions_fn

    def _regs(self, x, y):
        if self.regions_fn is None or self.cfg.alpha_z_mode != "watershed":
            return None
        return self.regions_fn(x, y)

    def _poly(self, x, y, elev, coords, extra=None):
        if self.cfg.kernel == "partial":
            P = np.column_stack([np.ones(len(x)), coords, np.asarray(elev, float)])
        else:
            P = np.column_stack([np.ones(len(x)), coords])
        if extra is not None:
            P = np.column_stack([P, np.asarray(extra, float)])
        return P

    def predict_field(self, x, y, elev, values, xq, yq, zq, extra=None, extra_q=None) -> np.ndarray:
        rs = self._regs(x, y)
        rq = self._regs(xq, yq)
        co = _coords(x, y, elev, self.cfg, rs)
        cq = _coords(xq, yq, zq, self.cfg, rq)
        P = self._poly(x, y, elev, co, extra)
        Pq = self._poly(xq, yq, zq, cq, extra_q)
        w, c, _ = fit_tps(
            co, np.asarray(values, float), self.cfg.lam, self.cfg.nugget, P, kernel=self.cfg.kernel,
        )
        return eval_tps(co, w, c, cq, self.cfg.kernel, self.cfg.predict_tile, Pq)

    def predict_timestamp(
        self, stn, val, elev, xq, yq, zq, var: str = "temp_mean",
        extra=None, extra_q=None, region=None, region_q=None,
    ) -> np.ndarray:
        val = np.asarray(val, float)
        if (
            self.cfg.wind_watershed
            and str(var).lower().startswith("wind")
            and region is not None
            and region_q is not None
        ):
            return clip_var(var, self._predict_by_region(
                stn, val, elev, xq, yq, zq, var, extra, extra_q, region, region_q,
            ))
        return clip_var(var, self._predict_one(stn, val, elev, xq, yq, zq, var, extra, extra_q))

    def _predict_by_region(self, stn, val, elev, xq, yq, zq, var, extra, extra_q, region, region_q):
        region = np.asarray(region)
        region_q = np.asarray(region_q)
        out = np.empty(np.asarray(xq).size, dtype=np.float64)
        elev = np.asarray(elev)
        for r in np.unique(region_q):
            qmask = region_q == r
            dmask = region == r
            if int(dmask.sum()) < max(self.cfg.min_stations, 5):
                dmask = np.ones(len(stn), dtype=bool)
            ex = None if extra is None else np.asarray(extra)[dmask]
            exq = None if extra_q is None else np.asarray(extra_q)[qmask]
            out[qmask] = self._predict_one(
                stn[dmask], val[dmask], elev[dmask],
                np.asarray(xq)[qmask], np.asarray(yq)[qmask], np.asarray(zq)[qmask],
                var, ex, exq,
            )
        return out

    def _predict_one(self, stn, val, elev, xq, yq, zq, var, extra, extra_q):
        if self.cfg.two_step and two_step_var(var):
            wet = (val > self.cfg.trace).astype(float)
            p = np.clip(self.predict_field(stn[:, 0], stn[:, 1], elev, wet, xq, yq, zq, extra, extra_q), 0.0, 1.0)
            wet_m = val > self.cfg.trace
            if wet_m.sum() >= 5:
                exw = None if extra is None else np.asarray(extra)[wet_m]
                amt = np.maximum(
                    self.predict_field(
                        stn[wet_m, 0], stn[wet_m, 1], np.asarray(elev)[wet_m],
                        val[wet_m], xq, yq, zq, exw, extra_q,
                    ),
                    0.0,
                )
            else:
                amt = np.zeros(np.asarray(xq).size)
            return np.where(p >= self.cfg.tau_wet, amt, 0.0)
        return self.predict_field(stn[:, 0], stn[:, 1], elev, val, xq, yq, zq, extra, extra_q)

    def _anom_krige(self, donors: dict, queries: dict, values, var: str):
        KrigingConfig, KrigingInterpolator = _kriging()
        if KrigingInterpolator is None:
            return None
        import pandas as pd
        values = np.asarray(values, float)
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
            model.fit(don, var, "daily", {"kriging": {}})
            model.two_step = False
            model.trace = 0.0
            return model.predict_frame(don, q, var)
        except Exception:
            return None

    def predict_eobs_timestamp(
        self, stn, val, elev, month_stn, month_val, month_elev,
        xq, yq, zq, var: str = "temp_mean", clc_s=None, clc_q=None, time=None,
    ) -> np.ndarray:
        is_zero = two_step_var(var)
        bg = self.predict_timestamp(month_stn, month_val, month_elev, xq, yq, zq, var=var)
        bg_s = self.predict_timestamp(
            month_stn, month_val, month_elev,
            stn[:, 0], stn[:, 1], elev, var=var,
        )
        val = np.asarray(val, float)
        if is_zero:
            ratio = np.zeros_like(val)
            good = bg_s > 1e-6
            ratio[good] = val[good] / bg_s[good]
            anom = ratio
        else:
            anom = val - bg_s

        donors = {
            "x": stn[:, 0], "y": stn[:, 1], "elev": elev,
            "clc": np.zeros(len(val), dtype=int) if clc_s is None else clc_s,
            "time": np.datetime64("2000-01-01") if time is None else time,
        }
        queries = {
            "x": xq, "y": yq, "elev": zq,
            "clc": np.zeros(len(xq), dtype=int) if clc_q is None else clc_q,
        }
        hat = self._anom_krige(donors, queries, anom, "anom")
        if hat is None:
            co = _coords(stn[:, 0], stn[:, 1], elev, self.cfg, self._regs(stn[:, 0], stn[:, 1]))
            cq = _coords(xq, yq, zq, self.cfg, self._regs(xq, yq))
            hat = _idw(co, anom, cq, self.cfg.k)
        if is_zero:
            return clip_var(var, np.maximum(bg, 0.0) * np.maximum(hat, 0.0))
        return clip_var(var, bg + hat)
