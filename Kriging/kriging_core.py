#!/usr/bin/env python3
"""
Linear RK / OK with local neighbourhood.

Trend m: intercept + x + y + elev + CLC dummies
         + elev:x + elev:y + elev:CLC  (v1 default).
Residuals: ordinary kriging in an anisotropic (x, y, αz·z) metric.
Precip: indicator RK then amount RK; hard mask at p_wet >= 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from kriging_data import apply_clc_map, is_precip, merge_rare_clc, precip_trace


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass
class KrigingConfig:
    k: int = 16
    k_indicator: int = 16
    family: str = "exponential"          # spherical | exponential | gaussian | matern15 | matern05
    theta_deg: float = 0.0               # horizontal ellipse rotation
    aniso_ratio: float = 1.0             # a_short / a_long in (0, 1]
    alpha_z: float = 0.0                 # metres elev ≈ alpha_z metres horizontal; 0 = ignore z
    interactions: str = "both"           # none | elev_xy | elev_clc | both
    trend: str = "linear"                # none (OK) | linear (RK)
    min_clc_count: int = 15
    min_stations: int = 10
    min_pairs: int = 200
    n_lags: int = 12
    seed: int = 22
    predict_tile: int = 8000


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------
def metric_coords(x, y, elev, theta_deg: float, aniso_ratio: float, alpha_z: float) -> np.ndarray:
    """Map (x, y, elev) into the distance space used by knn + variogram."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(elev, dtype=np.float64)
    th = np.deg2rad(float(theta_deg))
    c, s = np.cos(th), np.sin(th)
    xp = c * x + s * y
    yp = -s * x + c * y
    r = float(aniso_ratio)
    if r <= 0.0:
        r = 1.0
    yp = yp / r
    az = float(alpha_z)
    zp = (az * z) if az > 0.0 else np.zeros_like(z)
    return np.column_stack([xp, yp, zp])


# ---------------------------------------------------------------------------
# covariance families  C(h) = psill * R(h);  C(0) = nugget + psill
# ---------------------------------------------------------------------------
def _corr(h, rng, family: str) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    a = max(float(rng), 1e-6)
    u = h / a
    fam = family.lower()
    if fam == "spherical":
        out = np.zeros_like(u)
        m = u < 1.0
        uu = u[m]
        out[m] = 1.0 - (1.5 * uu - 0.5 * uu ** 3)
        return out
    if fam in ("exponential", "matern05"):
        return np.exp(-u)
    if fam == "gaussian":
        return np.exp(-(u ** 2))
    if fam == "matern15":
        t = np.sqrt(3.0) * u
        return (1.0 + t) * np.exp(-t)
    raise ValueError(f"unknown variogram family {family}")


def cov_from_params(h, nugget, psill, rng, family: str) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    out = np.full(h.shape, float(nugget) + float(psill), dtype=np.float64)
    pos = h > 0.0
    out[pos] = float(psill) * _corr(h[pos], rng, family)
    return out


def gamma_from_params(h, nugget, psill, rng, family: str) -> np.ndarray:
    c0 = float(nugget) + float(psill)
    return c0 - cov_from_params(h, nugget, psill, rng, family)


# ---------------------------------------------------------------------------
# design matrix
# ---------------------------------------------------------------------------
def _standardize_fit(x, y, elev):
    stats = {}
    cols = {"x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(y, dtype=np.float64),
            "elev": np.asarray(elev, dtype=np.float64)}
    out = {}
    for k, v in cols.items():
        mu = float(np.mean(v))
        sd = float(np.std(v))
        if sd < 1e-8:
            sd = 1.0
        stats[k] = (mu, sd)
        out[k] = (v - mu) / sd
    return out, stats


def _standardize_apply(x, y, elev, stats):
    def one(v, key):
        mu, sd = stats[key]
        return (np.asarray(v, dtype=np.float64) - mu) / sd
    return {"x": one(x, "x"), "y": one(y, "y"), "elev": one(elev, "elev")}


def build_design(xn, yn, en, clc, clc_levels, interactions: str, trend: str) -> np.ndarray:
    n = len(xn)
    if trend == "none":
        return np.ones((n, 1), dtype=np.float64)
    blocks = [np.ones(n), xn, yn, en]
    names_extra = []
    # CLC main effects always when trend is linear (Q4 A)
    for g in clc_levels:
        if g == 0:
            continue
        blocks.append((clc == g).astype(np.float64))
        names_extra.append(f"clc_{g}")
    if interactions in ("elev_xy", "both"):
        blocks.append(en * xn)
        blocks.append(en * yn)
    if interactions in ("elev_clc", "both"):
        for g in clc_levels:
            if g == 0:
                continue
            blocks.append(en * (clc == g).astype(np.float64))
    return np.column_stack(blocks)


def fit_ols(X: np.ndarray, z: np.ndarray):
    col_ok = np.std(X, axis=0) > 1e-12
    col_ok[0] = True
    Xr = X[:, col_ok]
    beta_r, *_ = np.linalg.lstsq(Xr, z, rcond=None)
    beta = np.zeros(X.shape[1], dtype=np.float64)
    beta[col_ok] = beta_r
    return beta, col_ok


# ---------------------------------------------------------------------------
# empirical variogram + WLS fit
# ---------------------------------------------------------------------------
def empirical_variogram(coords: np.ndarray, values: np.ndarray, n_lags: int = 12):
    n = len(values)
    if n < 4:
        return None
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    g = 0.5 * (values[:, None] - values[None, :]) ** 2
    iu, ju = np.triu_indices(n, k=1)
    dist = d[iu, ju]
    gam = g[iu, ju]
    finite = np.isfinite(dist) & np.isfinite(gam) & (dist > 0)
    dist, gam = dist[finite], gam[finite]
    if dist.size < 20:
        return None
    dmax = np.quantile(dist, 0.75)
    edges = np.linspace(0.0, dmax, n_lags + 1)
    centres, mean_g, counts = [], [], []
    for i in range(n_lags):
        m = (dist > edges[i]) & (dist <= edges[i + 1])
        if m.sum() < 5:
            continue
        centres.append(0.5 * (edges[i] + edges[i + 1]))
        mean_g.append(float(gam[m].mean()))
        counts.append(int(m.sum()))
    if len(centres) < 3:
        return None
    return {
        "h": np.asarray(centres),
        "gamma": np.asarray(mean_g),
        "n": np.asarray(counts, dtype=np.float64),
        "n_pairs": int(dist.size),
    }


def fit_variogram(emp: dict, family: str) -> dict:
    h, g, w = emp["h"], emp["gamma"], emp["n"]
    w = w / w.max()
    gmax = float(np.maximum(g.max(), 1e-12))
    hmax = float(np.maximum(h.max(), 1.0))

    def pack(nug, ps, rng):
        return np.array([nug, ps, rng], dtype=np.float64)

    def unpack(p):
        nug = float(np.clip(p[0], 1e-12, gmax))
        ps = float(np.clip(p[1], 1e-12, 4.0 * gmax))
        rng = float(np.clip(p[2], 0.05 * hmax, 4.0 * hmax))
        return nug, ps, rng

    def loss(p):
        nug, ps, rng = unpack(p)
        pred = gamma_from_params(h, nug, ps, rng, family)
        return float(np.sum(w * (g - pred) ** 2))

    x0 = pack(0.1 * gmax, 0.9 * gmax, 0.5 * hmax)
    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"maxiter": 400, "xatol": 1e-6, "fatol": 1e-8})
    nug, ps, rng = unpack(res.x)
    return {
        "nugget": nug,
        "psill": ps,
        "range": rng,
        "family": family,
        "wss": float(res.fun),
        "n_pairs": int(emp["n_pairs"]),
    }


def pool_empirical(fields: list[dict], n_lags: int) -> Optional[dict]:
    """Average per-field empirical variograms on a common lag grid."""
    usable = [e for e in fields if e is not None]
    if not usable:
        return None
    h_all = np.concatenate([e["h"] for e in usable])
    dmax = float(np.quantile(h_all, 0.8))
    edges = np.linspace(0.0, dmax, n_lags + 1)
    acc_g = np.zeros(n_lags)
    acc_n = np.zeros(n_lags)
    pairs = 0
    for e in usable:
        pairs += e["n_pairs"]
        for h, g, n in zip(e["h"], e["gamma"], e["n"]):
            b = np.searchsorted(edges, h, side="right") - 1
            if 0 <= b < n_lags:
                acc_g[b] += g * n
                acc_n[b] += n
    ok = acc_n >= 5
    if ok.sum() < 3:
        return None
    centres = 0.5 * (edges[:-1] + edges[1:])
    return {
        "h": centres[ok],
        "gamma": (acc_g[ok] / acc_n[ok]),
        "n": acc_n[ok],
        "n_pairs": int(pairs),
    }


# ---------------------------------------------------------------------------
# local ordinary kriging (batched)
# ---------------------------------------------------------------------------
def ordinary_krige(
    obs_c: np.ndarray,
    obs_v: np.ndarray,
    qry_c: np.ndarray,
    k: int,
    vgm: dict,
) -> np.ndarray:
    nobs = len(obs_v)
    if nobs == 0:
        return np.full(len(qry_c), np.nan)
    kk = int(min(max(k, 2), nobs))
    tree = cKDTree(obs_c)
    nq = len(qry_c)
    out = np.empty(nq, dtype=np.float64)
    tile = 4000
    fam = vgm["family"]
    nug, ps, rng = vgm["nugget"], vgm["psill"], vgm["range"]
    jitter = 1e-8 * (nug + ps + 1.0)
    for s in range(0, nq, tile):
        sl = slice(s, min(s + tile, nq))
        d_qo, idx = tree.query(qry_c[sl], k=kk, workers=-1)
        d_qo = np.atleast_2d(np.asarray(d_qo, dtype=np.float64))
        idx = np.atleast_2d(np.asarray(idx, dtype=np.int64))
        if kk == 1:
            d_qo = d_qo.reshape(-1, 1)
            idx = idx.reshape(-1, 1)
        neigh_c = obs_c[idx]
        neigh_v = obs_v[idx]
        delta = neigh_c[:, :, None, :] - neigh_c[:, None, :, :]
        d_nn = np.sqrt((delta ** 2).sum(-1))
        C = cov_from_params(d_nn, nug, ps, rng, fam)
        C[:, np.arange(kk), np.arange(kk)] = (nug + ps) + jitter
        cvec = cov_from_params(d_qo, nug, ps, rng, fam)
        nt = C.shape[0]
        A = np.zeros((nt, kk + 1, kk + 1), dtype=np.float64)
        A[:, :kk, :kk] = C
        A[:, :kk, kk] = 1.0
        A[:, kk, :kk] = 1.0
        rhs = np.ones((nt, kk + 1, 1), dtype=np.float64)
        rhs[:, :kk, 0] = cvec
        try:
            sol = np.linalg.solve(A, rhs)[..., 0]
        except np.linalg.LinAlgError:
            sol = np.empty((nt, kk + 1), dtype=np.float64)
            for i in range(nt):
                sol[i] = np.linalg.lstsq(A[i], rhs[i, :, 0], rcond=None)[0]
        lam = sol[:, :kk]
        out[sl] = (lam * neigh_v).sum(axis=1)
    return out


# ---------------------------------------------------------------------------
# interpolator
# ---------------------------------------------------------------------------
@dataclass
class KrigingInterpolator:
    cfg: KrigingConfig
    beta: Optional[np.ndarray] = None
    col_ok: Optional[np.ndarray] = None
    beta_ind: Optional[np.ndarray] = None
    col_ok_ind: Optional[np.ndarray] = None
    coord_stats: dict = field(default_factory=dict)
    clc_map: dict = field(default_factory=dict)
    clc_levels: list = field(default_factory=list)
    vgm_default: dict = field(default_factory=dict)
    vgm_by_regime: dict = field(default_factory=dict)
    vgm_ind_default: dict = field(default_factory=dict)
    vgm_ind_by_regime: dict = field(default_factory=dict)
    trace: float = 0.0
    var_name: str = ""
    two_step: bool = False

    def _metric(self, df: pd.DataFrame) -> np.ndarray:
        return metric_coords(
            df["x"].to_numpy(), df["y"].to_numpy(), df["elev"].to_numpy(),
            self.cfg.theta_deg, self.cfg.aniso_ratio, self.cfg.alpha_z,
        )

    def _clc(self, df: pd.DataFrame) -> np.ndarray:
        raw = df["clc_group"].to_numpy() if "clc_group" in df.columns else np.zeros(len(df), dtype=np.int32)
        if self.clc_map:
            return apply_clc_map(raw, self.clc_map)
        return raw

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        st = _standardize_apply(df["x"], df["y"], df["elev"], self.coord_stats)
        clc = self._clc(df)
        return build_design(
            st["x"], st["y"], st["elev"], clc, self.clc_levels,
            self.cfg.interactions, self.cfg.trend,
        )

    def _trend(self, df: pd.DataFrame, beta, col_ok) -> np.ndarray:
        if self.cfg.trend == "none" or beta is None:
            return np.zeros(len(df), dtype=np.float64)
        X = self._X(df)
        b = np.zeros_like(beta)
        b[col_ok] = beta[col_ok]
        return X @ b

    def _pick_vgm(self, cluster_id: int, indicator: bool = False) -> dict:
        table = self.vgm_ind_by_regime if indicator else self.vgm_by_regime
        fallback = self.vgm_ind_default if indicator else self.vgm_default
        v = table.get(int(cluster_id))
        if v is None or v.get("n_pairs", 0) < self.cfg.min_pairs:
            return fallback
        return v

    def fit(self, train: pd.DataFrame, var: str, time_res: str, cfg_yaml: dict) -> dict:
        self.var_name = var
        self.two_step = is_precip(var)
        self.trace = precip_trace(cfg_yaml, time_res, var) if self.two_step else 0.0
        z = train[var].to_numpy(dtype=np.float64)
        st, self.coord_stats = _standardize_fit(train["x"], train["y"], train["elev"])
        groups = train["clc_group"].to_numpy() if "clc_group" in train.columns else np.zeros(len(train), dtype=np.int32)
        remapped, self.clc_map = merge_rare_clc(groups, self.cfg.min_clc_count)
        self.clc_levels = sorted(set(remapped.tolist()))
        tmp = train.copy()
        tmp["clc_group"] = remapped

        if self.cfg.trend == "linear":
            X = build_design(
                st["x"], st["y"], st["elev"], remapped, self.clc_levels,
                self.cfg.interactions, self.cfg.trend,
            )
            self.beta, self.col_ok = fit_ols(X, z)
            resid = z - (X @ self.beta)
        else:
            self.beta = np.array([float(np.mean(z))])
            self.col_ok = np.array([True])
            resid = z - self.beta[0]

        tmp = tmp.assign(_resid=resid)
        self.vgm_default, self.vgm_by_regime = self._fit_vgms(tmp, "_resid")

        if self.two_step:
            wet = (z > self.trace).astype(np.float64)
            if self.cfg.trend == "linear":
                X = build_design(
                    st["x"], st["y"], st["elev"], remapped, self.clc_levels,
                    self.cfg.interactions, self.cfg.trend,
                )
                self.beta_ind, self.col_ok_ind = fit_ols(X, wet)
                resid_i = wet - (X @ self.beta_ind)
            else:
                self.beta_ind = np.array([float(np.mean(wet))])
                self.col_ok_ind = np.array([True])
                resid_i = wet - self.beta_ind[0]
            tmp = tmp.assign(_resid_i=resid_i)
            self.vgm_ind_default, self.vgm_ind_by_regime = self._fit_vgms(tmp, "_resid_i")
            # amount residuals only at wet stations (refit amount trend on wet rows)
            wet_df = tmp[z > self.trace]
            if len(wet_df) >= self.cfg.min_stations:
                Xw = build_design(
                    (wet_df["x"].to_numpy() - self.coord_stats["x"][0]) / self.coord_stats["x"][1],
                    (wet_df["y"].to_numpy() - self.coord_stats["y"][0]) / self.coord_stats["y"][1],
                    (wet_df["elev"].to_numpy() - self.coord_stats["elev"][0]) / self.coord_stats["elev"][1],
                    wet_df["clc_group"].to_numpy(), self.clc_levels,
                    self.cfg.interactions, self.cfg.trend,
                )
                zw = wet_df[var].to_numpy(dtype=np.float64)
                if self.cfg.trend == "linear":
                    self.beta, self.col_ok = fit_ols(Xw, zw)
                    wet_df = wet_df.assign(_resid_amt=zw - (Xw @ self.beta))
                else:
                    self.beta = np.array([float(np.mean(zw))])
                    self.col_ok = np.array([True])
                    wet_df = wet_df.assign(_resid_amt=zw - self.beta[0])
                self.vgm_default, self.vgm_by_regime = self._fit_vgms(wet_df, "_resid_amt")

        return {
            "n_train": int(len(train)),
            "n_clc": int(len(self.clc_levels)),
            "vgm": self.vgm_default,
            "n_regimes": int(len(self.vgm_by_regime)),
            "two_step": self.two_step,
            "trace": self.trace,
        }

    def _fit_vgms(self, df: pd.DataFrame, col: str) -> tuple[dict, dict]:
        by_reg: dict[int, dict] = {}
        per_field_all = []
        per_field_reg: dict[int, list] = {}
        for t, part in df.groupby("time", sort=False):
            if len(part) < 4:
                continue
            coords = self._metric(part)
            emp = empirical_variogram(coords, part[col].to_numpy(dtype=np.float64), self.cfg.n_lags)
            per_field_all.append(emp)
            cid = int(part["cluster_id"].iloc[0]) if "cluster_id" in part.columns else 0
            per_field_reg.setdefault(cid, []).append(emp)
        pooled = pool_empirical(per_field_all, self.cfg.n_lags)
        if pooled is None:
            default = {"nugget": 1e-6, "psill": 1.0, "range": 50000.0,
                       "family": self.cfg.family, "wss": np.nan, "n_pairs": 0}
        else:
            default = fit_variogram(pooled, self.cfg.family)
        for cid, fields in per_field_reg.items():
            p = pool_empirical(fields, self.cfg.n_lags)
            if p is None or p["n_pairs"] < self.cfg.min_pairs:
                continue
            by_reg[int(cid)] = fit_variogram(p, self.cfg.family)
        return default, by_reg

    def predict_frame(self, donors: pd.DataFrame, queries: pd.DataFrame, var: str) -> np.ndarray:
        if len(donors) < 2:
            return np.full(len(queries), np.nan)
        cid = int(donors["cluster_id"].iloc[0]) if "cluster_id" in donors.columns else 0
        q_c = self._metric(queries)
        if self.two_step:
            return self._predict_precip(donors, queries, var, cid, q_c)
        m_d = self._trend(donors, self.beta, self.col_ok)
        m_q = self._trend(queries, self.beta, self.col_ok)
        resid = donors[var].to_numpy(dtype=np.float64) - m_d
        d_c = self._metric(donors)
        ehat = ordinary_krige(d_c, resid, q_c, self.cfg.k, self._pick_vgm(cid, False))
        return m_q + ehat

    def _predict_precip(self, donors, queries, var, cid, q_c) -> np.ndarray:
        wet_ind = (donors[var].to_numpy(dtype=np.float64) > self.trace).astype(np.float64)
        m_i_d = self._trend(donors, self.beta_ind, self.col_ok_ind)
        m_i_q = self._trend(queries, self.beta_ind, self.col_ok_ind)
        resid_i = wet_ind - m_i_d
        d_c = self._metric(donors)
        e_i = ordinary_krige(d_c, resid_i, q_c, self.cfg.k_indicator, self._pick_vgm(cid, True))
        p_wet = np.clip(m_i_q + e_i, 0.0, 1.0)

        wet_mask = donors[var].to_numpy(dtype=np.float64) > self.trace
        amount = np.zeros(len(queries), dtype=np.float64)
        if wet_mask.sum() >= 2:
            wet = donors.loc[wet_mask]
            m_d = self._trend(wet, self.beta, self.col_ok)
            m_q = self._trend(queries, self.beta, self.col_ok)
            resid = wet[var].to_numpy(dtype=np.float64) - m_d
            ehat = ordinary_krige(self._metric(wet), resid, q_c, self.cfg.k, self._pick_vgm(cid, False))
            amount = m_q + ehat
        amount = np.maximum(amount, 0.0)
        out = np.where(p_wet >= 0.5, amount, 0.0)
        return out

    def state_dict(self) -> dict:
        return {
            "cfg": self.cfg.__dict__,
            "beta": self.beta,
            "col_ok": self.col_ok,
            "beta_ind": self.beta_ind,
            "col_ok_ind": self.col_ok_ind,
            "coord_stats": self.coord_stats,
            "clc_map": self.clc_map,
            "clc_levels": self.clc_levels,
            "vgm_default": self.vgm_default,
            "vgm_by_regime": self.vgm_by_regime,
            "vgm_ind_default": self.vgm_ind_default,
            "vgm_ind_by_regime": self.vgm_ind_by_regime,
            "trace": self.trace,
            "var_name": self.var_name,
            "two_step": self.two_step,
        }

    @classmethod
    def from_state(cls, state: dict) -> "KrigingInterpolator":
        obj = cls(KrigingConfig(**state["cfg"]))
        for k, v in state.items():
            if k == "cfg":
                continue
            setattr(obj, k, v)
        return obj
