#!/usr/bin/env python3
"""
Per-timestamp GAM (mode A).

Preferred solver: mgcv via fit_gam.R / predict_gam.R (REML).
Fallback: numpy B-splines if R/mgcv is missing.

Formulas searched on DEV:
  te_xy_s_elev   te(x,y)+s(elev)+terrain+CLC
  te_xy_ti_elev  + ti(x,elev)+ti(y,elev)
  te_xyelev      te(x,y,elev)+terrain+CLC
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline


@dataclass
class GAMConfig:
    formula: str = "te_xy_s_elev"
    n_splines: int = 10
    spline_order: int = 3
    use_clc: bool = True
    use_terrain: bool = True
    min_clc_count: int = 15
    two_step: bool = True
    tau_wet: float = 0.5
    rscript: str = "Rscript"
    fit_r: str = "fit_gam.R"
    min_stations: int = 10
    predict_tile: int = 50000
    seed: int = 22
    lam: float = 1.0  # numpy fallback only
    trace: float = 0.1
    r_timeout: float = 600.0


def two_step_var(var: str) -> bool:
    v = var.lower()
    return v.startswith("precip") or v.startswith("snow")


def _knots(x, n_basis, order):
    x = np.asarray(x, dtype=np.float64)
    ux = np.unique(x[np.isfinite(x)])
    if ux.size < 3:
        lo, hi = (float(ux[0]), float(ux[0]) + 1.0) if ux.size else (0.0, 1.0)
        return np.concatenate([np.full(order, lo), np.array([lo, hi]), np.full(order, hi)])
    n_basis = max(order + 1, min(int(n_basis), ux.size - 1))
    n_inner = max(2, n_basis - order + 1)
    qs = np.linspace(0.0, 1.0, n_inner)
    inner = np.quantile(x[np.isfinite(x)], qs)
    inner = np.unique(inner)
    if inner.size < 2:
        inner = np.array([float(ux[0]), float(ux[-1])])
    lo, hi = float(ux[0]), float(ux[-1])
    if hi <= lo:
        hi = lo + 1.0
    return np.concatenate([np.full(order, lo), inner, np.full(order, hi)])


def _bs(x, knots, order, n_basis):
    x = np.asarray(x, dtype=np.float64)
    B = np.zeros((x.size, n_basis), dtype=np.float64)
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        B[:, j] = BSpline(knots, c, order, extrapolate=True)(x)
    return B


def _penalty(n):
    if n < 3:
        return np.eye(n)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i], D[i, i + 1], D[i, i + 2] = 1.0, -2.0, 1.0
    return D.T @ D


class GAMInterpolator:
    def __init__(self, cfg: GAMConfig):
        self.cfg = cfg
        self._use_r = shutil.which(cfg.rscript) is not None
        self._r_dir = Path(__file__).resolve().parent
        self._knots = {}
        self._coef = None
        self._clc_levels = []
        self._r_model = None
        self._r_tmp = None
        self._backend = None

    def close(self):
        tmp = getattr(self, "_r_tmp", None)
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        self._r_tmp = None
        self._r_model = None

    def _frame(self, x, y, elev, values=None, clc=None, slope=None, sinasp=None, cosasp=None):
        df = pd.DataFrame({
            "x": np.asarray(x, dtype=np.float64),
            "ycoord": np.asarray(y, dtype=np.float64),
            "elev": np.asarray(elev, dtype=np.float64),
        })
        if values is not None:
            df["y"] = np.asarray(values, dtype=np.float64)
        if self.cfg.use_terrain:
            for name, arr, fill in (
                ("slope", slope, 0.0),
                ("sinasp", sinasp, 0.0),
                ("cosasp", cosasp, 1.0),
            ):
                if arr is None:
                    df[name] = fill
                    continue
                v = np.asarray(arr, dtype=np.float64)
                v = np.where(np.isfinite(v), v, fill)
                df[name] = v
        if self.cfg.use_clc and clc is not None:
            df["clc"] = np.asarray(clc, dtype=np.int32)
        return df

    def _fit_r(self, df: pd.DataFrame, family: str) -> bool:
        script = self._r_dir / self.cfg.fit_r
        if not script.exists() or not self._use_r:
            return False
        self.close()
        tmp = Path(tempfile.mkdtemp(prefix="gam_"))
        train = tmp / "train.parquet"
        df.to_parquet(train, index=False)
        out = tmp / "fit"
        cmd = [
            self.cfg.rscript, str(script), str(train), str(out),
            self.cfg.formula, family, str(int(self.cfg.n_splines)),
        ]
        try:
            subprocess.run(
                cmd, check=True, capture_output=True, text=True,
                timeout=float(self.cfg.r_timeout),
            )
        except Exception as exc:
            err = ""
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                err = exc.stderr[-500:]
            print(f"  mgcv failed, numpy fallback: {type(exc).__name__} {err}", flush=True)
            shutil.rmtree(tmp, ignore_errors=True)
            return False
        model = out / "model.rds"
        if not model.exists():
            shutil.rmtree(tmp, ignore_errors=True)
            return False
        self._r_model = model
        self._r_tmp = tmp
        return True

    def _predict_r(self, df: pd.DataFrame) -> np.ndarray | None:
        if self._r_model is None or self._r_tmp is None:
            return None
        pred_script = self._r_dir / "predict_gam.R"
        q = self._r_tmp / "query.parquet"
        o = self._r_tmp / "pred.parquet"
        df.to_parquet(q, index=False)
        try:
            subprocess.run(
                [self.cfg.rscript, str(pred_script), str(self._r_model), str(q), str(o)],
                check=True, capture_output=True, text=True,
                timeout=float(self.cfg.r_timeout),
            )
            return pd.read_parquet(o)["predicted"].to_numpy(dtype=np.float64)
        except Exception as exc:
            err = ""
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                err = exc.stderr[-400:]
            print(f"  mgcv predict failed: {type(exc).__name__} {err}", flush=True)
            return None

    def _fit_numpy(self, x, y, elev, values, clc, slope, sinasp, cosasp):
        n_s, k = self.cfg.n_splines, self.cfg.spline_order
        self._knots = {
            "x": _knots(x, n_s, k), "y": _knots(y, n_s, k), "z": _knots(elev, n_s, k),
        }
        if self.cfg.use_terrain:
            sl = np.zeros_like(x) if slope is None else np.asarray(slope, dtype=np.float64)
            sa = np.zeros_like(x) if sinasp is None else np.asarray(sinasp, dtype=np.float64)
            ca = np.ones_like(x) if cosasp is None else np.asarray(cosasp, dtype=np.float64)
            sl = np.where(np.isfinite(sl), sl, 0.0)
            sa = np.where(np.isfinite(sa), sa, 0.0)
            ca = np.where(np.isfinite(ca), ca, 1.0)
            self._knots["sl"] = _knots(sl, max(n_s // 2, 4), k)
            self._knots["sa"] = _knots(sa, max(n_s // 2, 4), k)
            self._knots["ca"] = _knots(ca, max(n_s // 2, 4), k)
        X, widths = self._design(x, y, elev, clc, slope, sinasp, cosasp, fit=True)
        p = np.zeros((X.shape[1], X.shape[1]))
        col = 1
        for w in widths[1:]:
            if w >= 3:
                p[col:col + w, col:col + w] = _penalty(w)
            col += w
        XtX = X.T @ X + float(self.cfg.lam) * p + 1e-8 * np.eye(X.shape[1])
        try:
            self._coef = np.linalg.solve(XtX, X.T @ values)
        except np.linalg.LinAlgError:
            self._coef = np.linalg.lstsq(XtX, X.T @ values, rcond=None)[0]
        self._backend = "numpy"

    def _design(self, x, y, elev, clc, slope, sinasp, cosasp, fit: bool):
        ns, k = self.cfg.n_splines, self.cfg.spline_order
        Bx = _bs(x, self._knots["x"], k, ns)
        By = _bs(y, self._knots["y"], k, ns)
        Bz = _bs(elev, self._knots["z"], k, ns)
        parts = [np.ones((x.size, 1))]
        fid = self.cfg.formula
        if fid == "te_xyelev":
            step = 2
            T = (Bx[:, ::step][:, :, None, None]
                 * By[:, ::step][:, None, :, None]
                 * Bz[:, ::step][:, None, None, :])
            parts.append(T.reshape(x.size, -1))
        else:
            step = 2
            tens = (Bx[:, ::step][:, :, None] * By[:, ::step][:, None, :]).reshape(x.size, -1)
            parts.append(tens)
            parts.append(Bz)
            if fid == "te_xy_ti_elev":
                parts.append((Bx[:, ::step][:, :, None] * Bz[:, ::step][:, None, :]).reshape(x.size, -1))
                parts.append((By[:, ::step][:, :, None] * Bz[:, ::step][:, None, :]).reshape(x.size, -1))
        if self.cfg.use_terrain and "sl" in self._knots:
            n2 = max(ns // 2, 4)
            sl = np.zeros_like(x) if slope is None else np.asarray(slope, dtype=np.float64)
            sa = np.zeros_like(x) if sinasp is None else np.asarray(sinasp, dtype=np.float64)
            ca = np.ones_like(x) if cosasp is None else np.asarray(cosasp, dtype=np.float64)
            sl = np.where(np.isfinite(sl), sl, 0.0)
            sa = np.where(np.isfinite(sa), sa, 0.0)
            ca = np.where(np.isfinite(ca), ca, 1.0)
            parts.append(_bs(sl, self._knots["sl"], k, n2))
            parts.append(_bs(sa, self._knots["sa"], k, n2))
            parts.append(_bs(ca, self._knots["ca"], k, n2))
        if self.cfg.use_clc and clc is not None:
            g = np.asarray(clc, dtype=np.int32)
            if fit:
                keep = []
                for lv in sorted(set(g.tolist())):
                    if lv != 0 and int((g == lv).sum()) >= self.cfg.min_clc_count:
                        keep.append(lv)
                self._clc_levels = keep
            if self._clc_levels:
                parts.append(np.column_stack([(g == lv).astype(np.float64) for lv in self._clc_levels]))
        widths = [p.shape[1] for p in parts]
        return np.column_stack(parts), widths

    def fit_gaussian(self, x, y, elev, values, clc=None, slope=None, sinasp=None, cosasp=None,
                     family: str = "gaussian"):
        self._fit_args = (
            np.asarray(x, float), np.asarray(y, float), np.asarray(elev, float),
            np.asarray(values, float), clc, slope, sinasp, cosasp,
        )
        df = self._frame(x, y, elev, values, clc, slope, sinasp, cosasp)
        if self._fit_r(df, family):
            self._backend = "mgcv"
            return self
        self._fit_numpy(*self._fit_args)
        return self

    def predict_raw(self, x, y, elev, clc=None, slope=None, sinasp=None, cosasp=None) -> np.ndarray:
        if getattr(self, "_backend", None) == "mgcv":
            df = self._frame(x, y, elev, None, clc, slope, sinasp, cosasp)
            hat = self._predict_r(df)
            if hat is not None:
                return hat
            if getattr(self, "_fit_args", None) is not None:
                self._fit_numpy(*self._fit_args)
            else:
                self._backend = "numpy"
        if not self._knots:
            raise RuntimeError("GAM has no coefficients (mgcv predict failed and numpy was not fit)")
        x = np.asarray(x, float)
        n = x.size
        out = np.empty(n)
        tile = self.cfg.predict_tile
        for i0 in range(0, n, tile):
            sl = slice(i0, i0 + tile)
            X, _ = self._design(
                x[sl], np.asarray(y, float)[sl], np.asarray(elev, float)[sl],
                None if clc is None else np.asarray(clc)[sl],
                None if slope is None else np.asarray(slope, float)[sl],
                None if sinasp is None else np.asarray(sinasp, float)[sl],
                None if cosasp is None else np.asarray(cosasp, float)[sl],
                fit=False,
            )
            out[sl] = X @ self._coef
        return out

    def predict_timestamp(
        self, x, y, elev, values, xq, yq, zq,
        clc=None, clc_q=None, slope=None, slope_q=None,
        sinasp=None, sinasp_q=None, cosasp=None, cosasp_q=None,
        var: str = "temp_mean", trace: float | None = None,
    ) -> np.ndarray:
        values = np.asarray(values, float)
        tr = self.cfg.trace if trace is None else float(trace)
        try:
            if self.cfg.two_step and two_step_var(var):
                wet = (values > tr).astype(float)
                self.fit_gaussian(x, y, elev, wet, clc, slope, sinasp, cosasp, family="binomial")
                p = np.clip(self.predict_raw(xq, yq, zq, clc_q, slope_q, sinasp_q, cosasp_q), 0.0, 1.0)
                wet_m = values > tr
                if wet_m.sum() >= 5:
                    self.fit_gaussian(
                        np.asarray(x)[wet_m], np.asarray(y)[wet_m], np.asarray(elev)[wet_m],
                        values[wet_m],
                        None if clc is None else np.asarray(clc)[wet_m],
                        None if slope is None else np.asarray(slope)[wet_m],
                        None if sinasp is None else np.asarray(sinasp)[wet_m],
                        None if cosasp is None else np.asarray(cosasp)[wet_m],
                        family="gaussian",
                    )
                    amt = np.maximum(self.predict_raw(xq, yq, zq, clc_q, slope_q, sinasp_q, cosasp_q), 0.0)
                else:
                    amt = np.zeros(np.asarray(xq).size)
                return np.where(p >= self.cfg.tau_wet, amt, 0.0)
            self.fit_gaussian(x, y, elev, values, clc, slope, sinasp, cosasp, family="gaussian")
            return self.predict_raw(xq, yq, zq, clc_q, slope_q, sinasp_q, cosasp_q)
        finally:
            self.close()
