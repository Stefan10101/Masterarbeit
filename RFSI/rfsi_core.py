#!/usr/bin/env python3
"""
rfsi_core.py
Random Forest Spatial Interpolation (Sekulić et al. 2020).

One forest per variable, trained on pooled station–time rows.
Each row uses the n nearest stations at the same timestamp
(values + distances) plus optional covariates at the target location.

Per-field fit is kept only as a helper for the old hyperparameter search.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
import warnings

warnings.filterwarnings("ignore")


DIST_SCALE = 1000.0  # m → km in the distance features


def neighbor_width(n_obs: int) -> int:
    """z_1..z_n + d_km_1..d_n + idw."""
    return int(n_obs) * 2 + 1


def clc_level1(codes) -> np.ndarray:
    c = np.asarray(codes, dtype=np.int32)
    out = np.zeros(c.shape, dtype=np.int32)
    high = c >= 100
    out[high] = c[high] // 100
    mid = (c >= 1) & (c <= 5)
    out[mid] = c[mid]
    out[(out < 0) | (out > 5)] = 0
    return out


def two_step_var(var: str) -> bool:
    v = str(var).lower()
    return v.startswith("precip") or v.startswith("snow")


def clip_var(var: str, pred) -> np.ndarray:
    v = str(var).lower()
    out = np.asarray(pred, dtype=np.float64)
    if v.startswith("precip") or v.startswith("snow") or v.startswith("wind"):
        return np.maximum(out, 0.0)
    if v.startswith("rh"):
        return np.clip(out, 0.0, 100.0)
    return out


def build_covariates(elev=None, clc_code=None, use_elev: bool = True, use_lc: bool = True,
                     extras: dict | None = None):
    """elev in km + CLC level-1 one-hot (6 cols) + optional extras (fixed order)."""
    parts = []
    if use_elev and elev is not None:
        e = np.asarray(elev, dtype=np.float32).reshape(-1)
        parts.append((e / 1000.0)[:, None])
    if use_lc and clc_code is not None:
        g = np.clip(clc_level1(clc_code).reshape(-1), 0, 5)
        parts.append(np.eye(6, dtype=np.float32)[g])
    if extras:
        for key in ("slope", "northness", "tpi", "tmean"):
            if key not in extras or extras[key] is None:
                continue
            v = np.asarray(extras[key], dtype=np.float32).reshape(-1)
            v = np.where(np.isfinite(v), v, 0.0)
            if key == "tmean":
                v = v / 10.0
            elif key == "tpi":
                v = v / 100.0
            elif key == "slope":
                v = v / 45.0
            parts.append(v[:, None])
    if not parts:
        return None
    return np.hstack(parts).astype(np.float32)


def build_neighbor_features(
    coords_obs: np.ndarray,
    z_obs: np.ndarray,
    coords_query: np.ndarray,
    n_obs: int = 12,
    tree: Optional[cKDTree] = None,
    exclude_self: bool = False,
) -> np.ndarray:
    """
    Features from n nearest observations: [z_1..z_n, d_km_1..d_n, idw].

    exclude_self=True drops the 0-distance match (training stations).
    Distances are in km. idw is inverse-distance^2 mean of those neighbours
    so the forest has an explicit local baseline (should not lose to IDW).
    """
    coords_obs = np.asarray(coords_obs, dtype=np.float64)
    z_obs = np.asarray(z_obs, dtype=np.float64)
    coords_query = np.asarray(coords_query, dtype=np.float64)

    if tree is None:
        tree = cKDTree(coords_obs)

    k = n_obs + 1 if exclude_self else n_obs
    k = min(k, len(coords_obs))
    distances, indices = tree.query(coords_query, k=k)

    if k == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    if exclude_self:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    n_got = distances.shape[1]
    z_neighbors = z_obs[indices]
    if n_got < n_obs:
        pad = n_obs - n_got
        z_neighbors = np.pad(z_neighbors, ((0, 0), (0, pad)), constant_values=np.nan)
        distances = np.pad(distances, ((0, 0), (0, pad)), constant_values=np.nan)

    d_km = distances / DIST_SCALE
    w = 1.0 / np.maximum(distances, 1.0) ** 2
    good = np.isfinite(z_neighbors) & np.isfinite(w)
    w = np.where(good, w, 0.0)
    z_w = np.where(good, z_neighbors, 0.0)
    idw = (w * z_w).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-12)
    return np.hstack([z_neighbors, d_km, idw[:, None]]).astype(np.float32)


def assemble_pooled_training(
    times: np.ndarray,
    coords: np.ndarray,
    z: np.ndarray,
    n_obs: int,
    X_cov: Optional[np.ndarray] = None,
    min_stations: int = 8,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Stack per-timestamp neighbour features into one training matrix."""
    times = np.asarray(times)
    coords = np.asarray(coords, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if X_cov is not None:
        X_cov = np.asarray(X_cov)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    order = np.argsort(times, kind="mergesort")
    times_s = times[order]
    uniq, starts = np.unique(times_s, return_index=True)
    starts = np.append(starts, len(times_s))

    for i in range(len(uniq)):
        sl = order[starts[i]:starts[i + 1]]
        if len(sl) < max(min_stations, n_obs + 1):
            continue
        X_nb = build_neighbor_features(
            coords[sl], z[sl], coords[sl],
            n_obs=n_obs, exclude_self=True,
        )
        if np.isnan(X_nb).any():
            ok = ~np.isnan(X_nb).any(axis=1)
            if ok.sum() < min_stations:
                continue
            sl_ok = sl[ok]
            X_nb = X_nb[ok]
        else:
            sl_ok = sl
        if X_cov is not None:
            X_parts.append(np.hstack([X_nb, X_cov[sl_ok]]))
        else:
            X_parts.append(X_nb)
        y_parts.append(z[sl_ok])

    if not X_parts:
        raise RuntimeError(
            f"No timestamp had >= {max(min_stations, n_obs + 1)} stations."
        )
    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.float32)
    return X, y, int(len(y))


class RFSI:
    def __init__(self, n_obs: int = 12, rf_params: Optional[dict] = None,
                 two_step: bool = False, tau_wet: float = 0.5, trace: float = 0.1,
                 var_name: str = ""):
        self.n_obs = int(n_obs)
        self.rf_params = rf_params or {
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        }
        self.two_step = bool(two_step)
        self.tau_wet = float(tau_wet)
        self.trace = float(trace)
        self.var_name = str(var_name)
        self.model = None
        self.model_ind = None
        self.n_features_: Optional[int] = None
        self.n_train_rows_: int = 0
        self.coords_train = None
        self.z_train = None
        self.tree = None
        self._rf_imported = False

    def _import_sklearn(self):
        if not self._rf_imported:
            from sklearn.ensemble import RandomForestRegressor
            self.RandomForestRegressor = RandomForestRegressor
            self._rf_imported = True

    def fit_pooled(
        self,
        times: np.ndarray,
        coords: np.ndarray,
        z: np.ndarray,
        X_cov: Optional[np.ndarray] = None,
        min_stations: int = 8,
    ) -> "RFSI":
        """Fit one forest on all station–time rows (paper architecture)."""
        self._import_sklearn()
        z = np.asarray(z, dtype=np.float64)
        if self.two_step:
            wet = (z > self.trace).astype(np.float64)
            Xi, yi, n = assemble_pooled_training(
                times, coords, wet, self.n_obs, X_cov=X_cov, min_stations=min_stations,
            )
            self.model_ind = self.RandomForestRegressor(**self.rf_params)
            self.model_ind.fit(Xi, yi)
            wet_m = z > self.trace
            if int(wet_m.sum()) >= max(min_stations * 3, 30):
                Xc = None if X_cov is None else np.asarray(X_cov)[wet_m]
                X, y, n = assemble_pooled_training(
                    np.asarray(times)[wet_m], np.asarray(coords)[wet_m], z[wet_m],
                    self.n_obs, X_cov=Xc, min_stations=max(4, min_stations // 2),
                )
            else:
                X, y, n = assemble_pooled_training(
                    times, coords, z, self.n_obs, X_cov=X_cov, min_stations=min_stations,
                )
        else:
            X, y, n = assemble_pooled_training(
                times, coords, z, self.n_obs, X_cov=X_cov, min_stations=min_stations,
            )
        self.n_train_rows_ = n
        self.n_features_ = int(X.shape[1])
        self.model = self.RandomForestRegressor(**self.rf_params)
        self.model.fit(X, y)
        return self

    def fit(self, coords: np.ndarray, z: np.ndarray, X_cov: Optional[np.ndarray] = None):
        """Single-field fit. Used by the old per-timestep search only."""
        self._import_sklearn()
        self.coords_train = np.asarray(coords, dtype=np.float64)
        self.z_train = np.asarray(z, dtype=np.float64)
        self.tree = cKDTree(self.coords_train)
        X_nb = build_neighbor_features(
            self.coords_train, self.z_train, self.coords_train,
            n_obs=self.n_obs, tree=self.tree, exclude_self=True,
        )
        X = np.hstack([X_nb, np.asarray(X_cov)]) if X_cov is not None else X_nb
        self.n_features_ = int(X.shape[1])
        self.n_train_rows_ = int(X.shape[0])
        self.model = self.RandomForestRegressor(**self.rf_params)
        self.model.fit(X, self.z_train)
        return self

    def predict_field(
        self,
        coords_obs: np.ndarray,
        z_obs: np.ndarray,
        coords_pred: np.ndarray,
        X_cov_pred: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict a grid/query set from the stations of one timestamp."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        coords_obs = np.asarray(coords_obs, dtype=np.float64)
        z_obs = np.asarray(z_obs, dtype=np.float64)
        coords_pred = np.asarray(coords_pred, dtype=np.float64)
        X_nb = build_neighbor_features(
            coords_obs, z_obs, coords_pred,
            n_obs=self.n_obs, exclude_self=False,
        )
        X = np.hstack([X_nb, np.asarray(X_cov_pred)]) if X_cov_pred is not None else X_nb
        if self.n_features_ is not None and X.shape[1] != self.n_features_:
            raise RuntimeError(
                f"Feature width {X.shape[1]} != trained width {self.n_features_}"
            )
        hat = np.asarray(self.model.predict(X), dtype=np.float64)
        if self.two_step and self.model_ind is not None:
            z_ind = (z_obs > self.trace).astype(np.float64)
            X_nb_i = build_neighbor_features(
                coords_obs, z_ind, coords_pred,
                n_obs=self.n_obs, exclude_self=False,
            )
            Xi = np.hstack([X_nb_i, np.asarray(X_cov_pred)]) if X_cov_pred is not None else X_nb_i
            p = np.clip(self.model_ind.predict(Xi), 0.0, 1.0)
            hat = np.where(p >= self.tau_wet, np.maximum(hat, 0.0), 0.0)
        return clip_var(self.var_name, hat)

    def predict_stations(
        self,
        coords_obs: np.ndarray,
        z_obs: np.ndarray,
        X_cov_obs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Station predictions with self excluded (frozen pooled model)."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        coords_obs = np.asarray(coords_obs, dtype=np.float64)
        z_obs = np.asarray(z_obs, dtype=np.float64)
        X_nb = build_neighbor_features(
            coords_obs, z_obs, coords_obs,
            n_obs=self.n_obs, exclude_self=True,
        )
        X = np.hstack([X_nb, np.asarray(X_cov_obs)]) if X_cov_obs is not None else X_nb
        hat = np.asarray(self.model.predict(X), dtype=np.float64)
        return clip_var(self.var_name, hat)


def extras_for_var(df, var: str, rh_t_mode: str = "none") -> dict | None:
    v = str(var).lower()
    out = {}
    if v.startswith("snow"):
        for c in ("slope", "northness"):
            if c in df.columns:
                out[c] = df[c].to_numpy()
    if v.startswith("wind") and "tpi" in df.columns:
        out["tpi"] = df["tpi"].to_numpy()
    if v.startswith("rh") and rh_t_mode in ("predicted", "observed") and "temp_mean" in df.columns:
        out["tmean"] = df["temp_mean"].to_numpy()
    return out or None

    def predict(
        self,
        coords_pred: np.ndarray,
        X_cov_pred: Optional[np.ndarray] = None,
        coords_obs: Optional[np.ndarray] = None,
        z_obs: Optional[np.ndarray] = None,
        exclude_self: bool = False,
    ) -> np.ndarray:
        """
        Backward-compatible predict.
        Pooled models must pass coords_obs/z_obs of the current field.
        """
        if coords_obs is None or z_obs is None:
            if self.coords_train is None or self.z_train is None:
                raise RuntimeError("Pass coords_obs and z_obs for a pooled model.")
            coords_obs, z_obs = self.coords_train, self.z_train
        if exclude_self:
            return self.predict_stations(coords_obs, z_obs, X_cov_obs=X_cov_pred)
        return self.predict_field(coords_obs, z_obs, coords_pred, X_cov_pred=X_cov_pred)
