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


def build_neighbor_features(
    coords_obs: np.ndarray,
    z_obs: np.ndarray,
    coords_query: np.ndarray,
    n_obs: int = 12,
    tree: Optional[cKDTree] = None,
    exclude_self: bool = False,
) -> np.ndarray:
    """
    Features from n nearest observations: [z_1..z_n, d_1..d_n].

    exclude_self=True drops the 0-distance match (training stations).
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

    return np.hstack([z_neighbors, distances]).astype(np.float32)


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
    def __init__(self, n_obs: int = 12, rf_params: Optional[dict] = None):
        self.n_obs = int(n_obs)
        self.rf_params = rf_params or {
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        }
        self.model = None
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
        return self.model.predict(X)

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
        return self.model.predict(X)

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
