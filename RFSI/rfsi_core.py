#!/usr/bin/env python3
"""
rfsi_core.py
Core implementation of Random Forest Spatial Interpolation (RFSI).

Key idea: Use neighboring station values + distances as features
together with optional covariates (elevation + landcover).
"""

import numpy as np
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings("ignore")


def build_neighbor_features(
    coords_train: np.ndarray,
    z_train: np.ndarray,
    coords_query: np.ndarray,
    n_obs: int = 12,
    tree: cKDTree = None,
    exclude_self: bool = False
) -> np.ndarray:
    """
    Build features from nearest neighbors (values + distances).
    
    If a pre-built cKDTree is passed, it avoids rebuilding it every time.
    """
    if tree is None:
        tree = cKDTree(coords_train)

    k = n_obs + 1 if exclude_self else n_obs
    distances, indices = tree.query(coords_query, k=k)

    if exclude_self:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    z_neighbors = z_train[indices]
    return np.hstack([z_neighbors, distances])


class RFSI:
    def __init__(self, n_obs: int = 12, rf_params: dict = None):
        self.n_obs = n_obs
        self.rf_params = rf_params or {
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1
        }
        self.model = None
        self.coords_train = None
        self.z_train = None
        self.tree = None
        self._rf_imported = False

    def _import_sklearn(self):
        if not self._rf_imported:
            from sklearn.ensemble import RandomForestRegressor
            self.RandomForestRegressor = RandomForestRegressor
            self._rf_imported = True

    def fit(self, coords: np.ndarray, z: np.ndarray, X_cov: np.ndarray = None):
        self._import_sklearn()
        self.coords_train = np.asarray(coords)
        self.z_train = np.asarray(z)
        self.tree = cKDTree(self.coords_train)

        X_nb = build_neighbor_features(
            self.coords_train, self.z_train, self.coords_train,
            n_obs=self.n_obs, tree=self.tree, exclude_self=True
        )

        if X_cov is not None:
            X = np.hstack([X_nb, np.asarray(X_cov)])
        else:
            X = X_nb

        self.model = self.RandomForestRegressor(**self.rf_params)
        self.model.fit(X, self.z_train)

    def predict(self, coords_pred: np.ndarray, X_cov_pred: np.ndarray = None):
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        coords_pred = np.asarray(coords_pred)

        X_nb = build_neighbor_features(
            self.coords_train, self.z_train, coords_pred,
            n_obs=self.n_obs, tree=self.tree, exclude_self=False
        )

        if X_cov_pred is not None:
            X = np.hstack([X_nb, np.asarray(X_cov_pred)])
        else:
            X = X_nb

        return self.model.predict(X)