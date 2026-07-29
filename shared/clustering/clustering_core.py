#!/usr/bin/env python3
"""
clustering_core.py
Core regime clustering logic: feature matrix assembly, k selection by
silhouette, medoid extraction, and rich diagnostic output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import yaml
from tqdm import tqdm

try:
    from minisom import MiniSom
    HAS_MINISOM = True
except ImportError:
    HAS_MINISOM = False

try:
    from .features import get_feature_func, FEATURE_FUNCS
except ImportError:
    from features import get_feature_func, FEATURE_FUNCS


def _prepare_station_arrays(
    df_t: pd.DataFrame,
    meta: pd.DataFrame,
    value_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Align station values with metadata for one timestamp.
    Returns values, elev, coords (N,2), meta_extra (aligned DataFrame).
    """
    # merge on station_name (meta may contain duplicate names → keep first)
    merged = (
        df_t[["station_name", value_col]]
        .merge(meta, on="station_name", how="inner")
        .dropna(subset=[value_col])
        .drop_duplicates(subset=["station_name"], keep="first")
    )
    if len(merged) < 5:
        return None, None, None, None

    values = merged[value_col].to_numpy(dtype=float)
    # prefer elev_dem if present, else hoehe
    if "elev_dem" in merged.columns:
        elev = merged["elev_dem"].to_numpy(dtype=float)
    else:
        elev = merged["hoehe"].to_numpy(dtype=float)

    # coordinates: prefer projected if available, else lat/lon
    if "x" in merged.columns and "y" in merged.columns:
        coords = merged[["x", "y"]].to_numpy(dtype=float)
    else:
        coords = merged[["lon", "lat"]].to_numpy(dtype=float)

    meta_extra = merged  # keep full row for aspect/slope etc.
    return values, elev, coords, meta_extra


def build_feature_matrix(
    data: pd.DataFrame,
    meta: pd.DataFrame,
    var: str,
    time_col: str,
    value_col: str,
    resolution: str,
    temp_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a (n_timestamps × n_features) DataFrame for one variable.
    Also returns the list of timestamps that produced valid features.
    """
    feature_func = get_feature_func(var)

    # group by time once
    grouped = data.groupby(time_col, sort=True)

    records = []
    prev_values = prev_elev = prev_coords = None
    prev_time = None

    # for humidity we may need simultaneous temperature
    has_temp = temp_col is not None and temp_col in data.columns

    for t, df_t in tqdm(grouped, desc=f"Features [{var}]", leave=False):
        values, elev, coords, meta_extra = _prepare_station_arrays(
            df_t, meta, value_col
        )
        if values is None:
            continue

        # temperature companion for humidity – align by station_name
        temp_values = None
        if has_temp and var in ("relative_humidity", "humidity"):
            rh_stations = meta_extra["station_name"].values
            temp_merged = (
                df_t[["station_name", temp_col]]
                .dropna(subset=[temp_col])
                .drop_duplicates(subset=["station_name"], keep="first")
            )
            temp_lookup = temp_merged.set_index("station_name")[temp_col]
            aligned = temp_lookup.reindex(rh_stations)
            if aligned.notna().sum() >= 5:
                temp_values = aligned.to_numpy(dtype=float)

        # time object
        if not isinstance(t, pd.Timestamp):
            try:
                t = pd.to_datetime(t)
            except Exception:
                # for year_week / year_month we still want a proxy
                t = pd.Timestamp(t) if not isinstance(t, pd.Timestamp) else t

        # lag only for half_hourly and when previous exists and is consecutive
        use_lag = (
            resolution == "half_hourly"
            and prev_values is not None
            and prev_time is not None
            and (t - prev_time) <= pd.Timedelta("1h")
        )

        # build kwargs that every feature function accepts
        kwargs = dict(
            values=values,
            elev=elev,
            coords=coords,
            time=t if isinstance(t, pd.Timestamp) else pd.Timestamp("2000-01-01"),
            meta_extra=meta_extra,
            prev_values=prev_values if use_lag else None,
            prev_elev=prev_elev if use_lag else None,
            prev_coords=prev_coords if use_lag else None,
        )
        # humidity is the only function that currently uses temp_values
        if temp_values is not None and var in ("relative_humidity", "humidity"):
            kwargs["temp_values"] = temp_values

        feat = feature_func(**kwargs)

        feat[time_col] = t
        feat["n_stations"] = len(values)
        records.append(feat)

        # update lag state
        prev_values = values
        prev_elev = elev
        prev_coords = coords
        prev_time = t if isinstance(t, pd.Timestamp) else None

    if not records:
        return pd.DataFrame()

    feat_df = pd.DataFrame(records)
    feat_df = feat_df.set_index(time_col)
    return feat_df


def _prepare_matrix(
    feat_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """Impute, drop constant columns, return X and remaining feature names."""
    feature_cols = [c for c in feat_df.columns if c not in ("n_stations",)]
    X = np.array(feat_df[feature_cols].to_numpy(dtype=float), copy=True)

    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col)
        if not np.isfinite(med):
            med = 0.0
        nan_mask = ~np.isfinite(col)
        if nan_mask.any():
            col[nan_mask] = med
            X[:, j] = col

    keep = np.std(X, axis=0) > 1e-12
    if not keep.all():
        dropped = [c for c, k in zip(feature_cols, keep) if not k]
        if dropped:
            print(f"  dropping constant/empty features: {dropped}")
        feature_cols = [c for c, k in zip(feature_cols, keep) if k]
        X = X[:, keep]

    if X.shape[1] == 0:
        raise RuntimeError("No usable features left after cleaning")
    return X, feature_cols


def _scale_and_pca(
    X: np.ndarray,
    pca_variance: Optional[float] = 0.95,
    random_state: int = 42,
) -> Tuple[np.ndarray, StandardScaler, Optional[PCA]]:
    """Standardise, optionally project with PCA retaining pca_variance fraction."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = None
    if pca_variance is not None and 0.0 < pca_variance < 1.0 and Xs.shape[1] > 2:
        pca = PCA(n_components=pca_variance, random_state=random_state)
        Xs = pca.fit_transform(Xs)
        print(f"  PCA: {X.shape[1]} features → {Xs.shape[1]} components "
              f"(explained var ≥ {pca_variance:.0%})")
    return Xs, scaler, pca


def select_kmeans(
    Xs: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
) -> Tuple[int, Dict[int, float], Any, np.ndarray]:
    scores = {}
    best_k, best_score, best_model, best_labels = k_min, -1.0, None, None
    for k in range(k_min, k_max + 1):
        if k >= len(Xs):
            break
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(Xs)
        score = float(silhouette_score(Xs, labels)) if len(np.unique(labels)) > 1 else -1.0
        scores[k] = score
        if score > best_score:
            best_k, best_score, best_model, best_labels = k, score, km, labels
    return best_k, scores, best_model, best_labels


def select_gmm(
    Xs: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
) -> Tuple[int, Dict[int, float], Any, np.ndarray]:
    """Choose k by minimum BIC (standard for GMM regime work)."""
    scores = {}
    best_k, best_bic, best_model, best_labels = k_min, np.inf, None, None
    for k in range(k_min, k_max + 1):
        if k >= len(Xs):
            break
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
            n_init=3,
            max_iter=200,
        )
        labels = gmm.fit_predict(Xs)
        bic = float(gmm.bic(Xs))
        scores[k] = bic
        if bic < best_bic:
            best_k, best_bic, best_model, best_labels = k, bic, gmm, labels
    return best_k, scores, best_model, best_labels


def select_som(
    Xs: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
) -> Tuple[int, Dict[int, float], Any, np.ndarray]:
    """
    Train a rectangular SOM. Grid size chosen so n_nodes is in [k_min, k_max]
    and closest to sqrt-range mid. Each node is one regime.
    """
    if not HAS_MINISOM:
        raise ImportError(
            "minisom is required for SOM clustering. Install with: pip install minisom"
        )

    # pick a nearly-square grid whose node count lies in [k_min, k_max]
    target = max(k_min, min(k_max, int(round(np.sqrt(k_min * k_max)))))
    side = max(2, int(round(np.sqrt(target))))
    n_nodes = side * side
    while n_nodes > k_max and side > 2:
        side -= 1
        n_nodes = side * side
    while n_nodes < k_min:
        side += 1
        n_nodes = side * side

    som = MiniSom(
        side, side, Xs.shape[1],
        sigma=1.0, learning_rate=0.5,
        random_seed=random_state,
    )
    som.random_weights_init(Xs)
    n_iter = min(1000, max(100, len(Xs) // 10))
    som.train_random(Xs, n_iter, verbose=False)

    # map each sample to a flat node id
    winners = np.array([som.winner(x) for x in Xs])
    labels = winners[:, 0] * side + winners[:, 1]

    # silhouette for reporting (not for selection)
    score = float(silhouette_score(Xs, labels)) if len(np.unique(labels)) > 1 else -1.0
    scores = {n_nodes: score}
    return n_nodes, scores, som, labels


def extract_medoids(
    Xs: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    n_medoids: int = 8,
) -> Dict[int, List]:
    """For each cluster return the n_medoids timestamps closest to the centroid."""
    medoids = {}
    for c in np.unique(labels):
        mask = labels == c
        pts = Xs[mask]
        ts = timestamps[mask]
        if len(pts) == 0:
            medoids[int(c)] = []
            continue
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        order = np.argsort(dists)
        chosen = order[: min(n_medoids, len(order))]
        medoids[int(c)] = list(ts[chosen])
    return medoids


def run_clustering_for_variable(
    feat_df: pd.DataFrame,
    var: str,
    k_min: int = 3,
    k_max: int = 8,
    n_medoids: int = 8,
    random_state: int = 42,
    cluster_method: str = "gmm",
    pca_variance: Optional[float] = 0.95,
) -> Dict[str, Any]:
    """
    Full pipeline for one variable.
    cluster_method: 'kmeans' | 'gmm' | 'som'
    pca_variance: fraction of variance to keep (None = skip PCA)
    """
    X, feature_cols = _prepare_matrix(feat_df)
    Xs, scaler, pca = _scale_and_pca(X, pca_variance=pca_variance, random_state=random_state)

    method = cluster_method.lower()
    if method == "kmeans":
        best_k, scores, model, labels = select_kmeans(Xs, k_min, k_max, random_state)
        score_name = "silhouette_scores"
        best_score_key = "best_silhouette"
        best_score_val = scores.get(best_k, float("nan"))
    elif method == "gmm":
        best_k, scores, model, labels = select_gmm(Xs, k_min, k_max, random_state)
        score_name = "bic_scores"
        best_score_key = "best_bic"
        best_score_val = scores.get(best_k, float("nan"))
    elif method == "som":
        best_k, scores, model, labels = select_som(Xs, k_min, k_max, random_state)
        score_name = "silhouette_scores"
        best_score_key = "best_silhouette"
        best_score_val = scores.get(best_k, float("nan"))
    else:
        raise ValueError(f"Unknown cluster_method '{cluster_method}'. Use kmeans|gmm|som")

    labels = np.asarray(labels, dtype=int)
    timestamps = feat_df.index.to_numpy()
    medoids = extract_medoids(Xs, labels, timestamps, n_medoids=n_medoids)

    # distance to assigned centroid (in the space used for clustering)
    dists = np.zeros(len(labels))
    for c in np.unique(labels):
        mask = labels == c
        centroid = Xs[mask].mean(axis=0)
        dists[mask] = np.linalg.norm(Xs[mask] - centroid, axis=1)

    assignments = pd.DataFrame({
        "timestamp": timestamps,
        "variable": var,
        "cluster_id": labels,
        "dist_to_centroid": dists,
        "n_stations": feat_df["n_stations"].to_numpy(),
    })

    cluster_stats = {}
    for c in sorted(np.unique(labels)):
        mask = labels == c
        cluster_stats[int(c)] = {
            "size": int(mask.sum()),
            "mean_dist": float(dists[mask].mean()) if mask.any() else float("nan"),
            "feature_means": {
                col: float(feat_df[col].iloc[mask].mean()) for col in feature_cols
            },
            "feature_stds": {
                col: float(feat_df[col].iloc[mask].std()) for col in feature_cols
            },
            "medoids": [str(t) for t in medoids.get(int(c), [])],
        }

    summary = {
        "variable": var,
        "cluster_method": method,
        "pca_variance": pca_variance,
        "n_pca_components": int(Xs.shape[1]) if pca is not None else None,
        "n_timestamps": len(feat_df),
        "best_k": int(best_k),
        score_name: {int(k): float(v) for k, v in scores.items()},
        best_score_key: float(best_score_val) if np.isfinite(best_score_val) else None,
        "feature_columns": feature_cols,
        "cluster_stats": cluster_stats,
        "n_medoids_requested": n_medoids,
    }

    feat_out = feat_df.copy()
    feat_out["cluster_id"] = labels
    feat_out["dist_to_centroid"] = dists

    print(f"  method={method}  best_k={best_k}  {best_score_key}={best_score_val:.4g}")

    return {
        "assignments": assignments,
        "features": feat_out,
        "summary": summary,
        "medoids": medoids,
        "model": model,
        "scaler": scaler,
        "pca": pca,
    }
