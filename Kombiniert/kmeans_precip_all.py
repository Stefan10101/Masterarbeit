#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Performance Analysis Pipeline v1.5 (Final)
===============================================
- Fixed station detection (uses station_name)
- Dynamic number of clusters via Elbow Method
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import warnings

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT

warnings.filterwarnings("ignore")

MASTER_DIR = Path(DATA_ROOT / "kombiniert")
PLOT_DIR   = Path(DATA_ROOT / "plots" / "kombiniert")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== CONFIG ====================
TEST_START = "2023-01-01"
TEST_END   = "2023-03-31"
N_NEIGHBORS = 10
MIN_TEST_DAYS = 20

LOG_FILE = PLOT_DIR / f"full_analysis_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def get_all_precip_stations():
    """Robust station detection using station_name"""
    stations = []
    for f in MASTER_DIR.glob("*_full_2020_2025_master.parquet"):
        try:
            meta = pd.read_parquet(f).iloc[0]

            # Use station_name (most common in your files)
            station_name = meta.get("station_name", meta.get("station", f.stem))

            if "precipitation" not in meta.index:
                continue

            stations.append({
                "file": f,
                "station": station_name,
                "lat": meta["lat"],
                "lon": meta["lon"],
                "hoehe": meta["hoehe"]
            })
        except Exception as e:
            logger.debug(f"Skipped {f.name}: {e}")
            continue

    logger.info(f"Found {len(stations)} stations with precipitation data")
    return stations


def build_neighbors(target_station, all_stations, n_neighbors=N_NEIGHBORS):
    """
    Rank neighboring stations using:
    - Distance
    - Elevation difference  
    - Precipitation correlation (higher = better)
    """
    t = next(s for s in all_stations if s["station"] == target_station)
    
    # Load target precipitation (daily for speed)
    target_df = pd.read_parquet(t["file"], columns=["timestamp", "precipitation"])
    target_daily = target_df.set_index("timestamp").resample("D")["precipitation"].sum()

    neighbors = []
    
    for s in all_stations:
        if s["station"] == target_station:
            continue
            
        try:
            # Load neighbor precipitation
            neigh_df = pd.read_parquet(s["file"], columns=["timestamp", "precipitation"])
            neigh_daily = neigh_df.set_index("timestamp").resample("D")["precipitation"].sum()
            
            # Calculate correlation (only on overlapping period)
            common_idx = target_daily.index.intersection(neigh_daily.index)
            if len(common_idx) < 100:  # Need enough data
                continue
                
            corr = target_daily.loc[common_idx].corr(neigh_daily.loc[common_idx])
            if pd.isna(corr):
                continue

            # Calculate geographic factors
            dist = haversine(t["lat"], t["lon"], s["lat"], s["lon"])
            elev_diff = abs(t["hoehe"] - s["hoehe"])
            
            # Combined score (lower = better)
            # 0.35 * distance + 0.25 * elevation + 0.40 * (1 - correlation)
            score = (0.35 * dist) + (0.25 * elev_diff) + (0.40 * (1 - corr))
            
            neighbors.append({
                **s,
                "distance_km": round(dist, 1),
                "elev_diff_m": round(elev_diff),
                "correlation": round(corr, 3),
                "score": round(score, 1)
            })
            
        except:
            continue

    # Sort by score (lower is better) and return top N
    ranked = sorted(neighbors, key=lambda x: x["score"])[:n_neighbors]
    
    logger.info(f"\n=== Top {n_neighbors} Most Similar Stations to {target_station} ===")
    for i, s in enumerate(ranked, 1):
        logger.info(f"{i:2d}. {s['station']:28s} | "
                    f"{s['distance_km']:6.1f} km | "
                    f"{s['elev_diff_m']:4.0f} m | "
                    f"corr: {s['correlation']:.3f} | "
                    f"score: {s['score']}")
    
    return ranked

def run_model_for_station(target_info, neighbors, test_start, test_end):
    try:
        target = pd.read_parquet(target_info["file"])

        available = ["precipitation"]
        for col in ["temperature", "relative_humidity", "wind_speed"]:
            if col in target.columns:
                available.append(col)

        df = target.set_index("timestamp")[available]

        for n in neighbors:
            try:
                neigh = pd.read_parquet(n["file"], columns=["timestamp", "precipitation"]).set_index("timestamp")
                df[f"precip_{n['station']}"] = neigh["precipitation"]
            except: continue

        # Feature engineering
        df = df.sort_index()
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            df[f"precip_lag_{lag}"] = df["precipitation"].shift(lag)
        for lag in [7, 14]:
            df[f"precip_lag_{lag*48}"] = df["precipitation"].shift(lag*48)

        for col in [c for c in df.columns if c.startswith("precip_") and c != "precipitation"]:
            for lag in [6, 24, 48]:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        df["target"] = df["precipitation"].shift(-1)
        df = df.dropna()

        test_mask = (df.index >= test_start) & (df.index <= test_end)
        if test_mask.sum() < MIN_TEST_DAYS * 48:
            return None

        train = df[~test_mask]
        test = df[test_mask]

        feature_cols = [c for c in df.columns if c not in ["target", "precipitation"]]
        X_train, y_train = train[feature_cols], train["target"]
        X_test, y_test = test[feature_cols], test["target"]

        model = XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            "station": target_info["station"],
            "lat": target_info["lat"],
            "lon": target_info["lon"],
            "hoehe": target_info["hoehe"],
            "rmse": round(rmse, 3),
            "mae": round(mae, 3),
            "r2": round(r2, 4),
            "n_test": len(y_test)
        }
    except Exception as e:
        logger.warning(f"Failed for {target_info['station']}: {e}")
        return None


def find_optimal_clusters(features_scaled, max_clusters=10):
    """Elbow method to find optimal number of clusters"""
    inertias = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        inertias.append(kmeans.inertia_)

    # Simple elbow detection
    diffs = np.diff(inertias)
    optimal_k = np.argmin(diffs) + 2
    logger.info(f"Optimal number of clusters (Elbow method): {optimal_k}")
    return optimal_k


def cluster_stations(results_df):
    features = results_df[["rmse", "r2", "hoehe"]].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    n_clusters = find_optimal_clusters(features_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    results_df["cluster"] = kmeans.fit_predict(features_scaled)

    logger.info(f"\n=== Cluster Summary ({n_clusters} clusters) ===")
    for c in range(n_clusters):
        cluster = results_df[results_df["cluster"] == c]
        logger.info(f"\nCluster {c} ({len(cluster)} stations):")
        logger.info(f"  Mean RMSE: {cluster['rmse'].mean():.2f} | Mean RÂ²: {cluster['r2'].mean():.3f}")
        logger.info(f"  Mean Height: {cluster['hoehe'].mean():.0f} m")

    return results_df


def main():
    logger.info("=== FULL PERFORMANCE ANALYSIS PIPELINE v1.5 ===")
    logger.info(f"Test period: {TEST_START} to {TEST_END}")

    all_stations = get_all_precip_stations()
    if not all_stations:
        logger.error("No stations found.")
        return

    results = []
    for i, station in enumerate(all_stations):
        logger.info(f"Processing {i+1}/{len(all_stations)}: {station['station']}")
        neighbors = build_neighbors(station["station"], all_stations)
        res = run_model_for_station(station, neighbors, TEST_START, TEST_END)
        if res:
            results.append(res)

    if not results:
        logger.error("No stations processed successfully.")
        return

    results_df = pd.DataFrame(results)
    results_df.to_csv(PLOT_DIR / "station_performance.csv", index=False)
    logger.info(f"Saved performance for {len(results_df)} stations")

    clustered = cluster_stations(results_df)
    clustered.to_csv(PLOT_DIR / "station_clusters.csv", index=False)
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
