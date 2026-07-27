#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Performance Analysis - All Variables v1.1 (Fixed)
======================================================
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
TEST_START = "2022-01-01"
TEST_END   = "2022-03-31"
N_NEIGHBORS = 8
MIN_TEST_DAYS = 20
TEST_MODE = False
N_TEST_STATIONS = 5
VARIABLES = ["temperature", "wind_speed", "relative_humidity", "snow_height"]

LOG_FILE = PLOT_DIR / f"all_variables_analysis_{datetime.now():%Y%m%d_%H%M%S}.log"
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


def get_all_stations_for_variable(variable):
    stations = []
    for f in MASTER_DIR.glob("*_full_2020_2025_master.parquet"):
        try:
            meta = pd.read_parquet(f).iloc[0]
            if variable not in meta.index:
                continue
            station_name = meta.get("station_name", meta.get("station", f.stem))
            stations.append({
                "file": f,
                "station": station_name,
                "lat": meta["lat"],
                "lon": meta["lon"],
                "hoehe": meta["hoehe"],
                "variable": variable
            })
        except:
            continue
    return stations


def build_neighbors(target_station, all_stations, variable, n_neighbors=N_NEIGHBORS):
    """Rank neighbors using distance + elevation + correlation of the target variable"""
    t = next(s for s in all_stations if s["station"] == target_station)
    
    target_df = pd.read_parquet(t["file"], columns=["timestamp", variable])
    target_daily = target_df.set_index("timestamp").resample("D")[variable].mean()

    neighbors = []
    for s in all_stations:
        if s["station"] == target_station: continue
        try:
            neigh_df = pd.read_parquet(s["file"], columns=["timestamp", variable])
            neigh_daily = neigh_df.set_index("timestamp").resample("D")[variable].mean()
            
            common_idx = target_daily.index.intersection(neigh_daily.index)
            if len(common_idx) < 100: continue
            
            corr = target_daily.loc[common_idx].corr(neigh_daily.loc[common_idx])
            if pd.isna(corr): continue

            dist = haversine(t["lat"], t["lon"], s["lat"], s["lon"])
            elev_diff = abs(t["hoehe"] - s["hoehe"])
            score = (0.25 * dist) + (0.15 * elev_diff) + (0.6 * (1 - corr))
            
            neighbors.append({
                **s,
                "distance_km": round(dist, 1),
                "elev_diff_m": round(elev_diff),
                "correlation": round(corr, 3),
                "score": round(score, 1)
            })
        except:
            continue

    ranked = sorted(neighbors, key=lambda x: x["score"])[:n_neighbors]
    
    logger.info(f"\n=== Top {n_neighbors} Most Similar Stations to {target_station} ({variable}) ===")
    for i, s in enumerate(ranked, 1):
        logger.info(f"{i:2d}. {s['station']:28s} | {s['distance_km']:6.1f} km | "
                    f"{s['elev_diff_m']:4.0f} m | corr: {s['correlation']:.3f} | score: {s['score']}")
    return ranked


def run_model_for_station(target_info, neighbors, test_start, test_end):
    try:
        variable = target_info["variable"]
        target = pd.read_parquet(target_info["file"])
        
        available = [variable]
        for col in ["temperature", "relative_humidity", "wind_speed", "snow_height"]:
            if col in target.columns and col != variable:
                available.append(col)

        df = target.set_index("timestamp")[available].rename(columns={variable: "target_var"})

        for n in neighbors:
            try:
                neigh = pd.read_parquet(n["file"], columns=["timestamp", variable]).set_index("timestamp")
                df[f"{variable}_{n['station']}"] = neigh[variable]
            except: continue

        df = df.sort_index()
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            df[f"target_lag_{lag}"] = df["target_var"].shift(lag)
        for lag in [7, 14]:
            df[f"target_lag_{lag*48}"] = df["target_var"].shift(lag*48)

        for col in [c for c in df.columns if c.startswith(f"{variable}_")]:
            for lag in [6, 24, 48]:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        df["target"] = df["target_var"].shift(-1)
        df = df.dropna()

        test_mask = (df.index >= test_start) & (df.index <= test_end)
        if test_mask.sum() < MIN_TEST_DAYS * 48:
            return None

        train = df[~test_mask]
        test = df[test_mask]

        feature_cols = [c for c in df.columns if c not in ["target", "target_var"]]
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
            "variable": variable,
            "lat": target_info["lat"],
            "lon": target_info["lon"],
            "hoehe": target_info["hoehe"],
            "rmse": round(rmse, 3),
            "mae": round(mae, 3),
            "r2": round(r2, 4),
            "n_test": len(y_test)
        }
    except Exception as e:
        logger.warning(f"Failed for {target_info['station']} ({variable}): {e}")
        return None


def find_optimal_clusters(features_scaled, max_clusters=10):
    n_samples = features_scaled.shape[0]
    max_k = min(max_clusters, n_samples - 1, 8)
    if max_k < 2: return 2
    
    inertias = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        inertias.append(kmeans.inertia_)
    
    diffs = np.diff(inertias)
    return np.argmin(diffs) + 2


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


def run_analysis_for_variable(variable):
    logger.info(f"\n{'='*60}")
    logger.info(f"STARTING ANALYSIS FOR: {variable.upper()}")
    logger.info(f"{'='*60}")
    
    all_stations = get_all_stations_for_variable(variable)
    
    if TEST_MODE:
        all_stations = all_stations[:N_TEST_STATIONS]
        logger.info(f"TEST MODE: Running on first {N_TEST_STATIONS} stations only")
    
    if not all_stations:
        logger.warning(f"No stations found for {variable}")
        return

    results = []
    for i, station in enumerate(all_stations):
        logger.info(f"Processing {i+1}/{len(all_stations)}: {station['station']}")
        neighbors = build_neighbors(station["station"], all_stations, variable)  # â† FIXED
        res = run_model_for_station(station, neighbors, TEST_START, TEST_END)
        if res:
            results.append(res)

    if not results:
        logger.warning(f"No successful results for {variable}")
        return

    results_df = pd.DataFrame(results)
    results_df.to_csv(PLOT_DIR / f"{variable}_performance.csv", index=False)
    logger.info(f"Saved {variable} performance for {len(results_df)} stations")

    clustered = cluster_stations(results_df)
    clustered.to_csv(PLOT_DIR / f"{variable}_clusters.csv", index=False)
    logger.info(f"{variable} analysis complete.")


def main():
    logger.info("=== FULL MULTI-VARIABLE PERFORMANCE ANALYSIS v1.1 ===")
    logger.info(f"Test period: {TEST_START} to {TEST_END}")
    logger.info(f"TEST MODE: {TEST_MODE}")

    for variable in VARIABLES:
        run_analysis_for_variable(variable)

    logger.info("\n=== ALL VARIABLES ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()
