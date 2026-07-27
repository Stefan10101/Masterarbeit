#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Precipitation Forecasting v3.2 (Custom Test Period)
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from sklearn.model_selection import TimeSeriesSplit
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
TARGET_STATION = "Innsbruck-Seegrube"
TEST_START = "2021-12-15"          # â† Start of test period
TEST_END   = "2022-02-01"          # â† End of test period
N_NEIGHBORS = 8
N_SPLITS = 5

LOG_FILE = PLOT_DIR / f"xgb_precip_period_{TARGET_STATION}_{datetime.now():%Y%m%d_%H%M%S}.log"
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


def build_station_hierarchy(target_station, n_neighbors=N_NEIGHBORS):
    target_file = list(MASTER_DIR.glob(f"*{target_station}*_full_2020_2025_master.parquet"))[0]
    target_meta = pd.read_parquet(target_file, columns=["lat", "lon", "hoehe"])
    t_lat, t_lon, t_elev = target_meta.iloc[0][["lat", "lon", "hoehe"]]

    all_stations = []
    for f in MASTER_DIR.glob("*_full_2020_2025_master.parquet"):
        if target_station in f.name: continue
        try:
            meta = pd.read_parquet(f, columns=["lat", "lon", "hoehe", "precipitation"])
            if "precipitation" not in meta.columns or meta["precipitation"].isna().all(): continue
            lat, lon, elev = meta.iloc[0][["lat", "lon", "hoehe"]]
            dist = haversine(t_lat, t_lon, lat, lon)
            elev_diff = abs(t_elev - elev)
            score = 0.6 * dist + 0.4 * elev_diff
            all_stations.append({"file": f, "station": f.stem.replace("_full_2020_2025_master", ""),
                                 "distance_km": round(dist,1), "elev_diff_m": round(elev_diff), "score": round(score,1)})
        except: continue

    ranked = sorted(all_stations, key=lambda x: x["score"])[:n_neighbors]
    logger.info(f"\n=== Top {n_neighbors} Most Similar Stations to {target_station} ===")
    for i, s in enumerate(ranked, 1):
        logger.info(f"{i:2d}. {s['station']:28s} | {s['distance_km']:6.1f} km | {s['elev_diff_m']:4.0f} m | score: {s['score']}")
    return ranked


def load_30min_data(target_station, ranked_neighbors):
    target_file = list(MASTER_DIR.glob(f"*{target_station}*_full_2020_2025_master.parquet"))[0]
    target = pd.read_parquet(target_file)

    available = ["precipitation"]
    for col in ["temperature", "relative_humidity", "wind_speed"]:
        if col in target.columns:
            available.append(col)

    df = target.set_index("timestamp")[available]

    for n in ranked_neighbors:
        try:
            neigh = pd.read_parquet(n["file"], columns=["timestamp", "precipitation"]).set_index("timestamp")
            df[f"precip_{n['station']}"] = neigh["precipitation"]
        except: continue

    return df.dropna()


def create_advanced_features(df):
    df = df.sort_index()

    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f"precip_lag_{lag}"] = df["precipitation"].shift(lag)
    for lag in [7, 14]:
        df[f"precip_lag_{lag*48}"] = df["precipitation"].shift(lag*48)

    for col in [c for c in df.columns if c.startswith("precip_") and c != "precipitation"]:
        for lag in [2, 6, 12, 24, 48]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    for window in [6, 12, 24, 48, 168]:
        df[f"precip_roll_sum_{window}"] = df["precipitation"].shift(1).rolling(window).sum()
        df[f"precip_roll_mean_{window}"] = df["precipitation"].shift(1).rolling(window).mean()

    df["hour"] = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear

    df["target"] = df["precipitation"].shift(-1)
    return df.dropna()


def time_series_cross_validation(X, y, n_splits=N_SPLITS):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=30, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        scores.append(rmse)
        logger.info(f"  Fold {fold+1}: RMSE = {rmse:.3f}")
    return np.mean(scores)


def final_evaluation(X_train, y_train, X_test, y_test, test_start, test_end):
    model = XGBRegressor(
        n_estimators=400, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=40, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"\n=== Final Results ({test_start} to {test_end}) ===")
    logger.info(f"RMSE: {rmse:.3f} mm/30min")
    logger.info(f"MAE:  {mae:.3f} mm/30min")
    logger.info(f"RÂ²:   {r2:.4f}")

    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index, y_test, label="Actual", alpha=0.7, linewidth=1)
    plt.plot(y_test.index, y_pred, label="XGBoost Predicted", alpha=0.7, linewidth=1)
    plt.title(f"XGBoost 30-min Precipitation Forecast\n{test_start} to {test_end} | RMSE = {rmse:.3f} mm/30min | RÂ² = {r2:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"xgb_precip_period_{TARGET_STATION}_{test_start}_{test_end}.png", dpi=300)
    plt.close()

    return model


def main():
    logger.info("=== ADVANCED PRECIPITATION MODEL v3.2 (Custom Period) ===")
    logger.info(f"Target: {TARGET_STATION}")
    logger.info(f"Test period: {TEST_START} to {TEST_END}")

    ranked = build_station_hierarchy(TARGET_STATION)
    df = load_30min_data(TARGET_STATION, ranked)
    featured = create_advanced_features(df)

    # Custom period split
    test_mask = (featured.index >= TEST_START) & (featured.index <= TEST_END)
    train_mask = ~test_mask

    train = featured[train_mask]
    test = featured[test_mask]

    feature_cols = [c for c in featured.columns if c not in ["target", "precipitation", "hour"]]
    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    logger.info(f"\nTraining samples: {len(train):,} | Test samples: {len(test):,}")

    logger.info("\n=== Time-Series Cross-Validation ===")
    cv_rmse = time_series_cross_validation(X_train, y_train)
    logger.info(f"Mean CV RMSE: {cv_rmse:.3f}")

    model = final_evaluation(X_train, y_train, X_test, y_test, TEST_START, TEST_END)

    logger.info("=== FINISHED ===")


if __name__ == "__main__":
    main()
