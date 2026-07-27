#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Precipitation Forecasting â€“ Global Station Hierarchy v2.0
===============================================================
Creates a similarity ranking of all precipitation stations
based on distance + elevation difference.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
TARGET_STATION = "Patscherkofel"
TEST_MONTH = "2024-01"
N_NEIGHBORS = 20                    # Top 10 most similar stations
N_ESTIMATORS = 200
RANDOM_STATE = 42

LOG_FILE = PLOT_DIR / f"rf_precip_hierarchy_{TARGET_STATION}_{datetime.now():%Y%m%d_%H%M%S}.log"
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
    """Build global similarity ranking of all precipitation stations."""
    target_file = list(MASTER_DIR.glob(f"*{target_station}*_full_2020_2025_master.parquet"))[0]
    target_meta = pd.read_parquet(target_file, columns=["lat", "lon", "hoehe"])
    t_lat, t_lon, t_elev = target_meta.iloc[0][["lat", "lon", "hoehe"]]

    all_stations = []
    for f in MASTER_DIR.glob("*_full_2020_2025_master.parquet"):
        if target_station in f.name:
            continue
        try:
            meta = pd.read_parquet(f, columns=["lat", "lon", "hoehe", "precipitation"])
            if "precipitation" not in meta.columns or meta["precipitation"].isna().all():
                continue

            lat, lon, elev = meta.iloc[0][["lat", "lon", "hoehe"]]
            dist = haversine(t_lat, t_lon, lat, lon)
            elev_diff = abs(t_elev - elev)
            score = 0.6 * dist + 0.4 * elev_diff     # weighted similarity

            all_stations.append({
                "file": f,
                "station": f.stem.replace("_full_2020_2025_master", ""),
                "distance_km": round(dist, 1),
                "elev_diff_m": round(elev_diff),
                "similarity_score": round(score, 1)
            })
        except:
            continue

    # Sort by similarity (lower score = better)
    ranked = sorted(all_stations, key=lambda x: x["similarity_score"])[:n_neighbors]

    logger.info(f"\n=== Station Hierarchy for {target_station} (Top {n_neighbors}) ===")
    for i, s in enumerate(ranked, 1):
        logger.info(f"{i:2d}. {s['station']:30s} | {s['distance_km']:5.1f} km | {s['elev_diff_m']:4.0f} m elev | score: {s['similarity_score']}")

    return ranked


def load_and_prepare_data(target_station, ranked_neighbors):
    """Load target + top neighbors and create features."""
    target_file = list(MASTER_DIR.glob(f"*{target_station}*_full_2020_2025_master.parquet"))[0]
    target = pd.read_parquet(target_file)
    daily = target.set_index("timestamp").resample("D").agg({
        "precipitation": "sum",
        "temperature": "mean",
        "relative_humidity": "mean",
        "wind_speed": "mean"
    }).dropna()

    # Target lags
    for lag in [1, 2, 3]:
        daily[f"target_precip_lag_{lag}"] = daily["precipitation"].shift(lag)

    # Add neighbors
    for n in ranked_neighbors:
        try:
            neigh = pd.read_parquet(n["file"], columns=["timestamp", "precipitation"])
            neigh_daily = neigh.set_index("timestamp").resample("D")["precipitation"].sum()
            daily[f"precip_{n['station']}"] = neigh_daily
            for lag in [1, 2, 3]:
                daily[f"precip_{n['station']}_lag_{lag}"] = neigh_daily.shift(lag)
        except:
            continue

    daily["month"] = daily.index.month
    daily["day_of_year"] = daily.index.dayofyear
    daily["target"] = daily["precipitation"]

    return daily.dropna()


def train_and_evaluate(df, test_month):
    train_mask = ~df.index.to_series().dt.strftime("%Y-%m").isin([test_month])
    train = df[train_mask]
    test = df[~train_mask]

    logger.info(f"\nTraining days: {len(train):,} | Test days: {len(test):,}")

    feature_cols = [c for c in df.columns if c not in ["target", "precipitation"] 
                    and pd.api.types.is_numeric_dtype(df[c])]

    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=16,
        min_samples_split=6,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    logger.info("Training model with spatial hierarchy features...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"\n=== Results for {test_month} ===")
    logger.info(f"RMSE: {rmse:.2f} mm/day")
    logger.info(f"MAE:  {mae:.2f} mm/day")
    logger.info(f"RÂ²:   {r2:.4f}")

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    logger.info("\nTop 15 features:")
    logger.info(importance.head(15))

    plt.figure(figsize=(14, 6))
    plt.bar(y_test.index, y_test, alpha=0.6, label="Actual", color="steelblue")
    plt.plot(y_test.index, y_pred, marker="o", alpha=0.8, label="Predicted", color="darkorange")
    plt.title(f"Global Hierarchy RF Daily Precipitation\n{TEST_MONTH} | RMSE = {rmse:.2f} mm/day")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"rf_precip_hierarchy_{TARGET_STATION}_{TEST_MONTH}.png", dpi=300)
    plt.close()

    logger.info("Plot saved.")
    return model, importance


def main():
    logger.info("=== GLOBAL STATION HIERARCHY MODEL v2.0 ===")
    logger.info(f"Target: {TARGET_STATION} | Test month: {TEST_MONTH}")

    ranked = build_station_hierarchy(TARGET_STATION)
    df = load_and_prepare_data(TARGET_STATION, ranked)
    model, importance = train_and_evaluate(df, TEST_MONTH)

    logger.info("=== FINISHED ===")


if __name__ == "__main__":
    main()
