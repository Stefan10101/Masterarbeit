#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temperature Forecasting â€“ Single Station v2.1 (Fixed)
=====================================================
Predict next 30-min temperature using Random Forest.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime
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
STATION_NAME = "Sonnblick"          # â† Change station here
TEST_MONTH   = "2023-07"            # â† Change month here
N_ESTIMATORS = 150
RANDOM_STATE = 42

LOG_FILE = PLOT_DIR / f"rf_{STATION_NAME}_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


def create_features(df):
    df = df.sort_values("timestamp").set_index("timestamp")

    # Drop non-numeric columns that may cause errors
    df = df.drop(columns=["station_name", "name", "parameter", "source_file", "is_missing"], errors="ignore")

    # Lags
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"temp_lag_{lag}"] = df["temperature"].shift(lag)
        df[f"rh_lag_{lag}"] = df["relative_humidity"].shift(lag)
        df[f"wind_lag_{lag}"] = df["wind_speed"].shift(lag)
        df[f"precip_lag_{lag}"] = df["precipitation"].shift(lag)

    # Rolling stats
    for w in [6, 24]:
        df[f"temp_roll_mean_{w}"] = df["temperature"].rolling(w).mean()
        df[f"temp_roll_std_{w}"] = df["temperature"].rolling(w).std()

    # Time features
    df["hour"] = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Target
    df["target"] = df["temperature"].shift(-1)

    return df.dropna()


def load_single_station(station_name):
    files = list(MASTER_DIR.glob(f"*{station_name}*_full_2020_2025_master.parquet"))
    if not files:
        raise FileNotFoundError(f"No file found for: {station_name}")

    file_path = files[0]
    logger.info(f"Loading: {file_path.name}")

    df = pd.read_parquet(file_path)

    required = ["timestamp", "temperature", "relative_humidity", "wind_speed", "precipitation"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


def train_and_evaluate(df, test_month):
    featured = create_features(df)
    logger.info(f"Samples after feature engineering: {len(featured):,}")

    # Train / Test split
    train_mask = ~featured.index.to_series().dt.strftime("%Y-%m").isin([test_month])
    train = featured[train_mask]
    test = featured[~train_mask]

    logger.info(f"Training samples: {len(train):,}")
    logger.info(f"Test samples ({test_month}): {len(test):,}")

    if len(test) == 0:
        raise ValueError(f"No data for month {test_month}")

    # Only numeric features
    feature_cols = [c for c in featured.columns 
                    if c not in ["target", "hour"] 
                    and pd.api.types.is_numeric_dtype(featured[c])]

    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    # Model
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=18,
        min_samples_split=8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    logger.info("Training Random Forest...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"\n=== Results for {test_month} ===")
    logger.info(f"RMSE: {rmse:.3f} Â°C")
    logger.info(f"MAE:  {mae:.3f} Â°C")
    logger.info(f"RÂ²:   {r2:.4f}")

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    logger.info("\nTop 10 most important features:")
    logger.info(importance.head(10))

    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index, y_test, label="Actual", alpha=0.75, linewidth=1.2)
    plt.plot(y_test.index, y_pred, label="Predicted", alpha=0.75, linewidth=1.2)
    plt.title(f"Random Forest 30-min Temperature Forecast\n{STATION_NAME} â€“ {test_month} | RMSE = {rmse:.2f}Â°C | RÂ² = {r2:.3f}")
    plt.xlabel("Time")
    plt.ylabel("Temperature (Â°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"rf_{STATION_NAME}_{test_month}.png", dpi=300)
    plt.close()

    logger.info(f"Plot saved: rf_{STATION_NAME}_{test_month}.png")

    return model, importance


def main():
    logger.info("=== RANDOM FOREST SINGLE STATION v2.1 (Fixed) ===")
    logger.info(f"Station: {STATION_NAME}")
    logger.info(f"Test month: {TEST_MONTH}")

    df = load_single_station(STATION_NAME)
    model, importance = train_and_evaluate(df, TEST_MONTH)

    logger.info("=== FINISHED ===")


if __name__ == "__main__":
    main()
