#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summer Grass Enrichment v1.0
Adds three key diagnostic attributes to the 180 problematic stations:
- max_winter_depth (cm)
- summer_zero_rate (%)
- num_summer_jumps_10cm
Plus a first automatic "likely_sensor_error" flag
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import multiprocessing as mp

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT


# ==================== CONFIG ====================
INPUT_CSV = Path(DATA_ROOT / "schneehoehe" / "paket" / "analysis_summer_grass_stations.csv")
INPUT_DIR = Path(DATA_ROOT / "schneehoehe" / "paket" / "full_2020_2025")
OUTPUT_CSV = Path(DATA_ROOT / "schneehoehe" / "paket" / "analysis_summer_grass_enriched.csv")

YEARS = list(range(2020, 2026))
SUMMER_START = {"month": 7, "day": 15}
SUMMER_END   = {"month": 9, "day": 15}
WINTER_MONTHS = [12, 1, 2, 3]

N_CORES = max(1, mp.cpu_count() - 1)

LOG_FILE = INPUT_DIR.parent / f"summer_grass_enrichment_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


def compute_diagnostics(args):
    row, input_dir = args
    station = row["station"]
    pq_path = input_dir / f"{station}_full_2020_2025_final.parquet"

    if not pq_path.exists():
        # try alternative naming
        pq_path = input_dir / f"{station.replace(' ', '_')}_full_2020_2025_final.parquet"
        if not pq_path.exists():
            return None

    try:
        df = pd.read_parquet(pq_path, columns=["timestamp", "value"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        # 1. max_winter_depth
        winter_mask = df.index.month.isin(WINTER_MONTHS)
        max_winter = float(df.loc[winter_mask, "value"].max()) if winter_mask.any() else np.nan

        # 2. summer_zero_rate + 3. num_summer_jumps_10cm
        total_summer_points = 0
        zero_summer_points = 0
        jumps = 0

        for y in YEARS:
            start = pd.Timestamp(year=y, month=SUMMER_START["month"], day=SUMMER_START["day"], tz="UTC")
            end   = pd.Timestamp(year=y, month=SUMMER_END["month"],   day=SUMMER_END["day"],   tz="UTC")

            summer = df.loc[(df.index >= start) & (df.index <= end), "value"]
            if len(summer) == 0:
                continue

            total_summer_points += len(summer)
            zero_summer_points += int((summer < 0.5).sum())   # allow tiny sensor noise

            # daily max jumps
            daily_max = summer.resample("D").max()
            daily_diff = daily_max.diff().abs()
            jumps += int((daily_diff > 10).sum())

        summer_zero_rate = (zero_summer_points / total_summer_points * 100) if total_summer_points > 0 else 0.0

        # simple first flag
        likely_sensor_error = (summer_zero_rate < 80) or (jumps > 5)

        return {
            "station": station,
            "max_winter_depth_cm": round(max_winter, 1) if pd.notna(max_winter) else None,
            "summer_zero_rate_%": round(summer_zero_rate, 1),
            "num_summer_jumps_10cm": jumps,
            "likely_sensor_error": likely_sensor_error
        }

    except Exception as e:
        logger.error(f"ERROR on {station}: {e}")
        return None


def main():
    logger.info("=== SUMMER GRASS ENRICHMENT v1.0 ===")

    df_prob = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(df_prob)} problematic stations")

    tasks = [(row, INPUT_DIR) for _, row in df_prob.iterrows()]

    with mp.Pool(N_CORES) as pool:
        results = pool.map(compute_diagnostics, tasks)

    enriched = [r for r in results if r is not None]
    df_enriched = pd.DataFrame(enriched)

    # merge back with original info
    df_final = df_prob.merge(df_enriched, on="station", how="left")

    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logger.info(f"Enriched file saved: {OUTPUT_CSV}")

    # quick summary
    n_error = df_final["likely_sensor_error"].sum()
    logger.info(f"Stations flagged as likely sensor error: {n_error} / {len(df_final)}")
    logger.info("Top 10 by summer non-zero points with new diagnostics:")
    print(df_final.sort_values("summer_nonzero_points", ascending=False).head(10)[
        ["station", "hoehe_m", "summer_nonzero_points", "max_winter_depth_cm",
         "summer_zero_rate_%", "num_summer_jumps_10cm", "likely_sensor_error"]
    ].to_string(index=False))

    logger.info("=== ENRICHMENT COMPLETE ===")
    logger.info("Open the new CSV and manually check the ~10 highest stations (Rettenbach, Sonnblick, etc.) â€” set likely_sensor_error = False for true permanent snow sites.")


if __name__ == "__main__":
    main()
