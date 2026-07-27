#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temperature Final Cleaner (Parallel) - v1.2 - PIPELINE v3 ADAPTED
=================================================================
Lightly adapted for new pipeline output structure (Full_2020-2025 nested folders + "height" column).
All original QC logic, thresholds, coordinate swap fix and algorithms 100% preserved.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import re
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


# ==================== CONFIG (adapted for pipeline v3) ====================

VARIABLE_SUFFIX = "_temp"

INPUT_FULL_DIR = Path(DATA_ROOT / "temperatur" / "data" / "data" / "full_2020-2025")
PAKET_ROOT     = Path(DATA_ROOT / "temperatur" / "paket")
OUTPUT_QC_DIR  = PAKET_ROOT / "full_2020_2025_qc"
OUTPUT_QC_DIR.mkdir(parents=True, exist_ok=True)

FREQ_MINUTES      = 30
STEPS_PER_HOUR    = 60 // FREQ_MINUTES

HARD_MIN          = -50.0
HARD_MAX          = 50.0
SPIKE_THRESHOLD   = 6.0
STUCK_HOURS       = 24.0
STUCK_TOL         = 0.1

ELEV_BANDS        = [0, 500, 1000, 1500, 2000, 4000]
BAND_SOFT_MAX     = {0: 40.0, 1: 37.0, 2: 34.0, 3: 30.0, 4: 25.0, 5: 25.0}
BAND_SOFT_MIN     = {0: -25.0, 1: -28.0, 2: -32.0, 3: -38.0, 4: -45.0, 5: -45.0}

# Logging must be set up BEFORE any logger calls (including the coordinate swap loader)
LOG_FILE = PAKET_ROOT / f"temperature_final_cleaner_v1.2_pipeline3_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# COORDINATE SWAP FIX (kept 100% unchanged - update path if report moved)
SWAP_REPORT_PATH = Path(DATA_ROOT / "temperatur" / "coordinate_swap_report.csv")

def load_swapped_coordinates():
    if not SWAP_REPORT_PATH.exists():
        logger.warning("coordinate_swap_report.csv not found")
        return []
    df = pd.read_csv(SWAP_REPORT_PATH)
    swapped_pairs = []
    for _, row in df.iterrows():
        if row["swapped"]:
            swapped_pairs.append((float(row["original_lat"]), float(row["original_lon"])))
    logger.info(f"Loaded {len(swapped_pairs)} swapped coordinate pairs")
    return swapped_pairs

SWAPPED_COORDINATES = load_swapped_coordinates()


# ==================== MICRO-GAP INTERPOLATION (unchanged) ====================
def interpolate_micro_gaps(df, max_gap_hours=2.0, col="value", logger=None, station_name=""):
    if df.empty or col not in df.columns or len(df) < 2:
        return df, 0, 0
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    s = df[col]
    valid_mask = s.notna()
    if valid_mask.sum() < 2:
        return df, 0, 0
    valid_df = df[valid_mask].copy()
    valid_times = valid_df.index
    new_dfs = []
    gaps_filled = 0
    points_interpolated = 0
    for i in range(len(valid_times) - 1):
        t1 = valid_times[i]
        t2 = valid_times[i + 1]
        delta_h = (t2 - t1).total_seconds() / 3600.0
        if not (0.5 < delta_h <= max_gap_hours):
            continue
        n_missing = int(round(delta_h / 0.5)) - 1
        if n_missing < 1 or n_missing > 3:
            continue
        v1 = valid_df.loc[t1, col]
        v2 = valid_df.loc[t2, col]
        if pd.isna(v1) or pd.isna(v2):
            continue
        interp_times = pd.date_range(start=t1 + pd.Timedelta("30min"), periods=n_missing, freq="30min")
        interp_times = interp_times[interp_times < t2]
        if len(interp_times) == 0:
            continue
        interp_vals = np.linspace(v1, v2, len(interp_times) + 2)[1:-1]
        temp_df = pd.DataFrame({col: interp_vals}, index=interp_times)
        for c in valid_df.columns:
            if c != col:
                temp_df[c] = valid_df.loc[t1, c]
        if "is_missing" in temp_df.columns:
            temp_df["is_missing"] = False
        new_dfs.append(temp_df)
        gaps_filled += 1
        points_interpolated += len(interp_times)
    if new_dfs:
        interp_df = pd.concat(new_dfs)
        combined = pd.concat([valid_df, interp_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        if logger:
            logger.info(f"  [{station_name}] Micro-gap interpolation: {gaps_filled} gaps filled, {points_interpolated} points interpolated (<=2h)")
        return combined, gaps_filled, points_interpolated
    return df, 0, 0


# ==================== DEDUP HELPER (unchanged) ====================
def get_station_coords(pq_path):
    try:
        df = pd.read_parquet(pq_path, columns=["lat", "lon"])
        return (round(df["lat"].iloc[0], 4), round(df["lon"].iloc[0], 4))
    except:
        return None


# ==================== HELPER (unchanged) ====================
def get_elev_band(hoehe: float) -> int:
    if pd.isna(hoehe):
        return 2
    for i in range(len(ELEV_BANDS) - 1):
        if ELEV_BANDS[i] <= hoehe < ELEV_BANDS[i + 1]:
            return i
    return len(ELEV_BANDS) - 2


# ==================== UPDATED QC FUNCTION (unchanged logic) ====================
def apply_temperature_qc(df: pd.DataFrame, logger=None, station_name: str = "") -> pd.DataFrame:
    if df.empty or "value" not in df.columns:
        return df

    # === ROBUST TIMESTAMP ===
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    elif isinstance(df.index, pd.DatetimeIndex):
        pass
    else:
        for col in ["time", "datetime", "DateTime", "obs_time"]:
            if col in df.columns:
                df = df.set_index(col)
                break
        else:
            if logger:
                logger.error(f"  [{station_name}] No timestamp column/index found")
            return df

    df.index = pd.to_datetime(df.index, utc=True).sort_values()
    df = df.dropna(subset=["value"])
    if df.empty:
        return df.reset_index() if isinstance(df.index, pd.DatetimeIndex) else df

    # === STEP 1: Data integrity ===
    if len(df) > 1:
        diffs = df.index.to_series().diff().dt.total_seconds() / 60
        gap_mask = (diffs > FREQ_MINUTES * 1.5) & (diffs < 60 * 24)
        n_gaps = int(gap_mask.sum())
        if logger and n_gaps > 0:
            logger.debug(f"  [{station_name}] {n_gaps} gaps detected")

    original_valid = int(df["value"].notna().sum())
    if original_valid == 0:
        return df.reset_index() if isinstance(df.index, pd.DatetimeIndex) else df

    flagged = {"range": 0, "clim": 0, "spike": 0, "stuck": 0}

    # Step 2: Hard range
    hard_mask = (df["value"] < HARD_MIN) | (df["value"] > HARD_MAX)
    flagged["range"] = int(hard_mask.sum())
    df.loc[hard_mask, "value"] = np.nan

    # Step 3: Climatological limits per elevation band
    hoehe = float(df["hoehe"].iloc[0]) if "hoehe" in df.columns else np.nan
    band = get_elev_band(hoehe)
    soft_max = BAND_SOFT_MAX[band]
    soft_min = BAND_SOFT_MIN[band]
    clim_mask = ((df["value"] > soft_max) | (df["value"] < soft_min)) & df["value"].notna()
    flagged["clim"] = int(clim_mask.sum())
    df.loc[clim_mask, "value"] = np.nan

    # Step 4: Spike
    df["diff"] = df["value"].diff().abs()
    spike_mask = (df["diff"] > SPIKE_THRESHOLD) & df["value"].notna()
    flagged["spike"] = int(spike_mask.sum())
    df.loc[spike_mask, "value"] = np.nan

    # Step 5: Stuck
    min_stuck_steps = int(STUCK_HOURS * STEPS_PER_HOUR)
    is_small_change = (df["value"].diff().abs() < STUCK_TOL).fillna(False).astype(int)
    df["run_id"] = (is_small_change.diff() != 0).cumsum()
    run_len = df.groupby("run_id")["run_id"].transform("size")
    stuck_in_run = (is_small_change == 1) & (run_len >= min_stuck_steps)
    long_run_ids = df.loc[stuck_in_run, "run_id"].unique()
    stuck_mask = df["run_id"].isin(long_run_ids) & df["value"].notna()
    flagged["stuck"] = int(stuck_mask.sum())
    df.loc[stuck_mask, "value"] = np.nan

    # === MICRO-GAP INTERPOLATION (after all QC) ===
    if len(df) > 0:
        df, n_gaps, n_pts = interpolate_micro_gaps(df, max_gap_hours=2.0, col="value",
                                                   logger=logger, station_name=station_name)
        if n_pts > 0 and logger:
            logger.info(f"  [{station_name}] Micro-gap interpolation: {n_gaps} gaps filled, {n_pts} points interpolated (<=2h)")

    # Cleanup
    temp_cols = [c for c in df.columns if c.startswith(("diff", "run_id"))]
    df = df.drop(columns=temp_cols, errors="ignore")

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    if "timestamp" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "timestamp"})

    if "is_missing" in df.columns:
        df["is_missing"] = df["value"].isna()

    total_flagged = sum(flagged.values())
    if logger is not None and total_flagged > 0:
        pct = total_flagged / original_valid * 100
        logger.info(f"QC v1.2 | flagged {total_flagged} ({pct:.1f}%) | "
                    f"range={flagged['range']}, clim(band {band})={flagged['clim']}, "
                    f"spike={flagged['spike']}, stuck={flagged['stuck']}")

    return df


# ==================== WORKER ====================
def process_one_station(args):
    pq_path, output_dir, output_suffix = args
    try:
        df = pd.read_parquet(pq_path)

        # === COLUMN NORMALIZATION for pipeline v3 ("height" -> "hoehe") ===
        if "height" in df.columns:
            df = df.rename(columns={"height": "hoehe"})

        station = df["station"].iloc[0] if "station" in df.columns else pq_path.stem.split("_")[0]

        # === COORDINATE SWAP FIX (kept 100% unchanged) ===
        if "lat" in df.columns and "lon" in df.columns:
            current_lat = float(df["lat"].iloc[0])
            current_lon = float(df["lon"].iloc[0])
            for orig_lat, orig_lon in SWAPPED_COORDINATES:
                if (abs(current_lat - orig_lat) < 0.0005 and abs(current_lon - orig_lon) < 0.0005):
                    df["lat"] = orig_lon
                    df["lon"] = orig_lat
                    logger.info(f"  â†’ Swapped coordinates for station (detected by coords): "
                                f"{current_lat:.4f}, {current_lon:.4f} â†’ {orig_lon:.4f}, {orig_lat:.4f}")
                    break

        df_clean = apply_temperature_qc(df, logger=logger, station_name=station)

        # === FINAL SAFEGUARD ===
        if "index" in df_clean.columns and "timestamp" not in df_clean.columns:
            df_clean = df_clean.rename(columns={"index": "timestamp"})
        if "timestamp" not in df_clean.columns:
            df_clean = df_clean.reset_index().rename(columns={"index": "timestamp"})

        safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', station)
        out_path = output_dir / f"{safe_name}{output_suffix}"

        if not df_clean.empty:
            cols = ["timestamp", "value", "is_missing", "station", "name", "hoehe", "lat", "lon", "parameter", "source_file"]
            existing_cols = [c for c in cols if c in df_clean.columns]
            df_clean[existing_cols].to_parquet(out_path.with_suffix(".parquet"), compression="snappy", index=False)

            json_data = {
                "name": str(df_clean["name"].iloc[0]) if "name" in df_clean.columns else station,
                "hoehe": float(df_clean["hoehe"].iloc[0]) if "hoehe" in df_clean.columns and pd.notna(df_clean["hoehe"].iloc[0]) else None,
                "lat": float(df_clean["lat"].iloc[0]) if "lat" in df_clean.columns and pd.notna(df_clean["lat"].iloc[0]) else None,
                "lon": float(df_clean["lon"].iloc[0]) if "lon" in df_clean.columns and pd.notna(df_clean["lon"].iloc[0]) else None,
                "qc_version": "v1.2_pipeline3 (dedup + micro-gap + robust ts; temperature + coordinate correction)",
                "data": [{"timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                          "value": float(v) if pd.notna(v) else None}
                         for ts, v in zip(df_clean["timestamp"], df_clean["value"])]
            }
            with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

        return f"OK: {station}"
    except Exception as e:
        return f"ERROR on {pq_path.name}: {e}"


# ==================== MAIN ====================
def main():
    logger.info("=== TEMPERATURE FINAL CLEANER v1.2 (pipeline v3 adapted) START ===")
    logger.info(f"Source: {INPUT_FULL_DIR}  ->  Destination: {OUTPUT_QC_DIR}")
    logger.info(f"Parallel workers: {mp.cpu_count() - 1}")

    if not INPUT_FULL_DIR.exists():
        logger.error(f"Folder not found: {INPUT_FULL_DIR}")
        return

    all_files = list(INPUT_FULL_DIR.rglob(f"*{VARIABLE_SUFFIX}.parquet"))
    logger.info(f"Scanning {len(all_files)} raw files for coordinate duplicates...")

    coord_to_files = {}
    for f in all_files:
        coords = get_station_coords(f)
        if coords:
            if coords not in coord_to_files:
                coord_to_files[coords] = []
            coord_to_files[coords].append(f)

    duplicates_found = 0
    for coords, files in coord_to_files.items():
        if len(files) > 1:
            duplicates_found += len(files) - 1
            logger.info(f"  DUPLICATE at {coords}: {len(files)} files -> keeping first")

    files_to_process = [files[0] for files in coord_to_files.values()]
    logger.info(f"After deduplication: {len(files_to_process)} unique stations (removed {duplicates_found} duplicates)")

    logger.info(f"Processing {len(files_to_process)} unique stations in parallel...")

    tasks = [(f, OUTPUT_QC_DIR, "_final_qc") for f in files_to_process]
    with mp.Pool(mp.cpu_count() - 1) as pool:
        results = pool.map(process_one_station, tasks)

    for r in results:
        if "ERROR" in r:
            logger.error(r)

    logger.info("=== TEMPERATURE FINAL CLEANER v1.2 (pipeline v3) FINISHED ===")
    logger.info(f"Output: {OUTPUT_QC_DIR}")
    logger.info(f"Log: {LOG_FILE}")


if __name__ == "__main__":
    main()

