#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wind Speed Final Cleaner (Parallel) - v1.1 - PIPELINE v3 ADAPTED
================================================================
Lightly adapted for new pipeline output structure (Full_2020-2025 nested folders + "height" column).
All original QC logic, thresholds and algorithms 100% preserved.
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

VARIABLE_SUFFIX = "_wind"

INPUT_FULL_DIR = Path(DATA_ROOT / "wind" / "data" / "data" / "full_2020-2025")
PAKET_ROOT     = Path(DATA_ROOT / "wind" / "paket")
OUTPUT_QC_DIR  = PAKET_ROOT / "full_2020_2025_qc"
OUTPUT_QC_DIR.mkdir(parents=True, exist_ok=True)

FREQ_MINUTES      = 30
STEPS_PER_HOUR    = 60 // FREQ_MINUTES

HARD_MIN          = 0.0
HARD_MAX          = 100.0

SPIKE_THRESHOLD   = 18.0
STUCK_HOURS       = 48.0
STUCK_TOL         = 0.3
ICING_HOURS       = 8.0
ICING_STD_MAX     = 1.0

ELEV_BANDS        = [0, 500, 1000, 1500, 2000, 4000]
BAND_SOFT_MAX     = {
    0: 38.0,
    1: 45.0,
    2: 52.0,
    3: 60.0,
    4: 68.0
}

LOG_FILE = PAKET_ROOT / f"wind_final_cleaner_v1.1_pipeline3_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


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


# ==================== STEP 3 HELPER (unchanged) ====================
def get_elev_band(hoehe: float) -> int:
    if pd.isna(hoehe):
        return 2
    for i in range(len(ELEV_BANDS) - 1):
        if ELEV_BANDS[i] <= hoehe < ELEV_BANDS[i + 1]:
            return i
    return len(ELEV_BANDS) - 2


# ==================== UPDATED QC FUNCTION (unchanged logic) ====================
def apply_wind_qc(df: pd.DataFrame, logger=None, station_name: str = "") -> pd.DataFrame:
    if df.empty or "value" not in df.columns:
        return df

    # === ROBUST TIMESTAMP (ported from snowheight v2.38) ===
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

    # === STEP 1: Data integrity & completeness ===
    if len(df) > 1:
        diffs = df.index.to_series().diff().dt.total_seconds() / 60
        gap_mask = (diffs > FREQ_MINUTES * 1.5) & (diffs < 60 * 24)
        n_gaps = int(gap_mask.sum())
        if logger and n_gaps > 0:
            logger.debug(f"  [{station_name}] {n_gaps} gaps detected (not removed)")

    original_valid = int(df["value"].notna().sum())
    if original_valid == 0:
        return df.reset_index() if isinstance(df.index, pd.DatetimeIndex) else df

    flagged = {"range": 0, "clim": 0, "spike": 0, "icing": 0, "stuck": 0}

    # === STEP 2: Gross range / physical limits ===
    hard_mask = (df["value"] < HARD_MIN) | (df["value"] > HARD_MAX)
    flagged["range"] = int(hard_mask.sum())
    df.loc[hard_mask, "value"] = np.nan

    # === STEP 3: Climatological soft limits â€” per elevation band ===
    hoehe = float(df["hoehe"].iloc[0]) if "hoehe" in df.columns else np.nan
    band = get_elev_band(hoehe)
    soft_max = BAND_SOFT_MAX[band]
    clim_mask = (df["value"] > soft_max) & df["value"].notna()
    flagged["clim"] = int(clim_mask.sum())
    df.loc[clim_mask, "value"] = np.nan

    # === STEP 6: Spike / rate-of-change ===
    df["diff"] = df["value"].diff().abs()
    spike_mask = (df["diff"] > SPIKE_THRESHOLD) & df["value"].notna()
    flagged["spike"] = int(spike_mask.sum())
    df.loc[spike_mask, "value"] = np.nan

    # === STEP 7: Stuck / flat-line ===
    min_stuck_steps = int(STUCK_HOURS * STEPS_PER_HOUR)
    is_small_change = (df["value"].diff().abs() < STUCK_TOL).fillna(False).astype(int)
    df["run_id"] = (is_small_change.diff() != 0).cumsum()
    run_len = df.groupby("run_id")["run_id"].transform("size")
    stuck_in_run = (is_small_change == 1) & (run_len >= min_stuck_steps)
    long_run_ids = df.loc[stuck_in_run, "run_id"].unique()
    stuck_mask = df["run_id"].isin(long_run_ids) & df["value"].notna()
    flagged["stuck"] = int(stuck_mask.sum())
    df.loc[stuck_mask, "value"] = np.nan

    # === STEP 5: Variability / icing-frozen sensor (only >2000 m) ===
    if hoehe > 2000:
        rolling_mean = df["value"].rolling(window=12, min_periods=6).mean()
        min_icing_steps = int(ICING_HOURS * STEPS_PER_HOUR)
        df["roll_std"] = df["value"].rolling(min_icing_steps, min_periods=min_icing_steps // 2).std()
        icing_mask = (df["roll_std"] < ICING_STD_MAX) & (rolling_mean < 1.0) & df["value"].notna()
        flagged["icing"] = int(icing_mask.sum())
        df.loc[icing_mask, "value"] = np.nan
    else:
        flagged["icing"] = 0

    # === MICRO-GAP INTERPOLATION (after all QC) ===
    if len(df) > 0:
        df, n_gaps, n_pts = interpolate_micro_gaps(df, max_gap_hours=2.0, col="value",
                                                   logger=logger, station_name=station_name)
        if n_pts > 0 and logger:
            logger.info(f"  [{station_name}] Micro-gap interpolation: {n_gaps} gaps filled, {n_pts} points interpolated (<=2h)")

    # Cleanup
    temp_cols = [c for c in df.columns if c.startswith(("diff", "run_id", "roll_std"))]
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
        logger.info(f"QC v1.1 | flagged {total_flagged} ({pct:.1f}%) | "
                    f"range={flagged['range']}, clim(band {band})={flagged['clim']}, "
                    f"spike={flagged['spike']}, icing={flagged['icing']}, stuck={flagged['stuck']}")

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

        df_clean = apply_wind_qc(df, logger=logger, station_name=station)

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
                "qc_version": "v1.1_pipeline3 (dedup + micro-gap + robust ts; wind: elev-grouped clim + QARTOD/MeteoSwiss logic)",
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
    logger.info("=== WIND SPEED FINAL CLEANER v1.1 (pipeline v3 adapted) START ===")
    logger.info(f"Source: {INPUT_FULL_DIR}  ->  Destination: {OUTPUT_QC_DIR}")
    logger.info(f"Thresholds (UNCHANGED): hard {HARD_MIN}â€“{HARD_MAX} m/s | spike >{SPIKE_THRESHOLD} m/s | "
                f"stuck â‰¥{STUCK_HOURS}h (tol {STUCK_TOL} m/s) | icing std <{ICING_STD_MAX} (only >2000 m + wind <1 m/s) | "
                f"elev bands: {ELEV_BANDS} â†’ soft_max {BAND_SOFT_MAX}")
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

    logger.info("=== WIND SPEED FINAL CLEANER v1.1 (pipeline v3) FINISHED ===")
    logger.info(f"Output: {OUTPUT_QC_DIR}")
    logger.info(f"Log: {LOG_FILE}")


if __name__ == "__main__":
    main()

