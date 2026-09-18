#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precipitation Final Cleaner (Parallel) - v1.2.1 - SPATIAL ZERO-MONTH QC (FIXED)
===============================================================================
- All v1.1 logic preserved (hard range, spike, stuck-non-zero, micro-gap interp, dedup).
- NEW: Spatial neighbour validation for suspicious zero months (exactly as described).
  For every month where a station sums to exactly 0 (raw), the 5 nearest stations are checked.
  If at least one neighbour has >0 precipitation that month â†’ month is flagged
  and all values in that month are set to NaN (after the per-value QC).
  If all neighbours are also 0 (or have no data) â†’ zeros are kept (region was dry).
- This catches stations that report 0 for the entire period or for individual months
  while nearby stations register rain.
- Monthly granularity + spatial context = precise and conservative.

FIX in v1.2.1: collection now requests the real column name "height" (files use "height",
not the renamed "hoehe"). This was causing 100% collection failure in v1.2.0.
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
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
from shared.ingest_paths import pipeline_output_dirs
from shared.qc_grid import interpolate_micro_gaps_keep_index, reindex_full_30min

_DIRS = pipeline_output_dirs("Niederschlag")
INPUT_FULL_DIR = _DIRS["full"]
PAKET_ROOT = _DIRS["root"]
OUTPUT_QC_DIR = _DIRS["qc"]

# ==================== CONFIG (v1.2.1 extended) ====================

VARIABLE_SUFFIX = "_precip"

FREQ_MINUTES      = 30
STEPS_PER_HOUR    = 60 // FREQ_MINUTES

HARD_MIN          = 0.0
HARD_MAX          = 50.0
SPIKE_THRESHOLD   = 20.0
STUCK_HOURS       = 3.0
STUCK_TOL         = 0.05

NEIGHBOR_K        = 2          # NEW in v1.2

LOG_FILE = PAKET_ROOT / f"precip_final_cleaner_v1.2.1_spatial_zero_qc_{datetime.now():%Y%m%d_%H%M%S}.log"
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


# ==================== NEW v1.2 HELPERS: HAVERSINE + NEAREST NEIGHBOURS ====================
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres. Returns inf if any coordinate is missing."""
    if None in (lat1, lon1, lat2, lon2):
        return np.inf
    if pd.isna([lat1, lon1, lat2, lon2]).any():
        return np.inf
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def find_k_nearest_neighbors(target_id: str, all_coords: dict, k: int = 5) -> list:
    """Return list of up to k nearest station_ids (by haversine distance)."""
    if target_id not in all_coords:
        return []
    tlat, tlon = all_coords[target_id]
    dists = []
    for sid, (lat, lon) in all_coords.items():
        if sid == target_id or lat is None or lon is None:
            continue
        d = haversine(tlat, tlon, lat, lon)
        if np.isfinite(d):
            dists.append((d, sid))
    dists.sort(key=lambda x: x[0])
    return [sid for _, sid in dists[:k]]


# ==================== UPDATED QC FUNCTION (unchanged logic) ====================
def apply_precip_qc(df: pd.DataFrame, logger=None, station_name: str = "") -> pd.DataFrame:
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
    df = df.sort_index()
    if df.empty:
        return df.reset_index() if isinstance(df.index, pd.DatetimeIndex) else df

    # === STEP 1: Data integrity ===
    if len(df) > 1:
        diffs = df.index.to_series().diff().dt.total_seconds() / 60
        gap_mask = (diffs > FREQ_MINUTES * 1.5) & (diffs < 60 * 24)
        n_gaps = int(gap_mask.sum())
        if logger and n_gaps > 0:
            logger.debug(f"  [{station_name}] {n_gaps} gaps detected (not removed)")

    original_valid = int(df["value"].notna().sum())
    if original_valid == 0:
        return df.reset_index() if isinstance(df.index, pd.DatetimeIndex) else df

    flagged = {"range": 0, "spike": 0, "stuck": 0}

    # === STEP 2: Gross range / physical limits ===
    hard_mask = (df["value"] < HARD_MIN) | (df["value"] > HARD_MAX)
    flagged["range"] = int(hard_mask.sum())
    df.loc[hard_mask, "value"] = np.nan

    # === STEP 3: Rate / spike check ===
    spike_mask = (df["value"] > SPIKE_THRESHOLD) & df["value"].notna()
    flagged["spike"] = int(spike_mask.sum())
    df.loc[spike_mask, "value"] = np.nan

    # === STEP 4: Stuck / flat-line NON-ZERO only ===
    min_stuck_steps = int(STUCK_HOURS * STEPS_PER_HOUR)
    is_small_change = ((df["value"].diff().abs() < STUCK_TOL).fillna(False) & 
                       (df["value"] > 0.01)).astype(int)
    df["run_id"] = (is_small_change.diff() != 0).cumsum()
    run_len = df.groupby("run_id")["run_id"].transform("size")
    stuck_in_run = (is_small_change == 1) & (run_len >= min_stuck_steps)
    long_run_ids = df.loc[stuck_in_run, "run_id"].unique()
    stuck_mask = df["run_id"].isin(long_run_ids) & df["value"].notna() & (df["value"] > 0.01)
    flagged["stuck"] = int(stuck_mask.sum())
    df.loc[stuck_mask, "value"] = np.nan

    # === MICRO-GAP INTERPOLATION (after all QC) ===
    if len(df) > 0:
        df, n_pts = interpolate_micro_gaps_keep_index(df, max_gap_hours=2.0, col="value")
        n_gaps = int(n_pts > 0)
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
        logger.info(f"QC v1.2.1 | flagged {total_flagged} ({pct:.1f}%) | "
                    f"range={flagged['range']}, spike={flagged['spike']}, "
                    f"stuck={flagged['stuck']} (â‰¥{STUCK_HOURS}h non-zero @ {STUCK_TOL} mm tol)")

    return df


# ==================== WORKER (updated for v1.2.1) ====================
def process_one_station(args):
    pq_path, output_dir, output_suffix, bad_months_set, station_id = args
    try:
        df = pd.read_parquet(pq_path)

        if "hoehe" in df.columns and "height" not in df.columns:
            df = df.rename(columns={"hoehe": "height"})

        df_clean = apply_precip_qc(df, logger=logger, station_name=station_id)

        # === NEW v1.2.1: Mask suspicious zero months (after per-value QC) ===
        if bad_months_set and not df_clean.empty:
            # ensure we have a proper timestamp column
            if isinstance(df_clean.index, pd.DatetimeIndex):
                df_clean = df_clean.reset_index().rename(columns={"index": "timestamp"})
            elif "timestamp" not in df_clean.columns:
                df_clean = df_clean.reset_index().rename(columns={"index": "timestamp"})

            df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], utc=True, errors="coerce")
            df_clean["month"] = df_clean["timestamp"].dt.to_period("M").astype(str)
            mask = df_clean["month"].isin(list(bad_months_set))
            n_masked = int(mask.sum())
            if n_masked > 0:
                df_clean.loc[mask, "value"] = np.nan
                if "is_missing" in df_clean.columns:
                    df_clean.loc[mask, "is_missing"] = True
                logger.info(f"  [{station_id}] Masked {n_masked} rows in {len(bad_months_set)} suspicious zero month(s)")

        # === FINAL SAFEGUARD ===
        if "index" in df_clean.columns and "timestamp" not in df_clean.columns:
            df_clean = df_clean.rename(columns={"index": "timestamp"})
        if "timestamp" not in df_clean.columns:
            df_clean = df_clean.reset_index().rename(columns={"index": "timestamp"})

        safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', station_id)
        out_path = output_dir / f"{safe_name}{output_suffix}"

        if not df_clean.empty:
            cols = ["timestamp", "value", "is_missing", "station", "name", "hoehe", "lat", "lon", "parameter", "source_file"]
            existing_cols = [c for c in cols if c in df_clean.columns]
            df_clean[existing_cols].to_parquet(out_path.with_suffix(".parquet"), compression="snappy", index=False)

            json_data = {
                "name": str(df_clean["name"].iloc[0]) if "name" in df_clean.columns else station_id,
                "hoehe": float(df_clean["hoehe"].iloc[0]) if "hoehe" in df_clean.columns and pd.notna(df_clean["hoehe"].iloc[0]) else None,
                "lat": float(df_clean["lat"].iloc[0]) if "lat" in df_clean.columns and pd.notna(df_clean["lat"].iloc[0]) else None,
                "lon": float(df_clean["lon"].iloc[0]) if "lon" in df_clean.columns and pd.notna(df_clean["lon"].iloc[0]) else None,
                "qc_version": "v1.2.1_spatial_zero_qc (dedup + micro-gap + robust ts + neighbour zero-month validation)",
                "data": [{"timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                          "value": float(v) if pd.notna(v) else None}
                         for ts, v in zip(df_clean["timestamp"], df_clean["value"])]
            }
            with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

        return f"OK: {station_id}"
    except Exception as e:
        return f"ERROR on {pq_path.name}: {e}"


# ==================== MAIN (v1.2.1 with spatial zero-month validation - FIXED) ====================
def main():
    logger.info("=== PRECIPITATION FINAL CLEANER v1.2.1 (spatial zero-month QC) START ===")
    logger.info(f"Source: {INPUT_FULL_DIR}  ->  Destination: {OUTPUT_QC_DIR}")
    logger.info(f"Thresholds (UNCHANGED): hard {HARD_MIN}â€“{HARD_MAX} mm/30min | spike >{SPIKE_THRESHOLD} mm/30min | "
                f"stuck â‰¥{STUCK_HOURS}h non-zero (tol {STUCK_TOL} mm) | 30-min data")
    logger.info(f"NEW in v1.2.x: Neighbour-based zero-month validation (k={NEIGHBOR_K}) on raw monthly sums")
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

    # ==================== NEW v1.2.1: COLLECT METADATA + IDENTIFY BAD MONTHS (FIXED column name) ====================
    logger.info("Collecting raw monthly totals and coordinates for neighbour validation...")

    all_coords = {}
    monthly_totals = {}          # station_id -> {"2020-01": sum, ...}
    station_info = {}            # station_id -> metadata + file_path
    bad_months_map = {}          # station_id -> set of bad 'YYYY-MM' strings

    for pq_path in files_to_process:
        try:
            # FIX: request the REAL column name present in the parquet files ("height")
            df = pd.read_parquet(pq_path, columns=["timestamp", "value", "lat", "lon", "station", "name", "height"])
            if df.empty or "value" not in df.columns:
                continue

            # rename for internal consistency (same as worker)
            if "height" in df.columns:
                df = df.rename(columns={"height": "hoehe"})

            station_id = str(df["station"].iloc[0])
            lat = float(df["lat"].iloc[0]) if pd.notna(df["lat"].iloc[0]) else None
            lon = float(df["lon"].iloc[0]) if pd.notna(df["lon"].iloc[0]) else None

            all_coords[station_id] = (lat, lon)
            station_info[station_id] = {
                "name": str(df["name"].iloc[0]) if "name" in df.columns else station_id,
                "hoehe": float(df["hoehe"].iloc[0]) if "hoehe" in df.columns and pd.notna(df["hoehe"].iloc[0]) else None,
                "lat": lat,
                "lon": lon,
                "file_path": pq_path
            }

            # monthly accumulated totals from RAW data
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp", "value"])
            if not df.empty:
                df["month"] = df["timestamp"].dt.to_period("M")
                monthly = df.groupby("month")["value"].sum()
                monthly_totals[station_id] = {str(m): float(s) for m, s in monthly.items()}
            else:
                monthly_totals[station_id] = {}
        except Exception as e:
            logger.warning(f"  Metadata collection failed for {pq_path.name}: {e}")
            continue

    logger.info(f"  Collected metadata for {len(all_coords)} stations.")

    # Compute k nearest neighbours (brute-force is efficient for typical station counts)
    logger.info(f"Computing {NEIGHBOR_K} nearest neighbours per station (haversine distance)...")
    neighbor_map = {sid: find_k_nearest_neighbors(sid, all_coords, NEIGHBOR_K)
                    for sid in all_coords}

    # Identify suspicious zero months
    logger.info("Identifying suspicious zero months (my sum==0 but â‰¥1 neighbour has >0)...")
    total_bad_months = 0
    stations_with_bad_months = 0
    for station_id, my_monthly in monthly_totals.items():
        bad = set()
        neighs = neighbor_map.get(station_id, [])
        for month_str, my_sum in my_monthly.items():
            if my_sum > 0:
                continue
            has_rain_nearby = any(
                monthly_totals.get(nid, {}).get(month_str, 0.0) > 0
                for nid in neighs
            )
            if has_rain_nearby:
                bad.add(month_str)
        bad_months_map[station_id] = bad
        if bad:
            stations_with_bad_months += 1
            total_bad_months += len(bad)
            logger.info(f"  [{station_id}] {len(bad)} suspicious zero month(s) will be masked to NaN")

    logger.info(f"Stations with at least one suspicious zero month: {stations_with_bad_months}")
    logger.info(f"Total suspicious zero months flagged across dataset: {total_bad_months}")
    # ==================== END NEW v1.2.1 BLOCK ====================

    logger.info(f"Processing {len(files_to_process)} unique stations in parallel...")

    # Build tasks including the pre-computed bad_months_set for each station
    tasks = []
    for station_id, info in station_info.items():
        bad_set = bad_months_map.get(station_id, set())
        tasks.append((info["file_path"], OUTPUT_QC_DIR, "_final_qc", bad_set, station_id))

    with mp.Pool(mp.cpu_count() - 1) as pool:
        results = pool.map(process_one_station, tasks)

    for r in results:
        if "ERROR" in r:
            logger.error(r)

    logger.info("=== PRECIPITATION FINAL CLEANER v1.2.1 (spatial zero-month QC) FINISHED ===")
    logger.info(f"Output: {OUTPUT_QC_DIR}")
    logger.info(f"Log: {LOG_FILE}")


if __name__ == "__main__":
    main()

