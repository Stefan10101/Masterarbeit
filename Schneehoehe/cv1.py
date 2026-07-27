#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snow Height Final Cleaner v2.38 (FINAL STABLE) - PIPELINE v3 ADAPTED
====================================================================
Lightly adapted for new pipeline output structure (Full_2020-2025 nested folders + "height" column).
All original QC logic, degrass, flatline, micro-gap and temp lookup 100% preserved.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import re
import multiprocessing as mp
from scipy.spatial.distance import cdist

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

VARIABLE_SUFFIX = "_snow"

INPUT_FULL_DIR = Path(DATA_ROOT / "schneehoehe" / "data" / "data" / "full_2020-2025")
PAKET_ROOT     = Path(DATA_ROOT / "schneehoehe" / "paket")
OUTPUT_QC_DIR  = PAKET_ROOT / "full_2020_2025_qc"
OUTPUT_QC_DIR.mkdir(parents=True, exist_ok=True)

# Temperature QC output (run temperature QC first)
TEMP_FINAL_ROOT = Path(DATA_ROOT / "temperatur" / "paket" / "full_2020_2025_qc")

TEMP_SEARCH_RADIUS_KM = 8.0
ELEVATION_TOLERANCE_M = 400
RATE_MAX = 0.001
TUKEY_WINDOW = 11
TUKEY_K = 1.8
FLATLINE_DAYS = 21
HIGH_ALTITUDE_THRESHOLD = 2900
MIN_FLATLINE_VALUE = 2.0

LOG_FILE = PAKET_ROOT / f"snowheight_final_cleaner_v2.38_pipeline3_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


# ==================== FLATLINE FILTER (unchanged) ====================
def flatline_filter(df, max_days=FLATLINE_DAYS, min_value=MIN_FLATLINE_VALUE):
    if df.empty or "value" not in df.columns:
        return pd.Series(True, index=df.index)
    s = df["value"]
    groups = (s != s.shift()).cumsum()
    group_sizes = s.groupby(groups).transform('size')
    group_values = s.groupby(groups).transform('first')
    max_points = max_days * 48
    is_flatline = (group_sizes > max_points) & (group_values != 0) & (group_values > min_value)
    return ~is_flatline


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


# ==================== DEGRASS v2.30 (unchanged) ====================
def degrass_filter_v2_30(df, station_name="", has_temp=False, elevation=np.nan, is_problematic=False, logger=None):
    if df.empty or "value" not in df.columns:
        return np.ones(len(df), dtype=bool), False
    if pd.notna(elevation) and elevation > HIGH_ALTITUDE_THRESHOLD:
        if logger:
            logger.info(f"  [{station_name}] >{HIGH_ALTITUDE_THRESHOLD}m â†’ skipping degrass")
        return np.ones(len(df), dtype=bool), False
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True).sort_values()
    df['date'] = df.index.normalize()
    ta_12h_daily = None
    if has_temp and "TA" in df.columns:
        try:
            ta = df["TA"].copy()
            ta_30min = ta.resample('30min').mean()
            ta_12h = ta_30min.rolling(window=24, center=True, min_periods=12).mean()
            ta_12h_daily = ta_12h.resample('D').mean()
        except Exception as e:
            if logger:
                logger.warning(f"  [{station_name}] 12-hour temperature mean FAILED: {e}")
    daily = df.groupby('date').agg({'value': 'max'})
    daily.columns = ['value_max']
    if not is_problematic:
        daily['is_snow_day'] = True
        used_aggressive = False
    else:
        years = sorted(df.index.year.unique())
        daily['is_snow_day'] = True
        used_aggressive = True
        for year in years:
            year_start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
            sep1 = pd.Timestamp(year=year, month=9, day=1, tz="UTC")
            nov1 = pd.Timestamp(year=year, month=11, day=1, tz="UTC")
            oct31 = pd.Timestamp(year=year, month=10, day=31, tz="UTC")
            year_data = daily.loc[(daily.index >= year_start) & (daily.index < nov1)]
            if len(year_data) < 20:
                continue
            jan_mar = year_data.loc[(year_data.index.month >= 1) & (year_data.index.month <= 3)]
            if len(jan_mar) == 0 or jan_mar['value_max'].isna().all():
                continue
            peak_idx = jan_mar['value_max'].idxmax()
            peak_value = jan_mar.loc[peak_idx, 'value_max']
            after_peak = year_data.loc[peak_idx:]
            first_20cm = after_peak[after_peak['value_max'] < 20.0]
            if len(first_20cm) == 0:
                end_date = pd.Timestamp(year=year, month=7, day=20, tz="UTC")
            else:
                first_20cm_date = first_20cm.index[0]
                value_at_20cm = first_20cm.iloc[0]['value_max']
                days_diff = (first_20cm_date - peak_idx).days
                if days_diff > 3 and peak_value > value_at_20cm:
                    slope = (value_at_20cm - peak_value) / days_diff
                    days_to_zero = int(abs(18.0 / slope)) if slope != 0 else 25
                    end_date = first_20cm_date + pd.Timedelta(days=days_to_zero)
                else:
                    end_date = pd.Timestamp(year=year, month=7, day=20, tz="UTC")
            if end_date > oct31:
                if logger:
                    logger.warning(f"  [{station_name}] Year {year}: end_date {end_date.date()} capped to 31 Oct")
                end_date = oct31
            daily.loc[(daily.index >= end_date) & (daily.index <= oct31) & (daily.index >= pd.Timestamp(year=year, month=4, day=1, tz="UTC")), 'is_snow_day'] = False
            if ta_12h_daily is not None:
                temp_ok = ta_12h_daily.reindex(daily.index).loc[(daily.index >= sep1) & (daily.index < nov1)] < 0
                snow_rising = daily['value_max'].loc[(daily.index >= sep1) & (daily.index < nov1)] > \
                              daily['value_max'].shift(3).loc[(daily.index >= sep1) & (daily.index < nov1)]
                early_start_mask = temp_ok & snow_rising
                daily.loc[(daily.index >= sep1) & (daily.index < nov1), 'is_snow_day'] = early_start_mask
            else:
                daily.loc[(daily.index >= sep1) & (daily.index < nov1), 'is_snow_day'] = False
    df['is_snow_day'] = df['date'].map(daily['is_snow_day']).fillna(True)
    summer_zeroed = ((~df['is_snow_day']) & (df.index.month >= 6) & (df.index.month <= 9)).sum()
    if logger and summer_zeroed > 0:
        method = "AGGRESSIVE per-year" if is_problematic else "normal"
        logger.info(f"  [{station_name}] v2.30: {summer_zeroed:,} summer points set to 0 ({method})")
    return df['is_snow_day'].values, used_aggressive


# ==================== apply_snowheight_qc (v2.38 - FINAL, unchanged logic) ====================
def apply_snowheight_qc(df, temp_lookup, station_name, logger=None):
    if df.empty or "value" not in df.columns:
        return df, False, None
    df = df.copy()
    # === ROBUST TIMESTAMP (v2.38) ===
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
            return df, False, None
    df.index = pd.to_datetime(df.index, utc=True).sort_values()
    df = df.dropna(subset=["value"])
    if df.empty:
        return df.reset_index(), False, None
    flatline_mask = flatline_filter(df)
    flatline_count = (~flatline_mask).sum()
    if flatline_count > 0 and logger:
        logger.info(f"  [{station_name}] Flatline filter: {flatline_count:,} points set to NaN")
    df = df[flatline_mask]
    if df.empty:
        return df.reset_index(), False, None
    YEARS = list(range(2020, 2026))
    total_summer_points = 0
    zero_summer_points = 0
    for y in YEARS:
        start = pd.Timestamp(year=y, month=7, day=15, tz="UTC")
        end = pd.Timestamp(year=y, month=9, day=15, tz="UTC")
        summer = df.loc[(df.index >= start) & (df.index <= end), "value"]
        if len(summer) == 0:
            continue
        total_summer_points += len(summer)
        zero_summer_points += int((summer < 0.5).sum())
    summer_zero_rate = (zero_summer_points / total_summer_points * 100) if total_summer_points > 0 else 100.0
    is_problematic = summer_zero_rate < 80.0
    snow_lat = df["lat"].iloc[0] if "lat" in df.columns else np.nan
    snow_lon = df["lon"].iloc[0] if "lon" in df.columns else np.nan
    snow_hoehe = df["hoehe"].iloc[0] if "hoehe" in df.columns else np.nan
    temp_df, temp_station_name = load_temperature_for_station(snow_lat, snow_lon, snow_hoehe, temp_lookup, station_name)
    has_temp = temp_df is not None
    if has_temp:
        temp_df = temp_df.copy()
        temp_df.index = pd.to_datetime(temp_df.index, utc=True).sort_values()
        if not isinstance(temp_df.index, pd.DatetimeIndex):
            temp_df.index = pd.to_datetime(temp_df.index, utc=True)
        df = df.join(temp_df["TA"], how="left")
    original_valid = int(df["value"].notna().sum())
    flagged = {"rate": 0, "tukey": 0, "degrass": 0, "flatline": flatline_count, "micro_gap_filled": 0}
    rate_mask = rate_filter(df["value"].values, df.index)
    flagged["rate"] = int((~rate_mask).sum())
    df = df[rate_mask]
    if len(df) > 0:
        tukey_mask = tukey_despike(df["value"].values)
        flagged["tukey"] = int((~tukey_mask).sum())
        df = df[tukey_mask]
    # Micro-gap interpolation
    if len(df) > 0:
        df, n_gaps, n_pts = interpolate_micro_gaps(df, max_gap_hours=2.0, col="value",
                                                   logger=logger, station_name=station_name)
        flagged["micro_gap_filled"] = n_pts
    used_aggressive = False
    if len(df) > 0:
        degrass_mask, used_aggressive = degrass_filter_v2_30(df, station_name=station_name,
                                                             has_temp=has_temp, elevation=snow_hoehe,
                                                             is_problematic=is_problematic, logger=logger)
        flagged["degrass"] = int((~degrass_mask).sum())
        df.loc[~degrass_mask, "value"] = 0.0
    df = df.reset_index()
    if "is_missing" in df.columns:
        df["is_missing"] = df["value"].isna()
    # === FINAL SAFEGUARD (v2.38) ===
    if "index" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"index": "timestamp"})
    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={"index": "timestamp"})
    return df, used_aggressive, temp_station_name


# ==================== Helper functions (unchanged) ====================
def rate_filter(values, times, max_rate=RATE_MAX):
    if len(values) == 0:
        return np.array([], dtype=bool)
    n = len(values)
    valid = np.ones(n, dtype=bool)
    last_valid = 0
    for i in range(1, n):
        dt = (times[i] - times[last_valid]).total_seconds()
        if dt <= 0:
            valid[i] = False
            continue
        slope = (values[i] - values[last_valid]) / dt
        if abs(slope) > max_rate:
            valid[i] = False
        else:
            last_valid = i
    return valid

def tukey_despike(values, window=TUKEY_WINDOW, k=TUKEY_K):
    s = pd.Series(values)
    med = s.rolling(window, center=True, min_periods=max(3, window//2)).median()
    std = s.rolling(window, center=True, min_periods=max(3, window//2)).std()
    spikes = np.abs(s - med) > (k * std)
    return (~spikes.fillna(False)).values

def build_temp_station_lookup():
    temp_files = list(TEMP_FINAL_ROOT.glob("*_final_qc.parquet"))
    stations = []
    for f in temp_files:
        try:
            df = pd.read_parquet(f, columns=["station", "lat", "lon", "hoehe"])
            if not df.empty:
                stations.append(df.iloc[0])
        except:
            continue
    return pd.DataFrame(stations)

def find_best_temp_station(snow_lat, snow_lon, snow_hoehe, temp_lookup):
    if temp_lookup.empty or pd.isna(snow_lat) or pd.isna(snow_lon) or pd.isna(snow_hoehe):
        return None, None, None
    coords = temp_lookup[["lat", "lon"]].values
    target = np.array([[snow_lat, snow_lon]])
    distances = cdist(target, coords, metric="euclidean") * 111
    temp_hoehe = temp_lookup["hoehe"].values
    elev_diff = np.abs(temp_hoehe - snow_hoehe)
    valid_mask = (distances[0] <= TEMP_SEARCH_RADIUS_KM) & (elev_diff <= ELEVATION_TOLERANCE_M)
    if not np.any(valid_mask):
        return None, None, None
    valid_distances = distances[0][valid_mask]
    valid_indices = np.where(valid_mask)[0]
    best_local_idx = np.argmin(valid_distances)
    best_global_idx = valid_indices[best_local_idx]
    return temp_lookup.iloc[best_global_idx]["station"], valid_distances[best_local_idx], None

def load_temperature_for_station(snow_lat, snow_lon, snow_hoehe, temp_lookup, station_name):
    best_station, distance, reason = find_best_temp_station(snow_lat, snow_lon, snow_hoehe, temp_lookup)
    if best_station is None:
        logger.info(f"  [{station_name}] No good temperature station within 8 km / 400 m â†’ using calendar rule only")
        return None, None
    temp_path = TEMP_FINAL_ROOT / f"{best_station}_final_qc.parquet"
    if not temp_path.exists():
        return None, None
    try:
        temp_df = pd.read_parquet(temp_path, columns=["timestamp", "value"]).rename(columns={"value": "TA"})
        temp_df = temp_df.set_index("timestamp")
        logger.info(f"  [{station_name}] Using temperature from {best_station} ({distance:.1f} km)")
        return temp_df, best_station
    except:
        return None, None

def get_station_coords(pq_path):
    try:
        df = pd.read_parquet(pq_path, columns=["lat", "lon"])
        return (round(df["lat"].iloc[0], 4), round(df["lon"].iloc[0], 4))
    except:
        return None


# ==================== WORKER ====================
def process_one_station(args):
    pq_path, output_dir, output_suffix, temp_lookup = args
    try:
        df = pd.read_parquet(pq_path)

        # === COLUMN NORMALIZATION for pipeline v3 ("height" -> "hoehe") ===
        if "height" in df.columns:
            df = df.rename(columns={"height": "hoehe"})

        station = df["station"].iloc[0] if "station" in df.columns else pq_path.stem.split("_")[0]
        df_clean, used_aggressive, temp_station = apply_snowheight_qc(df, temp_lookup, station, logger=logger)

        safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', station)
        out_path = output_dir / f"{safe_name}{output_suffix}"

        if not df_clean.empty:
            cols = ["timestamp", "value", "is_missing", "station", "name", "hoehe", "lat", "lon", "parameter", "source_file"]
            existing = [c for c in cols if c in df_clean.columns]
            df_clean[existing].to_parquet(out_path.with_suffix(".parquet"), compression="snappy", index=False)

            ts_col = df_clean.get("timestamp", df_clean.index)
            json_data = {
                "name": str(df_clean["name"].iloc[0]) if "name" in df_clean.columns else station,
                "hoehe": float(df_clean["hoehe"].iloc[0]) if "hoehe" in df_clean.columns and pd.notna(df_clean["hoehe"].iloc[0]) else None,
                "lat": float(df_clean["lat"].iloc[0]) if "lat" in df_clean.columns and pd.notna(df_clean["lat"].iloc[0]) else None,
                "lon": float(df_clean["lon"].iloc[0]) if "lon" in df_clean.columns and pd.notna(df_clean["lon"].iloc[0]) else None,
                "qc_version": "v2.38_pipeline3 (final stable)",
                "data": [{"timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                          "value": float(v) if pd.notna(v) else None}
                         for ts, v in zip(ts_col, df_clean["value"])]
            }
            with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

        if temp_station:
            status = "TEMP"
        else:
            status = "NO_TEMP"
        if used_aggressive:
            status = "AGGRESSIVE"
        return f"{status}: {station}"
    except Exception as e:
        return f"ERROR on {pq_path.name}: {e}"


# ==================== MAIN ====================
def main():
    logger.info("=== SNOW HEIGHT FINAL CLEANER v2.38 (pipeline v3 adapted) ===")
    logger.info("Temperature: 12-hour mean < 0Â°C + rising snow (only 1 Sep â€“ 1 Nov)")
    logger.info("Winter: only melt-curve")
    logger.info("Micro-gap interp: linear fill gaps <=2h (4 timesteps) on snowheight 'value'")

    if not INPUT_FULL_DIR.exists():
        logger.error(f"Folder not found: {INPUT_FULL_DIR}")
        return

    all_files = list(INPUT_FULL_DIR.rglob(f"*{VARIABLE_SUFFIX}.parquet"))
    logger.info(f"Scanning {len(all_files)} raw files for coordinate duplicates...")

    coord_to_files = {}
    for f in all_files:
        coords = get_station_coords(f)
        if coords:
            key = coords
            if key not in coord_to_files:
                coord_to_files[key] = []
            coord_to_files[key].append(f)

    duplicates_found = 0
    for coords, files in coord_to_files.items():
        if len(files) > 1:
            duplicates_found += len(files) - 1
            logger.info(f"  DUPLICATE at {coords}: {len(files)} files â†’ keeping first")

    files_to_process = [files[0] for files in coord_to_files.values()]
    logger.info(f"After deduplication: {len(files_to_process)} unique stations (removed {duplicates_found} duplicates)")

    temp_lookup = build_temp_station_lookup()
    logger.info(f"Loaded {len(temp_lookup)} temperature stations (from {TEMP_FINAL_ROOT})")

    high_alt_count = 0
    for f in files_to_process:
        try:
            df = pd.read_parquet(f, columns=["hoehe"])
            if df["hoehe"].iloc[0] > HIGH_ALTITUDE_THRESHOLD:
                high_alt_count += 1
        except:
            pass
    logger.info(f"Stations >{HIGH_ALTITUDE_THRESHOLD}m (degrass skipped): {high_alt_count}")

    logger.info(f"Processing {len(files_to_process)} unique stations...")

    tasks = [(f, OUTPUT_QC_DIR, "_final_qc", temp_lookup) for f in files_to_process]
    with mp.Pool(mp.cpu_count() - 1) as pool:
        results = pool.map(process_one_station, tasks)

    for r in results:
        if "ERROR" in r:
            logger.error(r)

    aggressive = sum(1 for r in results if r.startswith("AGGRESSIVE"))
    logger.info(f"Stations that used AGGRESSIVE per-year linear melt-curve: {aggressive} / {len(files_to_process)}")

    with_temp = sum(1 for r in results if r.startswith("TEMP") or r.startswith("AGGRESSIVE"))
    without_temp = len(files_to_process) - with_temp
    logger.info(f"Stations WITH good temperature sensor (â‰¤8 km / 400 m): {with_temp} / {len(files_to_process)}")
    logger.info(f"Stations WITHOUT good temperature sensor: {without_temp} / {len(files_to_process)}")

    logger.info("=== FINISHED v2.38_pipeline3 ===")
    logger.info(f"Output: {OUTPUT_QC_DIR}")


if __name__ == "__main__":
    main()

