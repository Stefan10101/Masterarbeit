#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RH Final Cleaner (Parallel) - v1.3 - PIPELINE v3 ADAPTED
=========================================================
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
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
from shared.ingest_paths import pipeline_output_dirs
from shared.qc_grid import interpolate_micro_gaps_keep_index, reindex_full_30min

_DIRS = pipeline_output_dirs("Luftfeuchte")
INPUT_FULL_DIR = _DIRS["full"]
PAKET_ROOT = _DIRS["root"]
OUTPUT_QC_DIR = _DIRS["qc"]

# ==================== CONFIG (adapted for pipeline v3) ====================

VARIABLE_SUFFIX = "_rh"

STUCK_HOURS      = 8.0
SATURATION_HOURS = 12.0
DRY_HOURS        = 18.0
SPIKE_THRESHOLD  = 25.0
STUCK_NEIGHBOUR_KM = 25.0
STUCK_NEIGHBOUR_TOL = 5.0

_RH_META = None  # DataFrame path,lat,lon,station — set in worker init

LOG_FILE = PAKET_ROOT / f"rh_final_cleaner_v1.3_pipeline3_{datetime.now():%Y%m%d_%H%M%S}.log"
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
    except Exception:
        return None


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _build_rh_meta(files):
    rows = []
    for pq in files:
        try:
            m = pd.read_parquet(pq, columns=["lat", "lon", "station"])
            rows.append({
                "path": pq,
                "lat": float(m["lat"].iloc[0]),
                "lon": float(m["lon"].iloc[0]),
                "station": str(m["station"].iloc[0]) if "station" in m.columns else pq.stem,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def _init_rh_worker(meta_df):
    global _RH_META
    _RH_META = meta_df


def _neighbour_confirms_rh(df, stuck_mask, long_run_ids) -> pd.Series:
    """True where a nearby station is within STUCK_NEIGHBOUR_TOL over the same run."""
    keep = pd.Series(False, index=df.index)
    if _RH_META is None or _RH_META.empty or "lat" not in df.columns:
        return keep
    lat0 = df["lat"].iloc[0]
    lon0 = df["lon"].iloc[0]
    if pd.isna(lat0) or pd.isna(lon0):
        return keep
    dist = _haversine_km(lat0, lon0, _RH_META["lat"].to_numpy(), _RH_META["lon"].to_numpy())
    neigh = _RH_META.loc[dist <= STUCK_NEIGHBOUR_KM].copy()
    self_name = str(df["station"].iloc[0]) if "station" in df.columns else ""
    neigh = neigh[neigh["station"] != self_name]
    if neigh.empty:
        return keep
    series = []
    for _, row in neigh.iterrows():
        try:
            nd = pd.read_parquet(row["path"], columns=["timestamp", "value"])
            nd["timestamp"] = pd.to_datetime(nd["timestamp"], utc=True)
            series.append(nd.set_index("timestamp")["value"])
        except Exception:
            continue
    if not series:
        return keep
    panel = pd.concat(series, axis=1)
    for rid in long_run_ids:
        run_idx = df.index[df["run_id"] == rid]
        if len(run_idx) == 0:
            continue
        own = df.loc[run_idx, "value"]
        aligned = panel.reindex(run_idx)
        if aligned.dropna(how="all").empty:
            continue
        med_abs = (aligned.sub(own, axis=0)).abs().median()
        if (med_abs <= STUCK_NEIGHBOUR_TOL).any():
            keep.loc[run_idx] = True
    return keep


# ==================== UPDATED QC FUNCTION (unchanged logic) ====================
def apply_final_rh_qc(df: pd.DataFrame, logger=None, station_name: str = "") -> pd.DataFrame:
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

    flagged = {"spike": 0, "stuck": 0, "saturation": 0, "dry": 0}

    # 1. STEP / SPIKE CHECK
    df["diff"] = df["value"].diff().abs()
    spike_mask = (df["diff"] > SPIKE_THRESHOLD) & df["value"].notna()
    flagged["spike"] = int(spike_mask.sum())
    df.loc[spike_mask, "value"] = np.nan

    # 2. PERSISTENCE / STUCK SENSOR
    tol = 0.10
    min_stuck_steps = int(STUCK_HOURS * 2)
    is_small_change = (df["value"].diff().abs() < tol).fillna(False).astype(int)
    df["run_id"] = (is_small_change.diff() != 0).cumsum()
    run_len = df.groupby("run_id")["run_id"].transform("size")
    stuck_in_run = (is_small_change == 1) & (run_len >= min_stuck_steps)
    long_run_ids = df.loc[stuck_in_run, "run_id"].unique()
    stuck_mask = df["run_id"].isin(long_run_ids) & df["value"].notna()
    if stuck_mask.any():
        keep = _neighbour_confirms_rh(df, stuck_mask, long_run_ids)
        stuck_mask = stuck_mask & ~keep
    flagged["stuck"] = int(stuck_mask.sum())
    df.loc[stuck_mask, "value"] = np.nan

    # 3. PROLONGED SATURATION
    sat_threshold = 99.5
    min_sat_steps = int(SATURATION_HOURS * 2)
    is_sat = (df["value"] >= sat_threshold).astype(int)
    df["sat_run_id"] = (is_sat.diff() != 0).cumsum()
    sat_run_len = df.groupby("sat_run_id")["sat_run_id"].transform("size")
    sat_in_run = (is_sat == 1) & (sat_run_len >= min_sat_steps)
    long_sat_ids = df.loc[sat_in_run, "sat_run_id"].unique()
    sat_mask = df["sat_run_id"].isin(long_sat_ids) & df["value"].notna()
    flagged["saturation"] = int(sat_mask.sum())
    df.loc[sat_mask, "value"] = np.nan

    # 4. PROLONGED DRY-OUT
    dry_threshold = 5.0
    min_dry_steps = int(DRY_HOURS * 2)
    is_dry = (df["value"] <= dry_threshold).astype(int)
    df["dry_run_id"] = (is_dry.diff() != 0).cumsum()
    dry_run_len = df.groupby("dry_run_id")["dry_run_id"].transform("size")
    dry_in_run = (is_dry == 1) & (dry_run_len >= min_dry_steps)
    long_dry_ids = df.loc[dry_in_run, "dry_run_id"].unique()
    dry_mask = df["dry_run_id"].isin(long_dry_ids) & df["value"].notna()
    flagged["dry"] = int(dry_mask.sum())
    df.loc[dry_mask, "value"] = np.nan

    # === MICRO-GAP INTERPOLATION (after all QC) ===
    if len(df) > 0:
        df, n_pts = interpolate_micro_gaps_keep_index(df, max_gap_hours=2.0, col="value")
        n_gaps = int(n_pts > 0)
        if n_pts > 0 and logger:
            logger.info(f"  [{station_name}] Micro-gap interpolation: {n_gaps} gaps filled, {n_pts} points interpolated (<=2h)")

    # Cleanup
    temp_cols = [c for c in df.columns if c.startswith(("diff", "run_id", "sat_run_id", "dry_run_id"))]
    df = df.drop(columns=temp_cols, errors="ignore")

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    if "timestamp" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "timestamp"})

    if "is_missing" in df.columns:
        df["is_missing"] = df["value"].isna()

    total_flagged = sum(flagged.values())
    if logger is not None and total_flagged > 0:
        pct = total_flagged / (total_flagged + df["value"].notna().sum()) * 100 if (total_flagged + df["value"].notna().sum()) > 0 else 0
        logger.info(f"QC v1.3 | flagged {total_flagged} ({pct:.1f}%) | "
                    f"spike={flagged['spike']}, stuck={flagged['stuck']} (>= {STUCK_HOURS}h), "
                    f"sat={flagged['saturation']} (>= {SATURATION_HOURS}h), dry={flagged['dry']}")

    return df


# ==================== WORKER ====================
def process_one_station(args):
    pq_path, output_dir, output_suffix = args
    try:
        df = pd.read_parquet(pq_path)

        if "hoehe" in df.columns and "height" not in df.columns:
            df = df.rename(columns={"hoehe": "height"})

        station = df["station"].iloc[0] if "station" in df.columns else pq_path.stem.split("_")[0]

        df_clean = apply_final_rh_qc(df, logger=logger, station_name=station)

        # === FINAL SAFEGUARD ===
        if "index" in df_clean.columns and "timestamp" not in df_clean.columns:
            df_clean = df_clean.rename(columns={"index": "timestamp"})
        if "timestamp" not in df_clean.columns:
            df_clean = df_clean.reset_index().rename(columns={"index": "timestamp"})

        safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', station)
        out_path = output_dir / f"{safe_name}{output_suffix}"

        if not df_clean.empty:
            cols = ["timestamp", "value", "is_missing", "station", "name", "height", "lat", "lon", "parameter", "source_file"]
            existing = [c for c in cols if c in df_clean.columns]
            df_clean[existing].to_parquet(out_path.with_suffix(".parquet"), compression="snappy", index=False)

            json_data = {
                "name": str(df_clean["name"].iloc[0]) if "name" in df_clean.columns else station,
                "hoehe": float(df_clean["hoehe"].iloc[0]) if "hoehe" in df_clean.columns and pd.notna(df_clean["hoehe"].iloc[0]) else None,
                "lat": float(df_clean["lat"].iloc[0]) if "lat" in df_clean.columns and pd.notna(df_clean["lat"].iloc[0]) else None,
                "lon": float(df_clean["lon"].iloc[0]) if "lon" in df_clean.columns and pd.notna(df_clean["lon"].iloc[0]) else None,
                "qc_version": "v1.3_pipeline3 (dedup + micro-gap + robust ts)",
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
    logger.info("=== RH FINAL CLEANER v1.3 (pipeline v3 adapted) START ===")
    logger.info(f"Source: {INPUT_FULL_DIR}  ->  Destination: {OUTPUT_QC_DIR}")
    logger.info(f"Thresholds (UNCHANGED): stuck >= {STUCK_HOURS}h | sat >= {SATURATION_HOURS}h | "
                f"dry >= {DRY_HOURS}h | spike > {SPIKE_THRESHOLD}%")
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

    meta = _build_rh_meta(files_to_process)
    logger.info(f"RH neighbour table: {len(meta)} stations, radius={STUCK_NEIGHBOUR_KM} km, tol={STUCK_NEIGHBOUR_TOL} %")
    tasks = [(f, OUTPUT_QC_DIR, "_final_qc") for f in files_to_process]
    with mp.Pool(max(1, mp.cpu_count() - 1), initializer=_init_rh_worker, initargs=(meta,)) as pool:
        results = pool.map(process_one_station, tasks)

    for r in results:
        if "ERROR" in r:
            logger.error(r)

    logger.info("=== RH FINAL CLEANER v1.3 (pipeline v3) FINISHED ===")
    logger.info(f"Output: {OUTPUT_QC_DIR}")
    logger.info(f"Log: {LOG_FILE}")


if __name__ == "__main__":
    main()

