#!/usr/bin/env python3
"""
Master Thesis - Snowheight Data Pipeline (v3.0)
Strict exact regrid, 30% coverage, new folder structure, thorough logging.
Master Thesis Snowheight Data Pipeline - full period only (cold seasons removed)
- DWD daily snow: 06:00 UTC exact only (no ffill), coverage filter disabled for DWD

"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import multiprocessing as mp
import json
import hashlib
import re
from typing import Optional, Dict, List, Callable

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
from shared.ingest_paths import (
    variable_source_dir,
    pipeline_output_dirs,
    list_network_jsons,
    canonical_network_name,
)
from paths import DATA_ROOT

# ==================== CONFIG ====================
VARIABLE = "SNOW"
PARAMETER = "HS"
ROOT = variable_source_dir("Schneehoehe")
_DIRS = pipeline_output_dirs("Schneehoehe")
AAAData = _DIRS["root"]
FULL_DIR = _DIRS["full"]
REJECTED_DIR = _DIRS["rejected"]
STATS_DIR = _DIRS["stats"]
LOG_DIR = DATA_ROOT / "Plots" / "logs" / "schneehoehe"
LOG_DIR.mkdir(parents=True, exist_ok=True)

COVERAGE_THRESHOLD = 20.0
FULL_PERIOD_START = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
FULL_PERIOD_END   = pd.Timestamp("2025-12-31 23:30:00", tz="UTC")

LOG_FILE = LOG_DIR / f"pipeline_{VARIABLE.lower()}_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==================== HELPERS (identical) ====================

def safe_station_name(name: str) -> str:
    name = re.sub(r'[^a-zA-Z0-9äöüÄÖÜß]', '_', name)
    name = name.replace('ä', 'a').replace('ö', 'o').replace('ü', 'u')
    name = name.replace('Ä', 'A').replace('Ö', 'O').replace('Ü', 'U')
    name = re.sub(r'_+', '_', name).strip('_')
    return name or "unknown_station"

def coverage_percent(df: pd.DataFrame) -> float:
    return float(df["value"].notna().mean() * 100) if not df.empty else 0.0

def check_and_fix_coordinates(lat: float, lon: float, station: str, network: str) -> tuple:
    if pd.isna(lat) or pd.isna(lon):
        return lat, lon
    if lon > lat:
        logger.warning(f"COORDINATE SWAP | {network}/{station} | lon={lon} > lat={lat} → swapping")
        return lon, lat
    return lat, lon

def ensure_utc(df: pd.DataFrame, network: str) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return df
    df = df.copy()
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
        return df

    if network in ["DWD", "ZAMG", "Meteosuisse"]:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    elif network in ["HydroT", "HydroVO"]:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Europe/Vienna", ambiguous="NaT", nonexistent="NaT")
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    elif network == "Suedtirol":
        df["timestamp"] = df["timestamp"].dt.tz_localize("Europe/Rome", ambiguous="NaT", nonexistent="NaT")
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    elif network == "LWD":
        df["timestamp"] = df["timestamp"] + pd.Timedelta(hours=1)
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        logger.warning(f"Unknown network for tz handling: {network} → assuming UTC")
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df

def regrid_to_30min_snow(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, exact_only: bool = False) -> pd.DataFrame:
    """Snowheight regrid.
    Normal (other networks): nearest 14 min + ffill (persists).
    DWD daily: exact match only at 06:00 UTC, no ffill, no tolerance.
    """
    if df.empty:
        idx = pd.date_range(start, end, freq="30min", tz="UTC")
        return pd.DataFrame({"timestamp": idx, "value": np.nan})

    df = df.copy().set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep='first')]

    target_idx = pd.date_range(start, end, freq="30min", tz="UTC")
    regridded = pd.DataFrame(index=target_idx, columns=["value"], dtype=float)
    regridded.index.name = "timestamp"

    if exact_only:
        # DWD daily: only the exact 06:00 slot gets the value. Everything else stays missing.
        regridded["value"] = df["value"].reindex(target_idx)
    else:
        regridded["value"] = df["value"].reindex(
            target_idx,
            method="nearest",
            tolerance=pd.Timedelta(minutes=14)
        ).ffill()

    regridded = regridded.reset_index()
    regridded["is_missing"] = regridded["value"].isna()

    for col in ["station", "name", "height", "lat", "lon", "parameter", "source_file"]:
        if col in df.columns and not df[col].empty:
            regridded[col] = df[col].iloc[0]

    return regridded
# ==================== NETWORK PARSERS (identical) ====================

def parse_dwd(path: Path) -> Optional[pd.DataFrame]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    df = pd.DataFrame(data.get("datapoints", []))
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # DWD daily snow (resolution="daily"): one value per day at 06:00 UTC (morning obs)
    resolution = meta.get("resolution", "").lower().strip()
    if resolution == "daily":
        df["timestamp"] = df["timestamp"].dt.normalize() + pd.Timedelta(hours=6)

    non_ts = [c for c in df.columns if c != "timestamp"]
    if non_ts:
        value_col = non_ts[0]
        df = df.rename(columns={value_col: "value"})
    else:
        logger.warning(f"No value column in DWD {path}")
        return None

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.loc[df["value"] == -999, "value"] = np.nan

    df["name"] = meta.get("name")
    df["height"] = pd.to_numeric(meta.get("hoehe") or meta.get("height"), errors="coerce")
    df["lat"] = pd.to_numeric(meta.get("breite"), errors="coerce")
    df["lon"] = pd.to_numeric(meta.get("laenge"), errors="coerce")
    return df
def parse_zamg(path: Path) -> Optional[pd.DataFrame]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    ts_list = data.get("timestamps", [])
    if not ts_list: return None
    ts = pd.to_datetime(ts_list, utc=True, errors="coerce")
    features = data.get("features", [])
    if not features: return None
    props = features[0].get("properties", {})
    params = props.get("parameters", {})
    value_key = None
    for k in ["tl", "ff", "hs", "rf", "rr", "sh"]:
        if k in params and "data" in params[k]:
            value_key = k
            break
    if value_key is None: return None
    val_data = params[value_key]["data"]
    df = pd.DataFrame({"timestamp": ts, "value": val_data})
    meta = data.get("station_metadata", {}) or {}
    df["name"] = meta.get("name") or meta.get("clean_name")
    df["height"] = pd.to_numeric(meta.get("altitude"), errors="coerce")
    geom = features[0].get("geometry", {})
    coords = geom.get("coordinates", [None, None])
    df["lon"] = pd.to_numeric(coords[0], errors="coerce")
    df["lat"] = pd.to_numeric(coords[1], errors="coerce")
    return df

def parse_meteosuisse(path: Path) -> Optional[pd.DataFrame]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data.get("data", []))
    if df.empty: return None
    df["timestamp"] = pd.to_datetime(df["time"], format="%d.%m.%Y %H:%M", errors="coerce")
    df = df.rename(columns={"value": "value"})
    df["name"] = data.get("name")
    df["height"] = pd.to_numeric(data.get("hoehe"), errors="coerce")
    df["lat"] = pd.to_numeric(data.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(data.get("lon"), errors="coerce")
    return df

def parse_lwd(path: Path) -> Optional[pd.DataFrame]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data.get("data", []))
    if df.empty: return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.rename(columns={"value": "value"})
    df["name"] = data.get("name")
    df["height"] = pd.to_numeric(data.get("hoehe"), errors="coerce")
    df["lat"] = pd.to_numeric(data.get("latitude"), errors="coerce")
    df["lon"] = pd.to_numeric(data.get("longitude"), errors="coerce")
    return df

def parse_hydro_generic(path: Path) -> Optional[pd.DataFrame]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data.get("data", []))
    if df.empty: return None
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    else:
        return None
    df = df.rename(columns={"value": "value"})
    df["name"] = data.get("name")
    df["height"] = pd.to_numeric(data.get("hoehe"), errors="coerce")
    df["lat"] = pd.to_numeric(data.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(data.get("lon"), errors="coerce")
    return df

def parse_suedtirol(path: Path) -> Optional[pd.DataFrame]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data.get("data", []))
    if df.empty: return None
    df["timestamp"] = pd.to_datetime(df["DATE"].str.replace(r"CET|CEST", "", regex=True).str.strip(), errors="coerce")
    df = df.rename(columns={"VALUE": "value"})
    df["name"] = data.get("name")
    df["height"] = pd.to_numeric(data.get("hoehe"), errors="coerce")
    df["lat"] = pd.to_numeric(data.get("latitude"), errors="coerce")
    df["lon"] = pd.to_numeric(data.get("longitude"), errors="coerce")
    return df

NETWORK_PARSERS: Dict[str, Callable[[Path], Optional[pd.DataFrame]]] = {
    "DWD": parse_dwd, "ZAMG": parse_zamg, "Meteosuisse": parse_meteosuisse,
    "LWD": parse_lwd, "HydroT": parse_hydro_generic, "HydroVO": parse_hydro_generic,
    "Suedtirol": parse_suedtirol,
}

# ==================== READ + LIGHT QC (Snowheight specific) ====================

def read_station_json(json_path: Path) -> Optional[pd.DataFrame]:
    station_raw = json_path.parent.name.strip()
    network = canonical_network_name(json_path.parent.parent.name.strip())
    safe_name = safe_station_name(station_raw)

    logger.info(f"READ | {network}/{station_raw} → {json_path.name}")

    parser = NETWORK_PARSERS.get(network)
    if parser is None:
        logger.error(f"READ FAIL | Unknown network: {network}")
        return None

    try:
        df = parser(json_path)
        if df is None or df.empty:
            logger.warning(f"READ FAIL | Empty: {network}/{station_raw}")
            return None

        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # === LIGHT QC for SNOWHEIGHT ===
        unreasonable_mask = (df["value"] < 0) | (df["value"] > 600)
        n_unreasonable = unreasonable_mask.sum()
        if n_unreasonable > 0:
            df.loc[unreasonable_mask, "value"] = np.nan
            logger.info(f"QC | {network}/{station_raw} → {n_unreasonable} values set to NaN ( <0 or >600 cm )")

        df = ensure_utc(df, network)

        df["is_missing"] = df["value"].isna()
        df["station"] = safe_name
        df["parameter"] = PARAMETER
        df["source_file"] = json_path.name

        if "lat" in df.columns and "lon" in df.columns:
            lat0, lon0 = df["lat"].iloc[0], df["lon"].iloc[0]
            new_lat, new_lon = check_and_fix_coordinates(lat0, lon0, station_raw, network)
            df["lat"] = new_lat
            df["lon"] = new_lon

        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first")
        valid = df["value"].notna().sum()
        logger.info(f"READ OK | {network}/{station_raw} → {len(df):,} rows ({valid} valid)")
        return df

    except Exception as e:
        logger.error(f"READ CRASH | {network}/{station_raw}: {type(e).__name__} - {e}")
        return None

# ==================== SAVE (identical) ====================

def save_cleaned(df: pd.DataFrame, station_folder: Path, variable: str):
    if df.empty: return
    station_folder.mkdir(parents=True, exist_ok=True)
    base = station_folder / f"{station_folder.name}_{variable}"

    cols = ["timestamp", "value", "is_missing", "station", "name", "height", "lat", "lon", "parameter", "source_file"]
    existing = [c for c in cols if c in df.columns]
    df[existing].to_parquet(base.with_suffix(".parquet"), compression="snappy", index=False)

    json_data = {
        "name": str(df["name"].iloc[0]) if "name" in df.columns else None,
        "height": float(df["height"].iloc[0]) if "height" in df.columns and pd.notna(df["height"].iloc[0]) else None,
        "lat": float(df["lat"].iloc[0]) if "lat" in df.columns and pd.notna(df["lat"].iloc[0]) else None,
        "lon": float(df["lon"].iloc[0]) if "lon" in df.columns and pd.notna(df["lon"].iloc[0]) else None,
        "data": [{"timestamp": ts.isoformat(), "value": float(v) if pd.notna(v) else None}
                 for ts, v in zip(df["timestamp"], df["value"])]
    }
    with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    logger.info(f"SAVED | {station_folder.name}_{variable} | {len(df):,} rows")

# ==================== PROCESS (identical) ====================

def process_station(json_path: Path) -> Dict:
    station = json_path.parent.name.strip()
    network = canonical_network_name(json_path.parent.parent.name.strip())
    logger.info(f"Processing {network}/{station}")

    stats = {
        "network": network,
        "station": station,
        "total_rows": 0,
        "valid_rows": 0,
        "full_coverage_pct": 0.0,
        "accepted_full": False,
        "rejected_reasons": []
    }

    df = read_station_json(json_path)
    if df is None or df.empty:
        stats["rejected_reasons"].append("read failed or empty")
        return stats

    if "value" not in df.columns:
        stats["rejected_reasons"].append("missing value column")
        return stats

    stats["total_rows"] = len(df)
    stats["valid_rows"] = df["value"].notna().sum()

    safe_station = re.sub(r'[^a-zA-Z0-9_.-]', '_', station)

    # DWD daily snow special case: exact 06:00 UTC only, no ffill, coverage filter OFF
    exact_only = (network == "DWD")

    df_full = regrid_to_30min_snow(df, FULL_PERIOD_START, FULL_PERIOD_END, exact_only=exact_only)
    stats["full_coverage_pct"] = coverage_percent(df_full)

    if exact_only or (stats["full_coverage_pct"] >= COVERAGE_THRESHOLD):
        stats["accepted_full"] = True
        save_cleaned(df_full, FULL_DIR / safe_station, "snow")
    else:
        stats["rejected_reasons"].append(f"full coverage {stats['full_coverage_pct']:.1f}% < {COVERAGE_THRESHOLD}%")

    return stats
# ==================== STATISTICS (identical) ====================
def save_statistics(results: List[Dict]):
    df = pd.DataFrame(results)

    # Make sure boolean columns exist
    if "accepted_full" not in df.columns:
        df["accepted_full"] = False

    # Write main statistics CSV
    df.to_csv(AAAData / "station_statistics_snowheight.csv", index=False, encoding="utf-8-sig")

    # Write rejected list (with network)
    with open(AAAData / "rejected_stations_snowheight.txt", "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            reasons = row.get("rejected_reasons", [])
            if reasons:
                net = row.get("network", "UNKNOWN")
                f.write(f"{net}/{row['station']}: {', '.join(reasons)}\n")

    # Write accepted list
    with open(AAAData / "accepted_full_2020_2025_snowheight.txt", "w", encoding="utf-8") as f:
        for s in df[df["accepted_full"]]["station"].dropna():
            f.write(f"{s}\n")

    # Optional: network summary (useful)
    if not df.empty and "network" in df.columns:
        logger.info("=== NETWORK ACCEPTANCE SUMMARY (Snowheight - full only) ===")
        for net, group in df.groupby("network"):
            total = len(group)
            acc = group["accepted_full"].sum()
            pct = round(acc / total * 100, 1) if total > 0 else 0.0
            logger.info(f"{net:12} | Total: {total:3} | Accepted: {int(acc):3} ({pct:5.1f}%)")

    logger.info("Saved statistics + acceptance/rejection lists (snowheight, full period only)")
# ==================== MAIN (identical) ====================

def main():
    logger.info(f"=== {VARIABLE} PIPELINE START (strict exact regrid, 30% threshold) ===")
    logger.info(f"Input: {ROOT} | Output Data folder: {AAAData}")

    json_files = list_network_jsons(ROOT, list(NETWORK_PARSERS.keys()))

    seen, unique_files = {}, []
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                raw = f.read(20000)
            key = f"{safe_station_name(jf.parent.name)}|{hashlib.md5(raw.encode()).hexdigest()}"
            if key not in seen:
                seen[key] = True
                unique_files.append(jf)
        except Exception:
            unique_files.append(jf)

    logger.info(f"Found {len(unique_files)} unique station JSON files")

    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        results = pool.map(process_station, unique_files)

    save_statistics(results)
    accepted = sum(bool(r.get("accepted_full", r.get("accepted", False))) for r in results)
    logger.info(f"=== {VARIABLE} PIPELINE FINISHED | Accepted: {accepted}/{len(results)} ===")
    logger.info(f"Log: {LOG_FILE}")

if __name__ == "__main__":
    main()

