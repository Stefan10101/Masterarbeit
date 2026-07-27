#!/usr/bin/env python3
"""
Master Thesis - Temperature Data Pipeline (v3.0)
Strict exact regrid, 30% coverage, new folder structure, thorough logging.
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

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT


# ==================== CONFIG ====================
VARIABLE = "TEMP"
PARAMETER = "LT"
ROOT = Path(DATA_ROOT / "temperatur")
LOG_DIR = Path(DATA_ROOT / "code" / "logs" / "temperatur")

DATA_ROOT = ROOT / "Data"
AAAData = DATA_ROOT / "Data"
FULL_DIR = AAAData / "Full_2020-2025"
REJECTED_DIR = AAAData / "rejected_stations"
STATS_DIR = AAAData / "Statistics"

for d in [FULL_DIR, REJECTED_DIR, STATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
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
    name = re.sub(r'[^a-zA-Z0-9Ã¤Ã¶Ã¼Ã„Ã–ÃœÃŸ]', '_', name)
    name = name.replace('Ã¤', 'a').replace('Ã¶', 'o').replace('Ã¼', 'u')
    name = name.replace('Ã„', 'A').replace('Ã–', 'O').replace('Ãœ', 'U')
    name = re.sub(r'_+', '_', name).strip('_')
    return name or "unknown_station"

def coverage_percent(df: pd.DataFrame) -> float:
    return float(df["value"].notna().mean() * 100) if not df.empty else 0.0

def check_and_fix_coordinates(lat: float, lon: float, station: str, network: str) -> tuple:
    if pd.isna(lat) or pd.isna(lon):
        return lat, lon
    if lon > lat:
        logger.warning(f"COORDINATE SWAP | {network}/{station} | lon={lon} > lat={lat} â†’ swapping")
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
        logger.warning(f"Unknown network for tz handling: {network} â†’ assuming UTC")
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df

def regrid_to_30min_strict(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, agg: str = "exact") -> pd.DataFrame:
    if df.empty:
        idx = pd.date_range(start, end, freq="30min", tz="UTC")
        return pd.DataFrame({"timestamp": idx, "value": np.nan})

    df = df.copy().set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    target_idx = pd.date_range(start, end, freq="30min", tz="UTC")
    regridded = pd.DataFrame(index=target_idx, columns=["value"], dtype=float)
    regridded.index.name = "timestamp"

    if agg == "sum":
        summed = df["value"].resample("30min", origin="start_day", label="right").sum()
        regridded["value"] = summed.reindex(target_idx)
    else:
        common = df.index.intersection(target_idx)
        regridded.loc[common, "value"] = df.loc[common, "value"].values

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
    if df.empty: return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    non_ts = [c for c in df.columns if c != "timestamp"]
    if non_ts:
        df = df.rename(columns={non_ts[0]: "value"})
    else:
        return None
    df["name"] = meta.get("name")
    df["height"] = pd.to_numeric(meta.get("hoehe"), errors="coerce")
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

# ==================== READ + LIGHT QC (Temperature specific) ====================

def read_station_json(json_path: Path) -> Optional[pd.DataFrame]:
    station_raw = json_path.parent.name.strip()
    network = json_path.parent.parent.name.strip()
    safe_name = safe_station_name(station_raw)

    logger.info(f"READ | {network}/{station_raw} â†’ {json_path.name}")

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

        # === LIGHT QC for TEMPERATURE ===
        unreasonable_mask = (df["value"] < -40) | (df["value"] > 50)
        n_unreasonable = unreasonable_mask.sum()
        if n_unreasonable > 0:
            df.loc[unreasonable_mask, "value"] = np.nan
            logger.info(f"QC | {network}/{station_raw} â†’ {n_unreasonable} values set to NaN (outside -40..+50 Â°C)")

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
        logger.info(f"READ OK | {network}/{station_raw} â†’ {len(df):,} rows ({valid} valid)")
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
    station_raw = json_path.parent.name.strip()
    network = json_path.parent.parent.name.strip()
    safe_name = safe_station_name(station_raw)
    station_folder = FULL_DIR / safe_name

    stats = {
        "network": network, "station": safe_name, "station_raw": station_raw,
        "total_rows": 0, "valid_rows": 0, "coverage_pct": 0.0,
        "accepted": False, "rejected_reason": ""
    }

    df = read_station_json(json_path)
    if df is None or df.empty:
        stats["rejected_reason"] = "read failed or empty after parser"
        return stats

    stats["total_rows"] = len(df)
    stats["valid_rows"] = int(df["value"].notna().sum())

    agg_mode = "sum" if VARIABLE == "PRECIP" else "exact"
    df_full = regrid_to_30min_strict(df, FULL_PERIOD_START, FULL_PERIOD_END, agg=agg_mode)
    stats["coverage_pct"] = coverage_percent(df_full)

    if stats["coverage_pct"] >= COVERAGE_THRESHOLD:
        stats["accepted"] = True
        save_cleaned(df_full, station_folder, VARIABLE.lower())
        logger.info(f"ACCEPTED | {network}/{station_raw} | coverage={stats['coverage_pct']:.1f}%")
    else:
        stats["rejected_reason"] = f"coverage {stats['coverage_pct']:.1f}% < {COVERAGE_THRESHOLD}%"
        logger.warning(f"REJECTED | {network}/{station_raw} | {stats['rejected_reason']}")

    return stats

# ==================== STATISTICS (identical) ====================

def save_statistics(all_stats: List[Dict]):
    df_stats = pd.DataFrame(all_stats)

    with open(STATS_DIR / f"coverage_summary_{VARIABLE.lower()}.txt", "w", encoding="utf-8") as f:
        f.write(f"Variable: {VARIABLE}\nThreshold: {COVERAGE_THRESHOLD}%\nPeriod: {FULL_PERIOD_START.date()} â€“ {FULL_PERIOD_END.date()}\n\n")
        f.write("station_raw;network;coverage_pct;accepted;rejected_reason\n")
        for _, row in df_stats.iterrows():
            f.write(f"{row['station_raw']};{row['network']};{row['coverage_pct']:.2f};{row['accepted']};{row['rejected_reason']}\n")

    with open(STATS_DIR / f"network_acceptance_{VARIABLE.lower()}.txt", "w", encoding="utf-8") as f:
        f.write(f"Variable: {VARIABLE}\n\n")
        for net, g in df_stats.groupby("network"):
            total = len(g)
            acc = g["accepted"].sum()
            pct = (acc / total * 100) if total > 0 else 0.0
            f.write(f"{net}: {acc}/{total} accepted ({pct:.1f}%)\n")

    with open(REJECTED_DIR / f"rejected_stations_{VARIABLE.lower()}.txt", "w", encoding="utf-8") as f:
        for _, row in df_stats.iterrows():
            if not row["accepted"]:
                f.write(f"{row['station_raw']}_{row['network']}: {row['rejected_reason']}\n")

    logger.info(f"STATISTICS WRITTEN to {STATS_DIR} and {REJECTED_DIR}")

# ==================== MAIN (identical) ====================

def main():
    logger.info(f"=== {VARIABLE} PIPELINE START (strict exact regrid, 30% threshold) ===")
    logger.info(f"Input: {ROOT} | Output Data folder: {AAAData}")

    json_files = []
    for net in NETWORK_PARSERS.keys():
        net_dir = ROOT / net
        if net_dir.exists():
            json_files.extend(list(net_dir.rglob("*.json")))

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
    accepted = sum(r.get("accepted", False) for r in results)
    logger.info(f"=== {VARIABLE} PIPELINE FINISHED | Accepted: {accepted}/{len(results)} ===")
    logger.info(f"Log: {LOG_FILE}")

if __name__ == "__main__":
    main()

