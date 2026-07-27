#!/usr/bin/env python3
"""
DWD Extraction to JSON
Reads ZIPs from the new ZIP folder and creates clean per-station JSON files
with strict 2020-01-01 to 2025-12-31 data only.
"""

import pandas as pd
import json
from pathlib import Path
import logging
from datetime import datetime
from zipfile import ZipFile
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT


# ====================== CONFIGURATION ======================
# ZIP source (where dwd_download_robust.py saves the files)
ZIPS_DIR = Path(DATA_ROOT / "dwd" / "zip")
ZIPS_DIR.mkdir(parents=True, exist_ok=True)

# Temporary extraction folder (can be deleted later)
EXTRACTED_BASE = Path(DATA_ROOT / "dwd" / "extracted")
EXTRACTED_BASE.mkdir(parents=True, exist_ok=True)

# Final JSON output folders
OUTPUT_BASES = {
    "rr": Path(DATA_ROOT / "niederschlag" / "dwd"),
    "ff": Path(DATA_ROOT / "wind" / "dwd"),
    "tu": Path(DATA_ROOT / "temperatur" / "dwd"),
    "rf": Path(DATA_ROOT / "luftfeuchte" / "dwd"),
    "sh": Path(DATA_ROOT / "schneehoehe" / "dwd"),
}

# Logging
LOG_DIR = Path(DATA_ROOT / "code" / "logs" / "dwd" / "extraction")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "dwd_extract.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

TARGET_START = pd.Timestamp("2020-01-01")
TARGET_END = pd.Timestamp("2025-12-31 23:59")
LAT_THRESHOLD = 48.5

# ====================== REST OF THE SCRIPT ======================
# (The functions below are the same as the working version I gave you earlier)

STATION_FILE_PATTERNS = {
    "rr": ["*precipitation*.txt", "BESCHREIBUNG_obsgermany-climate-10min-precipitation.txt", "zehn_min_rr_Beschreibung_Stationen.txt"],
    "ff": ["*Beschreibung*wind*.txt", "zehn_min_ff_Beschreibung_Stationen.txt"],
    "tu": ["*Beschreibung*tu*.txt", "zehn_min_tu_Beschreibung_Stationen.txt"],
    "rf": ["*Beschreibung*tu*.txt", "zehn_min_tu_Beschreibung_Stationen.txt"],
    "sh": ["RR_Tageswerte_Beschreibung_Stationen.txt", "*Beschreibung*more_precip*.txt"],
}

VARIABLE_CONFIG = {
    "rr": {
        "name": "precipitation", "zip_glob": "10minutenwerte_nieder_*.zip",
        "column": "RWS_10", "json_field": "precip_10min_mm",
        "description": "Niederschlag 10min", "resolution": "10 Minuten",
        "date_format": "%Y%m%d%H%M", "time_col": "MESS_DATUM"
    },
    "ff": {
        "name": "wind", "zip_glob": "10minutenwerte_wind_*.zip",
        "column": "FF_10", "json_field": "wind_speed_10min_ms",
        "description": "Windgeschwindigkeit 10min", "resolution": "10 Minuten",
        "date_format": "%Y%m%d%H%M", "time_col": "MESS_DATUM"
    },
    "tu": {
        "name": "air_temperature", "zip_glob": "10minutenwerte_TU_*.zip",
        "column": "TT_10", "json_field": "air_temperature_2m_degC",
        "description": "Lufttemperatur 10min", "resolution": "10 Minuten",
        "date_format": "%Y%m%d%H%M", "time_col": "MESS_DATUM"
    },
    "rf": {
        "name": "relative_humidity", "zip_glob": "10minutenwerte_TU_*.zip",
        "column": "RF_10", "json_field": "relative_humidity_2m_percent",
        "description": "Relative Luftfeuchte 10min", "resolution": "10 Minuten",
        "date_format": "%Y%m%d%H%M", "time_col": "MESS_DATUM"
    },
    "sh": {
        "name": "snow_height", "zip_glob": "tageswerte_RR_*.zip",
        "column": "SH_TAG", "json_field": "snow_height_cm",
        "description": "SchneehÃ¶he tÃ¤glich", "resolution": "daily",
        "date_format": "%Y%m%d", "time_col": "MESS_DATUM"
    },
}

# ====================== FUNCTIONS (same as before) ======================

def find_station_file(var_code: str) -> Path | None:
    patterns = STATION_FILE_PATTERNS.get(var_code, [])
    search_roots = [ZIPS_DIR.parent, Path.cwd(), Path.cwd().parent]
    for root in search_roots:
        if not root.exists(): continue
        for pat in patterns:
            matches = list(root.rglob(pat))
            for m in matches:
                if "beschreibung" in m.name.lower() or "stationen" in m.name.lower():
                    return m
            if matches: return matches[0]
    return None

def parse_station_file(station_path: Path) -> pd.DataFrame:
    logger.info(f"Parsing station file: {station_path.name}")
    lines = []
    with open(station_path, "r", encoding="latin1") as f:
        for line in f:
            if line.strip().startswith("Stations_id") or line.strip().startswith("-----------"): continue
            lines.append(" ".join(line.strip().split()))
    data = []
    for line in lines:
        parts = line.split()
        if len(parts) < 8: continue
        try:
            sid, von, bis, hoehe, breite = parts[0], parts[1], parts[2], parts[3], parts[4]
            laenge_idx = 5
            while laenge_idx < len(parts) and not (parts[laenge_idx].replace(".", "", 1).replace("-", "", 1).isdigit()): laenge_idx += 1
            if laenge_idx >= len(parts): continue
            laenge = parts[laenge_idx]
            name = " ".join(parts[laenge_idx + 1:-1]) if len(parts) > laenge_idx + 2 else " ".join(parts[laenge_idx + 1:])
            bundesland = parts[-1]
            data.append([sid, von, bis, hoehe, breite, laenge, name, bundesland])
        except: continue
    stations = pd.DataFrame(data, columns=["Stations_id", "von_datum", "bis_datum", "Stationshoehe", "geoBreite", "geoLaenge", "Stationsname", "Bundesland"])
    stations["Stations_id"] = stations["Stations_id"].astype(str).str.zfill(5)
    stations["geoBreite"] = pd.to_numeric(stations["geoBreite"], errors="coerce")
    stations["geoLaenge"] = pd.to_numeric(stations["geoLaenge"], errors="coerce")
    stations["Stationshoehe"] = pd.to_numeric(stations["Stationshoehe"], errors="coerce")
    stations["von_datum"] = pd.to_datetime(stations["von_datum"], format="%Y%m%d", errors="coerce")
    stations["bis_datum"] = pd.to_datetime(stations["bis_datum"], format="%Y%m%d", errors="coerce")
    stations = stations[(stations["geoBreite"] > 20) & (stations["geoBreite"] < 90)].copy()
    return stations

def get_valid_stations(stations: pd.DataFrame) -> tuple:
    if stations.empty: return set(), pd.DataFrame()
    valid = stations[(stations["geoBreite"] < LAT_THRESHOLD) & (stations["bis_datum"] >= TARGET_START) & (stations["von_datum"] <= TARGET_END)].copy()
    logger.info(f"  Filtered south of {LAT_THRESHOLD}Â°N + overlap 2020-2025: {len(valid)} stations")
    return set(valid["Stations_id"]), valid

def extract_zips(var_code: str, config: dict) -> Path:
    extracted_root = EXTRACTED_BASE / var_code
    extracted_root.mkdir(parents=True, exist_ok=True)
    zips = sorted(ZIPS_DIR.glob(config["zip_glob"]))
    logger.info(f"Found {len(zips)} ZIPs for {var_code.upper()}")
    extracted_count = 0
    for zip_path in zips:
        sid_match = re.search(r"_(\d{5})", zip_path.name)
        if not sid_match: continue
        sid = sid_match.group(1)
        station_dir = extracted_root / sid
        station_dir.mkdir(exist_ok=True)
        if any(station_dir.glob("produkt_*.txt")): continue
        try:
            with ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    if "produkt_" in member.lower() and member.endswith(".txt"):
                        zf.extract(member, station_dir)
                        extracted_count += 1
                        break
        except Exception as e:
            logger.warning(f"  Extraction failed {zip_path.name}: {e}")
    logger.info(f"  Extracted {extracted_count} new produkt files for {var_code}")
    return extracted_root

def process_station(sid: str, station_dir: Path, var_code: str, config: dict, valid_stations_df: pd.DataFrame, output_base: Path) -> int:
    data_files = list(station_dir.glob("produkt_*.txt"))
    if not data_files: return 0
    dfs = []
    for f in data_files:
        try:
            df = pd.read_csv(f, sep=";", encoding="latin1", skipinitialspace=True,
                             usecols=["STATIONS_ID", config["time_col"], config["column"]],
                             dtype={"STATIONS_ID": str, config["column"]: float}, on_bad_lines="skip")
            dfs.append(df)
        except: pass
    if not dfs: return 0
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=[config["time_col"]])
    df[config["time_col"]] = pd.to_datetime(df[config["time_col"]], format=config["date_format"], errors="coerce")
    df = df.dropna(subset=[config["time_col"]])

    # Strict date filter
    df = df[(df[config["time_col"]] >= TARGET_START) & (df[config["time_col"]] <= TARGET_END)]
    if df.empty: return 0

    if config["resolution"] != "daily":
        df["timestamp"] = df[config["time_col"]].dt.strftime("%Y-%m-%dT%H:%M:00")
    else:
        df["timestamp"] = df[config["time_col"]].dt.strftime("%Y-%m-%d")
    df = df[["timestamp", config["column"]]].rename(columns={config["column"]: config["json_field"]}).sort_values("timestamp")

    meta_row = valid_stations_df[valid_stations_df["Stations_id"] == sid].iloc[0]
    metadata = {
        "station_id": sid,
        "name": str(meta_row["Stationsname"]).strip(),
        "hoehe": float(meta_row["Stationshoehe"]) if pd.notna(meta_row["Stationshoehe"]) else None,
        "breite": float(meta_row["geoBreite"]),
        "laenge": float(meta_row["geoLaenge"]),
    }
    station_json = {
        "metadata": {**metadata, "data_start": df["timestamp"].iloc[0], "data_end": df["timestamp"].iloc[-1],
                     "resolution": config["resolution"], "num_measurements": len(df), "parameter": config["description"]},
        "datapoints": df.to_dict(orient="records")
    }

    station_name_clean = metadata["name"].replace(" ", "_").replace("/", "_").replace("-", "_").replace("(", "").replace(")", "").replace(".", "")
    output_folder = output_base / station_name_clean
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / (f"{var_code}_10min.json" if config["resolution"] != "daily" else "snow_daily.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(station_json, f, ensure_ascii=False, separators=(",", ":"))
    logger.info(f"  âœ“ {var_code.upper()} {sid} ({metadata['name']}): {len(df):,} measurements")
    return 1

def process_variable(var_code: str, config: dict):
    logger.info(f"\n{'='*60}\n=== Processing {var_code.upper()} ({config['description']}) ===\n{'='*60}")
    station_path = find_station_file(var_code)
    if not station_path or not station_path.exists():
        logger.error(f"  Station list not found for {var_code}")
        return 0
    stations = parse_station_file(station_path)
    valid_ids, valid_stations_df = get_valid_stations(stations)
    if not valid_ids:
        logger.warning(f"  No valid stations for {var_code}")
        return 0
    extracted_root = extract_zips(var_code, config)
    output_base = OUTPUT_BASES[var_code]
    output_base.mkdir(parents=True, exist_ok=True)
    station_folders = [d for d in sorted(extracted_root.iterdir()) if d.is_dir() and d.name.zfill(5) in valid_ids]
    logger.info(f"  Processing {len(station_folders)} stations in parallel...")
    processed = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_station, d.name.zfill(5), d, var_code, config, valid_stations_df, output_base): d for d in station_folders}
        for future in as_completed(futures):
            processed += future.result()
    logger.info(f"  FINISHED {var_code.upper()}: {processed} stations")
    return processed

def main():
    logger.info("=== DWD Extraction to JSON ===")
    logger.info(f"ZIPS source: {ZIPS_DIR}")
    total = 0
    for var_code in ["tu", "rf", "rr", "ff", "sh"]:
        if var_code in VARIABLE_CONFIG:
            try:
                total += process_variable(var_code, VARIABLE_CONFIG[var_code])
            except Exception as e:
                logger.exception(f"Error in {var_code}: {e}")
    logger.info(f"\n=== ALL DONE | Total stations processed: {total} ===")

if __name__ == "__main__":
    main()
