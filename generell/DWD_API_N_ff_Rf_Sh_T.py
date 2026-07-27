#!/usr/bin/env python3
"""
Robust DWD Download Script (consolidated & improved for master thesis)
- Finds ALL stations south of 48.5Â°N with data overlap 2020-2025
- Downloads recent/*_akt.zip (current/ongoing data, including latest 2025) + one historical zip per station
- Simple & safe zip selection
- Logs saved to the requested folder
"""

import requests
import logging
from pathlib import Path
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from datetime import datetime

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT


# ====================== USER CONFIG ======================
LOG_DIR = Path(DATA_ROOT / "code" / "logs" / "dwd" / "api")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "dwd_download.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

LAT_THRESHOLD = 48.5
TARGET_START = datetime(2020, 1, 1)
TARGET_END   = datetime(2025, 12, 31)

# ZIP files destination (as requested)
ZIP_DIR = Path(DATA_ROOT / "dwd" / "zip")
ZIP_DIR.mkdir(parents=True, exist_ok=True)

# Station lists are also saved here
STATION_LIST_DIR = ZIP_DIR

# ====================== VARIABLE CONFIG ======================
VARIABLES = {
    "tu": {
        "name": "air_temperature_humidity",
        "folder": "air_temperature",
        "station_list": "zehn_min_tu_Beschreibung_Stationen.txt",
        "zip_prefix": "10minutenwerte_TU_",
        "recent_name": "10minutenwerte_TU_{sid}_akt.zip",
        "id_regex": r"10minutenwerte_TU_(\d{5})",
        "base_url": "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/",
    },
    "rr": {
        "name": "precipitation",
        "folder": "precipitation",
        "station_list": "zehn_min_rr_Beschreibung_Stationen.txt",
        "zip_prefix": "10minutenwerte_nieder_",
        "recent_name": "10minutenwerte_nieder_{sid}_akt.zip",
        "id_regex": r"10minutenwerte_nieder_(\d{5})",
        "base_url": "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/",
    },
    "ff": {
        "name": "wind",
        "folder": "wind",
        "station_list": "zehn_min_ff_Beschreibung_Stationen.txt",
        "zip_prefix": "10minutenwerte_wind_",
        "recent_name": "10minutenwerte_wind_{sid}_akt.zip",
        "id_regex": r"10minutenwerte_wind_(\d{5})",
        "base_url": "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/",
    },
    "sh": {
        "name": "snow_height",
        "folder": "more_precip",
        "station_list": "RR_Tageswerte_Beschreibung_Stationen.txt",
        "zip_prefix": "tageswerte_RR_",
        "recent_name": "tageswerte_RR_{sid}_akt.zip",
        "id_regex": r"tageswerte_RR_(\d{5})",
        "base_url": "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/",
    },
}

def retry_request(url, retries=3, backoff=2, timeout=180):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(backoff ** attempt)
    return None

def parse_station_list_robust(station_path: Path) -> pd.DataFrame:
    lines = []
    with open(station_path, "r", encoding="latin1") as f:
        for line in f:
            if line.strip().startswith("Stations_id") or line.strip().startswith("---"):
                continue
            cleaned = " ".join(line.strip().split())
            lines.append(cleaned)

    data = []
    for line in lines:
        parts = line.split()
        if len(parts) < 8: continue
        try:
            sid = parts[0].strip()
            von = parts[1].strip()
            bis = parts[2].strip()
            hoehe = parts[3].strip()
            breite = parts[4].strip()
            laenge_idx = 5
            while laenge_idx < len(parts) and not (parts[laenge_idx].replace(".", "", 1).replace("-", "", 1).isdigit()):
                laenge_idx += 1
            if laenge_idx >= len(parts): continue
            laenge = parts[laenge_idx].strip()
            name = " ".join(parts[laenge_idx + 1:-1]) if len(parts) > laenge_idx + 2 else " ".join(parts[laenge_idx + 1:])
            bundesland = parts[-1]
            data.append([sid, von, bis, hoehe, breite, laenge, name, bundesland])
        except:
            continue

    df = pd.DataFrame(data, columns=["Stations_id", "von_datum", "bis_datum", "Stationshoehe",
                                     "geoBreite", "geoLaenge", "Stationsname", "Bundesland"])
    df["Stations_id"] = df["Stations_id"].astype(str).str.zfill(5)
    df["geoBreite"] = pd.to_numeric(df["geoBreite"], errors="coerce")
    df["von_datum"] = pd.to_datetime(df["von_datum"], format="%Y%m%d", errors="coerce")
    df["bis_datum"] = pd.to_datetime(df["bis_datum"], format="%Y%m%d", errors="coerce")
    return df

def get_south_stations(df: pd.DataFrame) -> set:
    if df.empty: return set()
    mask = (
        (df["geoBreite"] < LAT_THRESHOLD) &
        (df["von_datum"] <= TARGET_END) &
        ((df["bis_datum"] >= TARGET_START) | df["bis_datum"].isna())
    )
    south = df[mask].copy()
    logger.info(f"    South stations (<{LAT_THRESHOLD}Â°N + 2020-2025 overlap): {len(south)}")
    return set(south["Stations_id"])

def build_historical_mapping(config: dict, station_ids: set) -> dict:
    historical_url = f"{config['base_url']}{config['folder']}/historical/"
    logger.info(f"    Building historical mapping from: {historical_url}")
    r = retry_request(historical_url)
    if not r: return {}

    id_pattern = re.compile(config["id_regex"])
    mapping = {}
    for link in re.findall(r'href="([^"]*?\.zip)"', r.text):
        if config["zip_prefix"] not in link: continue
        match = id_pattern.search(link)
        if match:
            sid = match.group(1)
            if sid in station_ids:
                mapping[sid] = [link]
    logger.info(f"    Found historical ZIPs for {len(mapping)} of the south stations")
    return mapping

def download_file(url: str, dest: Path) -> bool:
    if dest.exists(): return False
    r = retry_request(url)
    if r and r.status_code == 200:
        with open(dest, "wb") as f:
            f.write(r.content)
        return True
    return False

def process_variable(var_key: str, config: dict):
    logger.info(f"\n{'='*70}")
    logger.info(f"=== Processing {var_key.upper()} ({config['name']}) ===")
    logger.info(f"{'='*70}")

    station_list_urls = [
        f"{config['base_url']}{config['folder']}/historical/{config['station_list']}",
        f"{config['base_url']}{config['folder']}/recent/{config['station_list']}",
    ]
    station_path = STATION_LIST_DIR / config['station_list']
    r = None
    for url in station_list_urls:
        r = retry_request(url)
        if r: break
    if not r:
        logger.error(f"  Could not download station list for {var_key}")
        return 0, 0
    with open(station_path, "wb") as f:
        f.write(r.content)

    stations = parse_station_list_robust(station_path)
    station_ids = get_south_stations(stations)
    if not station_ids: return 0, 0

    hist_mapping = build_historical_mapping(config, station_ids)

    def download_for_station(sid):
        new_count = 0
        recent_name = config["recent_name"].format(sid=sid)
        recent_url = f"{config['base_url']}{config['folder']}/recent/{recent_name}"
        if download_file(recent_url, ZIP_DIR / recent_name):
            new_count += 1

        for hist_name in hist_mapping.get(sid, []):
            hist_url = f"{config['base_url']}{config['folder']}/historical/{hist_name}"
            if download_file(hist_url, ZIP_DIR / hist_name):
                new_count += 1
        return sid, new_count

    logger.info(f"  Starting parallel download ({len(station_ids)} stations, max 8 workers)...")
    total_new_files = 0
    stations_with_data = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_for_station, sid): sid for sid in station_ids}
        for future in as_completed(futures):
            sid, cnt = future.result()
            total_new_files += cnt
            if cnt > 0: stations_with_data += 1

    logger.info(f"  FINISHED {var_key.upper()}: {total_new_files} new file(s) | {stations_with_data}/{len(station_ids)} stations got data")
    return total_new_files, stations_with_data

def main():
    logger.info("=== DWD Robust Download ===")
    logger.info(f"Target: {TARGET_START.date()} â€“ {TARGET_END.date()}, stations south of {LAT_THRESHOLD}Â°N")
    logger.info(f"ZIPs will be saved to: {ZIP_DIR}")

    grand_total_files = 0
    for var_key, config in VARIABLES.items():
        try:
            files, _ = process_variable(var_key, config)
            grand_total_files += files
        except Exception as e:
            logger.exception(f"Error in {var_key}: {e}")

    logger.info(f"\n{'='*70}")
    logger.info(f"TOTAL new files downloaded: {grand_total_files}")
    logger.info(f"ZIPs ready in: {ZIP_DIR}")
    logger.info("Now run dwd_extract_to_json.py")

if __name__ == "__main__":
    main()
