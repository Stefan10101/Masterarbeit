#!/usr/bin/env python3
r""DATA_ROOT / "
sÃ¼dtirol meteo api download script - extended version

downloads 10-minute data for the following variables for all sÃ¼dtirol stations that provide them:
    n  = precipitation
    wg = wind speed
    hs = snow height
    lf = relative humidity (luftfeuchte)
    lt = air temperature (new)

data is saved as one json file per variable per station.
already downloaded stations" / "variables with valid data are skipped automatically.

folder structure created:
    ..." / "daten" / "niederschlag" / "suedtirol" / "...
    ..." / "daten" / "wind" / "suedtirol" / "...
    ..." / "daten" / "schneehoehe" / "suedtirol" / "...
    ..." / "daten" / "luftfeuchte" / "suedtirol" / "...
    ..." / "daten" / "temperatur" / "suedtirol" / "...          <-- new

test mode supported (set test_mode = true for quick testing).
"""

import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT


# =============================================
# CONFIGURATION
# =============================================
BASE_URL = DATA_ROOT / "http:" / "daten.buergernetz.bz.it" / "services" / "meteo" / "v1"
YEARS = range(2020, 2026)

# Test settings - set False for full production run
TEST_MODE = False
TEST_MAX_CHECK = 30      # stations to check in test mode
TEST_MAX_DOWNLOAD = 10   # stations to download in test mode

# Output folders per variable (extended with temperature)
OUTPUT_BASES = {
    "N":  Path(DATA_ROOT / "niederschlag" / "suedtirol"),
    "WG": Path(DATA_ROOT / "wind" / "suedtirol"),
    "HS": Path(DATA_ROOT / "schneehoehe" / "suedtirol"),
    "LF": Path(DATA_ROOT / "luftfeuchte" / "suedtirol"),
    "LT": Path(DATA_ROOT / "temperatur" / "suedtirol"),  # NEW
}

# Sensor code â†’ variable name used in filename
SENSOR_MAP = {
    "N":  "precip",
    "WG": "windspeed",
    "HS": "snowheight",
    "LF": "relative_humidity",
    "LT": "temperature",          # NEW
}


def get_all_stations() -> List[Dict]:
    """Fetch all stations with metadata (one request)."""
    url = f"{BASE_URL}/stations"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    stations = []
    for feature in data.get("features", []):
        props = feature["properties"]
        code = props["SCODE"]
        name = props["NAME_D"].strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
        stations.append({
            "code": code,
            "name": name,
            "hoehe": props["ALT"],
            "longitude": props["LONG"],
            "latitude": props["LAT"]
        })
    return stations


def get_available_sensors(station_code: str) -> set:
    """Fetch all sensor types for one station with a single API call (efficient + rate-limit safe)."""
    url = f"{BASE_URL}/sensors"
    params = {"station_code": station_code}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return {s["TYPE"] for s in r.json()}
    except Exception:
        return set()


def download_yearly_data(station_code: str, sensor_code: str, year: int) -> list:
    """Download one year of data for a specific sensor."""
    url = f"{BASE_URL}/timeseries"
    params = {
        "station_code": station_code,
        "sensor_code": sensor_code,
        "output_format": "CSV",
        "date_from": f"{year}0101",
        "date_to": f"{year}1231"
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        lines = r.text.strip().splitlines()
        return lines[1:] if lines and lines[0].startswith("DATE,VALUE") else []
    except Exception:
        return []


def is_already_downloaded(station: Dict, sensor_code: str, var_name: str) -> bool:
    """Check if this variable for this station is already properly saved."""
    output_base = OUTPUT_BASES[sensor_code]
    station_folder = output_base / station["name"]
    outfile = station_folder / f"{station['name']}_{var_name}.json"

    if not outfile.exists():
        return False

    try:
        with open(outfile, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "data" in data and len(data["data"]) > 5:
            print(f"   â†’ {var_name}: already exists with {len(data['data'])} records â†’ skipping")
            return True
    except Exception:
        pass  # broken file â†’ re-download

    return False


def save_variable_json(station: Dict, sensor_code: str, var_name: str, all_rows: list):
    """Save data for one variable as its own JSON file."""
    data_points = []
    for row in all_rows:
        if row.strip():
            date, value = row.split(",", 1)
            try:
                val = float(value) if value.strip() else None
            except ValueError:
                val = None
            data_points.append({"DATE": date, "VALUE": val})

    full_data = {
        "name": station["name"],
        "hoehe": station["hoehe"],
        "longitude": station["longitude"],
        "latitude": station["latitude"],
        "data": data_points
    }

    output_base = OUTPUT_BASES[sensor_code]
    station_folder = output_base / station["name"]
    station_folder.mkdir(parents=True, exist_ok=True)

    outfile = station_folder / f"{station['name']}_{var_name}.json"

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)

    print(f"   â†’ {var_name}: {len(data_points)} records saved")


def main():
    print("=== SÃ¼dtirol Meteo API Download ===")
    print("Variables: Precipitation (N), Wind (WG), Snow height (HS), Relative humidity (LF), Air temperature (LT)")
    print(f"TEST_MODE = {TEST_MODE}\n")

    # 1. Get all stations
    print("Loading all stations...")
    all_stations = get_all_stations()
    stations_to_check = all_stations[:TEST_MAX_CHECK] if TEST_MODE else all_stations
    print(f"Total stations found: {len(all_stations)} | Checking: {len(stations_to_check)}\n")

    # 2. Efficient sensor check (one call per station)
    print("Checking available sensors for all stations (efficient single-call method)...")
    station_sensor_map: Dict[str, List[str]] = {}

    for station in stations_to_check:
        available = get_available_sensors(station["code"])
        target_sensors = [s for s in SENSOR_MAP if s in available]
        if target_sensors:
            station_sensor_map[station["code"]] = target_sensors
            print(f"  {station['name']} ({station['code']}) â†’ {target_sensors}")

    print(f"\nFound {len(station_sensor_map)} stations with at least one target sensor.\n")

    # 3. Download data (skip already downloaded)
    download_list = list(station_sensor_map.keys())
    if TEST_MODE:
        download_list = download_list[:TEST_MAX_DOWNLOAD]

    for idx, code in enumerate(download_list, 1):
        station = next(s for s in all_stations if s["code"] == code)
        name = station["name"]
        print(f"[{idx:3d}/{len(download_list)}] Processing {name} ({code})")

        for sensor_code in station_sensor_map[code]:
            var_name = SENSOR_MAP[sensor_code]

            if is_already_downloaded(station, sensor_code, var_name):
                continue

            print(f"   â†’ downloading {var_name}...")

            all_rows = []
            for year in YEARS:
                rows = download_yearly_data(code, sensor_code, year)
                all_rows.extend(rows)
                print(f"      {year}: {len(rows)} records")
                time.sleep(0.25)   # be gentle with the API

            save_variable_json(station, sensor_code, var_name, all_rows)
            time.sleep(0.15)

        print(f"   â†’ finished {name}\n")

    print("=== DOWNLOAD FINISHED ===")
    if TEST_MODE:
        print("Test mode active. Set TEST_MODE=False for full production run.")


if __name__ == "__main__":
    main()
