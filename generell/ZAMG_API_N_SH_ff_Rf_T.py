import requests
import json
import time
import re
from datetime import datetime
from pathlib import Path

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
BASE_URL = "https://dataset.api.hub.geosphere.at/v1"
RESOURCE_ID = "klima-v2-10min"

# Geographic filter: all stations west of 17Â° longitude (western Austria)
LON_CUTOFF = 13.5

TEST_LIMIT = None          # Set to e.g. 5 for testing, None for full run
YEARS = range(2020, 2026)

# --- NEW: central log location ---
LOG_DIR = Path(DATA_ROOT / "code" / "logs" / "zamg" / "api")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "zamg_download_log.txt"

# Output folders per variable
OUTPUT_BASES = {
    "RR": Path(DATA_ROOT / "niederschlag" / "zamg"),      # Precipitation
    "FF": Path(DATA_ROOT / "wind" / "zamg"),             # Wind speed
    "RF": Path(DATA_ROOT / "luftfeuchte" / "zamg"),      # Relative humidity
    "SH": Path(DATA_ROOT / "schneehoehe" / "zamg"),      # Snow height
    "tl": Path(DATA_ROOT / "temperatur" / "zamg")        # Air temperature (2m) - NEW
}

# Parameter mapping: API code â†’ filename suffix
# TL = Lufttemperatur 2m (air temperature at 2 m height) - confirmed for klima-v2-10min
PARAMETERS = {
    "RR": "precip",
    "FF": "windspeed",
    "RF": "relative_humidity",
    "SH": "snowheight",
    "tl": "air_temperature"          # NEW
}


def log(msg: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")


def sanitize_name(name: str) -> str:
    """Clean station name for folder and filename"""
    name = name.strip()
    name = re.sub(r'[^a-zA-Z0-9Ã¤Ã¶Ã¼Ã„Ã–ÃœÃŸ -]', '', name)
    name = (name.replace('Ã¤', 'ae').replace('Ã¶', 'oe').replace('Ã¼', 'ue')
            .replace('Ã„', 'Ae').replace('Ã–', 'Oe').replace('Ãœ', 'Ue').replace('ÃŸ', 'ss'))
    name = re.sub(r'\s+', '_', name)
    return name


def get_filtered_station_map() -> dict:
    """
    Fetch station metadata and return only stations west of LON_CUTOFF.
    Dynamic filtering â†’ no hardcoded ID list needed.
    """
    url = f"{BASE_URL}/station/historical/{RESOURCE_ID}/metadata"
    log(f"Fetching station metadata...")

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        stations = r.json().get("stations", [])
        log(f"Total stations from API: {len(stations)}")
    except Exception as e:
        log(f"Metadata fetch failed: {e}")
        return {}

    station_map = {}
    for s in stations:
        sid = s.get("id")
        lat = s.get("lat")
        lon = s.get("lon")
        altitude = s.get("altitude")

        if lat is None or lon is None or altitude is None:
            continue

        try:
            lat = float(lat)
            lon = float(lon)
            altitude = float(altitude)
        except (ValueError, TypeError):
            continue

        if lon >= LON_CUTOFF:
            continue

        raw_name = s.get("name", f"Station_{sid}")
        clean_name = sanitize_name(raw_name)

        station_map[str(sid)] = {
            "id": sid,
            "raw_name": raw_name,
            "clean_name": clean_name,
            "lat": lat,
            "lon": lon,
            "altitude": altitude
        }

    log(f"Found {len(station_map)} matching stations west of {LON_CUTOFF}Â° (western Austria)")
    return station_map


def download_variable(station_id: str, station_info: dict, param_code: str) -> bool:
    clean_name = station_info["clean_name"]
    output_base = OUTPUT_BASES[param_code]
    var_name = PARAMETERS[param_code]

    station_folder = output_base / clean_name
    station_folder.mkdir(parents=True, exist_ok=True)

    output_file = station_folder / f"{clean_name}_{var_name}.json"

    if output_file.exists():
        log(f"Already exists â†’ skipping {clean_name} / {var_name}")
        return True

    url = f"{BASE_URL}/station/historical/{RESOURCE_ID}"
    start = f"{min(YEARS)}-01-01T00:00"
    end   = f"{max(YEARS)}-12-31T23:59"

    params = {
        "station_ids": station_id,
        "parameters": param_code,
        "start": start,
        "end": end
    }

    log(f"Downloading {clean_name} â†’ {var_name} (ID {station_id})")

    try:
        r = requests.get(url, params=params, timeout=180)
        r.raise_for_status()
        data = r.json()

        data["station_metadata"] = {
            "id": station_info["id"],
            "name": station_info["raw_name"],
            "clean_name": clean_name,
            "latitude": station_info["lat"],
            "longitude": station_info["lon"],
            "altitude": station_info["altitude"]
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        log(f"Success: Saved {clean_name}_{var_name}.json")
        time.sleep(2)
        return True

    except Exception as e:
        log(f"Failed {clean_name} / {var_name}: {e}")
        return False


if __name__ == "__main__":
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=== Starting ZAMG Geosphere download ===")
    log("Variables: Precipitation (RR), Wind (FF), Relative Humidity (RF), Snow height (SH), Air Temperature (TL)")
    log("Geographic scope: ALL stations west of 17Â° longitude (western Austria)")
    log("Logs saved to: " + str(LOG_DIR))
    log("Station selection is dynamic (lon < 17.0)")

    station_map = get_filtered_station_map()

    if not station_map:
        log("No stations found â†’ exit")
    else:
        items = list(station_map.items())
        if TEST_LIMIT is not None:
            items = items[:TEST_LIMIT]
            log(f"TEST MODE: processing only first {TEST_LIMIT} stations")

        success = 0
        total_tasks = len(items) * len(PARAMETERS)

        for sid_str, info in sorted(items, key=lambda x: x[1]["clean_name"]):
            log(f"\nStation: {info['raw_name']} (ID: {info['id']})")

            for param_code in PARAMETERS:
                if download_variable(sid_str, info, param_code):
                    success += 1

        log(f"\nFinished. Successfully downloaded {success}/{total_tasks} variable files.")
