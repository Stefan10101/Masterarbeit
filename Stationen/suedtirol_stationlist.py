#!/usr/bin/env python3
"""
SÃ¼dtirol Station Metadata Collector (standalone, modeled after stationsliste_ZAMG.py and DWD_stationsliste.py)

Fetches all stations from the SÃ¼dtirol Meteo API and determines which of the target variables
they measure based on the sensors used in Suedtirol_API_N_SH_ff_Rf_LT.py:
    precipitation (N), wind speed (WG), snow height (HS), relative humidity (LF), air temperature (LT).

Value is 1 if the station has the corresponding sensor, else 0.
Only stations that measure at least one target variable are included.

Includes retry logic with exponential backoff for 429 rate-limit responses.

Outputs:
- Ready-to-copy RELEVANT_SUEDTIROL_CODES set (printed, sorted)
- Full station metadata JSON (with measurement flags) saved to Stationen/Suedtirol
- Log written to logs/Suedtirol/API
"""

import requests
import json
import time
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
# CONFIGURATION (edit if paths change)
# =============================================
LOG_DIR = Path(DATA_ROOT / "code" / "logs" / "suedtirol" / "api")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "suedtirol_stations_metadata.log"

STATION_DIR = Path(DATA_ROOT / "stationen" / "suedtirol")
STATION_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = STATION_DIR / "suedtirol_stations_with_measurement_flags.json"

BASE_URL = DATA_ROOT / "http:" / "daten.buergernetz.bz.it" / "services" / "meteo" / "v1"

# Target sensors and their flag names (updated to include air temperature)
TARGET_SENSORS = ["N", "WG", "HS", "LF", "LT"]
FLAG_MAP = {
    "N": "measures_precipitation",
    "WG": "measures_wind",
    "HS": "measures_snow_height",
    "LF": "measures_relative_humidity",
    "LT": "measures_temperature",          # NEW
}


def log(msg: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")


def get_all_stations() -> list[dict]:
    """Fetch all stations with metadata (adapted from download script)."""
    url = f"{BASE_URL}/stations"
    log(f"Fetching all stations from {url} ...")
    try:
        r = requests.get(url, timeout=30)
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
                "hoehe": props.get("ALT"),
                "longitude": props.get("LONG"),
                "latitude": props.get("LAT")
            })
        log(f"Total stations returned by API: {len(stations)}")
        return stations
    except Exception as e:
        log(f"Failed to fetch stations: {e}")
        return []


def get_available_sensors(station_code: str, retries: int = 5, backoff: float = 1.8) -> set:
    """
    Fetch sensor list for one station with retry + exponential backoff.
    Especially handles 429 (Too Many Requests) gracefully.
    """
    url = f"{BASE_URL}/sensors"
    params = {"station_code": station_code}

    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 429:
                wait = backoff ** (attempt + 1) * 1.5
                log(f"  429 rate limit for {station_code} â€” waiting {wait:.1f}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return {s["TYPE"] for s in r.json()}
        except Exception as e:
            if attempt == retries - 1:
                log(f"  Warning: could not fetch sensors for station {station_code} after {retries} attempts: {e}")
                return set()
            wait = backoff ** (attempt + 1)
            log(f"  Retry {attempt+1} for {station_code}: {e} (wait {wait:.1f}s)")
            time.sleep(wait)
    return set()


def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=== SÃ¼dtirol Station Metadata Collector ===")
    log("Target variables: precipitation (N), wind (WG), snow height (HS), relative humidity (LF), air temperature (LT)")
    log("Includes retry/backoff for API rate limiting (429)")
    log(f"Logs: {LOG_DIR}")
    log(f"Output JSON: {OUTPUT_JSON}")

    all_stations = get_all_stations()
    if not all_stations:
        log("No stations found. Exiting.")
        return

    final_stations = []
    skipped_no_sensors = 0

    for idx, station in enumerate(all_stations, 1):
        if idx % 20 == 0:
            log(f"  Processing station {idx}/{len(all_stations)} ...")

        available = get_available_sensors(station["code"])

        flags = {flag_name: 0 for flag_name in FLAG_MAP.values()}
        has_any = False
        for sensor_code, flag_name in FLAG_MAP.items():
            if sensor_code in available:
                flags[flag_name] = 1
                has_any = True

        if has_any:
            meta = {
                "id": station["code"],
                "name": station["name"],
                "lat": float(station["latitude"]) if station.get("latitude") is not None else None,
                "lon": float(station["longitude"]) if station.get("longitude") is not None else None,
                "altitude": float(station["hoehe"]) if station.get("hoehe") is not None else None,
                **flags
            }
            final_stations.append(meta)
        else:
            skipped_no_sensors += 1

        # Polite global delay to further reduce rate-limit risk
        time.sleep(0.12)

    log(f"Stations with at least one target sensor: {len(final_stations)}")
    log(f"Stations skipped (no target sensors or persistent API errors): {skipped_no_sensors}")

    if not final_stations:
        log("No relevant stations found. Exiting.")
        return

    # Sort by id (numeric if possible, else string) for deterministic order
    final_stations.sort(key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else x["id"])

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_stations, f, ensure_ascii=False, indent=2)
    log(f"Saved station metadata JSON to: {OUTPUT_JSON}")

    # Build ready-to-copy set (sorted)
    codes = []
    for s in final_stations:
        try:
            codes.append(int(s["id"]))
        except (ValueError, TypeError):
            codes.append(s["id"])
    codes = sorted(set(codes))

    print("\n" + "=" * 75)
    print("COPY THE FOLLOWING SET INTO YOUR SCRIPTS IF YOU WANT TO HARD-CODE THE CODES:")
    print("=" * 75)
    print("RELEVANT_SUEDTIROL_CODES = {")
    line = "    "
    for i, code in enumerate(codes):
        line += f"{code}, "
        if (i + 1) % 8 == 0:
            print(line.rstrip())
            line = "    "
    if line.strip():
        print(line.rstrip())
    print("}")
    print("=" * 75)
    print(f"Total unique station codes: {len(codes)}")

    log("\nDone. You can now use this JSON as your master station list for SÃ¼dtirol (target variables only).")


if __name__ == "__main__":
    main()
