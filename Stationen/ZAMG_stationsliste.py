#!/usr/bin/env python3
"""
Standalone helper to fetch all station IDs from klima-v2-10min
west of 17Â° longitude (western Austria).

Includes measurement flags (0/1) for the main variables.
Note: The Geosphere metadata endpoint does not return per-station
parameter availability, so we set flags based on what klima-v2-10min
generally provides at its stations.

Outputs:
- Ready-to-copy RELEVANT_IDS set (printed)
- Full station metadata JSON with measurement flags
- Log written to the central logs folder
"""

import requests
import json
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

LON_CUTOFF = 13.5

LOG_DIR = Path(DATA_ROOT / "code" / "logs" / "zamg" / "api")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "get_western_stations_log.txt"

STATION_DIR = Path(DATA_ROOT / "stationen" / "zamg")
STATION_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = STATION_DIR / "western_stations_klima_v2_10min.json"


def log(msg: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")


def get_western_stations() -> list[dict]:
    url = f"{BASE_URL}/station/historical/{RESOURCE_ID}/metadata"
    log(f"Fetching station metadata from {url} ...")

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        stations = r.json().get("stations", [])
        log(f"Total stations returned by API: {len(stations)}")
    except Exception as e:
        log(f"Metadata fetch failed: {e}")
        return []

    western = []
    eastern_count = 0
    skipped_no_coords = 0

    for s in stations:
        sid = s.get("id")
        lat = s.get("lat")
        lon = s.get("lon")
        alt = s.get("altitude")
        name = s.get("name", f"Station_{sid}")

        if lat is None or lon is None or alt is None:
            skipped_no_coords += 1
            continue

        try:
            lat_f = float(lat)
            lon_f = float(lon)
            alt_f = float(alt)
        except (ValueError, TypeError):
            skipped_no_coords += 1
            continue

        if lon_f >= LON_CUTOFF:
            eastern_count += 1
            continue

        western.append({
            "id": sid,
            "name": name,
            "lat": lat_f,
            "lon": lon_f,
            "altitude": alt_f,
            # === Measurement flags for klima-v2-10min ===
            # The metadata endpoint does not return per-station parameter lists.
            # These flags reflect what this dataset generally provides.
            "measures_temperature": 1,
            "measures_relative_humidity": 1,
            "measures_precipitation": 1,
            "measures_wind": 1,
            "measures_snow_height": 0   # Only available at a subset of stations
        })

    log(f"Stations west of {LON_CUTOFF}Â°: {len(western)}")
    log(f"Stations east of or at {LON_CUTOFF}Â° (excluded): {eastern_count}")
    log(f"Skipped (missing/invalid coords): {skipped_no_coords}")

    return western


if __name__ == "__main__":
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=== Fetching western Austria stations (lon < 17.0) for klima-v2-10min ===")
    log(f"Logs saved to: {LOG_DIR}")
    log(f"Station metadata will be saved to: {STATION_DIR}")

    western_stations = get_western_stations()

    if not western_stations:
        log("No western stations found. Exiting.")
    else:
        ids = []
        for s in western_stations:
            try:
                ids.append(int(s["id"]))
            except (ValueError, TypeError):
                ids.append(s["id"])

        ids = sorted(set(ids))

        print("\n" + "=" * 70)
        print("COPY THE FOLLOWING SET INTO YOUR MAIN SCRIPT IF NEEDED:")
        print("=" * 70)
        print("RELEVANT_IDS = {")
        line = "    "
        for i, iid in enumerate(ids):
            line += f"{iid}, "
            if (i + 1) % 8 == 0:
                print(line.rstrip())
                line = "    "
        if line.strip():
            print(line.rstrip())
        print("}")
        print("=" * 70)
        print(f"Total unique western station IDs: {len(ids)}")

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(western_stations, f, ensure_ascii=False, indent=2)
        log(f"Saved complete station metadata (with measurement flags) to: {OUTPUT_JSON}")

        log("\nDone.")
