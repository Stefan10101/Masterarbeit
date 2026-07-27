#!/usr/bin/env python3
"""
HYDRO_VORARLBERG Station Metadata Collector (standalone)

Scans LT*, N*, and SH* files in the HYDRO_VO folder and creates
a station list with metadata + measurement flags for:
    - Temperature (LT*)
    - Precipitation (N*)
    - Snow height (SH*)
"""

import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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
# CONFIGURATION - HYDRO_VORARLBERG
# =============================================
INPUT_DIR   = Path(DATA_ROOT / "hydro_vo")
LOG_DIR     = Path(DATA_ROOT / "code" / "logs" / "hydro_vorarlberg")
STATION_DIR = Path(DATA_ROOT / "stationen" / "hydro_vorarlberg")
OUTPUT_JSON = STATION_DIR / "hydro_vorarlberg_stations_with_measurement_flags.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)
STATION_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "hydro_vorarlberg_stations_metadata.log"


def log(msg: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")


def parse_hydro_metadata(csv_path: Path) -> dict | None:
    """Extract name, altitude and coordinates from HYDRO header."""
    try:
        with open(csv_path, 'r', encoding='latin1', errors='replace') as f:
            lines = [line.rstrip('\r\n') for line in f if line.strip()]
    except Exception:
        return None

    if not lines:
        return None

    # Station name
    name = csv_path.stem
    if ';' in lines[0]:
        parts = [p.strip() for p in lines[0].split(';') if p.strip()]
        if len(parts) >= 2:
            name = parts[1]

    # Altitude (HÃ¶he)
    hoehe = None
    in_hoehe = False
    for line in lines:
        stripped = line.strip()
        if "HÃ¶he:" in stripped or stripped.startswith("HÃ¶he"):
            in_hoehe = True
            continue
        if in_hoehe and "Geographische Koordinaten" in stripped:
            break
        if in_hoehe and ';' in line:
            parts = [p.strip() for p in line.split(';')]
            if len(parts) >= 2 and re.match(r'\d{2}\.\d{2}\.\d{4}', parts[0]):
                try:
                    hoehe = float(parts[1].replace(',', '.'))
                except ValueError:
                    pass
                break

    # Coordinates (DMS â†’ decimal)
    lat = lon = None
    in_coord = False
    for line in lines:
        stripped = line.strip()
        if "Geographische Koordinaten" in stripped:
            in_coord = True
            continue
        if in_coord and "Ursprungszeitreihe" in stripped:
            break
        if in_coord and ';' in stripped:
            parts = [p.strip() for p in stripped.split(';')]
            if len(parts) >= 3 and re.match(r'\d{2}\.\d{2}\.\d{4}', parts[0]):
                def dms_to_dec(dms: str):
                    nums = re.findall(r'\d+', dms)
                    if len(nums) == 3:
                        d, m, s = map(int, nums)
                        return round(d + m/60 + s/3600, 6)
                    return None
                lon = dms_to_dec(parts[1])
                lat = dms_to_dec(parts[2])
                break

    if lat is None or lon is None:
        return None

    return {"name": name, "hoehe": hoehe, "lat": lat, "lon": lon}


def get_variable_type(filename: str) -> str | None:
    """Detect variable from filename prefix."""
    stem = filename.upper()
    if stem.startswith("LT"):
        return "temperature"
    elif stem.startswith("N"):
        return "precipitation"
    elif stem.startswith("SH"):
        return "snow_height"
    return None


def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=== HYDRO_VORARLBERG Station Metadata Collector ===")
    log(f"Scanning folder: {INPUT_DIR}")

    # Find all relevant files
    patterns = ["LT*.[cC][sS][vV]", "N*.[cC][sS][vV]", "SH*.[cC][sS][vV]"]
    all_files = []
    for pattern in patterns:
        all_files.extend(INPUT_DIR.rglob(pattern))

    log(f"Found {len(all_files)} files (LT, N, SH)")

    # Group by station name and collect measured variables
    station_data = defaultdict(lambda: {"meta": None, "vars": set()})

    for csv_path in all_files:
        var_type = get_variable_type(csv_path.name)
        if not var_type:
            continue

        meta = parse_hydro_metadata(csv_path)
        if not meta:
            continue

        name = meta["name"]
        entry = station_data[name]

        if entry["meta"] is None:
            entry["meta"] = meta

        entry["vars"].add(var_type)

    # Build final station list
    stations = []
    for name, entry in station_data.items():
        if entry["meta"] is None:
            continue

        meta = entry["meta"]
        vars_set = entry["vars"]

        stations.append({
            "id": name,
            "name": name,
            "lat": meta["lat"],
            "lon": meta["lon"],
            "altitude": meta["hoehe"],
            "measures_temperature":       1 if "temperature" in vars_set else 0,
            "measures_relative_humidity": 0,
            "measures_precipitation":     1 if "precipitation" in vars_set else 0,
            "measures_wind":              0,
            "measures_snow_height":       1 if "snow_height" in vars_set else 0
        })

    if not stations:
        log("No stations found. Exiting.")
        return

    stations.sort(key=lambda x: x["id"])

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stations, f, ensure_ascii=False, indent=2)
    log(f"Saved {len(stations)} stations to: {OUTPUT_JSON}")

    # Ready-to-copy set
    ids = sorted(s["id"] for s in stations)

    print("\n" + "=" * 80)
    print("COPY THE FOLLOWING SET INTO YOUR SCRIPTS:")
    print("=" * 80)
    print("RELEVANT_HYDRO_VORARLBERG_STATIONS = {")
    line = "    "
    for i, iid in enumerate(ids):
        line += f'"{iid}", '
        if (i + 1) % 6 == 0:
            print(line.rstrip())
            line = "    "
    if line.strip():
        print(line.rstrip())
    print("}")
    print("=" * 80)
    print(f"Total unique stations: {len(ids)}")

    log("\nDone.")


if __name__ == "__main__":
    main()
