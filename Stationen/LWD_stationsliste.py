#!/usr/bin/env python3
"""
LWD Station Metadata Collector (standalone)

Scans all downloaded LWD CSV files and builds a master station list with:
- Basic metadata extracted from CSV headers (name, height, coordinates)
- 0/1 measurement flags for HS, WG, LF, LT (precipitation always 0 for schema coherence
  with other station lists)

This script is the LWD equivalent of suedtirol_stationlist.py.
It allows you to know exactly which stations have which variables without
having to open every file.

Outputs:
- Ready-to-copy STATION_CODES list (printed)
- Full station metadata JSON saved to Stationen/LWD
- Log written to logs/LWD
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional

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
LWD_FOLDER = Path(DATA_ROOT / "lwd")

LOG_DIR = Path(DATA_ROOT / "code" / "logs" / "lwd")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "lwd_station_metadata.log"

STATION_DIR = Path(DATA_ROOT / "stationen" / "lwd")
STATION_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = STATION_DIR / "lwd_stations_with_measurement_flags.json"

# Parameters we care about (must match the download script)
TARGET_PARAMS = ["HS", "WG", "LF", "LT"]

FLAG_MAP = {
    "HS": "measures_snow_height",
    "WG": "measures_wind",
    "LF": "measures_relative_humidity",
    "LT": "measures_temperature",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_metadata_from_csv(csv_path: Path) -> Optional[Dict]:
    """Extract station metadata from the header of one LWD CSV file."""
    metadata = {}
    try:
        with open(csv_path, "r", encoding="cp1252", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or "Datum/Uhrzeit" in stripped:
                    break
                if ";" in stripped:
                    key, value = [p.strip() for p in stripped.split(";", 1)]
                    key_clean = (key.replace("ï¿½", "Ã¤")
                                 .replace("Lï¿½nge", "LÃ¤nge")
                                 .replace("ï¿½C", "Â°C")
                                 .replace("ï¿½", "Ã¶"))
                    metadata[key_clean] = value
    except Exception:
        return None
    return metadata


def main():
    logger.info("=== LWD Station Metadata Collector ===")
    logger.info(f"Scanning folder: {LWD_FOLDER}")
    logger.info(f"Output JSON: {OUTPUT_JSON}")

    if not LWD_FOLDER.exists():
        logger.error(f"LWD folder does not exist: {LWD_FOLDER}")
        return

    # Group files by station
    station_files: Dict[str, List[Path]] = defaultdict(list)
    station_params: Dict[str, set] = defaultdict(set)

    for csv_file in LWD_FOLDER.glob("*.csv"):
        parts = csv_file.stem.split("_")
        if len(parts) < 2:
            continue
        station_code = parts[0]
        param = parts[1]
        if param in TARGET_PARAMS:
            station_files[station_code].append(csv_file)
            station_params[station_code].add(param)

    logger.info(f"Found {len(station_files)} unique stations with data")

    final_stations = []

    for station_code in sorted(station_files.keys()):
        # Take the first file to extract metadata
        first_file = sorted(station_files[station_code])[0]
        meta = extract_metadata_from_csv(first_file) or {}

        # Determine measurement flags
        flags = {flag_name: 0 for flag_name in FLAG_MAP.values()}
        for param in station_params[station_code]:
            if param in FLAG_MAP:
                flags[FLAG_MAP[param]] = 1
        flags["measures_precipitation"] = 0  # always 0 for LWD â†’ coherent schema with other station lists

        # Extract coordinates (note: LWD headers have them swapped in naming)
        raw_lat = meta.get("Geografische LÃ¤nge", "").strip()
        raw_lon = meta.get("Geografische Breite", "").strip()

        try:
            latitude = float(raw_lat.replace(",", ".")) if raw_lat else None
            longitude = float(raw_lon.replace(",", ".")) if raw_lon else None
        except (ValueError, TypeError):
            latitude = longitude = None

        station_entry = {
            "id": station_code,
            "name": meta.get("Stationsname", station_code).strip(),
            "lat": latitude,
            "lon": longitude,
            "altitude": int(meta.get("StationshÃ¶he", 0)) if meta.get("StationshÃ¶he") else None,
            **flags
        }

        final_stations.append(station_entry)

    if not final_stations:
        logger.warning("No stations found. Exiting.")
        return

    # Sort by station code
    final_stations.sort(key=lambda x: x["id"])

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_stations, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved station metadata JSON to: {OUTPUT_JSON}")
    logger.info(f"Total stations with data: {len(final_stations)}")

    # Print ready-to-copy list
    codes = [s["id"] for s in final_stations]
    print("\n" + "=" * 80)
    print("COPY THE FOLLOWING LIST INTO YOUR SCRIPTS:")
    print("=" * 80)
    print("STATION_CODES = [")
    for code in codes:
        print(f'    "{code}",')
    print("]")
    print("=" * 80)
    print(f"Total unique station codes: {len(codes)}")

    logger.info("=== LWD Station Metadata Collector finished ===")


if __name__ == "__main__":
    main()
