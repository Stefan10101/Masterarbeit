#!/usr/bin/env python3
"""
MeteoSuisse unified extractor - precipitation + relative humidity + snow height + temperature + wind speed

objective
---------
extract 10-minute data for multiple variables from meteosuisse json station files
(2020-2025) and write clean, compact json files per station and variable.

variables extracted:
- rre150z0  -> niederschlag (precipitation)
- ure200s0  -> relative luftfeuchtigkeit (relative humidity)
- htoauts0  -> schneehoehe (snow height)
- tre200s0  -> lufttemperatur 2m (air temperature)
- fkl010z0  -> windgeschwindigkeit skalar (mean wind speed)

all meteosuisse data is already in utc -> no timezone conversion required.

logs are written to:
  DATA_ROOT / "code" / "logs" / "meteosuisse" / "extract"

wind output:
  DATA_ROOT / "wind" / "meteosuisse"

design principles:
- efficient parallel processing (threadpoolexecutor)
- idempotent / resumable (skips files that already exist)
- clean, compact json output
- proper logging (file + console)
- easy to extend with new variables
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import time

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT


# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_DIR = Path(DATA_ROOT / "meteosuisse" / "tutti")

OUTPUT_PATHS = {
    "rre150z0":  Path(DATA_ROOT / "niederschlag" / "meteosuisse"),
    "ure200s0":  Path(DATA_ROOT / "luftfeuchte" / "meteosuisse"),
    "htoauts0":  Path(DATA_ROOT / "schneehoehe" / "meteosuisse"),
    "tre200s0":  Path(DATA_ROOT / "temperatur" / "meteosuisse"),
    "fkl010z0":  Path(DATA_ROOT / "wind" / "meteosuisse"),   # NEW
}

LOG_DIR = Path(DATA_ROOT / "code" / "logs" / "meteosuisse" / "extract")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"meteosuisse_extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

MAX_WORKERS = 12
TARGET_TIME_START = "2020-01-01"
TARGET_TIME_END   = "2025-12-31"

# Create all output directories
for path in OUTPUT_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

VARIABLES = {
    "rre150z0": {"name": "precipitation",       "unit": "mm"},
    "ure200s0": {"name": "relative_humidity",   "unit": "%"},
    "htoauts0": {"name": "snow_height",         "unit": "cm"},
    "tre200s0": {"name": "temperature",         "unit": "Â°C"},
    "fkl010z0": {"name": "wind_speed",          "unit": "m/s"},   # NEW
}

# =============================================================================
# LOAD STATION METADATA (all Switzerland)
# =============================================================================
meta_json_path = INPUT_DIR / "stations_metadata.json"
with open(meta_json_path, "r", encoding="utf-8") as f:
    station_meta_list = json.load(f)

logging.info(f"Loaded {len(station_meta_list)} stations (all Switzerland)")

# =============================================================================
# CORE WORKER FUNCTION
# =============================================================================
def load_station_json(json_file: Path) -> dict:
    """Load a station JSON. Accepts bare NaN/Infinity (legacy dumps) by rewriting them to null."""
    with open(json_file, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Legacy files may contain bare NaN / Infinity from pandas
        cleaned = (
            text.replace(":NaN", ":null")
            .replace(":Infinity", ":null")
            .replace(":-Infinity", ":null")
        )
        return json.loads(cleaned)


def process_station_file(json_file: Path):
    """
    Process one station's 10-min JSON file.
    Extracts all requested variables and writes compact JSON per variable.
    Returns (station_name, list_of_(variable_name, record_count))
    """
    try:
        full_data = load_station_json(json_file)

        meta = full_data["station_metadata"]
        station_abbr = meta["station_abbr"].lower()
        station_name = meta.get("station_name", station_abbr.upper()).replace(" ", "_").replace("/", "_")

        df = pd.DataFrame(full_data["data"])
        time_col = "reference_timestamp"

        if time_col not in df.columns:
            logging.warning(f"{station_name}: No '{time_col}' column found")
            return station_name, [], None

        df["time_dt"] = pd.to_datetime(df[time_col], format="%d.%m.%Y %H:%M", errors="coerce")

        results = []

        for col, var_info in VARIABLES.items():
            if col not in df.columns:
                continue

            output_base = OUTPUT_PATHS[col]
            station_dir = output_base / station_name
            out_path = station_dir / f"{station_name}_{var_info['name']}_10min.json"

            # Skip if output already exists (resumable / idempotent)
            if out_path.exists():
                continue

            mask = (
                (df["time_dt"] >= TARGET_TIME_START) &
                (df["time_dt"] <= TARGET_TIME_END) &
                df[col].notna()
            )
            df_filtered = df[mask].copy()

            if df_filtered.empty:
                continue

            output = {
                "name": meta.get("station_name", station_name),
                "hoehe": float(meta.get("station_height_masl", float("nan"))),
                "lon": float(meta["station_coordinates_wgs84_lon"]),
                "lat": float(meta["station_coordinates_wgs84_lat"]),
                "data": [
                    {"time": row[time_col], "value": float(row[col])}
                    for _, row in df_filtered.iterrows()
                ]
            }

            station_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, separators=(',', ':'))

            results.append((var_info['name'], len(output["data"])))

        return station_name, results, None

    except Exception as e:
        logging.error(f"Failed to process {json_file.name}: {e}")
        return None, [], json_file.name


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    logging.info("=== MeteoSuisse Unified Extractor started ===")
    logging.info(f"Time window: {TARGET_TIME_START} to {TARGET_TIME_END}")
    logging.info(f"Variables: {list(VARIABLES.keys())}")

    json_files = list(INPUT_DIR.glob("ogd-smn_*_10min_data.json"))
    logging.info(f"Found {len(json_files)} station files to process")

    start_time = time.time()
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_station_file, f): f for f in json_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing stations", unit="station"):
            station_name, var_results, bad_file = future.result()
            if bad_file:
                failed.append(bad_file)
            if var_results:
                for var_name, n_records in var_results:
                    logging.info(f"✓ {station_name} → {var_name}: {n_records:,} records")

    duration = time.time() - start_time
    logging.info(f"=== DONE in {duration/60:.1f} minutes ===")
    if failed:
        logging.warning(f"{len(failed)} station files failed (truncated or invalid JSON).")
        logging.warning("Delete those files and re-run meteoSuisseAPI.py, then re-run this extractor.")
        for name in sorted(failed):
            logging.warning(f"  {name}")
    logging.info(f"Full log written to: {LOG_FILE}")


if __name__ == "__main__":
    main()
