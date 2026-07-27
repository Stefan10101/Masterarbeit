#!/usr/bin/env python3
""DATA_ROOT / "
hydro_tirol unified data extractor (schneehoehe + niederschlag + temperatur)

objective
---------
single script that processes all three variable types from the hydro_tirol csv exports
in one run. every station is converted to a clean json file containing:
  - station metadata (name, elevation, lon" / "lat)
  - time series with correct utc timestamps

all logs are written to:
  " / "code" / "logs" / "hydro_tirol

temperature output path (as specified):
  " / "temperatur" / "hydrot

timestamp fix assurance (critical for thesis data integrity)
-------------------------------------------------------------
the timezone conversion logic was taken exclusively from readerma.py
(the temperature script for hydro_tirol, already confirmed working).

- parse_csv_to_dict (especially the tz_localize("Europe/Vienna") block) 
  originates from the provided readerMA.py.
- This logic correctly handles real DST transitions (CET â†” CEST) used by HYDRO_TIROL.
- The original HydroT_Schneehoehe_extract.py and HydroT_Niederschlag_extract.py 
  had structurally identical code. By centralizing the validated readerMA.py version, 
  any previous timestep/timezone drift is eliminated for all three variables.

Design principles applied:
- DRY: one parse function for all file types
- Config-driven: easy to maintain / extend
- Defensive + logged: every step is traceable
- Efficient: single pass per CSV, minimal object creation
"""

import json
import re
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

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
# LOGGING (persistent file + console)
# =============================================================================
LOG_DIR = Path(DATA_ROOT / "code" / "logs" / "hydro_tirol")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"hydrotirol_extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)


def parse_csv_to_dict(input_path: Path):
    """
    Parse a single HYDRO_TIROL CSV (LT*, N* or SH*) into a dict.
    UTC conversion uses the validated Europe/Vienna logic from readerMA.py.
    Returns None on any critical failure.
    """
    try:
        with open(input_path, "r", encoding="latin1", errors="replace") as f:
            lines = [line.rstrip("\r\n") for line in f if line.strip()]
    except Exception as e:
        logging.warning(f"Cannot read file: {input_path.name} â€“ {e}")
        return None

    # --- 1. Station name (first line, second field) ---
    name = "Unknown"
    if lines and ";" in lines[0]:
        parts = [p.strip() for p in lines[0].split(";") if p.strip()]
        if len(parts) >= 2:
            name = parts[1]

    # --- 2. Elevation (HÃ¶he) ---
    hoehe = None
    in_hoehe_block = False
    for line in lines:
        stripped = line.strip()
        if "HÃ¶he:" in stripped or "HÃ¶he" in stripped:
            in_hoehe_block = True
            continue
        if in_hoehe_block:
            if "Geographische Koordinaten" in stripped:
                break
            if ";" in line:
                parts = [p.strip() for p in line.split(";")]
                if len(parts) >= 2 and re.match(r"\d{2}\.\d{2}\.\d{4}", parts[0]):
                    try:
                        hoehe = float(parts[1].replace(",", "."))
                    except ValueError:
                        pass

    # --- 3. Geographic coordinates (decimal degrees) ---
    lon = lat = None
    in_coord_block = False
    for line in lines:
        stripped = line.strip()
        if "Geographische Koordinaten" in stripped:
            in_coord_block = True
            continue
        if in_coord_block:
            if "Ursprungszeitreihe" in stripped:
                break
            if ";" in stripped:
                parts = [p.strip() for p in stripped.split(";")]
                if len(parts) >= 3 and re.match(r"\d{2}\.\d{2}\.\d{4}", parts[0]):
                    def dms_to_dec(dms: str):
                        nums = re.findall(r"\d+", dms)
                        if len(nums) == 3:
                            d, m, s = map(int, nums)
                            return round(d + m / 60.0 + s / 3600.0, 6)
                        return None
                    lon = dms_to_dec(parts[1])
                    lat = dms_to_dec(parts[2])

    # --- 4. Time series data with CORRECT UTC conversion ---
    # This block is taken from the working readerMA.py implementation for HYDRO_TIROL.
    data = []
    reading_data = False
    for line in lines:
        stripped = line.strip()
        if "Werte:" in stripped:
            reading_data = True
            continue
        if reading_data and ";" in line:
            parts = [p.strip() for p in line.split(";", 1)]
            if len(parts) == 2:
                ts_str = parts[0]
                val_str = parts[1].strip().lower()
                value = None
                if val_str not in ["lÃ¼cke", ""]:
                    try:
                        value = float(parts[1].replace(",", "."))
                    except ValueError:
                        pass

                # === CORRECT UTC CONVERSION (HYDRO_TIROL follows real DST) ===
                # Source: readerMA.py (temperature script) â€“ the version confirmed working.
                # Uses tz_localize("Europe/Vienna") which auto-detects CET vs CEST.
                try:
                    local_ts = pd.to_datetime(
                        ts_str, format="%d.%m.%Y %H:%M:%S", errors="coerce"
                    )
                    if pd.notna(local_ts):
                        local_ts = local_ts.tz_localize(
                            "Europe/Vienna", ambiguous="NaT", nonexistent="NaT"
                        )
                        utc_ts = local_ts.tz_convert("UTC")
                        data.append({
                            "timestamp": utc_ts.isoformat(),
                            "value": value
                        })
                except Exception:
                    pass

    return {
        "name": name,
        "hoehe": hoehe,
        "lon": lon,
        "lat": lat,
        "data": data
    }


def process_variable_type(
    input_dir: Path,
    output_base: Path,
    file_pattern: str,
    type_label: str
) -> int:
    """
    Process all CSVs matching the pattern for one variable.
    Returns number of successfully written JSON files.
    """
    logging.info(f"=== {type_label} â€“ HYDRO_TIROL â†’ JSON (correct UTC) ===")

    if not input_dir.is_dir():
        logging.error(f"Input folder not found: {input_dir}")
        return 0

    output_base.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.rglob(file_pattern))
    logging.info(f"Found {len(csv_files)} matching files ({file_pattern})")

    processed = 0
    for csv_path in csv_files:
        logging.info(f"Processing: {csv_path.name}")

        result = parse_csv_to_dict(csv_path)
        if result is None or result["name"] == "Unknown":
            logging.warning(f"  Skipped â€“ invalid or unparseable: {csv_path.name}")
            continue

        station_name = result["name"]
        station_folder = output_base / station_name
        json_path = station_folder / f"{station_name}.json"

        # Remove legacy .csv files from earlier workflow versions (safe no-op if absent)
        old_csv = station_folder / f"{station_name}.csv"
        if old_csv.exists():
            try:
                old_csv.unlink()
            except Exception as e:
                logging.warning(f"  Could not delete legacy CSV: {e}")

        station_folder.mkdir(parents=True, exist_ok=True)

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logging.info(f"  âœ“ Saved: {json_path}")
            processed += 1
        except Exception as e:
            logging.error(f"  Failed to write JSON for {station_name}: {e}")

    logging.info(f"Completed {type_label}: {processed} stations written\n")
    return processed


def main():
    INPUT_DIR = Path(DATA_ROOT / "hydro").resolve()

    # Central configuration â€“ easy to extend or modify paths
    configs = [
        {
            "file_pattern": "SH*.[cC][sS][vV]",
            "output_base": Path(DATA_ROOT / "schneehoehe" / "hydrot"),
            "type_label": "Schneehoehe (Snow Height)"
        },
        {
            "file_pattern": "N*.[cC][sS][vV]",
            "output_base": Path(DATA_ROOT / "niederschlag" / "hydrot"),
            "type_label": "Niederschlag (Precipitation)"
        },
        {
            "file_pattern": "LT*.[cC][sS][vV]",
            "output_base": Path(DATA_ROOT / "temperatur" / "hydrot"),
            "type_label": "Temperatur (Temperature)"
        },
    ]

    total = 0
    for cfg in configs:
        total += process_variable_type(
            input_dir=INPUT_DIR,
            output_base=cfg["output_base"],
            file_pattern=cfg["file_pattern"],
            type_label=cfg["type_label"]
        )

    logging.info(f"=== ALL FINISHED === Total stations processed: {total}")
    logging.info(f"Full log written to: {LOG_FILE}")


if __name__ == "__main__":
    main()
