#!/usr/bin/env python3
"""
MeteoSuisse Station Metadata Collector (standalone)

Fetches all stations from the ogd-smn collection and determines which of the
5 target variables they measure by inspecting the column headers of one
10-minute data file per station (recent preferred, falls back to historical).

Measurement flags (0/1):
    measures_temperature, measures_relative_humidity,
    measures_precipitation, measures_wind, measures_snow_height

Only stations with at least one target variable are kept.
Uses lightweight header-only requests (no full file download) for efficiency.

Outputs:
- Ready-to-copy RELEVANT_METEOSUISSE_ABBRS set (printed, sorted, uppercase)
- Full station metadata JSON with measurement flags
- Log written to logs/MeteoSuisse/API
"""

import requests
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
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


# =============================================
# CONFIGURATION (edit paths if needed)
# =============================================
LOG_DIR = Path(DATA_ROOT / "code" / "logs" / "meteosuisse" / "api")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "meteosuisse_stations_metadata.log"

STATION_DIR = Path(DATA_ROOT / "stationen" / "meteosuisse")
STATION_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = STATION_DIR / "meteosuisse_stations_with_measurement_flags.json"

BASE_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
METADATA_URL = f"{BASE_URL}/ogd-smn_meta_stations.csv"

# Parameter codes used in 10-minute data files (t granularity)
# These are the ones that appear in the CSV header if the station measures them
PARAM_TO_FLAG = {
    "tre200s0": "measures_temperature",      # air temp 2m, instantaneous / 10min
    "ure200s0": "measures_relative_humidity",
    "rre150z0": "measures_precipitation",    # 10min precip sum
}

# Wind: any of these counts as "measures wind"
WIND_PARAMS = {"fkl010z0", "fkl010z1", "fkl010z3", "fve010z0", "dkl010z0"}

# Snow height (automatic): any of these
SNOW_PARAMS = {"htoauts0", "htoauths", "htoautd0"}


def log(msg: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")


def download_meta_csv(url: str, output_path: Path, encoding: str = "iso-8859-1") -> Path:
    """Download meta CSV and convert to UTF-8 (robust for German special chars)."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    content = resp.content.decode(encoding, errors="replace")
    output_path.write_text(content, encoding="utf-8")
    return output_path


def get_available_params(abbr: str, retries: int = 3) -> set[str]:
    """
    Fetch only the header line of one 10min data file for the station.
    Tries recent first, then historical. Returns set of column names present.
    Very lightweight (stops after first line).
    """
    abbr_l = str(abbr).lower().strip()
    candidates = [
        f"ogd-smn_{abbr_l}_t_recent.csv",
        f"ogd-smn_{abbr_l}_t_historical_2020-2029.csv",
    ]

    for fname in candidates:
        url = f"{BASE_URL}/{abbr_l}/{fname}"
        for attempt in range(retries):
            try:
                with requests.get(url, stream=True, timeout=25) as r:
                    if r.status_code == 404:
                        break  # try next filename
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if line:
                            header = line.decode("utf-8", errors="replace")
                            cols = {c.strip() for c in header.split(";") if c.strip()}
                            return cols
                break
            except Exception as e:
                if attempt == retries - 1:
                    log(f"  Warning: header fetch failed for {abbr} ({fname}): {e}")
                    return set()
                time.sleep(0.4 * (attempt + 1))
    return set()


def get_altitude_value(row: pd.Series) -> float | None:
    """Robust altitude extraction â€” now correctly uses station_height_masl."""
    # Primary column (confirmed from ogd-smn_meta_stations.csv)
    for col in ["station_height_masl", "station_height_barometer_masl",
                "station_height_m", "altitude_m", "station_altitude_m",
                "height_m", "altitude", "station_height", "hoehe"]:
        if col in row.index and pd.notna(row[col]):
            try:
                val = float(row[col])
                if 50 < val < 5000:          # safe range for Swiss stations
                    return round(val, 1)
            except (ValueError, TypeError):
                continue

    # Fallback: auto-detect any numeric column that looks like altitude
    for col in row.index:
        if any(kw in str(col).lower() for kw in ["height_masl", "height", "altitude", "alt", "hoehe", "elev"]):
            try:
                val = float(row[col])
                if 50 < val < 5000:
                    return round(val, 1)
            except (ValueError, TypeError):
                continue

    return None

def process_station(row: pd.Series) -> dict | None:
    """Return metadata dict with flags if station has at least one target var, else None."""
    abbr = str(row.get("station_abbr", "")).strip().upper()
    if not abbr:
        return None

    available = get_available_params(abbr)
    if not available:
        return None

    flags = {
        "measures_temperature": 0,
        "measures_relative_humidity": 0,
        "measures_precipitation": 0,
        "measures_wind": 0,
        "measures_snow_height": 0,
    }
    has_any = False

    for param, flag in PARAM_TO_FLAG.items():
        if param in available:
            flags[flag] = 1
            has_any = True

    if WIND_PARAMS & available:
        flags["measures_wind"] = 1
        has_any = True

    if SNOW_PARAMS & available:
        flags["measures_snow_height"] = 1
        has_any = True

    if not has_any:
        return None

    lat = pd.to_numeric(row.get("station_coordinates_wgs84_lat"), errors="coerce")
    lon = pd.to_numeric(row.get("station_coordinates_wgs84_lon"), errors="coerce")
    alt = get_altitude_value(row)

    return {
        "id": abbr,
        "name": str(row.get("station_name", abbr)).strip(),
        "lat": float(lat) if pd.notna(lat) else None,
        "lon": float(lon) if pd.notna(lon) else None,
        "altitude": alt,
        **flags,
    }


def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=== MeteoSuisse Station Metadata Collector (ogd-smn) ===")
    log("Target variables: temperature (tre200s0), RH (ure200s0), precip (rre150z0), wind, snow height (auto)")
    log("Uses lightweight header-only requests for efficiency")
    log(f"Logs: {LOG_DIR}")
    log(f"Output JSON: {OUTPUT_JSON}")

    # === STEP 1: Download & load station metadata ===
    meta_path = LOG_DIR / "ogd-smn_meta_stations.csv"
    try:
        download_meta_csv(METADATA_URL, meta_path)
        df = pd.read_csv(meta_path, sep=";", encoding="utf-8", low_memory=False)
        log(f"Total stations in metadata: {len(df)}")
    except Exception as e:
        log(f"Failed to load station metadata: {e}")
        return

    # === STEP 2: Process all stations in parallel (lightweight) ===
    final_stations = []
    skipped = 0

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(process_station, row): idx for idx, row in df.iterrows()}

        for i, future in enumerate(as_completed(futures), 1):
            if i % 25 == 0:
                log(f"  Processed {i}/{len(futures)} stations ...")

            result = future.result()
            if result is not None:
                final_stations.append(result)
            else:
                skipped += 1

    log(f"Stations with at least one target variable: {len(final_stations)}")
    log(f"Stations skipped (no target params or fetch error): {skipped}")

    if not final_stations:
        log("No relevant stations found. Exiting.")
        return

    # Sort deterministically by id
    final_stations.sort(key=lambda x: x["id"])

    # === STEP 3: Save JSON ===
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_stations, f, ensure_ascii=False, indent=2)
    log(f"Saved station metadata JSON to: {OUTPUT_JSON}")

    # === STEP 4: Ready-to-copy set ===
    abbrs = sorted({s["id"] for s in final_stations})

    print("\n" + "=" * 80)
    print("COPY THE FOLLOWING SET INTO YOUR SCRIPTS IF YOU WANT TO HARD-CODE THE ABBREVIATIONS:")
    print("=" * 80)
    print("RELEVANT_METEOSUISSE_ABBRS = {")
    line = "    "
    for i, abbr in enumerate(abbrs):
        line += f'"{abbr}", '
        if (i + 1) % 8 == 0:
            print(line.rstrip())
            line = "    "
    if line.strip():
        print(line.rstrip())
    print("}")
    print("=" * 80)
    print(f"Total unique station abbreviations: {len(abbrs)}")

    log("\nDone. JSON is ready as master station list for MeteoSuisse (target variables only).")


if __name__ == "__main__":
    main()
