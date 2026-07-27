#!/usr/bin/env python3
"""
DWD Station Metadata Collector (standalone, modeled after ZAMG stationsliste_ZAMG.py)

Collects all relevant stations south of 48.5Â°N that have data overlap with 2020-2025.
For each station we record which of the 5 variables it measures:
    measures_temperature, measures_relative_humidity (both from 'tu'),
    measures_precipitation, measures_wind, measures_snow_height.

Value is 1 if the station has measurements for that variable in the target period, else 0.

Outputs:
- Ready-to-copy RELEVANT_DWD_IDS set (printed, sorted)
- Full station metadata JSON (with measurement flags) saved to the Stationen/DWD folder
- Log written to the central Extraction logs folder
"""

import requests
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

# =============================================
# CONFIGURATION (edit if paths change)
# =============================================
LOG_DIR = Path(DATA_ROOT / "code" / "logs" / "dwd" / "extraction")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "dwd_stations_metadata.log"

STATION_DIR = Path(DATA_ROOT / "stationen" / "dwd")
STATION_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = STATION_DIR / "dwd_south_stations_48.5N_2020-2025.json"

LAT_THRESHOLD = 48.5
TARGET_START = datetime(2020, 1, 1)
TARGET_END = datetime(2025, 12, 31)

# We collect from these variables to get the union of all relevant stations
VARS_FOR_STATIONS = ["tu", "rr", "ff", "sh"]

# Mapping from internal var_key to the measurement flag column names in the final JSON
VARIABLE_TO_FLAGS = {
    "tu": ["measures_temperature", "measures_relative_humidity"],
    "rr": ["measures_precipitation"],
    "ff": ["measures_wind"],
    "sh": ["measures_snow_height"],
}

# Mapping for station list download (same as in download/extract scripts)
STATION_SOURCES = {
    "tu": {
        "base_url": "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/air_temperature/",
        "station_list": "zehn_min_tu_Beschreibung_Stationen.txt"
    },
    "rr": {
        "base_url": "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/precipitation/",
        "station_list": "zehn_min_rr_Beschreibung_Stationen.txt"
    },
    "ff": {
        "base_url": "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/wind/",
        "station_list": "zehn_min_ff_Beschreibung_Stationen.txt"
    },
    "sh": {
        "base_url": "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/more_precip/",
        "station_list": "RR_Tageswerte_Beschreibung_Stationen.txt"
    },
}


def log(msg: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")


def retry_request(url, retries=3, backoff=2, timeout=120):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except Exception as e:
            log(f"  Attempt {attempt+1} failed for {url}: {e}")
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

            time.sleep(backoff ** attempt)
    return None


def parse_station_list_robust(station_path: Path) -> pd.DataFrame:
    """Same proven robust parser used in download/extraction scripts."""
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
        if len(parts) < 8:
            continue
        try:
            sid = parts[0].strip()
            von = parts[1].strip()
            bis = parts[2].strip()
            hoehe = parts[3].strip()
            breite = parts[4].strip()
            laenge_idx = 5
            while laenge_idx < len(parts) and not (parts[laenge_idx].replace(".", "", 1).replace("-", "", 1).isdigit()):
                laenge_idx += 1
            if laenge_idx >= len(parts):
                continue
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
    df["geoLaenge"] = pd.to_numeric(df["geoLaenge"], errors="coerce")
    df["Stationshoehe"] = pd.to_numeric(df["Stationshoehe"], errors="coerce")
    df["von_datum"] = pd.to_datetime(df["von_datum"], format="%Y%m%d", errors="coerce")
    df["bis_datum"] = pd.to_datetime(df["bis_datum"], format="%Y%m%d", errors="coerce")
    return df


def get_valid_stations_for_var(var_key: str) -> list[dict]:
    """Download (or reuse) station list for one variable and return filtered south stations."""
    src = STATION_SOURCES[var_key]
    station_url = f"{src['base_url']}historical/{src['station_list']}"
    station_path = LOG_DIR.parent.parent / src['station_list']   # temp location next to logs

    log(f"  Fetching station list for {var_key} ...")
    r = retry_request(station_url)
    if not r:
        # try recent/ as fallback
        station_url2 = f"{src['base_url']}recent/{src['station_list']}"
        r = retry_request(station_url2)
        if not r:
            log(f"  Could not download station list for {var_key}")
            return []

    with open(station_path, "wb") as f:
        f.write(r.content)

    df = parse_station_list_robust(station_path)
    if df.empty:
        return []

    mask = (
        (df["geoBreite"] < LAT_THRESHOLD) &
        (df["von_datum"] <= TARGET_END) &
        ((df["bis_datum"] >= TARGET_START) | df["bis_datum"].isna())
    )
    south_df = df[mask].copy()

    stations = []
    for _, row in south_df.iterrows():
        stations.append({
            "id": row["Stations_id"],
            "name": str(row["Stationsname"]).strip(),
            "lat": float(row["geoBreite"]),
            "lon": float(row["geoLaenge"]),
            "altitude": float(row["Stationshoehe"]) if pd.notna(row["Stationshoehe"]) else None
        })
    return stations


def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=== DWD South-of-48.5Â°N Station Metadata Collector ===")
    log(f"Target period: 2020-01-01 to 2025-12-31 | Lat < {LAT_THRESHOLD}Â°N")
    log(f"Logs: {LOG_DIR}")
    log(f"Output JSON: {OUTPUT_JSON}")

    # Track which stations measure which variables
    var_to_sids = {v: set() for v in VARS_FOR_STATIONS}
    all_stations = {}  # sid -> metadata dict (without flags yet)

    for var_key in VARS_FOR_STATIONS:
        log(f"\nProcessing variable: {var_key}")
        var_stations = get_valid_stations_for_var(var_key)
        log(f"  Found {len(var_stations)} valid stations for {var_key}")

        for s in var_stations:
            sid = s["id"]
            var_to_sids[var_key].add(sid)

            if sid not in all_stations:
                all_stations[sid] = s
            else:
                # keep the entry that has non-None altitude if possible
                existing = all_stations[sid]
                if existing.get("altitude") is None and s.get("altitude") is not None:
                    all_stations[sid] = s

    # Enrich every station with 0/1 measurement flags for all 5 variables
    for sid, meta in all_stations.items():
        meta["measures_temperature"]       = 1 if sid in var_to_sids["tu"] else 0
        meta["measures_relative_humidity"] = 1 if sid in var_to_sids["tu"] else 0
        meta["measures_precipitation"]     = 1 if sid in var_to_sids["rr"] else 0
        meta["measures_wind"]              = 1 if sid in var_to_sids["ff"] else 0
        meta["measures_snow_height"]       = 1 if sid in var_to_sids["sh"] else 0

    # Final list sorted by id (numeric)
    final_list = sorted(all_stations.values(), key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else x["id"])

    log(f"\nTotal unique south stations across all variables: {len(final_list)}")

    if not final_list:
        log("No stations found. Exiting.")
        return

    # Save JSON (now includes the 5 measurement flag columns)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)
    log(f"Saved station metadata JSON to: {OUTPUT_JSON}")

    # Print ready-to-copy RELEVANT_DWD_IDS set (useful for hard-coding in other scripts)
    ids = []
    for s in final_list:
        try:
            ids.append(int(s["id"]))
        except (ValueError, TypeError):
            ids.append(s["id"])
    ids = sorted(set(ids))

    print("\n" + "=" * 75)
    print("COPY THE FOLLOWING SET INTO YOUR SCRIPTS IF YOU WANT TO HARD-CODE THE IDs:")
    print("=" * 75)
    print("RELEVANT_DWD_IDS = {")
    line = "    "
    for i, iid in enumerate(ids):
        line += f"{iid}, "
        if (i + 1) % 10 == 0:   # 10 per line for readability
            print(line.rstrip())
            line = "    "
    if line.strip():
        print(line.rstrip())
    print("}")
    print("=" * 75)
    print(f"Total unique station IDs: {len(ids)}")

    log("\nDone. You can now use this JSON as your master station list for DWD south of 48.5Â°N.")


if __name__ == "__main__":
    main()
