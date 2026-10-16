#!/usr/bin/env python3
"""
MeteoNetwork unpacker
---------------------
Reads yearly station files from Network_providers/meteonetwork/
and writes clean per-variable JSON:

  temperatur/meteonetwork/<clean>/<clean>_temperature_data.json
  niederschlag/meteonetwork/...
  wind/meteonetwork/...
  luftfeuchte/meteonetwork/...
  schneehoehe/meteonetwork/...

Timestamps expected as ISO UTC strings.
"""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_ROOT = PROJECT_ROOT

RAW_BASE = DATA_ROOT / "Network_providers" / "meteonetwork"

OUTPUT_BASES = {
    "temperature": DATA_ROOT / "temperatur" / "meteonetwork",
    "precipitation": DATA_ROOT / "niederschlag" / "meteonetwork",
    "snow_height": DATA_ROOT / "schneehoehe" / "meteonetwork",
    "wind": DATA_ROOT / "wind" / "meteonetwork",
    "relative_humidity": DATA_ROOT / "luftfeuchte" / "meteonetwork",
}
for p in OUTPUT_BASES.values():
    p.mkdir(parents=True, exist_ok=True)

LOG_DIR = DATA_ROOT / "code" / "logs" / "meteonetwork" / "unpack"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"meteonetwork_unpack_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("meteonetwork_unpack")

# map possible API field names → our variable
FIELD_MAP = {
    "temperature": ["temperature", "temp", "temp_c"],
    "relative_humidity": ["rh", "relative_humidity", "humidity"],
    "wind": ["wind_speed", "wind"],
    "precipitation": ["precipitation", "rain", "precip"],
    "snow_height": ["snow_height", "snow", "snow_depth"],
}


def clean_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s.strip())
    return s.strip("_").lower() or "unknown"


def pick(row: dict, keys: list):
    for k in keys:
        if k in row and row[k] not in (None, "", "null"):
            try:
                return float(row[k])
            except (TypeError, ValueError):
                return None
    return None


def main():
    log.info("=== MeteoNetwork unpack start ===")
    # station_code → {var → [(ts, value), ...]}
    by_station = defaultdict(lambda: defaultdict(list))

    for station_dir in RAW_BASE.iterdir():
        if not station_dir.is_dir():
            continue
        for year_file in station_dir.glob("*_*.json"):
            if year_file.name == "stations_metadata.json":
                continue
            try:
                rows = json.loads(year_file.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning(f"skip {year_file}: {e}")
                continue
            code = station_dir.name
            for row in rows:
                ts = row.get("observation_time_utc") or row.get("observation_time_local")
                if not ts:
                    continue
                for var, keys in FIELD_MAP.items():
                    val = pick(row, keys)
                    if val is not None:
                        by_station[code][var].append({"time": ts, "value": val})

    for code, vars_ in by_station.items():
        clean = clean_name(code)
        for var, series in vars_.items():
            if not series:
                continue
            out_dir = OUTPUT_BASES[var] / clean
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{clean}_{var}_data.json"
            # sort and drop duplicates
            series = sorted(series, key=lambda x: x["time"])
            seen = set()
            unique = []
            for item in series:
                if item["time"] not in seen:
                    seen.add(item["time"])
                    unique.append(item)
            out_path.write_text(json.dumps(unique), encoding="utf-8")
            log.info(f"  {clean} {var}: {len(unique)} rows")

    log.info("=== MeteoNetwork unpack finished ===")


if __name__ == "__main__":
    main()
