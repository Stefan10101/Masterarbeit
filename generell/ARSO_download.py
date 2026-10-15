#!/usr/bin/env python3
"""
ARSO systematic download (2020-2025)
------------------------------------
Fetches automatic-station (type 4) data from the ARSO webmet archive.
Half-hourly in 3-month chunks (avoids 502s). Sequential.

Raw files land in:
  <Daten>/Network_providers/ARSO/halfhourly/<station_id>_<YYYY>_<Q>.json
  <Daten>/Network_providers/ARSO/daily/<station_id>.json
  <Daten>/Network_providers/ARSO/stations_metadata.json

Resumable. Retries on 502/503/504/timeout.
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_ROOT = PROJECT_ROOT

RAW_BASE = DATA_ROOT / "Network_providers" / "ARSO"
HH_DIR = RAW_BASE / "halfhourly"
DAILY_DIR = RAW_BASE / "daily"
for d in (HH_DIR, DAILY_DIR):
    d.mkdir(parents=True, exist_ok=True)

LOG_DIR = DATA_ROOT / "code" / "logs" / "arso" / "download"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"arso_download_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("arso_download")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = "https://meteo.arso.gov.si/webmet/archive"
YEARS = list(range(2020, 2026))
START = "2020-01-01"
END = "2025-12-31"

HH_VARS = "15,18,26,21"   # mean T, mean RH, precip, mean wind
HH_TYPE = "halfhourly"

DAILY_VARS = "88,85"      # total snow depth, 24-h precip sum
DAILY_TYPE = "daily"

SLEEP_SEC = 1.0
TIMEOUT = 90
MAX_RETRIES = 4
RETRY_SLEEP = 8
TEST_LIMIT = None         # set to e.g. 2 for a quick test

# 3-month quarters: (label, start_month, end_month)
QUARTERS = [
    ("Q1", 1, 3),
    ("Q2", 4, 6),
    ("Q3", 7, 9),
    ("Q4", 10, 12),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ARSO-thesis-downloader/1.0)",
    "Accept": "*/*",
}


def sanitize(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\s\-äöüÄÖÜß]", "", name, flags=re.UNICODE)
    replacements = {
        "ä": "ae", "ö": "oe", "ü": "ue",
        "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss",
        "č": "c", "š": "s", "ž": "z",
        "Č": "C", "Š": "S", "Ž": "Z",
    }
    for a, b in replacements.items():
        name = name.replace(a, b)
    name = re.sub(r"\s+", "_", name)
    return name or "unknown"


def parse_locations_xml(xml_text: str) -> list[dict]:
    stations = []
    pattern = re.compile(
        r'_(\d+):\s*\{\s*name:"([^"]+)"\s*,\s*lon:([\d.\-]+)\s*,\s*lat:([\d.\-]+)\s*,\s*alt:([\d.\-]+)'
    )
    for m in pattern.finditer(xml_text):
        sid, name, lon, lat, alt = m.groups()
        stations.append({
            "id": sid,
            "name": name,
            "clean_name": sanitize(name),
            "lon": float(lon),
            "lat": float(lat),
            "alt": float(alt),
            "type": 4,
        })
    return stations


def fetch_stations() -> list[dict]:
    url = f"{BASE}/locations.xml?d1={START}&d2={END}&type=4"
    log.info("Fetching automatic station list …")
    r = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
    r.raise_for_status()
    stations = parse_locations_xml(r.text)
    log.info(f"Found {len(stations)} automatic stations")
    return stations


def fetch_data(station_id: str, api_type: str, vars_: str, d1: str, d2: str) -> str | None:
    url = (
        f"{BASE}/data.xml?lang=en&vars={vars_}&type={api_type}"
        f"&id={station_id}&d1={d1}&d2={d2}"
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
            if r.status_code in (502, 503, 504):
                log.warning(f"  {r.status_code} attempt {attempt}/{MAX_RETRIES} {station_id} {d1}-{d2}")
                time.sleep(RETRY_SLEEP * attempt)
                continue
            r.raise_for_status()
            text = r.text
            m = re.search(r"AcademaPUJS.set\(\s*(\{.+)\s*\)", text, re.DOTALL)
            if not m:
                log.warning(f"No JSON payload for {station_id} {api_type} {d1}-{d2}")
                return None
            return m.group(1)
        except requests.exceptions.Timeout:
            log.warning(f"  timeout attempt {attempt}/{MAX_RETRIES} {station_id} {d1}-{d2}")
            time.sleep(RETRY_SLEEP * attempt)
        except Exception as e:
            log.error(f"Request failed {station_id} {api_type} {d1}-{d2}: {e}")
            if attempt == MAX_RETRIES:
                return None
            time.sleep(RETRY_SLEEP)
    log.error(f"Gave up after {MAX_RETRIES} attempts: {station_id} {d1}-{d2}")
    return None


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def quarter_ranges(year: int):
    """Yield (d1, d2, label) for each 3-month block of the year."""
    for label, m_start, m_end in QUARTERS:
        d1 = f"{year}-{m_start:02d}-01"
        # last day of end month
        if m_end == 12:
            d2 = f"{year}-12-31"
        else:
            # simple: use last day of the end month
            import calendar
            last = calendar.monthrange(year, m_end)[1]
            d2 = f"{year}-{m_end:02d}-{last:02d}"
        yield d1, d2, label


def download_halfhourly(station: dict) -> None:
    sid = station["id"]
    for year in YEARS:
        for d1, d2, qlabel in quarter_ranges(year):
            out = HH_DIR / f"{sid}_{year}_{qlabel}.json"
            if out.exists() and out.stat().st_size > 200:
                log.info(f"  skip hh {sid} {year}-{qlabel}")
                continue
            log.info(f"  hh {station['name']} ({sid}) {year}-{qlabel}")
            payload = fetch_data(sid, HH_TYPE, HH_VARS, d1, d2)
            if payload is None:
                continue
            obj = {
                "station": station,
                "resolution": "halfhourly",
                "year": year,
                "quarter": qlabel,
                "vars": HH_VARS,
                "payload": payload,
                "fetched_at": datetime.utcnow().isoformat(),
            }
            save_json(out, obj)
            time.sleep(SLEEP_SEC)


def download_daily(station: dict) -> None:
    sid = station["id"]
    out = DAILY_DIR / f"{sid}.json"
    if out.exists() and out.stat().st_size > 200:
        log.info(f"  skip daily {sid}")
        return
    log.info(f"  daily {station['name']} ({sid})")
    payload = fetch_data(sid, DAILY_TYPE, DAILY_VARS, START, END)
    if payload is None:
        return
    obj = {
        "station": station,
        "resolution": "daily",
        "vars": DAILY_VARS,
        "payload": payload,
        "fetched_at": datetime.utcnow().isoformat(),
    }
    save_json(out, obj)
    time.sleep(SLEEP_SEC)


def main():
    log.info("=== ARSO download start (3-month chunks, sequential) ===")
    stations = fetch_stations()
    meta_path = RAW_BASE / "stations_metadata.json"
    save_json(meta_path, {"stations": stations, "generated": datetime.utcnow().isoformat()})

    if TEST_LIMIT:
        stations = stations[:TEST_LIMIT]
        log.info(f"TEST MODE: first {TEST_LIMIT} stations only")

    for i, st in enumerate(stations, 1):
        log.info(f"[{i}/{len(stations)}] {st['name']} (id={st['id']})")
        try:
            download_halfhourly(st)
            download_daily(st)
        except Exception as e:
            log.exception(f"Station {st['id']} failed: {e}")
        time.sleep(0.3)

    log.info("=== ARSO download finished ===")


if __name__ == "__main__":
    main()
