#!/usr/bin/env python3
"""
MeteoNetwork downloader (2020-2025)
-----------------------------------
One file per station per year:
  Network_providers/meteonetwork/{station_code}/{station_code}_{year}.json

Fetches /data-archive/{code} day-by-day (BULK token required),
aggregates into the yearly file, then removes the temporary day files.

Variables kept: temperature, rh, wind_speed, precipitation, snow_height.
Alpine Convention stations only.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_ROOT = PROJECT_ROOT

RAW_BASE = DATA_ROOT / "Network_providers" / "meteonetwork"
RAW_BASE.mkdir(parents=True, exist_ok=True)

LOG_DIR = DATA_ROOT / "code" / "logs" / "meteonetwork" / "download"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"meteonetwork_download_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("meteonetwork_download")

BASE = "https://api.meteonetwork.it/v3"
ALPINE = {"min_lon": 6.5, "max_lon": 15.5, "min_lat": 44.5, "max_lat": 47.5}
MAX_STATIONS = 150
YEARS = list(range(2020, 2026))
SLEEP = 3.0
RETRY_WAIT = 60
TIMEOUT = 60
HEADERS = {"User-Agent": "MeteoNetwork-thesis-downloader/1.0"}

# fields we care about (API names may vary)
KEEP = {
    "temperature", "temp", "temp_c",
    "rh", "relative_humidity", "humidity",
    "wind_speed", "wind", "wind_gust",
    "precipitation", "rain", "rain_rate", "precip",
    "snow_height", "snow", "snow_depth",
    "observation_time_utc", "observation_time_local", "station_code",
}

EMAIL = os.environ.get("METEONETWORK_EMAIL")
PASS = os.environ.get("METEONETWORK_PASS")
TOKEN = os.environ.get("METEONETWORK_TOKEN")


def get_token() -> str | None:
    if TOKEN:
        log.info("Using METEONETWORK_TOKEN from environment")
        return TOKEN
    if not EMAIL or not PASS:
        log.error("Set METEONETWORK_TOKEN or METEONETWORK_EMAIL + METEONETWORK_PASS")
        return None
    try:
        r = requests.post(
            f"{BASE}/login",
            data={"email": EMAIL, "password": PASS},
            timeout=TIMEOUT,
            headers=HEADERS,
        )
        r.raise_for_status()
        data = r.json()
        token = data.get("access_token") or data.get("token") or data
        if isinstance(token, dict):
            token = token.get("access_token") or token.get("token")
        if not token or not isinstance(token, str):
            log.error(f"Unexpected login response: {data}")
            return None
        log.info("Authenticated")
        return token
    except Exception as e:
        log.error(f"Login failed: {e}")
        return None


def auth_headers(token: str) -> dict:
    h = HEADERS.copy()
    h["Authorization"] = f"Bearer {token}"
    return h


def in_alpine(lon, lat) -> bool:
    try:
        return (
            ALPINE["min_lon"] <= float(lon) <= ALPINE["max_lon"]
            and ALPINE["min_lat"] <= float(lat) <= ALPINE["max_lat"]
        )
    except (TypeError, ValueError):
        return False


def slim(row: dict) -> dict:
    """Keep only fields of interest."""
    return {k: v for k, v in row.items() if k in KEEP or k.startswith("temp") or k.startswith("rain") or k.startswith("snow") or k.startswith("wind") or k.startswith("rh")}


def fetch_stations(token: str) -> list:
    try:
        r = requests.get(
            f"{BASE}/stations",
            params={"country": "IT"},
            timeout=TIMEOUT,
            headers=auth_headers(token),
        )
        if r.status_code == 401:
            log.warning("/stations requires BULK token")
            return []
        r.raise_for_status()
        stations = r.json()
        if not isinstance(stations, list):
            stations = stations.get("data") or stations.get("stations") or []
        alpine = [
            s for s in stations
            if in_alpine(s.get("longitude") or s.get("lon"), s.get("latitude") or s.get("lat"))
        ]
        if len(alpine) > MAX_STATIONS:
            log.info(f"Capping {len(alpine)} Alpine stations to {MAX_STATIONS}")
            alpine = alpine[:MAX_STATIONS]
        log.info(f"Alpine stations: {len(alpine)}")
        (RAW_BASE / "stations_metadata.json").write_text(
            json.dumps(alpine, indent=2), encoding="utf-8"
        )
        return alpine
    except Exception as e:
        log.warning(f"station list failed: {e}")
        return []


def fetch_period(token: str, code: str, start: str, end: str) -> list | None:
    """Try a date range. Returns rows or None if the API rejects the range."""
    try:
        r = requests.get(
            f"{BASE}/data-archive/{code}",
            params={"observation_date": start, "observation_date_to": end},
            timeout=TIMEOUT,
            headers=auth_headers(token),
        )
        if r.status_code == 429:
            log.warning("rate limited")
            time.sleep(RETRY_WAIT)
            return None
        if r.status_code in (400, 404, 405, 422):
            return None  # range not supported
        if r.status_code == 204:
            return []
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            data = data.get("data") or data.get("observations") or []
        return [slim(row) for row in data if isinstance(row, dict)]
    except Exception:
        return None


def fetch_day(token: str, code: str, date: str) -> list:
    for attempt in range(3):
        try:
            r = requests.get(
                f"{BASE}/data-archive/{code}",
                params={"observation_date": date},
                timeout=TIMEOUT,
                headers=auth_headers(token),
            )
            if r.status_code == 429:
                log.warning(f"  rate limited, waiting {RETRY_WAIT}s")
                time.sleep(RETRY_WAIT)
                continue
            if r.status_code in (204, 404):
                return []
            if r.status_code == 401:
                log.warning("archive requires BULK token")
                return []
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                data = data.get("data") or data.get("observations") or []
            return [slim(row) for row in data if isinstance(row, dict)]
        except Exception as e:
            log.warning(f"  {code} {date}: {e}")
            time.sleep(RETRY_WAIT)
    return []
    for attempt in range(3):
        try:
            r = requests.get(
                f"{BASE}/data-archive/{code}",
                params={"observation_date": date},
                timeout=TIMEOUT,
                headers=auth_headers(token),
            )
            if r.status_code == 429:
                log.warning(f"  rate limited, waiting {RETRY_WAIT}s")
                time.sleep(RETRY_WAIT)
                continue
            if r.status_code in (204, 404):
                return []
            if r.status_code == 401:
                log.warning("archive requires BULK token")
                return []
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                data = data.get("data") or data.get("observations") or []
            return [slim(row) for row in data if isinstance(row, dict)]
        except Exception as e:
            log.warning(f"  {code} {date}: {e}")
            time.sleep(RETRY_WAIT)
    return []


def main():
    log.info("=== MeteoNetwork download start ===")
    token = get_token()
    if not token:
        return

    stations = fetch_stations(token)
    if not stations:
        log.info("No stations. Request a BULK token and rerun.")
        return

    for st in stations:
        code = st.get("station_code") or st.get("code")
        if not code:
            continue
        station_dir = RAW_BASE / code
        station_dir.mkdir(exist_ok=True)

        for year in YEARS:
            year_path = station_dir / f"{code}_{year}.json"
            if year_path.exists():
                log.info(f"  {code} {year} already present")
                continue

            # try full year in one request
            rows = fetch_period(token, code, f"{year}-01-01", f"{year}-12-31")
            if rows is not None:
                year_path.write_text(json.dumps(rows), encoding="utf-8")
                log.info(f"  {code} {year} (single request) → {len(rows)} rows")
                time.sleep(SLEEP)
                continue

            # fall back to monthly
            year_rows = []
            for month in range(1, 13):
                start = f"{year}-{month:02d}-01"
                if month == 12:
                    end = f"{year}-12-31"
                else:
                    end = f"{year}-{month+1:02d}-01"
                m_rows = fetch_period(token, code, start, end)
                if m_rows is not None:
                    year_rows.extend(m_rows)
                    time.sleep(SLEEP)
                    continue
                # fall back to daily for this month
                d = datetime(year, month, 1)
                end_d = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
                while d < end_d:
                    year_rows.extend(fetch_day(token, code, d.strftime("%Y-%m-%d")))
                    d += timedelta(days=1)
                    time.sleep(SLEEP)
            year_path.write_text(json.dumps(year_rows), encoding="utf-8")
            log.info(f"  {code} {year} → {len(year_rows)} rows")

    log.info("=== MeteoNetwork download finished ===")


if __name__ == "__main__":
    main()
