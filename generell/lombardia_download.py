#!/usr/bin/env python3
"""
ARPA Lombardia systematic download (2020-2025)
----------------------------------------------
Downloads 10-min station data from the Regione Lombardia open data portal
(dati.lombardia.it / Socrata) and stores raw files under
Network_providers/lombardia/.

Station metadata: resource nf78-nj6b (Stazioni Idro Nivo Meteorologiche)
Data: per-variable historical datasets (see DATASETS below) + current month.

Only stations inside the Alpine Convention bounding box + buffer are kept.
Resumable. Logs to code/logs/lombardia/download/.

Style matches ZAMG / Südtirol / MeteoFrance scrapers.
"""

import json
import logging
import re
import time
from collections import defaultdict
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

RAW_BASE = DATA_ROOT / "Network_providers" / "lombardia"
RAW_BASE.mkdir(parents=True, exist_ok=True)

LOG_DIR = DATA_ROOT / "code" / "logs" / "lombardia" / "download"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"lombardia_download_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("lombardia_download")

# ---------------------------------------------------------------------------
# Alpine box + config
# ---------------------------------------------------------------------------
ALPINE = {
    "min_lon": 6.0,
    "max_lon": 16.5,
    "min_lat": 43.5,
    "max_lat": 48.8,
}

YEARS = list(range(2020, 2026))
SLEEP = 0.5
TIMEOUT = 90
HEADERS = {"User-Agent": "Lombardia-thesis-downloader/1.0"}
STATIONS_RESOURCE = "nf78-nj6b"


def in_alpine(lon, lat) -> bool:
    try:
        return (
            ALPINE["min_lon"] <= float(lon) <= ALPINE["max_lon"]
            and ALPINE["min_lat"] <= float(lat) <= ALPINE["max_lat"]
        )
    except (TypeError, ValueError):
        return False


def sanitize(name: str) -> str:
    name = re.sub(r"[^\w\s\-äöüÄÖÜß]", "", str(name).strip(), flags=re.UNICODE)
    for a, b in {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}.items():
        name = name.replace(a, b)
    return re.sub(r"\s+", "_", name) or "unknown"


def socrata_get(resource: str, params: dict | None = None) -> list:
    """Fetch all rows from a Socrata resource (handles pagination)."""
    url = f"https://www.dati.lombardia.it/resource/{resource}.json"
    all_rows = []
    offset = 0
    limit = 50000
    while True:
        p = {"$limit": limit, "$offset": offset}
        if params:
            p.update(params)
        r = requests.get(url, params=p, timeout=60, headers=HEADERS)

        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < limit:
            break
        offset += limit
        time.sleep(SLEEP)
    return all_rows


# Resource IDs from ARPALData (url_W_year_par.R)
RESOURCES = {
    "temperature": {
        (2011, 2020): "d4kj-kbpj",
        (2021, 2026): "w9wd-u6jh",
    },
    "precipitation": {
        (2011, 2020): "2kar-pnuk",
        (2021, 2026): "pstb-pga6",
    },
    "wind": {
        (2011, 2020): "x9gp-z9xx",
        (2021, 2026): "hu5q-68e3",
    },
    "relative_humidity": {
        (2011, 2020): "xpun-8722",
        (2021, 2026): "823w-fh4c",
    },
    "snow_height": {
        (2011, 2020): "id9e-7wf3",
        (2021, 2026): "uqbu-tt6m",
    },
}

TIPOLOGIA_TO_VAR = {
    "Temperatura": "temperature",
    "Precipitazione": "precipitation",
    "Velocità Vento": "wind",
    "Umidità Relativa": "relative_humidity",
    "Altezza neve": "snow_height",
    "Altezza Neve": "snow_height",
}



def get_resource(var: str, year: int) -> str | None:
    for (y1, y2), rid in RESOURCES.get(var, {}).items():
        if y1 <= year <= y2:
            return rid
    return None


def fetch_sensor_map() -> dict:
    """Return {station_id: {var: [sensor_ids]}} for Alpine stations."""
    log.info("Fetching full sensor metadata …")
    rows = socrata_get(STATIONS_RESOURCE)
    sensor_map = defaultdict(lambda: defaultdict(list))
    alpine_stations = set()

    for row in rows:
        sid = str(row.get("idstazione") or row.get("IdStazione") or "")
        lon = row.get("lng") or row.get("lon")
        lat = row.get("lat")
        if not sid or not in_alpine(lon, lat):
            continue
        alpine_stations.add(sid)
        tip = row.get("tipologia") or row.get("Tipologia") or ""
        var = TIPOLOGIA_TO_VAR.get(tip)
        sens = row.get("idsensore") or row.get("IdSensore")
        if var and sens:
            sensor_map[sid][var].append(str(sens))

    log.info(f"Alpine stations with sensors: {len(alpine_stations)}")
    return sensor_map, alpine_stations


def download_variable_year(var: str, year: int, sensor_ids: list[str], out_path: Path):
    rid = get_resource(var, year)
    if not rid:
        log.warning(f"  no resource for {var} {year}")
        return
    if out_path.exists():
        log.info(f"  {var} {year} already present")
        return

    log.info(f"  downloading {var} {year} ({len(sensor_ids)} sensors) …")
    all_rows = []
    # Smaller chunks and monthly slices to avoid long queries
    sensor_chunks = [sensor_ids[i:i+50] for i in range(0, len(sensor_ids), 50)]
    for m in range(1, 13):
        month_start = f"{year}-{m:02d}-01T00:00:00"
        if m == 12:
            month_end = f"{year+1}-01-01T00:00:00"
        else:
            month_end = f"{year}-{m+1:02d}-01T00:00:00"
        for chunk in sensor_chunks:
            ids = ",".join(f"'{s}'" for s in chunk)
            where = f"idsensore in ({ids}) and data >= '{month_start}' and data < '{month_end}'"
            try:
                rows = socrata_get(rid, {"$where": where, "$limit": 100000})
                all_rows.extend(rows)
                if rows:
                    log.info(f"    {year}-{m:02d} chunk: +{len(rows)} rows")
            except Exception as e:
                log.warning(f"    {year}-{m:02d} chunk failed: {e}")
            time.sleep(SLEEP)

    if all_rows:
        out_path.write_text(json.dumps(all_rows, ensure_ascii=False), encoding="utf-8")
        log.info(f"  {var} {year}: {len(all_rows):,} rows → {out_path.name}")
    else:
        log.info(f"  {var} {year}: no data")


def main():
    log.info("=== Lombardia download start ===")
    sensor_map, alpine_ids = fetch_sensor_map()


    # Save station list (unique stations)
    stations = []
    seen = set()
    # re-fetch minimal for metadata file (or rebuild from sensor_map if needed)
    meta_rows = socrata_get(STATIONS_RESOURCE)
    for row in meta_rows:
        sid = str(row.get("idstazione") or "")
        if sid in alpine_ids and sid not in seen:
            seen.add(sid)
            stations.append({
                "id": sid,
                "name": row.get("nomestazione") or sid,
                "clean": sanitize(row.get("nomestazione") or sid),
                "lon": float(row.get("lng") or 0),
                "lat": float(row.get("lat") or 0),
                "alt": row.get("quota"),
                "provincia": row.get("provincia"),
            })
    meta_path = RAW_BASE / "stations_metadata.json"
    meta_path.write_text(json.dumps(stations, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Wrote {len(stations)} stations to {meta_path}")

    # Download per variable / year
    for var in RESOURCES:
        log.info(f"--- {var} ---")
        all_sensors = []
        for sid, vars_ in sensor_map.items():
            all_sensors.extend(vars_.get(var, []))
        all_sensors = list(dict.fromkeys(all_sensors))  # unique
        log.info(f"  sensors: {len(all_sensors)}")
        for year in YEARS:
            out_path = RAW_BASE / f"{var}_{year}.json"
            download_variable_year(var, year, all_sensors, out_path)

    log.info("=== Lombardia download finished ===")


if __name__ == "__main__":
    main()
