#!/usr/bin/env python3
"""
Météo-France API downloader for Alpine stations (2020-2025).

Uses the Données Climatologiques API (DPClim) via portail-api.meteofrance.fr.
Orders and downloads hourly (or daily) data for selected Alpine stations,
stores raw CSV + metadata per station under meteofrance/raw/.

Requires free account and subscription to "API Données climatologiques".
Set METEOFRANCE_CLIENT_AUTH env var (the Basic auth string from the portal).

Only stations in Alpine departments (or lon > 5.0) are considered.
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===============================================
# PATHS (relative to Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_ROOT = PROJECT_ROOT

# =============================================
# CONFIG
# =============================================
BASE_URL = "https://public-api.meteofrance.fr/public/DPClim/v1"
TOKEN_URL = "https://portail-api.meteofrance.fr/token"

CLIENT_AUTH = os.environ.get("METEOFRANCE_CLIENT_AUTH", "")
API_TOKEN = os.environ.get("METEOFRANCE_TOKEN", "")  # direct Bearer token / API key

if not CLIENT_AUTH and not API_TOKEN:
    raise RuntimeError("Set METEOFRANCE_TOKEN (direct API key) or METEOFRANCE_CLIENT_AUTH")

YEARS = range(2020, 2026)
STEP = "1h"  # "1h" or "1d"
ALPINE_DEPTS = ["01", "04", "05", "06", "07", "25", "26", "38", "39", "73", "74", "90"]  # Ain, Alpes-*, Isère, Savoie, Jura, etc.
MIN_LON = 5.5
MAX_LON = 7.5
MIN_LAT = 43.8
MAX_LAT = 46.5

TEST_LIMIT = None  # set to small int for testing

RAW_DIR = DATA_ROOT / "meteofrance" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = DATA_ROOT / "code" / "logs" / "meteofrance" / "api"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "meteofrance_download_log.txt"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_token() -> str:
    if API_TOKEN:
        return API_TOKEN
    if not CLIENT_AUTH:
        raise RuntimeError("No METEOFRANCE_TOKEN or METEOFRANCE_CLIENT_AUTH set")
    headers = {"Authorization": f"Basic {CLIENT_AUTH}"}
    data = {"grant_type": "client_credentials"}
    r = requests.post(TOKEN_URL, headers=headers, data=data, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def list_stations(token: str, dept: str, step: str = "1h") -> list:
    freq = {"6m": "infrahoraire-6m", "1h": "horaire", "1d": "quotidienne"}[step]
    url = f"{BASE_URL}/liste-stations/{freq}"
    params = {"id-departement": dept}
    # try both common auth styles
    for headers in (
        {"Authorization": f"Bearer {token}"},
        {"apikey": token, "Authorization": f"Bearer {token}"},
        {"Authorization": token},
    ):
        r = requests.get(url, params=params, headers=headers, timeout=60)
        if r.status_code == 200:
            break
    if r.status_code != 200:
        log(f"list_stations {dept} failed {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
    data = r.json()
    # API can return a list directly, or a dict wrapping the list
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        stations = data.get("stations") or data.get("listeStations") or data.get("station")
        if isinstance(stations, list):
            return stations
        if isinstance(stations, dict):
            return stations.get("station", []) if isinstance(stations.get("station"), list) else []
    return []


def get_station_info(token: str, station_id: str) -> dict:
    url = f"{BASE_URL}/information-station"
    params = {"id-station": station_id}
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    if r.status_code != 200:
        return {}
    return r.json()


def order_and_download(token: str, station_id: str, step: str, date_deb: str, date_fin: str, out_path: Path) -> bool:
    if out_path.exists():
        return True
    fail_marker = out_path.with_suffix(".fail")
    if fail_marker.exists():
        return False  # already known to fail; skip
    freq_map = {"6m": "infrahoraire-6m", "1h": "horaire", "1d": "quotidienne"}
    freq = freq_map[step]
    order_url = f"{BASE_URL}/commande-station/{freq}"
    params = {
        "id-station": station_id,
        "date-deb-periode": date_deb,
        "date-fin-periode": date_fin,
    }
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(order_url, params=params, headers=headers, timeout=60)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        log(f"Order request failed {station_id}: {type(e).__name__}")
        return False
    if r.status_code != 202:
        log(f"Order failed {station_id}: {r.status_code} {r.text[:200]}")
        # permanent-looking failures: mark so we don't retry forever
        if r.status_code in (400, 404):
            fail_marker.touch()
        return False
    order_id = r.json().get("elaboreProduitAvecDemandeResponse", {}).get("return")
    if not order_id:
        log(f"No order id for {station_id}")
        fail_marker.touch()
        return False

    # poll for file
    dl_url = f"{BASE_URL}/commande/fichier"
    for attempt in range(60):  # ~5 min max
        time.sleep(5)
        try:
            dr = requests.get(dl_url, params={"id-cmde": order_id}, headers=headers, timeout=60, stream=True)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            log(f"Poll timeout/conn error {station_id} attempt {attempt}: {type(e).__name__}")
            continue
        if dr.status_code == 201:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                for chunk in dr.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        elif dr.status_code >= 400:
            log(f"Download poll failed {station_id}: {dr.status_code}")
            if dr.status_code in (400, 404):
                fail_marker.touch()
            return False
    log(f"Timeout waiting for {station_id}")
    return False


def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    log("=== Météo-France Alpine stations download ===")
    token = get_token()
    log("Token obtained")

    # collect candidate stations
    all_stations = {}
    sample_printed = False
    for dept in ALPINE_DEPTS:
        try:
            sts = list_stations(token, dept, STEP)
            log(f"Dept {dept}: {len(sts)} stations")
            if sts and not sample_printed:
                log(f"Sample station keys: {list(sts[0].keys())}")
                log(f"Sample station: {sts[0]}")
                sample_printed = True
            for s in sts:
                sid = str(s.get("id") or s.get("Id_station") or s.get("station_id", ""))
                if not sid:
                    continue
                lon = s.get("longitude") or s.get("Longitude") or s.get("lon")
                lat = s.get("latitude") or s.get("Latitude") or s.get("lat")
                if lon is None or lat is None:
                    info = get_station_info(token, sid)
                    lon = info.get("longitude") or info.get("lon")
                    lat = info.get("latitude") or info.get("lat")
                try:
                    lon_f = float(lon) if lon else None
                    lat_f = float(lat) if lat else None
                except (TypeError, ValueError):
                    lon_f = lat_f = None
                if (
                    lon_f is None or lat_f is None
                    or lon_f < MIN_LON or lon_f > MAX_LON
                    or lat_f < MIN_LAT or lat_f > MAX_LAT
                ):
                    continue
                name = s.get("nom") or s.get("Nom_usuel") or s.get("name") or sid
                all_stations[sid] = {
                    "id": sid,
                    "name": name,
                    "lat": lat_f,
                    "lon": lon_f,
                    "dept": dept,
                    "altitude": s.get("altitude") or s.get("Altitude"),
                }
            time.sleep(1)
        except Exception as e:
            log(f"Error listing dept {dept}: {e}")

    log(f"Alpine stations selected (lon {MIN_LON}-{MAX_LON}, lat {MIN_LAT}-{MAX_LAT}): {len(all_stations)}")
    if not all_stations:
        log("No stations found. Exit.")
        return

    items = list(all_stations.items())
    if TEST_LIMIT:
        items = items[:TEST_LIMIT]
        log(f"TEST mode: {TEST_LIMIT} stations")

    success_stations = 0
    for sid, info in items:
        meta_path = RAW_DIR / f"{sid}_meta.json"
        if not meta_path.exists():
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)

        year_ok = 0
        fail_count = 0
        for year in YEARS:
            out_csv = RAW_DIR / f"{sid}_{STEP}_{year}.csv"
            fail_marker = out_csv.with_suffix(".fail")
            if out_csv.exists():
                year_ok += 1
                continue
            if fail_marker.exists():
                fail_count += 1
                if fail_count >= 3:
                    log(f"Skipping rest of {sid}: already {fail_count} failures")
                    break
                continue
            date_deb = f"{year}-01-01T00:00:00Z"
            date_fin = f"{year}-12-31T23:59:59Z"
            log(f"Processing {info['name']} ({sid}) year {year}")
            try:
                ok = order_and_download(token, sid, STEP, date_deb, date_fin, out_csv)
                if ok:
                    year_ok += 1
                elif fail_marker.exists():
                    fail_count += 1
                    if fail_count >= 3:
                        log(f"Skipping rest of {sid}: {fail_count} failures")
                        # mark remaining years as fail so they are skipped next run
                        for y2 in YEARS:
                            if y2 <= year:
                                continue
                            m2 = (RAW_DIR / f"{sid}_{STEP}_{y2}.csv").with_suffix(".fail")
                            if not (RAW_DIR / f"{sid}_{STEP}_{y2}.csv").exists():
                                m2.touch()
                        break
            except Exception as e:
                log(f"Unexpected error {sid} {year}: {e}")
            time.sleep(1)

        if year_ok == len(YEARS):
            success_stations += 1
            log(f"Complete: {sid} all years")
        else:
            log(f"Partial: {sid} {year_ok}/{len(YEARS)} years")

    log(f"Finished. {success_stations}/{len(items)} stations fully downloaded.")


if __name__ == "__main__":
    main()
