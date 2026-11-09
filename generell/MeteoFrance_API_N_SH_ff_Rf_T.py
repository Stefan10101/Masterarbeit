#!/usr/bin/env python3
"""
Météo-France API downloader for stations inside the analysis domain.

Uses the Données Climatologiques API (DPClim) via portail-api.meteofrance.fr.
Orders and downloads hourly (or daily) data for French stations that fall
inside the Alpine Convention perimeter + 100 km buffer (same domain used
by the rest of the project). Stores raw CSV + metadata per station under
meteofrance/raw/.

Requires free account and subscription to "API Données climatologiques".
Set METEOFRANCE_CLIENT_AUTH env var (the Basic auth string from the portal)
or METEOFRANCE_TOKEN (direct Bearer / API key).
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime

# ===============================================
# PATHS (relative to Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_ROOT = PROJECT_ROOT

# shared domain filter (Alpine Convention perimeter + 100 km)
# implemented locally so the downloader does not depend on evaluation_sets (yaml)
sys.path.insert(0, str(SCRIPT.parents[1]))
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

PERIMETER_GEOJSON = SCRIPT.parents[1] / "shared" / "splits" / "geometry_cache" / "alpine_convention_perimeter.geojson"


def stations_inside_domain(stations_df, lon_col="lon", lat_col="lat", buffer_km=100.0):
    """Keep stations inside Alpine Convention perimeter + buffer_km (EPSG:31287)."""
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas required for domain filter")
    if not PERIMETER_GEOJSON.exists():
        raise FileNotFoundError(f"perimeter geojson missing: {PERIMETER_GEOJSON}")
    peri = gpd.read_file(PERIMETER_GEOJSON)
    if peri.crs is None:
        peri = peri.set_crs("EPSG:4326")
    peri = peri.to_crs("EPSG:31287")
    peri = peri.copy()
    peri["geometry"] = peri.geometry.buffer(0)
    buffered = peri.buffer(buffer_km * 1000.0).union_all()
    df = stations_df.dropna(subset=[lon_col, lat_col]).copy()
    pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    ).to_crs("EPSG:31287")
    inside = pts.intersects(buffered)
    return df.loc[inside.to_numpy()].copy()


HAS_DOMAIN = HAS_GEOPANDAS and PERIMETER_GEOJSON.exists()
if not HAS_DOMAIN:
    print("Warning: domain filter unavailable (geopandas or perimeter geojson missing); falling back to wide bbox")

# =============================================
# CONFIG
# =============================================
BASE_URL = "https://public-api.meteofrance.fr/public/DPClim/v1"
TOKEN_URL = "https://portail-api.meteofrance.fr/token"

CLIENT_AUTH = os.environ.get("METEOFRANCE_CLIENT_AUTH", "")
API_TOKEN = os.environ.get("METEOFRANCE_TOKEN", "")  # direct Bearer token / API key

# optional local token file (avoids pasting a long JWT into the shell every run)
_TOKEN_FILE = SCRIPT.parent / "meteofrance_token.txt"
if not API_TOKEN and _TOKEN_FILE.exists():
    API_TOKEN = _TOKEN_FILE.read_text(encoding="utf-8").strip()

if not CLIENT_AUTH and not API_TOKEN:
    raise RuntimeError(
        "Set METEOFRANCE_CLIENT_AUTH (Basic string from portal curl) "
        "or METEOFRANCE_TOKEN (API-key JWT).\n"
        "Ordering (commande-station) requires the OAuth client credentials:\n"
        "  PowerShell:  $env:METEOFRANCE_CLIENT_AUTH = 'the-string-after-Basic'\n"
        "Listing alone works with the API-key JWT in METEOFRANCE_TOKEN "
        "or in " + str(_TOKEN_FILE)
    )

YEARS = range(2020, 2026)
STEP = "1h"  # "1h" or "1d"

# French departments that can contain stations inside the Alpine domain + 100 km.
# Geometry filter (stations_inside_domain) keeps only those actually inside.
# Extra departments cost only a list call; downloads are still domain-limited.
ALPINE_DEPTS = [
    "01", "04", "05", "06", "07",          # Ain, Alpes-de-HP, Hautes-Alpes, Alpes-Maritimes, Ardèche
    "13", "15", "21", "25", "26",          # Bouches-du-Rhône, Cantal, Côte-d'Or, Doubs, Drôme
    "38", "39", "42", "43",                # Isère, Jura, Loire, Haute-Loire
    "63", "68", "69", "71",                # Puy-de-Dôme, Haut-Rhin, Rhône, Saône-et-Loire
    "73", "74", "83", "84", "90",          # Savoie, Haute-Savoie, Var, Vaucluse, Territoire de Belfort
]

# Fallback bounding box (perimeter + ~100 km) used only if the domain filter cannot be imported.
# Perimeter itself is roughly lon 4.9–16.5, lat 43.4–48.4; +100 km ≈ ±1.3° lon / ±0.9° lat.
MIN_LON = 3.5
MAX_LON = 18.0
MIN_LAT = 42.4
MAX_LAT = 49.5

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
    """Return a usable token.

    Prefer OAuth2: if METEOFRANCE_CLIENT_AUTH is set, exchange it for a
    short-lived access token (Bearer, ~1 h). The API-key JWT works for
    listing but is rejected by commande-station; ordering needs the OAuth token.
    Call again to refresh after expiry.
    """
    if CLIENT_AUTH:
        headers = {"Authorization": f"Basic {CLIENT_AUTH}"}
        data = {"grant_type": "client_credentials"}
        r = requests.post(TOKEN_URL, headers=headers, data=data, timeout=30)
        r.raise_for_status()
        tok = r.json().get("access_token")
        if not tok:
            raise RuntimeError("Token endpoint returned no access_token")
        return tok
    if API_TOKEN:
        return API_TOKEN
    raise RuntimeError("No METEOFRANCE_TOKEN or METEOFRANCE_CLIENT_AUTH set")


def refresh_token(auth: dict) -> str:
    """Refresh the OAuth token in place. Returns the new token."""
    if not CLIENT_AUTH:
        return auth["token"]
    log("Refreshing OAuth token")
    auth["token"] = get_token()
    return auth["token"]


def _auth_headers(token: str) -> list:
    """Header dicts to try. Bearer first when we have an OAuth token;
    apikey first when we only have the portal API-key JWT.
    """
    if CLIENT_AUTH:
        return [
            {"Authorization": f"Bearer {token}", "accept": "application/json"},
            {"apikey": token, "accept": "application/json"},
        ]
    return [
        {"apikey": token, "accept": "application/json"},
        {"Authorization": f"Bearer {token}", "accept": "application/json"},
        {"apikey": token, "Authorization": f"Bearer {token}"},
        {"Authorization": token},
    ]


def list_stations(token: str, dept: str, step: str = "1h") -> list:
    freq = {"6m": "infrahoraire-6m", "1h": "horaire", "1d": "quotidienne"}[step]
    url = f"{BASE_URL}/liste-stations/{freq}"
    params = {"id-departement": dept}
    r = None
    for headers in _auth_headers(token):
        r = requests.get(url, params=params, headers=headers, timeout=60)
        if r.status_code == 200:
            break
    if r is None or r.status_code != 200:
        code = r.status_code if r is not None else "no-response"
        body = r.text[:300] if r is not None else ""
        log(f"list_stations {dept} failed {code}: {body}")
        if r is not None:
            r.raise_for_status()
        raise RuntimeError(f"list_stations {dept} failed")
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
    for headers in _auth_headers(token):
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 200:
            return r.json()
    return {}


def order_and_download(auth: dict, station_id: str, step: str, date_deb: str, date_fin: str, out_path: Path) -> bool:
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

    def _place_order() -> tuple:
        token = auth["token"]
        headers = None
        r = None
        for h in _auth_headers(token):
            try:
                r = requests.get(order_url, params=params, headers=h, timeout=60)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                log(f"Order request failed {station_id}: {type(e).__name__}")
                return None, None
            if r.status_code in (200, 202):
                headers = h
                break
        return headers, r

    headers, r = _place_order()
    # token expired mid-run: refresh once and retry the order
    if (headers is None) and r is not None and r.status_code == 401 and CLIENT_AUTH:
        refresh_token(auth)
        headers, r = _place_order()

    if headers is None:
        status = r.status_code if r is not None else "no-response"
        body = r.text[:200] if r is not None else ""
        log(f"Order failed {station_id}: {status} {body}")
        if r is not None and r.status_code == 401:
            log("401 on commande-station after refresh. Check METEOFRANCE_CLIENT_AUTH.")
            return False
        if r is not None and r.status_code in (400, 404):
            fail_marker.touch()
        return False
    if r.status_code != 202:
        log(f"Order unexpected status {station_id}: {r.status_code}")
        return False
    order_id = r.json().get("elaboreProduitAvecDemandeResponse", {}).get("return")
    if not order_id:
        log(f"No order id for {station_id}")
        fail_marker.touch()
        return False

    # poll for file; refresh token on 401 and keep polling
    dl_url = f"{BASE_URL}/commande/fichier"
    refreshed_on_poll = False
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
        if dr.status_code == 401 and CLIENT_AUTH and not refreshed_on_poll:
            log(f"Poll 401 for {station_id}; refreshing token and retrying")
            refresh_token(auth)
            headers = _auth_headers(auth["token"])[0]
            refreshed_on_poll = True
            continue
        if dr.status_code >= 400:
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
    auth = {"token": get_token()}
    log("Token obtained")

    # collect candidate stations
    all_stations = {}
    sample_printed = False
    for dept in ALPINE_DEPTS:
        try:
            sts = list_stations(auth["token"], dept, STEP)
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
                    info = get_station_info(auth["token"], sid)
                    lon = info.get("longitude") or info.get("lon")
                    lat = info.get("latitude") or info.get("lat")
                try:
                    lon_f = float(lon) if lon else None
                    lat_f = float(lat) if lat else None
                except (TypeError, ValueError):
                    lon_f = lat_f = None
                if lon_f is None or lat_f is None:
                    continue
                # wide pre-filter only when domain geometry is unavailable
                if not HAS_DOMAIN and (
                    lon_f < MIN_LON or lon_f > MAX_LON
                    or lat_f < MIN_LAT or lat_f > MAX_LAT
                ):
                    continue
                name = s.get("nom") or s.get("Nom_usuel") or s.get("name") or sid
                # altitude: list-stations endpoint returns "alt" (not "altitude")
                alt_raw = s.get("alt") or s.get("altitude") or s.get("Altitude")
                alt_f = None
                if alt_raw is not None:
                    try:
                        alt_f = float(alt_raw)
                    except (TypeError, ValueError):
                        alt_f = None
                all_stations[sid] = {
                    "id": sid,
                    "name": name,
                    "lat": lat_f,
                    "lon": lon_f,
                    "dept": dept,
                    "altitude": alt_f,
                }
            time.sleep(1)
        except Exception as e:
            log(f"Error listing dept {dept}: {e}")

    # precise domain filter: Alpine Convention perimeter + 100 km
    if HAS_DOMAIN and all_stations:
        import pandas as pd
        df = pd.DataFrame(list(all_stations.values()))
        n_before = len(df)
        df = stations_inside_domain(df, lon_col="lon", lat_col="lat", buffer_km=100.0)
        all_stations = {row["id"]: row for row in df.to_dict("records")}
        log(f"Domain filter (perimeter + 100 km): {n_before} -> {len(all_stations)}")
    else:
        log(f"Using wide bbox fallback (lon {MIN_LON}-{MAX_LON}, lat {MIN_LAT}-{MAX_LAT})")

    log(f"Stations selected inside domain: {len(all_stations)}")
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
        # always (re)write so altitude and other metadata stay correct on re-runs
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
                ok = order_and_download(auth, sid, STEP, date_deb, date_fin, out_csv)
                if ok:
                    year_ok += 1
                elif fail_marker.exists():
                    fail_count += 1
                    if fail_count >= 3:
                        log(f"Skipping rest of {sid}: {fail_count} failures")
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
