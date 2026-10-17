#!/usr/bin/env python3
"""
Refresh altitude in existing Météo-France meta JSON files.

Uses the same DPClim list-stations endpoint (field "alt") that the downloader uses.
Updates files in meteofrance/raw/*_meta.json in place.

Requires METEOFRANCE_CLIENT_AUTH or METEOFRANCE_TOKEN.
"""

import json
import os
import requests
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_ROOT = PROJECT_ROOT

RAW_DIR = DATA_ROOT / "meteofrance" / "raw"
BASE_URL = "https://public-api.meteofrance.fr/public/DPClim/v1"
TOKEN_URL = "https://portail-api.meteofrance.fr/token"

CLIENT_AUTH = os.environ.get("METEOFRANCE_CLIENT_AUTH", "")
API_TOKEN = os.environ.get("METEOFRANCE_TOKEN", "")
ALPINE_DEPTS = ["01", "04", "05", "06", "07", "25", "26", "38", "39", "73", "74", "90"]


def get_token() -> str:
    if API_TOKEN:
        return API_TOKEN
    if not CLIENT_AUTH:
        raise RuntimeError("Set METEOFRANCE_CLIENT_AUTH or METEOFRANCE_TOKEN")
    r = requests.post(
        TOKEN_URL,
        headers={"Authorization": f"Basic {CLIENT_AUTH}"},
        data={"grant_type": "client_credentials"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def list_stations(token: str, dept: str) -> list:
    url = f"{BASE_URL}/liste-stations/horaire"
    r = requests.get(
        url,
        params={"id-departement": dept},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        stations = data.get("stations") or data.get("listeStations") or data.get("station")
        if isinstance(stations, list):
            return stations
    return []


def main():
    token = get_token()
    alt_by_id = {}
    for dept in ALPINE_DEPTS:
        for s in list_stations(token, dept):
            sid = str(s.get("id", ""))
            alt = s.get("alt")
            if sid and alt is not None:
                try:
                    alt_by_id[sid] = float(alt)
                except (TypeError, ValueError):
                    pass
        print(f"dept {dept}: alt map size {len(alt_by_id)}")

    updated = 0
    missing = 0
    for meta_path in RAW_DIR.glob("*_meta.json"):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        sid = str(meta.get("id", ""))
        if sid in alt_by_id:
            meta["altitude"] = alt_by_id[sid]
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            updated += 1
        else:
            missing += 1
    print(f"Updated {updated} meta files, {missing} still without altitude")


if __name__ == "__main__":
    main()
