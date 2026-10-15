#!/usr/bin/env python3
"""
MeteoHub systematic download (2020-2025)
----------------------------------------
Official API flow (user guide):
  1. POST /auth/login → token
  2. POST /api/data → 202 + request_id
  3. GET  /api/requests → poll until SUCCESS
  4. GET  /api/data/{filename} → download

Datasets (OBS): arpafvg, dpcn-lombardia, open-trentino,
                dpcn-bolzanoboze, dpcn-piemonte, dpcn-veneto

Quarterly windows to stay under the 1 GB output limit.
Respects ~10 requests/hour quota.
Resumable. Logs to code/logs/meteohub/download/.

Set METEOHUB_USER and METEOHUB_PASS.
"""

import json
import logging
import os
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

RAW_BASE = DATA_ROOT / "Network_providers" / "meteohub"
RAW_BASE.mkdir(parents=True, exist_ok=True)

LOG_DIR = DATA_ROOT / "code" / "logs" / "meteohub" / "download"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"meteohub_download_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("meteohub_download")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = "https://meteohub.agenziaitaliameteo.it"
DATASETS = [
    "arpafvg",
    "dpcn-lombardia",
    "open-trentino",
    "dpcn-bolzanoboze",
    "dpcn-piemonte",
    "dpcn-veneto",
]
YEARS = list(range(2020, 2026))
QUARTERS = [
    ("01", "03"),
    ("04", "06"),
    ("07", "09"),
    ("10", "12"),
]
POLL_INTERVAL = 15          # seconds between status checks
MAX_POLLS = 40              # ~10 min per request
SLEEP_BETWEEN = 2.0
TIMEOUT = 120
HEADERS = {"User-Agent": "MeteoHub-thesis-downloader/1.0"}

USER = os.environ.get("METEOHUB_USER")
PASS = os.environ.get("METEOHUB_PASS")


def get_token() -> str | None:
    if not USER or not PASS:
        log.error("METEOHUB_USER / METEOHUB_PASS not set")
        return None
    url = f"{BASE}/auth/login"
    try:
        r = requests.post(
            url,
            json={"username": USER, "password": PASS},
            timeout=TIMEOUT,
            headers=HEADERS,
        )
        r.raise_for_status()
        try:
            data = r.json()
        except Exception:
            data = r.text.strip()
        # response can be a bare token string or a dict
        if isinstance(data, str):
            token = data.strip('"')
        elif isinstance(data, dict):
            token = data.get("access_token") or data.get("token") or data.get("accessToken")
        else:
            token = None
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


def submit_extraction(token: str, dataset: str, year: int, q_from: str, q_to: str) -> int | None:
    """POST /api/data. Returns request_id or None."""
    url = f"{BASE}/api/data"
    body = {
        "request_name": f"{dataset}_{year}_{q_from}_{q_to}",
        "reftime": {
            "from": f"{year}-{q_from}-01T00:00:00Z",
            "to": f"{year}-{q_to}-31T23:59:59Z",
        },
        "dataset_names": [dataset],
        "output_format": "json",
        "only_reliable": True,
    }
    try:
        r = requests.post(url, json=body, timeout=TIMEOUT, headers=auth_headers(token))
        if r.status_code == 202:
            data = r.json()
            rid = data.get("request_id")
            log.info(f"  submitted {dataset} {year}-{q_from}/{q_to} → request_id={rid}")
            return rid
        if r.status_code == 401 and "max number of requests" in r.text.lower():
            log.warning("Rate limit reached — stopping for this hour")
            return -1  # signal to stop
        log.warning(f"  submit {r.status_code}: {r.text[:300]}")
        return None
    except Exception as e:
        log.warning(f"  submit failed: {e}")
        return None


def poll_request(token: str, request_id: int) -> dict | None:
    """Poll /api/requests until SUCCESS or FAILED. Returns the request dict."""
    url = f"{BASE}/api/requests"
    for i in range(MAX_POLLS):
        try:
            r = requests.get(url, timeout=TIMEOUT, headers=auth_headers(token))
            r.raise_for_status()
            requests_list = r.json()
            # response can be list or dict with items
            if isinstance(requests_list, dict):
                requests_list = requests_list.get("requests") or requests_list.get("items") or []
            for req in requests_list:
                if str(req.get("id") or req.get("request_id")) == str(request_id):
                    status = (req.get("status") or "").upper()
                    log.info(f"    poll {i+1}: status={status}")
                    if status in ("SUCCESS", "FAILED", "ERROR"):
                        return req
                    break
        except Exception as e:
            log.warning(f"    poll error: {e}")
        time.sleep(POLL_INTERVAL)
    log.warning(f"    timeout waiting for {request_id}")
    return None


def delete_request(token: str, request_id: int) -> bool:
    """DELETE /api/requests/{id} to free storage quota."""
    url = f"{BASE}/api/requests/{request_id}"
    try:
        r = requests.delete(url, timeout=TIMEOUT, headers=auth_headers(token))
        if r.status_code in (200, 204):
            log.info(f"    deleted remote request {request_id}")
            return True
        log.warning(f"    delete {r.status_code}: {r.text[:200]}")
        return False
    except Exception as e:
        log.warning(f"    delete failed: {e}")
        return False
    url = f"{BASE}/api/data/{filename}"
    try:
        r = requests.get(url, timeout=TIMEOUT * 2, headers=auth_headers(token), stream=True)
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        log.info(f"    downloaded → {out_path.name} ({out_path.stat().st_size:,} bytes)")
        return True
    except Exception as e:
        log.warning(f"    download failed: {e}")
        return False


def process_quarter(token: str, dataset: str, year: int, q_from: str, q_to: str) -> bool:
    out_path = RAW_BASE / f"{dataset}_{year}_{q_from}_{q_to}.json"
    if out_path.exists():
        log.info(f"  {dataset} {year}-{q_from}/{q_to} already present")
        return True

    rid = submit_extraction(token, dataset, year, q_from, q_to)
    if rid == -1:
        return False  # rate limit
    if not rid:
        return False

    req = poll_request(token, rid)
    if not req or (req.get("status") or "").upper() != "SUCCESS":
        log.warning(f"  {dataset} {year}-{q_from}/{q_to} failed or incomplete")
        return False

    filename = req.get("fileoutput") or req.get("filename")
    if not filename:
        log.warning(f"  no filename in response: {req}")
        return False

    ok = download_file(token, filename, out_path)
    if ok:
        delete_request(token, rid)
    return ok


def main():
    log.info("=== MeteoHub download start ===")
    token = get_token()
    if not token:
        return

    for dataset in DATASETS:
        log.info(f"=== Dataset {dataset} ===")
        for year in YEARS:
            for q_from, q_to in QUARTERS:
                ok = process_quarter(token, dataset, year, q_from, q_to)
                if not ok and "rate limit" in str(ok).lower():
                    log.info("Stopping due to rate limit")
                    return
                time.sleep(SLEEP_BETWEEN)

    log.info("=== MeteoHub download finished ===")


if __name__ == "__main__":
    main()
