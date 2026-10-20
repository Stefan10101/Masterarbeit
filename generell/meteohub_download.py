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

Quarterly windows to stay under the output limit.
No hourly request limit (account upgraded).
Historical extractions have lower priority — longer waits expected.
FVG (arpafvg) data starts July 2026 — skipped for 2020–2025.
Resumable. Logs to code/logs/meteohub/download/.

Set METEOHUB_USER and METEOHUB_PASS.
"""

import json
import logging
import os
import time
from calendar import monthrange
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
MAX_CONCURRENT = 5          # keep this many requests in flight
POLL_INTERVAL = 30          # seconds between status checks
BATCH_WAIT_MAX = 2 * 60 * 60  # 2 h max per request before giving up
SLEEP_BETWEEN = 1.0
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
    # last valid day of the end month
    last_day = monthrange(year, int(q_to))[1]
    body = {
        "request_name": f"{dataset}_{year}_{q_from}_{q_to}",
        "reftime": {
            "from": f"{year}-{q_from}-01T00:00:00Z",
            "to": f"{year}-{q_to}-{last_day:02d}T23:59:59Z",
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


def get_all_requests(token: str) -> list:
    url = f"{BASE}/api/requests"
    try:
        r = requests.get(url, timeout=TIMEOUT, headers=auth_headers(token))
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            data = data.get("requests") or data.get("items") or []
        return data if isinstance(data, list) else []
    except Exception as e:
        log.warning(f"list requests failed: {e}")
        return []


def download_and_delete(token: str, rid: int, req: dict, meta: tuple) -> bool:
    dataset, year, q_from, q_to = meta
    out_path = RAW_BASE / f"{dataset}_{year}_{q_from}_{q_to}.json"
    status = (req.get("status") or "").upper()
    if status != "SUCCESS":
        log.warning(f"  {dataset} {year}-{q_from}/{q_to} status={status}, deleting")
        delete_request(token, rid)
        return False
    filename = req.get("fileoutput") or req.get("filename")
    if not filename:
        log.warning(f"  no filename for {rid}")
        delete_request(token, rid)
        return False
    ok = download_file(token, filename, out_path)
    delete_request(token, rid)
    return ok


def delete_request(token: str, request_id: int) -> bool:
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


def download_file(token: str, filename: str, out_path: Path) -> bool:
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


def iter_jobs():
    for dataset in DATASETS:
        for year in YEARS:
            # arpafvg only from July 2026
            if dataset == "arpafvg" and year < 2026:
                continue
            for q_from, q_to in QUARTERS:
                out_path = RAW_BASE / f"{dataset}_{year}_{q_from}_{q_to}.json"
                if out_path.exists():
                    continue
                yield dataset, year, q_from, q_to


def main():
    log.info("=== MeteoHub download start ===")
    token = get_token()
    if not token:
        return

    jobs = iter(iter_jobs())
    pending = {}          # request_id → (dataset, year, q_from, q_to, submitted_at)
    exhausted = False

    def submit_next():
        nonlocal exhausted
        if exhausted:
            return
        try:
            dataset, year, q_from, q_to = next(jobs)
        except StopIteration:
            exhausted = True
            return
        rid = submit_extraction(token, dataset, year, q_from, q_to)
        if rid and rid != -1:
            pending[rid] = (dataset, year, q_from, q_to, time.time())
            log.info(f"  in flight: {len(pending)}")
        time.sleep(SLEEP_BETWEEN)

    # fill initial pool
    for _ in range(MAX_CONCURRENT):
        submit_next()
        if exhausted:
            break

    while pending:
        all_reqs = get_all_requests(token)
        by_id = {str(r.get("id") or r.get("request_id")): r for r in all_reqs}

        finished = []
        for rid, meta in list(pending.items()):
            dataset, year, q_from, q_to, submitted_at = meta
            req = by_id.get(str(rid))
            if not req:
                if time.time() - submitted_at > BATCH_WAIT_MAX:
                    log.warning(f"  {dataset} {year}-{q_from}/{q_to} timeout, dropping")
                    finished.append(rid)
                continue
            status = (req.get("status") or "").upper()
            log.info(f"  {dataset} {year}-{q_from}/{q_to} → {status}")
            if status in ("SUCCESS", "FAILED", "ERROR"):
                download_and_delete(token, rid, req, meta[:4])
                finished.append(rid)
            elif time.time() - submitted_at > BATCH_WAIT_MAX:
                log.warning(f"  {dataset} {year}-{q_from}/{q_to} timeout, dropping")
                finished.append(rid)

        for rid in finished:
            pending.pop(rid, None)
            submit_next()

        if pending:
            time.sleep(POLL_INTERVAL)

    log.info("=== MeteoHub download finished ===")


if __name__ == "__main__":
    main()
