#!/usr/bin/env python3
"""
MeteoHub systematic download (2020-2025) — storage-aware, variable-first draft
------------------------------------------------------------------------------
Official API flow:
  1. POST /auth/login → token
  2. POST /api/data → 202 + request_id
  3. GET  /api/requests → poll until SUCCESS
  4. GET  /api/data/{filename} → download
  5. DELETE /api/requests/{id} → free storage

Upgrades (5 GB online storage, no request rate limit):
- Storage-aware concurrency (check /api/usage before submitting).
- Variable-first extraction with Alpine bounding-box + quantity/product filters.
- Probe mode to discover filter syntax and build a station catalog.
- Immediate download + delete so the 5 GB acts as a circulating buffer.
- Resumable. Logs under code/logs/meteohub/download/.

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

CATALOG_PATH = RAW_BASE / "station_catalog.json"
PROBE_DIR = RAW_BASE / "probe"
PROBE_DIR.mkdir(parents=True, exist_ok=True)

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
    # add more northern OBS ids after probing if useful (mnw, agrmet, ...)
]
YEARS = list(range(2020, 2026))
MONTHS = [f"{m:02d}" for m in range(1, 13)]

# Thesis variables → BUFR codes (snow excluded)
VARIABLES = {
    "temperature": ["B12101"],
    "precipitation": ["B13011"],
    "wind": ["B11002"],
    "relative_humidity": ["B13003"],
}

# 2-month windows
PERIODS = [
    ("01", "02"),
    ("03", "04"),
    ("05", "06"),
    ("07", "08"),
    ("09", "10"),
    ("11", "12"),
]
# auxiliary codes we may also want: B11001 (wind dir), B13215 (unknown, appeared in sample)

# Alpine Convention box (same as unpacker)
ALPINE_BBOX = {
    "lonmin": 6.0,
    "latmin": 43.5,
    "lonmax": 16.5,
    "latmax": 48.8,
}

# Concurrency / storage
MAX_CONCURRENT = 8           # monthly is smaller; keep headroom under 5 GB
STORAGE_SOFT_LIMIT_GB = 3.5  # leave room for in-flight results
POLL_INTERVAL = 20
BATCH_WAIT_MAX = 3 * 60 * 60
SLEEP_BETWEEN = 0.5
TIMEOUT = 120
HEADERS = {"User-Agent": "MeteoHub-thesis-downloader/2.0"}

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


def get_usage_gb(token: str) -> float:
    """Return current storage usage in GB, or 0 on failure."""
    url = f"{BASE}/api/usage"
    try:
        r = requests.get(url, timeout=TIMEOUT, headers=auth_headers(token))
        r.raise_for_status()
        data = r.json()
        # guide says values in bytes; handle common shapes
        if isinstance(data, dict):
            used = data.get("used") or data.get("storage_used") or data.get("bytes") or 0
        else:
            used = 0
        return float(used) / (1024 ** 3)
    except Exception as e:
        log.warning(f"usage check failed: {e}")
        return 0.0


def submit_extraction(token: str, body: dict) -> int | None:
    """POST /api/data. Returns request_id or None."""
    url = f"{BASE}/api/data"
    try:
        r = requests.post(url, json=body, timeout=TIMEOUT, headers=auth_headers(token))
        if r.status_code == 202:
            data = r.json()
            rid = data.get("request_id")
            log.info(f"  submitted {body.get('request_name')} → request_id={rid}")
            return rid
        log.warning(f"  submit {r.status_code}: {r.text[:400]}")
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
        r = requests.get(url, timeout=TIMEOUT * 3, headers=auth_headers(token), stream=True)
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        log.info(f"    downloaded → {out_path.name} ({out_path.stat().st_size:,} bytes)")
        return True
    except Exception as e:
        log.warning(f"    download failed: {e}")
        return False


def download_and_delete(token: str, rid: int, req: dict, out_path: Path) -> bool:
    status = (req.get("status") or "").upper()
    if status != "SUCCESS":
        log.warning(f"  status={status}, deleting {rid}")
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


# ---------------------------------------------------------------------------
# Probe & catalog
# ---------------------------------------------------------------------------
def probe_filters_and_stations(token: str) -> None:
    """
    One-time probe:
    - tiny extraction (1 day) → NDJSON
    - extract station names + BUFR codes present
    - write a small catalog summary
    """
    log.info("=== Probe start ===")
    body = {
        "request_name": "probe_tiny_1d",
        "reftime": {
            "from": "2024-07-01T00:00:00Z",
            "to": "2024-07-01T23:59:59Z",
        },
        "dataset_names": ["dpcn-lombardia"],
        "output_format": "json",
        "only_reliable": True,
    }
    rid = submit_extraction(token, body)
    if not rid:
        log.warning("Probe submit failed")
        return

    for _ in range(60):
        time.sleep(10)
        reqs = get_all_requests(token)
        by_id = {str(r.get("id") or r.get("request_id")): r for r in reqs}
        req = by_id.get(str(rid))
        if not req:
            continue
        status = (req.get("status") or "").upper()
        log.info(f"  probe status: {status}")
        if status in ("SUCCESS", "FAILED", "ERROR"):
            out = PROBE_DIR / "probe_tiny.json"
            download_and_delete(token, rid, req, out)
            if out.exists():
                _summarize_ndjson(out)
            break
    log.info("=== Probe finished ===")


def _summarize_ndjson(path: Path) -> None:
    """Walk NDJSON probe file and write a short station/code summary."""
    from collections import Counter, defaultdict
    codes = Counter()
    stations = {}
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            n += 1
            name = None
            lat = lon = alt = None
            for item in obj.get("data") or []:
                vars_ = item.get("vars") or {}
                for code, payload in vars_.items():
                    codes[code] += 1
                    if code == "B01019":
                        name = payload.get("v")
                    elif code == "B05001":
                        lat = payload.get("v")
                    elif code == "B06001":
                        lon = payload.get("v")
                    elif code == "B07030":
                        alt = payload.get("v")
            if name:
                stations[name] = {"lat": lat, "lon": lon, "alt": alt, "network": obj.get("network")}
    summary = {
        "lines": n,
        "stations": len(stations),
        "top_codes": codes.most_common(40),
        "sample_stations": dict(list(stations.items())[:5]),
    }
    out = path.with_suffix(".summary.json")
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"  probe summary → {out.name} ({len(stations)} stations, {len(codes)} codes)")


def build_filter(var_key: str) -> dict:
    """
    Filters for a single variable.
    BUFR datasets accept: bounding_box, level, network, product, timerange.
    """
    codes = VARIABLES.get(var_key, [var_key])
    return {
        "bounding_box": ALPINE_BBOX,
        "product": [{"code": c} for c in codes],
    }


def make_body(dataset: str, year: int, m_from: str, m_to: str, var_key: str | None = None) -> dict:
    last_day = monthrange(year, int(m_to))[1]
    body = {
        "request_name": f"{dataset}_{year}_{m_from}_{m_to}" + (f"_{var_key}" if var_key else ""),
        "reftime": {
            "from": f"{year}-{m_from}-01T00:00:00Z",
            "to": f"{year}-{m_to}-{last_day:02d}T23:59:59Z",
        },
        "dataset_names": [dataset],
        "output_format": "json",
        "only_reliable": True,
    }
    if var_key:
        body["filters"] = build_filter(var_key)
    return body


def iter_jobs(variable_first: bool = True):
    """Yield (dataset, year, m_from, m_to, var_key or None, out_path)."""
    if variable_first:
        for var_key in VARIABLES:
            for dataset in DATASETS:
                for year in YEARS:
                    if dataset == "arpafvg" and year < 2026:
                        continue
                    for m_from, m_to in PERIODS:
                        out_path = RAW_BASE / f"{dataset}_{var_key}_{year}_{m_from}_{m_to}.json"
                        if out_path.exists():
                            continue
                        yield dataset, year, m_from, m_to, var_key, out_path
    else:
        for dataset in DATASETS:
            for year in YEARS:
                if dataset == "arpafvg" and year < 2026:
                    continue
                for m_from, m_to in PERIODS:
                    out_path = RAW_BASE / f"{dataset}_{year}_{m_from}_{m_to}.json"
                    if out_path.exists():
                        continue
                    yield dataset, year, m_from, m_to, None, out_path


def sweep_existing(token: str) -> None:
    """Download + delete any already-SUCCESS requests so quota is free at start."""
    log.info("sweeping existing SUCCESS requests")
    reqs = get_all_requests(token)
    for req in reqs:
        if (req.get("status") or "").upper() != "SUCCESS":
            continue
        rid = req.get("id")
        name = req.get("name") or str(rid)
        out_path = RAW_BASE / f"{name}.json"
        log.info(f"  recovering {name}")
        download_and_delete(token, rid, req, out_path)


def main(probe: bool = False, variable_first: bool = True):
    log.info("=== MeteoHub download start ===")
    token = get_token()
    if not token:
        return

    if probe:
        probe_filters_and_stations(token)
        return

    # free quota from any leftover SUCCESS results
    sweep_existing(token)

    jobs = iter(iter_jobs(variable_first=variable_first))
    pending = {}  # rid → (out_path, submitted_at, meta)
    exhausted = False

    def submit_next():
        nonlocal exhausted
        if exhausted:
            return
        used = get_usage_gb(token)
        if used >= STORAGE_SOFT_LIMIT_GB:
            log.info(f"  storage {used:.2f} GB ≥ soft limit, waiting")
            return
        try:
            dataset, year, m_from, m_to, var_key, out_path = next(jobs)
        except StopIteration:
            exhausted = True
            return
        body = make_body(dataset, year, m_from, m_to, var_key)
        rid = submit_extraction(token, body)
        if rid:
            pending[rid] = (out_path, time.time(), (dataset, year, m_from, m_to, var_key))
            log.info(f"  in flight: {len(pending)} (storage ~{used:.2f} GB)")
        time.sleep(SLEEP_BETWEEN)

    for _ in range(MAX_CONCURRENT):
        submit_next()
        if exhausted:
            break

    while pending:
        all_reqs = get_all_requests(token)
        by_id = {str(r.get("id") or r.get("request_id")): r for r in all_reqs}

        finished = []
        for rid, (out_path, submitted_at, meta) in list(pending.items()):
            req = by_id.get(str(rid))
            if not req:
                if time.time() - submitted_at > BATCH_WAIT_MAX:
                    log.warning(f"  {meta} timeout, dropping")
                    finished.append(rid)
                continue
            status = (req.get("status") or "").upper()
            log.info(f"  {meta} → {status}")
            if status in ("SUCCESS", "FAILED", "FAILURE", "ERROR"):
                if status != "SUCCESS":
                    err = req.get("error_message") or req.get("error") or req.get("message") or ""
                    log.warning(f"    failure detail: {err or req}")
                download_and_delete(token, rid, req, out_path)
                finished.append(rid)
            elif time.time() - submitted_at > BATCH_WAIT_MAX:
                log.warning(f"  {meta} timeout, dropping")
                finished.append(rid)

        for rid in finished:
            pending.pop(rid, None)
            submit_next()

        if pending:
            time.sleep(POLL_INTERVAL)

    log.info("=== MeteoHub download finished ===")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true", help="Run one-time probe + catalog only")
    p.add_argument("--full-dump", action="store_true", help="Fall back to unfiltered quarterly dumps")
    args = p.parse_args()
    main(probe=args.probe, variable_first=not args.full_dump)
