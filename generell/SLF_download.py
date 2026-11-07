#!/usr/bin/env python3
"""
SLF / IMIS downloader

Fetches station metadata from the public measurement API and downloads the
historical per-station CSVs from measurement-data.slf.ch.

No authentication required.
CC BY 4.0 — cite https://doi.org/10.16904/envidat.406 for scientific use.

Outputs (under DATA_ROOT / "slf"):
  stations.json                  # clean metadata (code, label, lon, lat, elev, type, ...)
  csv/{CODE}.csv                 # full historical series
  csv/{CODE}_pluvio.csv          # precipitation (only where available)
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths (same pattern as other downloaders)
# ---------------------------------------------------------------------------
SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_ROOT = PROJECT_ROOT

OUT_DIR = DATA_ROOT / "slf"
CSV_DIR = OUT_DIR / "csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = DATA_ROOT / "code" / "logs" / "slf" / "download"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"slf_download_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("slf_dl")

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
API_BASE = "https://measurement-api.slf.ch/public/api"
DATA_BASE = "https://measurement-data.slf.ch/imis/data/by_station"

STATIONS_URL = f"{API_BASE}/imis/stations"
MAX_WORKERS = 8
TIMEOUT = 60

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fetch_stations() -> list[dict]:
    r = requests.get(STATIONS_URL, timeout=TIMEOUT)
    r.raise_for_status()
    stations = r.json()
    log.info(f"Fetched {len(stations)} IMIS stations from API")
    return stations


def download_one(code: str, suffix: str = "") -> tuple[str, str, int]:
    """Download one CSV. suffix='' for main series, '_pluvio' for precip.
    Returns (code, status, bytes).
    """
    fname = f"{code}{suffix}.csv"
    url = f"{DATA_BASE}/{fname}"
    dest = CSV_DIR / fname

    if dest.exists() and dest.stat().st_size > 100:
        return code, "skipped", dest.stat().st_size

    try:
        r = requests.get(url, timeout=TIMEOUT, stream=True)
        if r.status_code == 404:
            return code, "missing", 0
        r.raise_for_status()
        size = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
                    size += len(chunk)
        return code, "ok", size
    except Exception as e:
        if dest.exists():
            dest.unlink(missing_ok=True)
        return code, f"error:{e}", 0


def main() -> None:
    log.info("=== SLF / IMIS download started ===")

    stations = fetch_stations()
    meta_path = OUT_DIR / "stations.json"
    meta_path.write_text(json.dumps(stations, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Wrote station metadata → {meta_path}")

    codes = [s["code"] for s in stations]
    log.info(f"Downloading main series for {len(codes)} stations")

    # Main series
    ok = skipped = missing = err = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(download_one, c): c for c in codes}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="main csv"):
            code, status, size = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            elif status == "missing":
                missing += 1
            else:
                err += 1
                log.warning(f"{code}: {status}")

    log.info(f"Main series: ok={ok} skipped={skipped} missing={missing} error={err}")

    # Precipitation (pluvio) — optional, many stations lack it
    log.info("Downloading precipitation (pluvio) series where available")
    ok = skipped = missing = err = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(download_one, c, "_pluvio"): c for c in codes}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="pluvio csv"):
            code, status, size = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            elif status == "missing":
                missing += 1
            else:
                err += 1

    log.info(f"Pluvio series: ok={ok} skipped={skipped} missing={missing} error={err}")
    log.info("=== DONE ===")
    log.info(f"Metadata: {meta_path}")
    log.info(f"CSVs:     {CSV_DIR}")
    log.info(f"Log:      {LOG_FILE}")


if __name__ == "__main__":
    main()
