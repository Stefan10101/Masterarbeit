#!/usr/bin/env python3
"""
Météo-France unpacker.

Reads raw CSV files from meteofrance/raw/ (produced by the API downloader)
and extracts the standard variables into the network folders:

  niederschlag/meteofrance/<station>/
  temperatur/meteofrance/<station>/
  wind/meteofrance/<station>/
  luftfeuchte/meteofrance/<station>/
  schneehoehe/meteofrance/<station>/

Output format matches other networks (compact JSON with station_metadata + data).
Idempotent: skips stations that already have all variable files.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===============================================
# PATHS
# ===============================================
SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_ROOT = PROJECT_ROOT

# =============================================
# CONFIG
# =============================================
RAW_DIR = DATA_ROOT / "meteofrance" / "raw"
OUTPUT_BASES = {
    "precip": DATA_ROOT / "niederschlag" / "meteofrance",
    "temp": DATA_ROOT / "temperatur" / "meteofrance",
    "wind": DATA_ROOT / "wind" / "meteofrance",
    "rh": DATA_ROOT / "luftfeuchte" / "meteofrance",
    "snow": DATA_ROOT / "schneehoehe" / "meteofrance",
}

# Typical column names in Météo-France climatological CSV (adapt if needed)
COLUMN_MAP = {
    "precip": ["RR", "PRECIP", "precipitation", "RR1"],
    "temp": ["T", "TEMP", "temperature", "TN", "TX", "T_MOY"],
    "wind": ["FF", "VENT", "wind_speed", "FF_MOY"],
    "rh": ["U", "HUMIDITE", "relative_humidity", "U_MOY"],
    "snow": ["NEIGE", "HAUTEUR_NEIGE", "snow_height", "NEIGE_SOL"],
}

LOG_DIR = DATA_ROOT / "code" / "logs" / "meteofrance" / "unpack"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"meteofrance_unpack_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

TARGET_START = "2020-01-01"
TARGET_END = "2025-12-31"

for p in OUTPUT_BASES.values():
    p.mkdir(parents=True, exist_ok=True)


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in name).replace(" ", "_")


def process_station(csv_path: Path) -> tuple:
    sid = csv_path.stem.split("_")[0]
    meta_path = RAW_DIR / f"{sid}_meta.json"
    if not meta_path.exists():
        return sid, []
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    clean_name = sanitize(meta.get("name", sid))

    try:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8", low_memory=False)
    except Exception:
        df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)

    # try to find time column
    time_col = None
    for c in ["DATE", "date", "DATE_HEURE", "AAAAMMJJHH", "time"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        logger.warning(f"{sid}: no time column")
        return sid, []

    df["time_dt"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[(df["time_dt"] >= TARGET_START) & (df["time_dt"] <= TARGET_END)]

    results = []
    for var, candidates in COLUMN_MAP.items():
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            continue
        out_dir = OUTPUT_BASES[var] / clean_name
        out_file = out_dir / f"{clean_name}_{var}.json"
        if out_file.exists():
            results.append((var, -1))  # skipped
            continue
        sub = df[["time_dt", col]].dropna().copy()
        if sub.empty:
            continue
        records = [
            {"time": t.isoformat(), "value": float(v)}
            for t, v in zip(sub["time_dt"], sub[col])
        ]
        payload = {
            "station_metadata": meta,
            "variable": var,
            "unit": {"precip": "mm", "temp": "degC", "wind": "m/s", "rh": "%", "snow": "cm"}.get(var, ""),
            "data": records,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        results.append((var, len(records)))
    return clean_name, results


def main():
    csvs = list(RAW_DIR.glob("*_1h.csv")) + list(RAW_DIR.glob("*_1d.csv"))
    logger.info(f"Found {len(csvs)} raw files")
    if not csvs:
        return

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(process_station, p): p for p in csvs}
        for fut in as_completed(futures):
            name, res = fut.result()
            if res:
                logger.info(f"{name}: {res}")

    logger.info("Unpack finished")


if __name__ == "__main__":
    main()
