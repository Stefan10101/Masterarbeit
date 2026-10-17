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
    "precip": ["RR1", "RR"],
    "temp": ["T", "TN", "TX"],
    "wind": ["FF", "FF2"],
    "rh": ["U"],
    "snow": ["NEIGETOT", "HNEIGEF", "HNEIGEFI1", "HNEIGEFI3"],
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
    logger.info(f"Looking in: {RAW_DIR}")
    logger.info(f"RAW_DIR exists: {RAW_DIR.exists()}")
    # yearly files from the downloader: {sid}_{step}_{year}.csv
    csvs = list(RAW_DIR.glob("*_1h_*.csv")) + list(RAW_DIR.glob("*_1d_*.csv"))
    if not csvs:
        # fallback: any csv in the folder
        csvs = list(RAW_DIR.glob("*.csv"))
        logger.info(f"Primary glob empty, fallback found {len(csvs)} csv files")
    # group by station (first token before _)
    by_sid = {}
    for p in csvs:
        sid = p.stem.split("_")[0]
        by_sid.setdefault(sid, []).append(p)

    logger.info(f"Found {len(csvs)} raw files for {len(by_sid)} stations")
    if not by_sid:
        return

    def process_one(sid_files):
        sid, files = sid_files
        meta_path = RAW_DIR / f"{sid}_meta.json"
        if not meta_path.exists():
            return sid, []
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        clean_name = sanitize(meta.get("name", sid))

        dfs = []
        for csv_path in sorted(files):
            try:
                df = pd.read_csv(csv_path, sep=";", encoding="utf-8", low_memory=False)
            except Exception:
                try:
                    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
                except Exception as e:
                    logger.warning(f"{sid}: read fail {csv_path.name}: {e}")
                    continue
            dfs.append(df)
        if not dfs:
            return clean_name, []
        df = pd.concat(dfs, ignore_index=True).copy()

        # log columns once per station if nothing matches later
        time_col = None
        for c in ["DATE", "date", "DATE_HEURE", "AAAAMMJJHH", "time", "AAAAMMJJ", "AAAAMMJJHHMM"]:
            if c in df.columns:
                time_col = c
                break
        if time_col is None:
            for c in df.columns:
                if "date" in c.lower() or "time" in c.lower() or "aaaamm" in c.lower():
                    time_col = c
                    break
        if time_col is None:
            logger.warning(f"{sid}: no time column. cols={list(df.columns)[:20]}")
            return clean_name, []

        # DATE is YYYYMMDDHH (hourly) or YYYYMMDD (daily)
        raw_dates = df[time_col].astype(str).str.replace(r"\.0$", "", regex=True)
        df["time_dt"] = pd.to_datetime(raw_dates, format="%Y%m%d%H", errors="coerce")
        if df["time_dt"].isna().all():
            df["time_dt"] = pd.to_datetime(raw_dates, format="%Y%m%d", errors="coerce")
        df = df[(df["time_dt"] >= TARGET_START) & (df["time_dt"] <= TARGET_END)].copy()

        results = []
        matched_any = False
        for var, candidates in COLUMN_MAP.items():
            col = next((c for c in candidates if c in df.columns), None)
            if col is None:
                continue
            matched_any = True
            out_dir = OUTPUT_BASES[var] / clean_name
            out_file = out_dir / f"{clean_name}_{var}.json"
            if out_file.exists():
                results.append((var, -1))
                continue
            # European decimal comma
            series = df[col].astype(str).str.replace(",", ".", regex=False)
            vals = pd.to_numeric(series, errors="coerce")
            sub = pd.DataFrame({"time_dt": df["time_dt"], "value": vals}).dropna()
            if sub.empty:
                continue
            records = [
                {"time": t.isoformat(), "value": float(v)}
                for t, v in zip(sub["time_dt"], sub["value"])
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
        if not matched_any:
            logger.warning(f"{sid}: no variable columns matched. available={list(df.columns)[:30]}")
        return clean_name, results

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(process_one, item): item[0] for item in by_sid.items()}
        for fut in as_completed(futures):
            name, res = fut.result()
            if res:
                logger.info(f"{name}: {res}")

    logger.info("Unpack finished")


if __name__ == "__main__":
    main()
