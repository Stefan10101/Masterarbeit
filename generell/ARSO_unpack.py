#!/usr/bin/env python3
"""
ARSO unpacker
-------------
Reads raw files from <Daten>/Network_providers/ARSO/ and writes clean
per-variable JSON files:

  <Daten>/temperatur/arso/<clean_name>/<clean_name>_temperature_data.json
  <Daten>/niederschlag/arso/...
  <Daten>/schneehoehe/arso/...
  <Daten>/wind/arso/...
  <Daten>/luftfeuchte/arso/...

Timestamps converted from ARSO minutes-since-1800-01-01 to UTC ISO.
"""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_ROOT = PROJECT_ROOT

RAW_BASE = DATA_ROOT / "Network_providers" / "ARSO"
HH_DIR = RAW_BASE / "halfhourly"
DAILY_DIR = RAW_BASE / "daily"

OUTPUT_BASES = {
    "temperature": DATA_ROOT / "temperatur" / "arso",
    "precipitation": DATA_ROOT / "niederschlag" / "arso",
    "snow_height": DATA_ROOT / "schneehoehe" / "arso",
    "wind": DATA_ROOT / "wind" / "arso",
    "relative_humidity": DATA_ROOT / "luftfeuchte" / "arso",
}
for p in OUTPUT_BASES.values():
    p.mkdir(parents=True, exist_ok=True)

LOG_DIR = DATA_ROOT / "code" / "logs" / "arso" / "unpack"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"arso_unpack_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("arso_unpack")

EPOCH = datetime(1800, 1, 1)

# Half-hourly order requested: 15,18,26,21
HH_PARAM_MAP = {
    "p0": "temperature",
    "p1": "relative_humidity",
    "p2": "precipitation",
    "p3": "wind",
}

# Daily order requested: 88,85
DAILY_PARAM_MAP = {
    "p0": "snow_height",
    "p1": "precipitation_daily",
}


def parse_payload(payload_str: str) -> dict | None:
    if not payload_str:
        return None
    cleaned = re.sub(r"([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1 "\2":', payload_str)
    cleaned = re.sub(r"([{,])\s*(_[A-Za-z0-9]+)\s*:", r'\1 "\2":', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse failed: {e}")
        return None


def minutes_to_iso(minute_key: str) -> str | None:
    try:
        minutes = int(minute_key.lstrip("_"))
        dt = EPOCH + timedelta(minutes=minutes)
        return dt.isoformat()
    except Exception:
        return None


def to_float(val) -> float | None:
    if val is None or val == "" or val == "/" or val == "null":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def collect_halfhourly() -> dict:
    series = defaultdict(lambda: defaultdict(list))
    if not HH_DIR.exists():
        log.warning(f"No halfhourly dir: {HH_DIR}")
        return series
    for path in sorted(HH_DIR.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.error(f"Cannot read {path}: {e}")
            continue
        station = obj.get("station", {})
        sid = station.get("id")
        if not sid:
            continue
        data = parse_payload(obj.get("payload", ""))
        if not data:
            continue
        points = data.get("points", {})
        station_points = points.get(f"_{sid}") or points.get(sid) or {}
        for ts_key, vals in station_points.items():
            iso = minutes_to_iso(ts_key)
            if not iso:
                continue
            for pkey, var in HH_PARAM_MAP.items():
                if pkey in vals:
                    series[sid][var].append((iso, to_float(vals[pkey])))
        series[sid]["_meta"] = station
    return series


def collect_daily() -> dict:
    series = defaultdict(lambda: defaultdict(list))
    if not DAILY_DIR.exists():
        log.warning(f"No daily dir: {DAILY_DIR}")
        return series
    for path in sorted(DAILY_DIR.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.error(f"Cannot read {path}: {e}")
            continue
        station = obj.get("station", {})
        sid = station.get("id")
        if not sid:
            continue
        data = parse_payload(obj.get("payload", ""))
        if not data:
            continue
        points = data.get("points", {})
        station_points = points.get(f"_{sid}") or points.get(sid) or {}
        for ts_key, vals in station_points.items():
            iso = minutes_to_iso(ts_key)
            if not iso:
                continue
            for pkey, var in DAILY_PARAM_MAP.items():
                if pkey in vals:
                    series[sid][var].append((iso, to_float(vals[pkey])))
        if "_meta" not in series[sid]:
            series[sid]["_meta"] = station
    return series


def write_variable(sid: str, meta: dict, var_name: str, records: list) -> None:
    if not records or var_name not in OUTPUT_BASES:
        return
    df = pd.DataFrame(records, columns=["timestamp", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    clean = meta.get("clean_name") or re.sub(r"\s+", "_", str(meta.get("name", sid)).strip())
    out_dir = OUTPUT_BASES[var_name] / clean
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clean}_{var_name}_data.json"

    payload = {
        "name": meta.get("name"),
        "code": sid,
        "hoehe": meta.get("alt"),
        "longitude": meta.get("lon"),
        "latitude": meta.get("lat"),
        "source": "ARSO",
        "resolution": "halfhourly" if var_name != "snow_height" else "daily",
        "data": [
            {"timestamp": ts.isoformat(), "value": None if pd.isna(v) else float(v)}
            for ts, v in zip(df["timestamp"], df["value"])
        ],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    valid = int(df["value"].notna().sum())
    log.info(f"  {clean} / {var_name}: {len(df):,} rows ({valid:,} valid)")


def main():
    log.info("=== ARSO unpack start ===")
    log.info(f"Raw source: {RAW_BASE}")
    hh = collect_halfhourly()
    daily = collect_daily()

    all_sids = set(hh.keys()) | set(daily.keys())
    log.info(f"Stations with data: {len(all_sids)}")

    for sid in sorted(all_sids):
        meta = hh.get(sid, {}).get("_meta") or daily.get(sid, {}).get("_meta") or {}
        log.info(f"Station {sid} ({meta.get('name', '?')})")

        for var in ("temperature", "relative_humidity", "precipitation", "wind"):
            recs = hh.get(sid, {}).get(var, [])
            write_variable(sid, meta, var, recs)

        snow_recs = daily.get(sid, {}).get("snow_height", [])
        write_variable(sid, meta, "snow_height", snow_recs)

    log.info("=== ARSO unpack finished ===")


if __name__ == "__main__":
    main()
