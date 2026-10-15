#!/usr/bin/env python3
"""
MeteoHub unpacker
-----------------
Reads raw files from Network_providers/meteohub/
and writes clean per-variable JSON into the standard folders:

  <Daten>/temperatur/meteohub/<clean>/<clean>_temperature_data.json
  <Daten>/niederschlag/meteohub/...
  <Daten>/wind/meteohub/...
  <Daten>/luftfeuchte/meteohub/...
  <Daten>/schneehoehe/meteohub/...

Only stations inside the Alpine Convention box (+buffer) are written.
Same output style as ARSO_unpack.py / alpine_unpack.py.
"""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
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

RAW_BASE = DATA_ROOT / "Network_providers" / "meteohub"

OUTPUT_BASES = {
    "temperature": DATA_ROOT / "temperatur" / "meteohub",
    "precipitation": DATA_ROOT / "niederschlag" / "meteohub",
    "wind": DATA_ROOT / "wind" / "meteohub",
    "relative_humidity": DATA_ROOT / "luftfeuchte" / "meteohub",
    "snow_height": DATA_ROOT / "schneehoehe" / "meteohub",
}
for p in OUTPUT_BASES.values():
    p.mkdir(parents=True, exist_ok=True)

LOG_DIR = DATA_ROOT / "code" / "logs" / "meteohub" / "unpack"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"meteohub_unpack_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("meteohub_unpack")

ALPINE = {
    "min_lon": 6.0,
    "max_lon": 16.5,
    "min_lat": 43.5,
    "max_lat": 48.8,
}

# Common MeteoHub / BUFR-derived variable name mappings (extend as needed)
VAR_MAP = {
    "temperature": ["temp", "temperature", "air_temperature", "T", "temp_mean"],
    "precipitation": ["precip", "precipitation", "rainfall", "P", "precip_sum"],
    "wind": ["wind", "wind_speed", "ff", "wind_velocity"],
    "relative_humidity": ["humidity", "relative_humidity", "RH", "rh", "luftfeuchte"],
    "snow_height": ["snow", "snow_height", "snow_depth", "HS", "schnee"],
}


def in_alpine(lon, lat) -> bool:
    try:
        return (
            ALPINE["min_lon"] <= float(lon) <= ALPINE["max_lon"]
            and ALPINE["min_lat"] <= float(lat) <= ALPINE["max_lat"]
        )
    except (TypeError, ValueError):
        return False


def sanitize(name: str) -> str:
    name = re.sub(r"[^\w\s\-äöüÄÖÜß]", "", str(name).strip(), flags=re.UNICODE)
    for a, b in {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}.items():
        name = name.replace(a, b)
    return re.sub(r"\s+", "_", name) or "unknown"


def to_float(val):
    if val is None or val == "" or val == "/" or val == "null":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def normalize_var(raw_name: str) -> str | None:
    if not raw_name:
        return None
    low = str(raw_name).lower()
    for target, aliases in VAR_MAP.items():
        if any(a in low for a in aliases):
            return target
    return None


def collect() -> dict:
    """Collect {sid: {var: [(iso, val), ...], _meta: {...}}} from raw files."""
    series = defaultdict(lambda: defaultdict(list))
    if not RAW_BASE.exists():
        log.warning(f"No raw dir: {RAW_BASE}")
        return series

    for path in sorted(RAW_BASE.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.error(f"Cannot read {path}: {e}")
            continue

        # Flexible parsing: support both list-of-records and nested structures
        records = obj if isinstance(obj, list) else obj.get("data") or obj.get("observations") or []
        if isinstance(records, dict):
            records = records.get("features") or records.get("stations") or []

        for rec in records:
            if not isinstance(rec, dict):
                continue
            # Station identity
            props = rec.get("properties") or rec
            sid = str(props.get("id") or props.get("station_id") or props.get("scode") or "")
            if not sid:
                continue
            lon = props.get("lon") or props.get("longitude") or props.get("LONG")
            lat = props.get("lat") or props.get("latitude") or props.get("LAT")
            if not in_alpine(lon, lat):
                continue

            name = props.get("name") or props.get("NAME") or sid
            series[sid]["_meta"] = {
                "id": sid,
                "name": name,
                "clean": sanitize(name),
                "lon": lon,
                "lat": lat,
                "alt": props.get("alt") or props.get("elevation") or props.get("ALT"),
            }

            # Values
            ts = rec.get("timestamp") or rec.get("time") or rec.get("date") or props.get("timestamp")
            for k, v in rec.items():
                var = normalize_var(k)
                if var and ts:
                    series[sid][var].append((ts, to_float(v)))
            # Nested value dicts
            for k, v in (rec.get("values") or {}).items():
                var = normalize_var(k)
                if var and ts:
                    series[sid][var].append((ts, to_float(v)))

    return series


def write_variable(sid: str, meta: dict, var_name: str, records: list) -> None:
    if not records or var_name not in OUTPUT_BASES:
        return
    df = pd.DataFrame(records, columns=["timestamp", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    clean = meta.get("clean") or sanitize(meta.get("name", sid))
    out_dir = OUTPUT_BASES[var_name] / clean
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clean}_{var_name}_data.json"

    payload = {
        "name": meta.get("name"),
        "code": sid,
        "hoehe": meta.get("alt"),
        "longitude": meta.get("lon"),
        "latitude": meta.get("lat"),
        "source": "meteohub",
        "resolution": "halfhourly_or_better",
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
    log.info("=== MeteoHub unpack start ===")
    series = collect()
    log.info(f"Stations in Alpine box: {len(series)}")
    for sid in sorted(series.keys()):
        meta = series[sid].get("_meta", {})
        for var in OUTPUT_BASES:
            write_variable(sid, meta, var, series[sid].get(var, []))
    log.info("=== MeteoHub unpack finished ===")


if __name__ == "__main__":
    main()
