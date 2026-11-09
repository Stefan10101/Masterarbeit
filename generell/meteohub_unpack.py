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
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
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

CODE_DIR = SCRIPT
while CODE_DIR.name != "CODE" and CODE_DIR.parent != CODE_DIR:
    CODE_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

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

N_WORKERS = max(1, min(8, __import__("os").cpu_count() or 4))

# Domain polygon, loaded once
_DOMAIN = None

def _load_domain():
    global _DOMAIN
    if _DOMAIN is not None:
        return _DOMAIN
    try:
        import geopandas as gpd
        from shared.splits.evaluation_sets import load_alpine_perimeter
        peri = load_alpine_perimeter()
        if peri.crs is None:
            peri = peri.set_crs("EPSG:4326")
        peri = peri.to_crs("EPSG:31287")
        peri["geometry"] = peri.geometry.buffer(0)
        _DOMAIN = peri.buffer(100_000).union_all()
        log.info("Domain: Alpine perimeter + 100 km")
    except Exception as e:
        log.warning(f"Domain polygon unavailable ({e}); using bbox")
        _DOMAIN = None
    return _DOMAIN

# BUFR codes used by MeteoHub (Arkimet JSON)
BUFR_VAR = {
    "B13013": ("snow_height", "m", 100.0),      # total snow depth, m -> cm
    "B12101": ("temperature", "K", None),        # air temperature, K -> C
    "B13011": ("precipitation", "mm", 1.0),      # total precip, kg/m2 = mm
    "B11002": ("wind", "m/s", 1.0),              # wind speed
    "B13003": ("relative_humidity", "%", 1.0),   # relative humidity
}
BUFR_NAME = "B01019"
BUFR_LAT = "B05001"
BUFR_LON = "B06001"
BUFR_ELEV = "B07030"


def in_alpine(lon, lat) -> bool:
    try:
        lon_f, lat_f = float(lon), float(lat)
    except (TypeError, ValueError):
        return False
    dom = _load_domain()
    if dom is None:
        return 6.0 <= lon_f <= 16.5 and 43.5 <= lat_f <= 48.8
    from shapely.geometry import Point
    from pyproj import Transformer
    if not hasattr(in_alpine, "_tr"):
        in_alpine._tr = Transformer.from_crs("EPSG:4326", "EPSG:31287", always_xy=True)
    x, y = in_alpine._tr.transform(lon_f, lat_f)
    return dom.intersects(Point(x, y))


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


def _region_of(path: Path) -> str:
    """Region prefix from the filename (dpcn-bolzanoboze_... -> dpcn-bolzanoboze)."""
    return path.stem.split("_")[0]


def _load_records(path: Path) -> list:
    """Load a raw file. Handles a single JSON document or NDJSON (one object per line)."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # NDJSON: one JSON object per line
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return obj.get("data") or obj.get("observations") or obj.get("features") or [obj]
    return []


def _bufr_val(vars_dict: dict, code: str):
    """Extract the 'v' value from a BUFR vars entry."""
    entry = vars_dict.get(code)
    if isinstance(entry, dict):
        return entry.get("v")
    return entry


def collect_region(paths: list) -> dict:
    """Collect one region's files into {sid: {var: [(iso, val), ...], _meta: {...}}}.

    Each line is a BUFR-in-JSON record:
      data[0].vars = station metadata (B01019 name, B05001 lat, B06001 lon, B07030 elev)
      data[1].vars = measurement (B13013 snow, B12101 temp, ...)
    Frequencies vary by region (10/15/30 min); timestamps are kept as-is.
    """
    series = defaultdict(lambda: defaultdict(list))
    for path in paths:
        log.info(f"reading {path.name}")
        try:
            records = _load_records(path)
        except Exception as e:
            log.error(f"Cannot read {path}: {e}")
            continue

        for rec in records:
            if not isinstance(rec, dict):
                continue
            data = rec.get("data")
            if not isinstance(data, list) or len(data) < 2:
                continue
            meta_vars = data[0].get("vars") or {}
            meas_vars = data[1].get("vars") or {}

            name = _bufr_val(meta_vars, BUFR_NAME)
            if not name:
                continue
            lat = _bufr_val(meta_vars, BUFR_LAT)
            lon = _bufr_val(meta_vars, BUFR_LON)
            elev = _bufr_val(meta_vars, BUFR_ELEV)
            if lat is None and rec.get("lat") is not None:
                lat = float(rec["lat"]) / 1e5
            if lon is None and rec.get("lon") is not None:
                lon = float(rec["lon"]) / 1e5
            if not in_alpine(lon, lat):
                continue

            ts = rec.get("date")
            if not ts:
                continue

            sid = sanitize(str(name))
            series[sid]["_meta"] = {
                "id": sid,
                "name": str(name),
                "clean": sid,
                "lon": float(lon) if lon is not None else None,
                "lat": float(lat) if lat is not None else None,
                "alt": float(elev) if elev is not None else None,
            }

            for code, (var, unit, scale) in BUFR_VAR.items():
                raw = _bufr_val(meas_vars, code)
                if raw is None:
                    continue
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    continue
                if unit == "K":
                    val = val - 273.15
                elif scale and scale != 1.0:
                    val = val * scale
                series[sid][var].append((ts, val))
    return series


def write_region(region: str, series: dict) -> int:
    """Write all stations of one region. Returns number of files written."""
    n = 0
    for sid in sorted(series.keys()):
        meta = series[sid].get("_meta", {})
        for var in OUTPUT_BASES:
            recs = series[sid].get(var, [])
            if recs:
                write_variable(sid, meta, var, recs)
                n += 1
    log.info(f"region {region}: wrote {n} variable files for {len(series)} stations")
    return n


def _merge_series(parts: list) -> dict:
    """Merge partial series dicts from workers into one."""
    merged = defaultdict(lambda: defaultdict(list))
    for part in parts:
        for sid, content in part.items():
            for key, val in content.items():
                if key == "_meta":
                    merged[sid]["_meta"] = val
                else:
                    merged[sid][key].extend(val)
    return merged


def process_file_chunk(paths: list) -> dict:
    """Worker: read a chunk of files, return partial series dict."""
    return collect_region(paths)


def main():
    log.info("=== MeteoHub unpack start ===")
    if not RAW_BASE.exists():
        log.warning(f"No raw dir: {RAW_BASE}")
        return

    _load_domain()

    by_region = defaultdict(list)
    for path in sorted(RAW_BASE.glob("*.json")):
        by_region[_region_of(path)].append(path)
    regions = sorted(by_region)
    log.info(f"Regions: {regions} | workers: {N_WORKERS}")

    total_files = 0
    for region in regions:
        paths = by_region[region]
        log.info(f"region {region}: {len(paths)} files")
        # split files across workers
        chunks = [paths[i::N_WORKERS] for i in range(N_WORKERS)]
        chunks = [c for c in chunks if c]
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            parts = list(pool.map(process_file_chunk, chunks))
        series = _merge_series(parts)
        n = write_region(region, series)
        total_files += n
        log.info(f"done {region}: {n} files, {len(series)} stations")
        del series, parts
    log.info(f"=== MeteoHub unpack finished ({total_files} files) ===")


def resample_30min(df: pd.DataFrame, var_name: str) -> pd.DataFrame:
    """Resample native observations to a regular 30-min grid.

    precip: sum into each 30-min bin (total preserved).
    snow: if the station is sparse (<=2 obs/day median), collapse to the
          daily mean stamped at 06:00; otherwise mean per 30-min bin.
    temp/wind/RH: mean per 30-min bin, then linear-fill a single missing
                  :30 slot between two known values (limit=1).
    """
    if df.empty:
        return df
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if var_name == "precipitation":
        out = df["value"].resample("30min").sum(min_count=1)
    elif var_name == "snow_height":
        per_day = df["value"].resample("D").count()
        med = per_day[per_day > 0].median() if (per_day > 0).any() else 0
        if med <= 2:
            daily = df["value"].resample("D").mean()
            daily.index = daily.index + pd.Timedelta(hours=6)
            out = daily
        else:
            out = df["value"].resample("30min").mean()
    else:
        out = df["value"].resample("30min").mean()
        out = out.interpolate(method="linear", limit=1)

    out = out.dropna()
    return out.reset_index().rename(columns={"index": "timestamp", "value": "value"})


def write_variable(sid: str, meta: dict, var_name: str, records: list) -> None:
    if not records or var_name not in OUTPUT_BASES:
        return
    df = pd.DataFrame(records, columns=["timestamp", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df = resample_30min(df, var_name)
    if df.empty:
        return

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
        "resolution": "30min",
        "data": [
            {"timestamp": ts.isoformat(), "value": None if pd.isna(v) else float(v)}
            for ts, v in zip(df["timestamp"], df["value"])
        ],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    valid = int(df["value"].notna().sum())
    log.info(f"  {clean} / {var_name}: {len(df):,} rows ({valid:,} valid)")


if __name__ == "__main__":
    main()
