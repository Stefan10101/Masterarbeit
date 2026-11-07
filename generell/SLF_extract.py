#!/usr/bin/env python3
"""
SLF / IMIS extractor

Reads the historical CSVs produced by SLF_download.py, filters to 2020–2025,
and writes per-station / per-variable JSON files in the same format used by
the other providers (MeteoSuisse, ZAMG, …).

30-minute native resolution. Times are already UTC.

Output layout (under DATA_ROOT):
  Temperatur/slf/{name}/{name}_temperature_30min.json
  Luftfeuchte/slf/{name}/{name}_relative_humidity_30min.json
  Wind/slf/{name}/{name}_wind_speed_30min.json
  Schneehoehe/slf/{name}/{name}_snow_height_30min.json
  Niederschlag/slf/{name}/{name}_precipitation_30min.json   (only if pluvio exists)

JSON structure (identical to other extractors):
  {
    "name": "...",
    "hoehe": 2558.0,
    "lon": 9.8194,
    "lat": 46.6989,
    "data": [{"time": "2020-01-01 00:00", "value": -2.1}, ...]
  }
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_ROOT = PROJECT_ROOT

SLF_DIR = DATA_ROOT / "slf"
CSV_DIR = SLF_DIR / "csv"
STATIONS_JSON = SLF_DIR / "stations.json"

OUTPUT_PATHS = {
    "temperature":       DATA_ROOT / "Temperatur" / "slf",
    "relative_humidity": DATA_ROOT / "Luftfeuchte" / "slf",
    "wind_speed":        DATA_ROOT / "Wind" / "slf",
    "snow_height":       DATA_ROOT / "Schneehoehe" / "slf",
    "precipitation":     DATA_ROOT / "Niederschlag" / "slf",
}
for p in OUTPUT_PATHS.values():
    p.mkdir(parents=True, exist_ok=True)

LOG_DIR = DATA_ROOT / "code" / "logs" / "slf" / "extract"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"slf_extract_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("slf_ex")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_START = "2020-01-01"
TARGET_END   = "2025-12-31"
MAX_WORKERS  = 8

# Column → (output variable name, unit)
# Main CSV
MAIN_COLS = {
    "TA_30MIN_MEAN": ("temperature",       "°C"),
    "RH_30MIN_MEAN": ("relative_humidity", "%"),
    "VW_30MIN_MEAN": ("wind_speed",        "m/s"),
    "HS":            ("snow_height",       "cm"),
}
# Pluvio CSV. Prefer the already-aggregated 30-min sum (RRE = corrected/real).
# Stations that only have 10-min columns are summed into the same 30-min grid.
PLUVIO_30 = ["RRE_30MIN_SUM", "RR_30MIN_SUM"]
PLUVIO_10 = ["RRE_10MIN_SUM", "RR_10MIN_SUM"]


def _to_30min_sum(df: pd.DataFrame, col30: str | None, col10: str | None) -> pd.DataFrame:
    """Return a 30-min precipitation series.

    Prefer the native 30-min column where it has a value. For timestamps that
    only exist at 10-min resolution, sum the three 10-min values that fall
    into each 30-min bin (:00 and :30). Precipitation is a sum, so this does
    not lose mass and does not double-count.
    """
    parts = []

    if col30 and col30 in df.columns:
        s30 = df[["measure_date", col30]].dropna(subset=[col30]).copy()
        if not s30.empty:
            s30 = s30.rename(columns={col30: "value"})
            s30["bin"] = s30["measure_date"].dt.floor("30min")
            parts.append(s30[["bin", "value"]])

    if col10 and col10 in df.columns:
        s10 = df[["measure_date", col10]].dropna(subset=[col10]).copy()
        if not s10.empty:
            s10 = s10.rename(columns={col10: "value"})
            s10["bin"] = s10["measure_date"].dt.floor("30min")
            # only keep 10-min bins that have no native 30-min value
            if parts:
                have = set(parts[0]["bin"])
                s10 = s10[~s10["bin"].isin(have)]
            if not s10.empty:
                summed = (
                    s10.groupby("bin", as_index=False)["value"]
                    .sum(min_count=1)
                )
                parts.append(summed)

    if not parts:
        return pd.DataFrame(columns=["measure_date", "value"])

    out = pd.concat(parts, ignore_index=True)
    out = out.groupby("bin", as_index=False)["value"].sum(min_count=1)
    out = out.rename(columns={"bin": "measure_date"})
    return out.sort_values("measure_date")


def clean_name(label: str, code: str) -> str:
    """Filesystem-safe station name, unique by appending code."""
    base = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(label).strip())
    base = base.strip("_") or code
    return f"{base}_{code}".lower()


def load_meta() -> dict[str, dict]:
    if not STATIONS_JSON.exists():
        raise SystemExit(f"Missing {STATIONS_JSON} — run SLF_download.py first")
    stations = json.loads(STATIONS_JSON.read_text(encoding="utf-8"))
    return {s["code"]: s for s in stations}


def write_json(path: Path, name: str, hoehe: float, lon: float, lat: float,
               times: pd.Series, values: pd.Series) -> int:
    if path.exists():
        return 0  # resumable
    records = [
        {"time": t.strftime("%Y-%m-%d %H:%M"), "value": float(v)}
        for t, v in zip(times, values)
        if pd.notna(v)
    ]
    if not records:
        return 0
    out = {
        "name": name,
        "hoehe": float(hoehe) if pd.notna(hoehe) else None,
        "lon": float(lon),
        "lat": float(lat),
        "data": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return len(records)


def process_station(code: str, meta: dict) -> list[tuple[str, int]]:
    results = []
    st = meta.get(code)
    if not st:
        return results

    name = clean_name(st.get("label", code), code)
    hoehe = st.get("elevation")
    lon = st.get("lon")
    lat = st.get("lat")

    # --- main series ---
    main_csv = CSV_DIR / f"{code}.csv"
    if main_csv.exists():
        try:
            df = pd.read_csv(main_csv, parse_dates=["measure_date"], low_memory=False)
            df = df[(df["measure_date"] >= TARGET_START) & (df["measure_date"] <= TARGET_END)]
            for col, (var, _unit) in MAIN_COLS.items():
                if col not in df.columns:
                    continue
                sub = df[["measure_date", col]].dropna(subset=[col])
                if sub.empty:
                    continue
                out_path = OUTPUT_PATHS[var] / name / f"{name}_{var}_30min.json"
                n = write_json(out_path, st.get("label", name), hoehe, lon, lat,
                               sub["measure_date"], sub[col])
                if n:
                    results.append((var, n))
        except Exception as e:
            log.warning(f"{code} main: {e}")

    # --- precipitation (pluvio) ---
    # Native IMIS resolution is 30 min. Some pluvio files only contain
    # 10-min sums (RRE_10MIN_SUM). Those are summed into the same :00/:30
    # grid so the series matches T/RH/wind/snow and the pipeline regrid.
    pluvio_csv = CSV_DIR / f"{code}_pluvio.csv"
    if pluvio_csv.exists():
        try:
            df = pd.read_csv(pluvio_csv, parse_dates=["measure_date"], low_memory=False)
            df = df[(df["measure_date"] >= TARGET_START) & (df["measure_date"] <= TARGET_END)]
            col30 = next((c for c in PLUVIO_30 if c in df.columns and df[c].notna().any()), None)
            col10 = next((c for c in PLUVIO_10 if c in df.columns and df[c].notna().any()), None)
            agg = _to_30min_sum(df, col30, col10)
            if not agg.empty:
                out_path = OUTPUT_PATHS["precipitation"] / name / f"{name}_precipitation_30min.json"
                n = write_json(out_path, st.get("label", name), hoehe, lon, lat,
                               agg["measure_date"], agg["value"])
                if n:
                    results.append(("precipitation", n))
        except Exception as e:
            log.warning(f"{code} pluvio: {e}")

    return results


def main() -> None:
    log.info("=== SLF / IMIS extract started ===")
    log.info(f"Window: {TARGET_START} → {TARGET_END}")

    meta = load_meta()
    codes = sorted(meta.keys())
    log.info(f"Stations in metadata: {len(codes)}")

    total = {k: 0 for k in OUTPUT_PATHS}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_station, c, meta): c for c in codes}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="extract"):
            code = futures[fut]
            try:
                res = fut.result()
                for var, n in res:
                    total[var] += n
                    log.info(f"✓ {code} → {var}: {n:,}")
            except Exception as e:
                log.error(f"{code}: {e}")

    log.info("=== DONE ===")
    for var, n in total.items():
        log.info(f"  {var}: {n:,} records")
    log.info(f"Log: {LOG_FILE}")


if __name__ == "__main__":
    main()
