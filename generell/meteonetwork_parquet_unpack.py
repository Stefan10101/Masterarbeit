#!/usr/bin/env python3
"""
MeteoNetwork parquet → half-hourly JSON
---------------------------------------
Native data are irregular (mostly 1–5 min). rain_rate is mm/h.

- temperature / rh / wind_speed : mean over 30 min
- precipitation               : rain_rate * Δt_hours, then sum over 30 min
- wind_gust dropped
- local → UTC via station time_zone_name
- Alpine Convention perimeter + 100 km filter
"""
from pathlib import Path
from datetime import datetime
import logging
import re
import json
import sys

import pandas as pd
import numpy as np

SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_ROOT = PROJECT_ROOT

sys.path.insert(0, str(SCRIPT.parents[1]))
from shared.splits.evaluation_sets import stations_inside_domain

RAW = DATA_ROOT / "Network_providers" / "meteonetwork"
STATIONS_PQ = RAW / "alps_mnw_native_raw_2020-2025_stations.parquet"
DATA_PQ = RAW / "alps_mnw_native_raw_2020-2025.parquet"

OUTPUT_BASES = {
    "temperature": DATA_ROOT / "Temperatur" / "meteonetwork",
    "precipitation": DATA_ROOT / "Niederschlag" / "meteonetwork",
    "wind": DATA_ROOT / "Wind" / "meteonetwork",
    "relative_humidity": DATA_ROOT / "Luftfeuchte" / "meteonetwork",
}
for p in OUTPUT_BASES.values():
    p.mkdir(parents=True, exist_ok=True)

LOG_DIR = DATA_ROOT / "code" / "logs" / "meteonetwork" / "unpack"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"meteonetwork_hh_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("mnw_hh")


def clean_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(s).strip())
    return s.strip("_").lower() or "unknown"


def main():
    log.info("Loading stations")
    stations = pd.read_parquet(STATIONS_PQ)
    stations = stations[stations["is_valid"].isin(["Y", "E"])].copy()
    stations = stations_inside_domain(
        stations, lon_col="longitude", lat_col="latitude", buffer_km=100.0
    )
    log.info(f"Stations after Alpine perimeter + 100 km: {len(stations)}")

    meta = stations.set_index("station_id")[
        ["site_name", "place_name", "elevation", "latitude", "longitude", "time_zone_name"]
    ].to_dict("index")

    log.info("Loading observations")
    df = pd.read_parquet(
        DATA_PQ,
        columns=["station_id", "datetime", "temperature", "rh", "wind_speed", "rain_rate"],
    )
    df = df[df["station_id"].isin(stations["station_id"])]
    log.info(f"Rows after filter: {len(df):,}")

    for sid, grp in df.groupby("station_id"):
        info = meta.get(sid, {})
        tz = info.get("time_zone_name", "Europe/Berlin")
        clean = clean_name(sid)

        # localize → UTC
        try:
            dt = grp["datetime"].dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
        except Exception:
            dt = grp["datetime"].dt.tz_localize("UTC")
        g = grp.copy()
        g["datetime"] = dt
        g = g.dropna(subset=["datetime"]).sort_values("datetime")

        # interval length in hours (first row uses median)
        delta_h = g["datetime"].diff().dt.total_seconds() / 3600.0
        med = delta_h.median()
        if pd.isna(med) or med <= 0:
            med = 5 / 60
        delta_h = delta_h.fillna(med).clip(upper=1.0)  # cap crazy gaps

        # rain_rate is mm/h → amount in the interval
        g["precip_amount"] = g["rain_rate"] * delta_h

        g = g.set_index("datetime")

        # 30-min aggregate
        res = g.resample("30min").agg({
            "temperature": "mean",
            "rh": "mean",
            "wind_speed": "mean",
            "precip_amount": "sum",
        })

        var_map = {
            "temperature": ("temperature", "temperature"),
            "rh": ("relative_humidity", "relative_humidity"),
            "wind_speed": ("wind", "wind"),
            "precip_amount": ("precipitation", "precipitation"),
        }
        for src, (var, _) in var_map.items():
            series = res[src].dropna()
            if series.empty:
                continue
            records = [
                {"timestamp": ts.isoformat(), "value": float(v)}
                for ts, v in series.items()
            ]
            payload = {
                "name": info.get("site_name") or info.get("place_name") or sid,
                "code": sid,
                "hoehe": int(info["elevation"]) if pd.notna(info.get("elevation")) else None,
                "longitude": float(info["longitude"]) if pd.notna(info.get("longitude")) else None,
                "latitude": float(info["latitude"]) if pd.notna(info.get("latitude")) else None,
                "source": "meteonetwork",
                "resolution": "halfhourly",
                "data": records,
            }
            out_dir = OUTPUT_BASES[var] / clean
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{clean}_{var}_data.json"
            out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            log.info(f"  {clean} {var}: {len(records)} half-hours")

    log.info("Done")


if __name__ == "__main__":
    main()