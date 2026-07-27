#!/usr/bin/env python3
"""
LWD JSON exporter (fixed version)
- Robust value parsing (prevents valid numbers turning into null)
- Single, safe timestamp parse + CETâ†’UTC shift (no data loss / corruption)
- Clear logging of dropped / null rates
- Always overwrites JSONs (no skip logic)
"""

import json
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import StringIO
from collections import defaultdict

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT


# ==================== CONFIG ====================
LWD_FOLDER = Path(DATA_ROOT / "lwd")

TARGET_BASE = {
    "HS": Path(DATA_ROOT / "schneehoehe" / "lwd"),
    "WG": Path(DATA_ROOT / "wind" / "lwd"),
    "LF": Path(DATA_ROOT / "luftfeuchte" / "lwd"),
    "LT": Path(DATA_ROOT / "temperatur" / "lwd"),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.FileHandler("weather_json_export_fixed.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

for folder in TARGET_BASE.values():
    folder.mkdir(parents=True, exist_ok=True)

# ==================== FIXED FUNCTIONS ====================

def parse_yearly_file(path: Path) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    """Parse one yearly LWD CSV with robust value + timestamp handling."""
    try:
        text = path.read_text(encoding="cp1252")
    except Exception:
        text = path.read_text(encoding="utf-8", errors="replace")

    lines = text.splitlines()
    metadata: Dict[str, str] = {}
    data_start = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if "Datum/Uhrzeit" in stripped:
            data_start = i
            break
        if ";" in stripped:
            key, value = [p.strip() for p in stripped.split(";", 1)]
            key_clean = (key.replace("ï¿½", "Ã¤")
                         .replace("Lï¿½nge", "LÃ¤nge")
                         .replace("ï¿½C", "Â°C")
                         .replace("ï¿½", "Ã¶"))
            metadata[key_clean] = value

    if data_start is None:
        logger.warning(f"No data section found in {path.name}")
        return None, None

    csv_text = "\n".join(lines[data_start:])

    df = pd.read_csv(
        StringIO(csv_text),
        sep=";",
        decimal=",",
        parse_dates=[0],
        date_format="%d.%m.%Y %H:%M:%S",
        dayfirst=True,                 # extra safety for European DD.MM
        skipinitialspace=True,         # robust against " ; 4,2"
        on_bad_lines="warn",
        dtype_backend="numpy_nullable"
    )

    if len(df.columns) < 2:
        logger.error(f"Too few columns in {path.name}: {list(df.columns)}")
        return None, None

    df.columns = ["timestamp", "value"]

    # === ROBUST VALUE PARSING (prevents valid numbers â†’ null) ===
    # 1. Normalize German decimal "," â†’ "." so to_numeric always succeeds
    # 2. Then handle missing markers
    val_str = (df["value"].astype(str)
               .str.strip()
               .str.replace(",", ".", regex=False)
               .replace(["---", "-", "", "nan", "NaN", "None", "<NA>"], pd.NA))

    df["value"] = pd.to_numeric(val_str, errors="coerce")

    # === SAFE TIMESTAMP + CETâ†’UTC (single parse, no corruption) ===
    # parse_dates already gave us datetime64 with possible NaT
    ts = df["timestamp"]

    if ts.dt.tz is None:
        # Shift naive CET (+01:00) â†’ UTC by adding 1h. NaT stays NaT.
        ts = ts + pd.Timedelta(hours=1)
        df["timestamp"] = ts.dt.tz_localize("UTC")
    else:
        df["timestamp"] = ts.dt.tz_convert("UTC")

    # Count & log rows lost to bad timestamps (transparency)
    nat_count = df["timestamp"].isna().sum()
    if nat_count > 0:
        logger.warning(f"{path.name}: {nat_count} rows dropped (unparseable timestamp)")

    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    return metadata, df


def process_station_parameter(station_code: str, param: str, files: List[Path]):
    """Merge yearly files, remove duplicate timestamps (keep last), and write JSON."""
    base_folder = TARGET_BASE.get(param)
    if not base_folder:
        logger.error(f"Unknown parameter {param} for station {station_code}")
        return

    logger.info(f"Processing {station_code} - {param} ({len(files)} files)")

    all_dfs: List[pd.DataFrame] = []
    station_metadata = None
    real_station_name = station_code

    for file in sorted(files):
        meta, df = parse_yearly_file(file)
        if df is None or df.empty:
            continue
        all_dfs.append(df)
        if station_metadata is None and meta:
            station_metadata = meta
            real_station_name = meta.get("Stationsname", station_code).strip()

    if not all_dfs or station_metadata is None:
        logger.error(f"No valid data/metadata for {station_code} - {param}")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # === Remove duplicate timestamps (keep the last version) ===
    before = len(combined)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    removed = before - len(combined)
    if removed > 0:
        logger.info(f"  Removed {removed:,} duplicate timestamps (kept last)")

    # === Coordinates ===
    lon_str = station_metadata.get("Geografische LÃ¤nge") or station_metadata.get("longitude", "")
    lat_str = station_metadata.get("Geografische Breite") or station_metadata.get("latitude", "")

    try:
        longitude = float(lon_str.replace(",", ".")) if lon_str else None
        latitude  = float(lat_str.replace(",", ".")) if lat_str else None
    except (ValueError, TypeError):
        longitude = latitude = None
        logger.warning(f"Could not parse coordinates for {station_code}")

    station_folder = base_folder / real_station_name
    station_folder.mkdir(parents=True, exist_ok=True)

    station_json = {
        "name": real_station_name,
        "code": station_code,
        "hoehe": int(station_metadata.get("StationshÃ¶he", 0)) if station_metadata.get("StationshÃ¶he") else None,
        "longitude": longitude,
        "latitude": latitude,
        "data": [
            {
                "timestamp": ts.isoformat(),
                "value": float(val) if pd.notna(val) else None
            }
            for ts, val in zip(combined["timestamp"], combined["value"])
        ]
    }

    json_path = station_folder / f"{real_station_name}_{param}_data.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(station_json, f, ensure_ascii=False, indent=2)

    valid_count = combined["value"].notna().sum()
    total = len(combined)
    valid_pct = (valid_count / total * 100) if total > 0 else 0
    logger.info(f"âœ“ Saved {total:,} records ({valid_count:,} valid = {valid_pct:.1f}%) â†’ {json_path.name}")
# ==================== MAIN ====================
if __name__ == "__main__":
    logger.info("=== Starting LWD JSON export (FIXED: robust value + safe timestamp) ===")

    file_groups: Dict[Tuple[str, str], List[Path]] = defaultdict(list)

    for csv_file in LWD_FOLDER.glob("*.csv"):
        parts = csv_file.stem.split("_")
        if len(parts) < 3:
            continue
        station_code = parts[0]
        param = parts[1]
        if param in TARGET_BASE:
            file_groups[(station_code, param)].append(csv_file)

    logger.info(f"Found {len(file_groups)} station-parameter combinations")

    for (station_code, param), files in file_groups.items():
        try:
            process_station_parameter(station_code, param, files)
        except Exception as e:
            logger.error(f"Failed to process {station_code} - {param}: {e}", exc_info=True)

    logger.info("=== JSON export finished (all files overwritten) ===")

