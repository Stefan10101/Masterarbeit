import json
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT


# =============================================
# CONFIGURATION - clear & efficient
# =============================================
INPUT_DIR = Path(DATA_ROOT / "meteosuisse" / "tutti")
OUTPUT_BASE = Path(DATA_ROOT / "temperatur")

MIN_LON = 8.0
MAX_WORKERS = 12
TARGET_TIME_START = "2020-01-01"
TARGET_TIME_END   = "2025-12-31"

OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# =============================================
# STEP 1: Load filtered station metadata
# =============================================
meta_json_path = INPUT_DIR / "stations_east_of_8deg_metadata.json"
with open(meta_json_path, "r", encoding="utf-8") as f:
    station_meta_list = json.load(f)

print(f"Loaded {len(station_meta_list)} stations east of {MIN_LON}Â° E")

# Fast lookup: station_abbr (lowercase) -> metadata
meta_dict = {m["station_abbr"].lower(): m for m in station_meta_list}

# =============================================
# STEP 2: Process each station JSON â†’ only air temperature + per-station subfolder
# =============================================
print(f"\nExtracting only air temperature (tre200s0) | Time filter: {TARGET_TIME_START} to {TARGET_TIME_END}...")

def process_station_file(json_file: Path):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            full_data = json.load(f)
        
        station_id = full_data["station_metadata"]["station_abbr"].lower()
        meta = full_data["station_metadata"]
        
        # Create station-specific subfolder (e.g. AEG, SMA, etc.)
        station_dir = OUTPUT_BASE / station_id.upper()
        station_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data into DataFrame for fast filtering
        df = pd.DataFrame(full_data["data"])
        
        time_col = "reference_timestamp"
        temp_col = "tre200s0"
        
        if time_col not in df.columns or temp_col not in df.columns:
            return station_id, 0
        
        # Fast datetime conversion
        df["time_dt"] = pd.to_datetime(df[time_col], format="%d.%m.%Y %H:%M", errors="coerce")
        
        # Filter: date range + valid temperature only
        mask = (
            (df["time_dt"] >= TARGET_TIME_START) &
            (df["time_dt"] <= TARGET_TIME_END) &
            df[temp_col].notna()
        )
        df_filtered = df[mask].copy()
        
        if df_filtered.empty:
            return station_id, 0
        
        # Build exact target structure
        output = {
            "name": meta.get("station_name", station_id.upper()),
            "hoehe": float(meta.get("station_height_masl", float("nan"))),
            "lon": float(meta["station_coordinates_wgs84_lon"]),
            "lat": float(meta["station_coordinates_wgs84_lat"]),
            "data": [
                {
                    "time": row[time_col],
                    "value": float(row[temp_col])
                }
                for _, row in df_filtered.iterrows()
            ]
        }
        
        # Save inside station subfolder
        out_path = station_dir / f"{station_id}_airtemp_10min.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, separators=(',', ':'))  # compact JSON
        
        return station_id, len(output["data"])
        
    except Exception as e:
        print(f"  Error processing {json_file.name}: {e}")
        return None, 0

# Find all downloaded station JSON files
json_files = list(INPUT_DIR.glob("ogd-smn_*_10min_data.json"))

print(f"Found {len(json_files)} station files to process.")

start_time = time.time()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_station_file, f): f for f in json_files}
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting air temp", unit="station"):
        sid, n_records = future.result()
        if n_records > 0:
            print(f"  âœ“ {sid.upper()} â†’ {n_records:,} records")

duration = time.time() - start_time
print(f"\n=== PROCESSING COMPLETE in {duration/60:.1f} minutes ===")
print(f"Output structure:")
print(f"   {OUTPUT_BASE}")
print(f"   â”œâ”€â”€ AEG/")
print(f"   â”‚   â””â”€â”€ AEG_airtemp_10min.json")
print(f"   â”œâ”€â”€ SMA/")
print(f"   â”‚   â””â”€â”€ SMA_airtemp_10min.json")
print(f"   ...")
print("Each JSON contains only air temperature (tre200s0) from 01.01.2020 to 31.12.2025")
