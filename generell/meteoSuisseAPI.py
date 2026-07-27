import json
from pathlib import Path
import requests
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
# CONFIGURATION - straightforward & efficient
# =============================================
OUTPUT_DIR = Path(DATA_ROOT / "meteosuisse" / "temperatur")
BASE_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
METADATA_URL = f"{BASE_URL}/ogd-smn_meta_stations.csv"
PARAMETERS_URL = f"{BASE_URL}/ogd-smn_meta_parameters.csv"

MIN_LON = 8.0
MAX_WORKERS = 16
CHUNK_SIZE = 16384

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================
# STEP 1: Fetch metadata with correct encoding (ISO-8859-1 / latin1)
# =============================================
print("STEP 1: Downloading station and parameter metadata (ISO-8859-1 encoding)...")

def download_csv_with_encoding(url: str, output_path: Path, encoding: str = "iso-8859-1"):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    # Decode with correct encoding and re-encode to UTF-8 for clean pandas handling
    content = resp.content.decode(encoding, errors="replace")
    output_path.write_text(content, encoding="utf-8")
    return output_path

# Download and convert to UTF-8
meta_path = OUTPUT_DIR / "ogd-smn_meta_stations.csv"
param_path = OUTPUT_DIR / "ogd-smn_meta_parameters.csv"

download_csv_with_encoding(METADATA_URL, meta_path)
download_csv_with_encoding(PARAMETERS_URL, param_path)

# Load with UTF-8 (now safe)
df_stations = pd.read_csv(meta_path, sep=";", encoding="utf-8")
print(f"Total automatic weather stations: {len(df_stations)}")

# =============================================
# STEP 2: Filter east of 8Â° longitude + save full metadata as JSON
# =============================================
east_df = df_stations[df_stations["station_coordinates_wgs84_lon"] > MIN_LON].copy()
print(f"Stations east of {MIN_LON}Â° E: {len(east_df)}")

# Save complete filtered metadata as JSON (all columns preserved)
east_metadata = east_df.to_dict(orient="records")
meta_json_path = OUTPUT_DIR / "stations_east_of_8deg_metadata.json"
with open(meta_json_path, "w", encoding="utf-8") as f:
    json.dump(east_metadata, f, indent=2, ensure_ascii=False)

print(f"Full station metadata saved: {meta_json_path.name}")

station_list = [abbr.lower() for abbr in east_df["station_abbr"]]

# =============================================
# STEP 3: Parallel download + conversion to self-contained JSON
# =============================================
print(f"\nSTEP 3: Processing {len(station_list)} stations â†’ 10-min data as JSON...")

session = requests.Session()

def process_station(station_id: str):
    files = [
        f"ogd-smn_{station_id}_t_historical_2020-2029.csv",
        f"ogd-smn_{station_id}_t_recent.csv"
    ]
    
    station_data = []
    
    for fname in files:
        url = f"{BASE_URL}/{station_id}/{fname}"
        try:
            r = session.get(url, stream=True, timeout=90)
            if r.status_code == 200:
                temp_csv = OUTPUT_DIR / f"temp_{station_id}_{fname}"
                with open(temp_csv, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        f.write(chunk)
                
                # Read with correct separator (data files also use ;)
                df_data = pd.read_csv(temp_csv, sep=";", encoding="utf-8", low_memory=False)
                station_data.extend(df_data.to_dict(orient="records"))
                
                temp_csv.unlink(missing_ok=True)
                
            elif r.status_code != 404:
                print(f"  Warning {station_id}: {fname} â†’ {r.status_code}")
        except Exception as e:
            print(f"  Error {station_id}: {fname} â†’ {e}")
    
    if station_data:
        # Attach full station metadata
        meta_row = east_df[east_df["station_abbr"].str.lower() == station_id].iloc[0].to_dict()
        
        output = {
            "station_metadata": meta_row,
            "parameter_metadata": pd.read_csv(param_path, sep=";", encoding="utf-8").to_dict(orient="records"),
            "data": station_data
        }
        
        json_path = OUTPUT_DIR / f"ogd-smn_{station_id}_10min_data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, separators=(',', ':'))  # compact JSON
        
        return station_id, len(station_data), json_path.name
    
    return station_id, 0, None

# Run in parallel
start_time = time.time()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_station, sid): sid for sid in station_list}
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="station"):
        sid, n_records, json_file = future.result()
        if n_records > 0:
            print(f"  âœ“ {sid} â†’ {n_records:,} records â†’ {json_file}")

duration = time.time() - start_time
print(f"\n=== FINISHED in {duration/60:.1f} minutes ===")
print(f"All data written to: {OUTPUT_DIR}")
print("Each JSON contains:")
print("   â€¢ Full station metadata (all columns)")
print("   â€¢ Parameter explanations")
print("   â€¢ All 10-min records (incl. air temperature tre200s0)")
