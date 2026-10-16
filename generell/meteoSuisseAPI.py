import json
import math
from pathlib import Path
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import os

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
OUTPUT_DIR = Path(DATA_ROOT / "meteosuisse" / "tutti")
BASE_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
METADATA_URL = f"{BASE_URL}/ogd-smn_meta_stations.csv"
PARAMETERS_URL = f"{BASE_URL}/ogd-smn_meta_parameters.csv"

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
# STEP 2: Keep all Swiss stations + save full metadata as JSON
# =============================================
all_df = df_stations.copy()
print(f"Stations in catalogue (all Switzerland): {len(all_df)}")

# Save complete metadata as JSON (all columns preserved)
all_metadata = all_df.to_dict(orient="records")
meta_json_path = OUTPUT_DIR / "stations_metadata.json"
with open(meta_json_path, "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, indent=2, ensure_ascii=False)

print(f"Full station metadata saved: {meta_json_path.name}")

station_list = [abbr.lower() for abbr in all_df["station_abbr"]]

# Skip stations that already have a finished, valid JSON
def output_json_path(station_id: str) -> Path:
    return OUTPUT_DIR / f"ogd-smn_{station_id}_10min_data.json"

def is_complete_station_json(path: Path) -> bool:
    """Fast check: file exists, is non-trivial, and ends with } (complete write).
    Does not parse the file. Truncated writes from an interrupted run fail this.
    """
    try:
        if not path.exists() or path.stat().st_size < 50:
            return False
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
            if last == b"\n":
                f.seek(-2, os.SEEK_END)
                last = f.read(1)
            return last == b"}"
    except Exception:
        return False

def json_safe(obj):
    """Replace NaN / Inf with None so json.dump writes valid JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj

print("Checking existing station files...")
already_done = {sid for sid in station_list if is_complete_station_json(output_json_path(sid))}
to_download = [sid for sid in station_list if sid not in already_done]
invalid_existing = [
    sid for sid in station_list
    if output_json_path(sid).exists() and sid not in already_done
]

print(f"Already downloaded (complete): {len(already_done)}")
print(f"Invalid / truncated (will re-download): {len(invalid_existing)}")
print(f"To download: {len(to_download)}")

# =============================================
# STEP 3: Parallel download + conversion to self-contained JSON
# =============================================
print(f"\nSTEP 3: Processing {len(to_download)} stations → 10-min data as JSON...")

def process_station(station_id: str):
    json_path = output_json_path(station_id)
    if is_complete_station_json(json_path):
        return station_id, 0, None, "skipped"

    files = [
        f"ogd-smn_{station_id}_t_historical_2020-2029.csv",
        f"ogd-smn_{station_id}_t_recent.csv"
    ]

    station_data = []
    # Own session per worker: requests.Session is not thread-safe
    with requests.Session() as session:
        for fname in files:
            url = f"{BASE_URL}/{station_id}/{fname}"
            try:
                r = session.get(url, stream=True, timeout=90)
                if r.status_code == 200:
                    temp_csv = OUTPUT_DIR / f"temp_{station_id}_{fname}"
                    with open(temp_csv, "wb") as f:
                        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                            f.write(chunk)

                    df_data = pd.read_csv(temp_csv, sep=";", encoding="utf-8", low_memory=False)
                    station_data.extend(df_data.to_dict(orient="records"))

                    temp_csv.unlink(missing_ok=True)

                elif r.status_code != 404:
                    print(f"  Warning {station_id}: {fname} → {r.status_code}")
            except Exception as e:
                print(f"  Error {station_id}: {fname} → {e}")

    if station_data:
        meta_row = all_df[all_df["station_abbr"].str.lower() == station_id].iloc[0].to_dict()
        param_meta = pd.read_csv(param_path, sep=";", encoding="utf-8").to_dict(orient="records")

        output = {
            "station_metadata": json_safe(meta_row),
            "parameter_metadata": json_safe(param_meta),
            "data": json_safe(station_data),
        }

        # Atomic write: only the final path exists when the write is complete
        tmp_path = json_path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            os.replace(tmp_path, json_path)
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            print(f"  Error writing {station_id}: {e}")
            return station_id, 0, None, "error"

        return station_id, len(station_data), json_path.name, "ok"

    return station_id, 0, None, "empty"

# Run in parallel
start_time = time.time()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_station, sid): sid for sid in to_download}

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="station"):
        sid, n_records, json_file, status = future.result()
        if status == "ok" and n_records > 0:
            print(f"  ✓ {sid} → {n_records:,} records → {json_file}")

duration = time.time() - start_time
print(f"\n=== FINISHED in {duration/60:.1f} minutes ===")
print(f"All data written to: {OUTPUT_DIR}")
print("Each JSON contains:")
print("   • Full station metadata (all columns)")
print("   • Parameter explanations")
print("   • All 10-min records (incl. air temperature tre200s0)")
