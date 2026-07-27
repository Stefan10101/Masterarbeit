import pandas as pd
from pathlib import Path
import json
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


# =============================================
# CONFIGURATION
# =============================================
DATA_ROOT = Path(DATA_ROOT)
OUTPUT_CSV = DATA_ROOT / "station.csv"

VARIABLE_FOLDERS = ["Luftfeuchte", "Wind", "Niederschlag", "Schneehoehe", "Temperatur"]

VALID_PROVIDER_FOLDERS = {"DWD", "HydroT", "HydroVO", "LWD", "Meteosuisse", "Suedtirol", "Zamg", "ZAMG"}

PROVIDER_MAP = {
    "DWD": "DWD",
    "HydroT": "HYDRO_TIROL",
    "HydroVO": "HYDRO_VO",
    "LWD": "LWD",
    "Meteosuisse": "MeteoSuisse",
    "Suedtirol": "Suedtirol",
    "Zamg": "ZAMG",
    "ZAMG": "ZAMG"
}

# =============================================
# ENHANCED METADATA EXTRACTION (now ZAMG-proof)
# =============================================
def extract_metadata_from_json(json_path: Path):
    """Robust extractor that works for ALL providers, especially ZAMG."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lat = lon = hoehe = None

        # Helper: recursive search in nested dicts/lists
        def find_coords(obj):
            nonlocal lat, lon, hoehe
            if isinstance(obj, dict):
                for k, v in obj.items():
                    k_lower = k.lower()
                    if lat is None and any(x in k_lower for x in ["lat", "latitude", "breite", "geoBreite"]):
                        try:
                            lat = float(v)
                        except:
                            pass
                    if lon is None and any(x in k_lower for x in ["lon", "longitude", "laenge", "geoLaenge"]):
                        try:
                            lon = float(v)
                        except:
                            pass
                    if hoehe is None and any(x in k_lower for x in ["hoehe", "hoehe_masl", "altitude", "elevation", "stationshoehe", "alt"]):
                        try:
                            hoehe = float(v)
                        except:
                            pass
                    find_coords(v)
            elif isinstance(obj, list):
                for item in obj:
                    find_coords(item)

        find_coords(data)

        # Final rounding
        lat = round(lat, 6) if lat is not None else None
        lon = round(lon, 6) if lon is not None else None
        hoehe = round(hoehe, 1) if hoehe is not None else None

        return lat, lon, hoehe

    except Exception:
        return None, None, None


# =============================================
# MAIN - deduplicated + ZAMG fixed
# =============================================
def build_clean_station_csv():
    best_records = {}   # key = (provider, station_name.lower())

    print("Scanning all folders and extracting metadata (ZAMG now supported)...\n")

    for var_name in VARIABLE_FOLDERS:
        var_path = DATA_ROOT / var_name
        if not var_path.exists():
            continue
        print(f"â†’ Scanning {var_name}")

        for provider_dir in var_path.iterdir():
            if not provider_dir.is_dir() or provider_dir.name not in VALID_PROVIDER_FOLDERS:
                continue

            provider_name = PROVIDER_MAP.get(provider_dir.name, provider_dir.name)

            for station_dir in provider_dir.iterdir():
                if not station_dir.is_dir():
                    continue

                station_name = station_dir.name.strip()
                if not station_name or station_name.lower() in {"paket", "full_2020_2025"}:
                    continue

                key = (provider_name, station_name.lower())

                # Get best metadata
                lat = lon = hoehe = None
                for jf in station_dir.glob("*.json"):
                    lat, lon, hoehe = extract_metadata_from_json(jf)
                    if lat is not None and lon is not None:
                        break  # one good JSON is enough

                new_row = {
                    "station_name": station_name,
                    "latitude": lat,
                    "longitude": lon,
                    "elevation_masl": hoehe,
                    "provider": provider_name
                }

                # Keep the best version (prefer the one with coordinates)
                if key not in best_records or \
                   (best_records[key]["latitude"] is None and lat is not None):
                    best_records[key] = new_row

    df = pd.DataFrame.from_dict(best_records, orient="index")
    df = df.sort_values(["provider", "station_name"]).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"\nâœ… DONE â†’ {len(df):,} unique station entries")
    print(f"   Saved to: {OUTPUT_CSV}\n")

    print("Provider breakdown:")
    print(df["provider"].value_counts().to_string())

    print("\nPreview (first 15):")
    cols = ["station_name", "latitude", "longitude", "elevation_masl", "provider"]
    print(df.head(15)[cols].to_string(index=False))

    dup = df.duplicated(subset=["provider", "station_name"]).sum()
    print(f"\nDuplicates removed: {dup}")


if __name__ == "__main__":
    build_clean_station_csv()
