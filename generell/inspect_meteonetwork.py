"""
inspect_meteonetwork.py
Inspect the two Meteonetwork native parquet files (pandas only, no direct pyarrow).
"""
from pathlib import Path
import pandas as pd

STATIONS = Path(r"C:\Users\stefa\Documents\UNI\Master\Masterarbeit\Daten\Network_providers\meteonetwork\alps_mnw_native_raw_2020-2025_stations.parquet")
DATA = Path(r"C:\Users\stefa\Documents\UNI\Master\Masterarbeit\Daten\Network_providers\meteonetwork\alps_mnw_native_raw_2020-2025.parquet")

def inspect(path: Path, label: str) -> None:
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    print(f"Path:   {path}")
    print(f"Exists: {path.exists()}")
    if not path.exists():
        return
    print(f"Size:   {path.stat().st_size / 1e6:.1f} MB")

    # Load full file (inspection). If this OOMs, we can switch to chunked later.
    df = pd.read_parquet(path)
    print(f"\nLoaded shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    print(f"\nNull counts:\n{df.isnull().sum()}")

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    # Time range
    time_cols = [c for c in df.columns if any(k in c.lower() for k in ("time", "date", "timestamp", "datetime"))]
    for c in time_cols:
        try:
            s = pd.to_datetime(df[c], errors="coerce")
            valid = s.notna().sum()
            if valid > 0:
                print(f"\n{c} range: {s.min()} → {s.max()}  (valid: {valid})")
        except Exception as e:
            print(f"{c}: could not parse as datetime ({e})")

    # Candidate ID / station columns
    id_cols = [c for c in df.columns if any(k in c.lower() for k in ("station", "id", "site", "code", "name"))]
    for c in id_cols:
        nunique = df[c].nunique()
        print(f"\nUnique {c}: {nunique}")
        if nunique < 40:
            vals = sorted(df[c].dropna().unique().tolist())
            print(f"  values: {vals}")

if __name__ == "__main__":
    inspect(STATIONS, "STATIONS METADATA")
    inspect(DATA, "RAW OBSERVATIONS")