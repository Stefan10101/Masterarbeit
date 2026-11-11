#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Station counts from QC/stations_overview.csv.

Writes a summary CSV with:
- total station count
- station count per variable
- station count per network provider (cleaned labels)

Run from the CODE directory or anywhere; paths are resolved via paths.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from paths import DATA_ROOT, get_metadata_path

# Overview names as stored in the "variables" column of stations_overview.csv
VARIABLES = [
    "temperature",
    "precipitation",
    "wind_speed",
    "relative_humidity",
    "snow_height",
]

# Clean display names for providers (matches station_overview_plots.py)
PROVIDER_LABELS = {
    "DWD": "German Weather Service",
    "MeteoFrance": "MeteoFrance",
    "ZAMG": "Geosphere",
    "zamg": "Geosphere",
    "LWD": "Avalanche Service Tirol",
    "Meteosuisse": "MeteoSuisse",
    "MeteoSuisse": "MeteoSuisse",
    "ARSO": "Slovenian Environment Agency",
    "HydroT": "Hydrological Service Tirol",
    "HYDRO_TIROL": "Hydrological Service Tirol",
    "HydroVO": "Hydrological Service Vorarlberg",
    "HYDRO_VORARLBERG": "Hydrological Service Vorarlberg",
    "Suedtirol": "Autonomous Province South Tyrol",
    "Suedtriol": "Autonomous Province South Tyrol",
    "Meteonetwork": "MeteoNetwork",
    "meteonetwork": "MeteoNetwork",
    "SLF": "SLF",
    "slf": "SLF",
    "Meteohub": "MeteoHub",
    "meteohub": "MeteoHub",
}


def rename_provider(name: str) -> str:
    if not isinstance(name, str) or name.strip() == "" or name == "Unknown":
        return "Unknown"
    return PROVIDER_LABELS.get(name, name)


def has_variable(cell, overview_name: str) -> bool:
    if pd.isna(cell):
        return False
    parts = [p.strip() for p in str(cell).split(",")]
    return overview_name in parts


def main() -> None:
    overview_path = get_metadata_path()
    if not overview_path.exists():
        raise FileNotFoundError(f"Overview CSV not found: {overview_path}")

    df = pd.read_csv(overview_path)
    n_total = len(df)

    rows = [{"section": "total", "name": "all", "n_stations": n_total}]

    for var in VARIABLES:
        n = int(df["variables"].apply(lambda c: has_variable(c, var)).sum())
        rows.append({"section": "variable", "name": var, "n_stations": n})

    df["provider_label"] = df["network_provider"].map(rename_provider)
    provider_counts = (
        df["provider_label"].value_counts(dropna=False).sort_values(ascending=False)
    )
    for name, n in provider_counts.items():
        rows.append({"section": "provider", "name": str(name), "n_stations": int(n)})

    out = pd.DataFrame(rows)
    out_path = DATA_ROOT / "Plots" / "Overview" / "station_counts.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Overview: {overview_path}")
    print(f"Total stations: {n_total}")
    print("\nPer variable:")
    for var in VARIABLES:
        n = next(r["n_stations"] for r in rows if r["section"] == "variable" and r["name"] == var)
        print(f"  {var}: {n}")
    print("\nPer provider:")
    for name, n in provider_counts.items():
        print(f"  {name}: {n}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
