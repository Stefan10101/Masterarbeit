#!/usr/bin/env python3
"""
Download HydroBASINS Europe (standard, levels 01-12) and extract to Source/Basins.

Usage (from CODE/):
  python shared/download_hydrobasins_europe.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from paths import get_source_root  # noqa: E402


URL = "https://data.hydrosheds.org/file/hydrobasins/standard/hybas_eu_lev01-12_v1c.zip"
OUT_DIR = get_source_root() / "Basins"
ZIP_PATH = OUT_DIR / "hybas_eu_lev01-12_v1c.zip"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"downloading {URL}")
        r = requests.get(URL, stream=True, timeout=120)
        r.raise_for_status()
        with ZIP_PATH.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        print(f"saved {ZIP_PATH}")
    else:
        print(f"zip already present: {ZIP_PATH}")

    print("extracting …")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(OUT_DIR)
    print(f"extracted under {OUT_DIR}")
    # list a few shapefiles
    shps = sorted(OUT_DIR.rglob("*.shp"))
    print(f"found {len(shps)} shapefiles")
    for p in shps[:8]:
        print(f"  {p.relative_to(OUT_DIR)}")


if __name__ == "__main__":
    main()
