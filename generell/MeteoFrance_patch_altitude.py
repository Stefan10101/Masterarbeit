#!/usr/bin/env python3
"""
Patch altitude into already-unpacked Météo-France JSON files.

Reads altitude from meteofrance/raw/*_meta.json and writes it into
station_metadata.altitude of existing files under:
  niederschlag/meteofrance, temperatur/meteofrance, wind/meteofrance,
  luftfeuchte/meteofrance, schneehoehe/meteofrance
"""

import json
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_ROOT = PROJECT_ROOT

RAW_DIR = DATA_ROOT / "meteofrance" / "raw"
OUTPUT_ROOTS = [
    DATA_ROOT / "niederschlag" / "meteofrance",
    DATA_ROOT / "temperatur" / "meteofrance",
    DATA_ROOT / "wind" / "meteofrance",
    DATA_ROOT / "luftfeuchte" / "meteofrance",
    DATA_ROOT / "schneehoehe" / "meteofrance",
]


def main():
    # id -> altitude from refreshed meta
    alt_by_id = {}
    name_by_id = {}
    for meta_path in RAW_DIR.glob("*_meta.json"):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        sid = str(meta.get("id", ""))
        if sid and meta.get("altitude") is not None:
            alt_by_id[sid] = meta["altitude"]
            name_by_id[sid] = meta.get("name", "")

    # also build name -> altitude for matching output folders
    alt_by_name = {}
    for sid, name in name_by_id.items():
        clean = "".join(c if c.isalnum() or c in " -_" else "_" for c in name).replace(" ", "_")
        alt_by_name[clean] = alt_by_id[sid]
        alt_by_name[name] = alt_by_id[sid]

    updated = 0
    skipped = 0
    for root in OUTPUT_ROOTS:
        if not root.exists():
            continue
        for station_dir in root.iterdir():
            if not station_dir.is_dir():
                continue
            alt = alt_by_name.get(station_dir.name)
            if alt is None:
                skipped += 1
                continue
            for jf in station_dir.glob("*.json"):
                with open(jf, encoding="utf-8") as f:
                    data = json.load(f)
                sm = data.get("station_metadata")
                if not isinstance(sm, dict):
                    continue
                if sm.get("altitude") == alt:
                    continue
                sm["altitude"] = alt
                with open(jf, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
                updated += 1
    print(f"Patched altitude in {updated} JSON files, {skipped} station folders unmatched")


if __name__ == "__main__":
    main()
