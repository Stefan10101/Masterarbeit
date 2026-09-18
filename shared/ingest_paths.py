#!/usr/bin/env python3
"""Resolve Variables/ + QC/ with fallback to the pre-migration tree."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from paths import DATA_ROOT, get_raw_data_dir, get_variables_root

VAR_DIR_NAMES = {
    "temperature": "Temperatur",
    "precipitation": "Niederschlag",
    "wind_speed": "Wind",
    "relative_humidity": "Luftfeuchte",
    "snow_height": "Schneehoehe",
}

NETWORK_ALIASES = {
    "dwd": "DWD",
    "zamg": "ZAMG",
    "meteosuisse": "Meteosuisse",
    "lwd": "LWD",
    "hydrot": "HydroT",
    "hydrovo": "HydroVO",
    "suedtirol": "Suedtirol",
}


def _first_existing(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def variable_source_dir(var_folder: str) -> Path:
    """Raw provider JSON tree for one variable."""
    return _first_existing(
        get_variables_root() / var_folder,
        DATA_ROOT / var_folder,
    )


def qc_variable_dir(var_folder: str) -> Path:
    """Per-variable QC root under QC/<Var>/."""
    return get_raw_data_dir() / var_folder


def pipeline_output_dirs(var_folder: str) -> dict[str, Path]:
    root = qc_variable_dir(var_folder)
    dirs = {
        "root": root,
        "full": root / "full_2020_2025",
        "qc": root / "full_2020_2025_qc",
        "rejected": root / "rejected_stations",
        "stats": root / "statistics",
        "plots": DATA_ROOT / "Plots" / var_folder,
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def list_network_jsons(source_root: Path, network_names: list[str]) -> list[Path]:
    """rglob JSON under provider folders, case-insensitive on the folder name."""
    if not source_root.exists():
        return []
    actual = {p.name.lower(): p for p in source_root.iterdir() if p.is_dir()}
    files: list[Path] = []
    for net in network_names:
        folder = actual.get(net.lower())
        if folder is None:
            continue
        files.extend(folder.rglob("*.json"))
    return files


def canonical_network_name(folder_name: str) -> str:
    return NETWORK_ALIASES.get(folder_name.lower(), folder_name)
