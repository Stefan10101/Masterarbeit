#!/usr/bin/env python3
"""Snow height station plots. Values from the daily aggregate."""
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from shared.plotting.variable_station_plots import run

if __name__ == "__main__":
    run({
        "code": "snow_mean",
        "overview": "snow_height",
        "folder": "Schneehoehe",
        "label": "Snow height",
        "unit": "cm",
        "master_col": "snow_height",
        "has_qc": True,
    })
