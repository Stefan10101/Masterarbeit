#!/usr/bin/env python3
"""Temperature station plots. Values from the daily aggregate."""
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from shared.plotting.variable_station_plots import run

if __name__ == "__main__":
    run({
        "code": "temp_mean",
        "overview": "temperature",
        "folder": "Temperatur",
        "label": "Temperature",
        "unit": "°C",
        "master_col": "temperature",
        "has_qc": True,
    })
