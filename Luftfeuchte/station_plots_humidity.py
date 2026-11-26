#!/usr/bin/env python3
"""Relative humidity station plots. Values from the daily aggregate."""
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from shared.plotting.variable_station_plots import run

if __name__ == "__main__":
    run({
        "code": "rh_mean",
        "overview": "relative_humidity",
        "folder": "Luftfeuchte",
        "label": "Relative humidity",
        "unit": "%",
        "master_col": "relative_humidity",
        "has_qc": True,
    })
