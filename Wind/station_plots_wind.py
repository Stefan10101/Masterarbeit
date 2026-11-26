#!/usr/bin/env python3
"""Wind station plots. Values from the daily aggregate."""
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from shared.plotting.variable_station_plots import run

if __name__ == "__main__":
    run({
        "code": "wind_mean",
        "overview": "wind_speed",
        "folder": "Wind",
        "label": "Wind speed",
        "unit": "m/s",
        "master_col": "wind_speed",
        "has_qc": True,
    })
