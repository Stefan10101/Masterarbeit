#!/usr/bin/env python3
"""SWE station plots. Values from the daily aggregate (derived variable)."""
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from shared.plotting.variable_station_plots import run

if __name__ == "__main__":
    run({
        "code": "swe",
        "folder": "SWE",
        "label": "SWE",
        "unit": "mm",
        "has_qc": False,
    })
