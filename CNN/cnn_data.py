#!/usr/bin/env python3
"""Panel + metrics for CNN. Reuses Kriging panel loader."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

_spec = importlib.util.spec_from_file_location(
    "kriging_data", CODE_DIR / "Kriging" / "kriging_data.py"
)
_kd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kd)

load_config = _kd.load_config
load_panel = _kd.load_panel
compute_metrics = _kd.compute_metrics
print_split_metrics = _kd.print_split_metrics
data_sources = _kd.data_sources
clc_group = _kd.clc_group
precip_trace = _kd.precip_trace


def load_cnn_config(path: Path | None = None) -> dict:
    return _kd.load_config(path or Path(__file__).resolve().parent / "config.yaml")
