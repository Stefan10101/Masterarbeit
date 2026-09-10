#!/usr/bin/env python3
"""Panel + metrics for TPS. Reuses Kriging panel loader."""

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
uses_two_step = _kd.uses_two_step
as_naive_utc = _kd.as_naive_utc
precip_trace = _kd.precip_trace
attach_temperature = _kd.attach_temperature
attach_pack_terrain = _kd.attach_pack_terrain
pack_from_master = _kd.pack_from_master
subset_times = _kd.subset_times
subset_years = _kd.subset_years
subset_splits = _kd.subset_splits


def compute_block(cfg: dict) -> dict:
    block = dict(cfg.get("tps", {}).get("compute", {}) or {})
    block.setdefault("tune_months", "seasonal4")
    block.setdefault("tune_folds", 5)
    return block


def load_tps_config(path: Path | None = None) -> dict:
    return _kd.load_config(path or Path(__file__).resolve().parent / "config.yaml")
