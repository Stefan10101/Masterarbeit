#!/usr/bin/env python3
"""Panel + metrics + cheap time cuts for Frei. Reuses Kriging panel loader."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd

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
precip_trace = _kd.precip_trace

SEASONAL4_MONTHS = (2, 5, 8, 11)


def load_frei_config(path: Path | None = None) -> dict:
    return _kd.load_config(path or Path(__file__).resolve().parent / "config.yaml")


def attach_temperature(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Left-join temp_mean at the same station and timestamp."""
    if "temp_mean" in panel.columns:
        return panel
    tpanel = load_panel(cfg, "temp_mean")[["station_name", "time", "temp_mean"]]
    return panel.merge(tpanel, on=["station_name", "time"], how="left")


def subset_times(panel: pd.DataFrame, spec) -> pd.DataFrame:
    """
    spec: None/'all' | 'seasonal4' | 'YYYY-MM,YYYY-MM,...' | list of those strings.
    seasonal4 keeps Feb/May/Aug/Nov in whatever years the panel already has.
    """
    if spec is None or spec == "" or spec == "all":
        return panel
    if isinstance(spec, (list, tuple)):
        if len(spec) == 1 and spec[0] in ("all", "seasonal4"):
            spec = spec[0]
        else:
            keys = {str(s).strip() for s in spec}
            stamp = pd.to_datetime(panel["time"]).dt.strftime("%Y-%m")
            return panel.loc[stamp.isin(keys)].copy()
    spec = str(spec).strip().lower()
    if spec in ("all", "none"):
        return panel
    times = pd.to_datetime(panel["time"])
    if spec == "seasonal4":
        return panel.loc[times.dt.month.isin(SEASONAL4_MONTHS)].copy()
    keys = {s.strip() for s in spec.split(",") if s.strip()}
    if keys == {"seasonal4"}:
        return panel.loc[times.dt.month.isin(SEASONAL4_MONTHS)].copy()
    stamp = times.dt.strftime("%Y-%m")
    return panel.loc[stamp.isin(keys)].copy()


def subset_years(panel: pd.DataFrame, years) -> pd.DataFrame:
    if not years:
        return panel
    if isinstance(years, str):
        years = [y.strip() for y in years.split(",") if y.strip()]
    want = {int(y) for y in years}
    return panel.loc[pd.to_datetime(panel["time"]).dt.year.isin(want)].copy()


def subset_splits(panel: pd.DataFrame, splits) -> pd.DataFrame:
    if splits is None or splits == "all" or splits == {"all"}:
        return panel
    if isinstance(splits, str):
        splits = {s.strip() for s in splits.split(",") if s.strip()}
    if not splits or splits == {"all"}:
        return panel
    return panel.loc[panel["split"].isin(splits)].copy()


def compute_block(cfg: dict) -> dict:
    block = dict(cfg.get("frei", {}).get("compute", {}) or {})
    block.setdefault("tune_mode", "coord")
    block.setdefault("tune_months", "seasonal4")
    block.setdefault("tune_folds", 5)
    block.setdefault("produce_splits", "all")
    return block
