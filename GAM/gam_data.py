#!/usr/bin/env python3
"""Panel + metrics for GAM. Reuses Kriging panel loader."""

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
compute_metrics = _kd.compute_metrics
print_split_metrics = _kd.print_split_metrics
data_sources = _kd.data_sources
clc_group = _kd.clc_group


def load_panel(cfg, var):
    panel = _kd.load_panel(cfg, var)
    try:
        import pandas as pd
        from paths import get_domain_stations_path
        from shared.terrain import attach_terrain_to_panel
        sta = pd.read_parquet(get_domain_stations_path(cfg.get("paths", {}).get("stations_from", "RFSI"), "full"))
        panel = attach_terrain_to_panel(panel, sta)
    except Exception:
        for c, fill in (("slope", 0.0), ("sinasp", 0.0), ("cosasp", 1.0)):
            if c not in panel.columns:
                panel[c] = fill
    return panel


def load_gam_config(path: Path | None = None) -> dict:
    return _kd.load_config(path or Path(__file__).resolve().parent / "config.yaml")
