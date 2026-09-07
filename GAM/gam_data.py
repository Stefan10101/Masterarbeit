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
precip_trace = _kd.precip_trace


def load_panel(cfg, var):
    panel = _kd.load_panel(cfg, var)
    try:
        import pandas as pd
        from paths import get_domain_stations_path, get_stations_metadata_path
        from shared.terrain import attach_terrain_to_panel
        sta_method = cfg.get("paths", {}).get("stations_from", "RFSI")
        sta_path = get_domain_stations_path(sta_method, "full")
        sta = pd.read_parquet(sta_path)
        meta = get_stations_metadata_path()
        if meta.exists():
            extra = pd.read_csv(meta)
            name_col = next((c for c in ("station_name", "name", "Station") if c in extra.columns), None)
            if name_col is not None:
                extra = extra.rename(columns={name_col: "station_name"})
                keep = [c for c in extra.columns if c in
                        ("station_name", "slope", "slope_deg", "hangneigung",
                         "aspect", "aspect_deg", "exposition")]
                extra = extra[keep].drop_duplicates("station_name")
                sta = sta.merge(extra, on="station_name", how="left", suffixes=("", "_meta"))
        panel = attach_terrain_to_panel(panel, sta)
    except Exception as exc:
        print(f"  terrain attach failed ({type(exc).__name__}: {exc}); using flat defaults", flush=True)
        for c, fill in (("slope", 0.0), ("sinasp", 0.0), ("cosasp", 1.0)):
            if c not in panel.columns:
                panel[c] = fill
    return panel


def load_gam_config(path: Path | None = None) -> dict:
    return _kd.load_config(path or Path(__file__).resolve().parent / "config.yaml")
