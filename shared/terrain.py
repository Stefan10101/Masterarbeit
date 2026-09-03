#!/usr/bin/env python3
"""Slope / aspect helpers for GAM and CNN."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aspect_trig(aspect_deg) -> tuple[np.ndarray, np.ndarray]:
    rad = np.deg2rad(np.asarray(aspect_deg, dtype=np.float64))
    return np.sin(rad), np.cos(rad)


def slope_aspect_from_dem(dem: np.ndarray, dx: float, dy: float):
    gy, gx = np.gradient(np.asarray(dem, dtype=np.float64), dy, dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = (np.degrees(np.arctan2(-gx, gy)) + 360.0) % 360.0
    return slope.astype(np.float32), aspect.astype(np.float32)


def merge_station_terrain(stations: pd.DataFrame) -> pd.DataFrame:
    """Attach slope / sinasp / cosasp if the station table has them."""
    out = stations.copy()
    slope_col = next((c for c in ("slope", "slope_deg", "hangneigung") if c in out.columns), None)
    asp_col = next((c for c in ("aspect", "aspect_deg", "exposition") if c in out.columns), None)
    if slope_col and "slope" not in out.columns:
        out["slope"] = out[slope_col]
    if asp_col:
        s, c = aspect_trig(out[asp_col])
        out["sinasp"] = s
        out["cosasp"] = c
    if "slope" not in out.columns:
        out["slope"] = 0.0
    if "sinasp" not in out.columns:
        out["sinasp"] = 0.0
        out["cosasp"] = 1.0
    return out


def attach_terrain_to_panel(panel: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    extra = merge_station_terrain(stations)
    keep = [c for c in ("station_name", "slope", "sinasp", "cosasp") if c in extra.columns]
    extra = extra[keep].drop_duplicates("station_name")
    out = panel.merge(extra, on="station_name", how="left")
    for c in ("slope", "sinasp", "cosasp"):
        if c not in out.columns:
            out[c] = 0.0 if c != "cosasp" else 1.0
        else:
            fill = 1.0 if c == "cosasp" else 0.0
            out[c] = out[c].fillna(fill)
    return out
