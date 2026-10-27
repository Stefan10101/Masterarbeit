#!/usr/bin/env python3
"""Slope / aspect helpers and shared terrain-cache attachment."""

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


CORE_COVARIATES = (
    "elev",
    "smooth_elev",
    "slope",
    "northness",
    "tpi",
    "clc",
    "dist_to_coast",
    "signed_dist_to_ridge",
)

# Map config key -> column name(s) in the cache
_COVARIATE_COLS = {
    "elev": ["elev"],
    "smooth_elev": ["smooth_elev"],
    "slope": ["slope"],
    "northness": ["northness", "sinasp", "cosasp"],
    "tpi": ["tpi"],
    "clc": ["clc_level2"],
    "dist_to_coast": ["dist_to_coast"],
    "signed_dist_to_ridge": ["signed_dist_to_ridge"],
}


def enabled_covariates(cfg: dict) -> list[str]:
    """Return cache column names enabled by the method's covariates block.

    Missing block or missing key defaults to True (full core set).
    """
    block = cfg.get("covariates") or {}
    cols: list[str] = []
    for key in CORE_COVARIATES:
        if block.get(key, True):
            cols.extend(_COVARIATE_COLS[key])
    return cols


def attach_enabled_terrain(stations: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Merge only the covariates enabled in cfg from the shared station cache."""
    return attach_shared_terrain(stations, enabled_covariates(cfg))


def attach_shared_terrain(
    stations: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Merge preferred static features from Source/terrain/station_features.parquet.

    Cache values overwrite any existing columns of the same name.
    Falls back silently if the cache is missing or a column is absent.
    """
    from paths import get_station_terrain_path

    path = get_station_terrain_path()
    if not path.exists():
        return stations
    cache = pd.read_parquet(path)
    if "station_name" not in cache.columns:
        return stations
    want = columns or [
        "elev", "smooth_elev", "tpi", "slope", "aspect",
        "sinasp", "cosasp", "northness",
        "dist_to_coast", "signed_dist_to_ridge", "clc_level2",
    ]
    keep = [c for c in want if c in cache.columns]
    if not keep:
        return stations
    cache = cache[["station_name"] + keep].drop_duplicates("station_name")
    out = stations.drop(columns=[c for c in keep if c in stations.columns], errors="ignore")
    return out.merge(cache, on="station_name", how="left")
