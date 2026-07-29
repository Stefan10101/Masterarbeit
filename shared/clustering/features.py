#!/usr/bin/env python3
"""
features.py
Feature extraction for meteorological regime clustering.

All features are computed from the stations that report a valid value
at a given timestamp. Station terrain attributes come from the rich
metadata table (elev_dem, slope, aspect, ...).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cyclic_encode(values: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sin, cos) encoding of a cyclic variable."""
    angle = 2.0 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that returns NaN on insufficient data."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return np.nan
    a = a[mask]
    b = b[mask]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _spatial_autocorr(
    values: np.ndarray,
    coords: np.ndarray,
    k: int = 5,
) -> float:
    """
    Approximate global spatial autocorrelation (Moran-like).
    For each station compute its value deviation and the deviation of the
    mean of its k nearest neighbours; then correlate those two series.
    """
    n = len(values)
    if n < k + 2:
        return np.nan
    tree = cKDTree(coords)
    _, idxs = tree.query(coords, k=min(k + 1, n))
    global_mean = np.nanmean(values)
    devs = []
    lags = []
    for i in range(n):
        if not np.isfinite(values[i]):
            continue
        neigh = idxs[i, 1:]
        local_vals = values[neigh]
        local_vals = local_vals[np.isfinite(local_vals)]
        if len(local_vals) < 3:
            continue
        lag = np.mean(local_vals)
        devs.append(values[i] - global_mean)
        lags.append(lag - global_mean)
    if len(devs) < 5:
        return np.nan
    return _safe_corr(np.asarray(devs), np.asarray(lags))


def _elev_gradient(
    values: np.ndarray,
    elev: np.ndarray,
    pct: float = 0.20,
) -> float:
    """Mean of highest pct stations minus mean of lowest pct stations."""
    n = len(values)
    if n < 10:
        return np.nan
    order = np.argsort(elev)
    n_edge = max(3, int(n * pct))
    low = values[order[:n_edge]]
    high = values[order[-n_edge:]]
    if np.isfinite(low).sum() < 3 or np.isfinite(high).sum() < 3:
        return np.nan
    return float(np.nanmean(high) - np.nanmean(low))


def _high_low_ratio(
    values: np.ndarray,
    elev: np.ndarray,
    pct: float = 0.20,
) -> float:
    n = len(values)
    if n < 10:
        return np.nan
    order = np.argsort(elev)
    n_edge = max(3, int(n * pct))
    low = values[order[:n_edge]]
    high = values[order[-n_edge:]]
    low_m = np.nanmean(low)
    high_m = np.nanmean(high)
    if not np.isfinite(low_m) or abs(low_m) < 1e-6:
        return np.nan
    return float(high_m / low_m)


# ---------------------------------------------------------------------------
# Per-variable feature extractors
# Each returns a dict {feature_name: scalar}
# ---------------------------------------------------------------------------

def features_temperature(
    values: np.ndarray,
    elev: np.ndarray,
    coords: np.ndarray,
    time: pd.Timestamp,
    meta_extra: Optional[pd.DataFrame] = None,
    prev_values: Optional[np.ndarray] = None,
    prev_elev: Optional[np.ndarray] = None,
    prev_coords: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    hour = time.hour + time.minute / 60.0
    doy = time.dayofyear
    hour_sin, hour_cos = _cyclic_encode(np.array([hour]), 24.0)
    doy_sin, doy_cos = _cyclic_encode(np.array([doy]), 365.25)

    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "elev_corr": _safe_corr(values, elev),
        "elev_gradient": _elev_gradient(values, elev),
        "spatial_autocorr": _spatial_autocorr(values, coords),
        "hour_sin": float(hour_sin[0]),
        "hour_cos": float(hour_cos[0]),
        "doy_sin": float(doy_sin[0]),
        "doy_cos": float(doy_cos[0]),
    }


def features_precipitation(
    values: np.ndarray,
    elev: np.ndarray,
    coords: np.ndarray,
    time: pd.Timestamp,
    meta_extra: Optional[pd.DataFrame] = None,
    prev_values: Optional[np.ndarray] = None,
    prev_elev: Optional[np.ndarray] = None,
    prev_coords: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    wet_mask = values > 0.01
    n_wet = int(wet_mask.sum())
    wet_vals = values[wet_mask]
    wet_elev = elev[wet_mask]

    doy = time.dayofyear
    doy_sin, doy_cos = _cyclic_encode(np.array([doy]), 365.25)

    feat = {
        "wet_fraction": float(n_wet / max(len(values), 1)),
        "wet_mean": float(np.nanmean(wet_vals)) if n_wet > 0 else 0.0,
        "elev_corr_all": _safe_corr(values, elev),
        "elev_corr_wet": _safe_corr(wet_vals, wet_elev) if n_wet >= 5 else np.nan,
        "cv_wet": float(np.nanstd(wet_vals) / (np.nanmean(wet_vals) + 1e-6)) if n_wet >= 3 else np.nan,
        "spatial_autocorr": _spatial_autocorr(values, coords),
        "domain_max": float(np.nanmax(values)),
        "doy_sin": float(doy_sin[0]),
        "doy_cos": float(doy_cos[0]),
    }
    return feat


def features_wind(
    values: np.ndarray,
    elev: np.ndarray,
    coords: np.ndarray,
    time: pd.Timestamp,
    meta_extra: Optional[pd.DataFrame] = None,
    prev_values: Optional[np.ndarray] = None,
    prev_elev: Optional[np.ndarray] = None,
    prev_coords: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    hour = time.hour + time.minute / 60.0
    doy = time.dayofyear
    hour_sin, hour_cos = _cyclic_encode(np.array([hour]), 24.0)
    doy_sin, doy_cos = _cyclic_encode(np.array([doy]), 365.25)

    feat = {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "elev_corr": _safe_corr(values, elev),
        "high_low_ratio": _high_low_ratio(values, elev),
        "spatial_autocorr": _spatial_autocorr(values, coords),
        "hour_sin": float(hour_sin[0]),
        "hour_cos": float(hour_cos[0]),
        "doy_sin": float(doy_sin[0]),
        "doy_cos": float(doy_cos[0]),
    }

    # lag features (only meaningful when previous field is supplied)
    if prev_values is not None and len(prev_values) == len(values):
        feat["lag_elev_corr"] = _safe_corr(prev_values, prev_elev) if prev_elev is not None else np.nan
        feat["lag_mean"] = float(np.nanmean(prev_values))
        # spatial persistence: correlation of current field with previous field
        # (stations must be aligned – caller guarantees same station order)
        feat["field_persistence"] = _safe_corr(values, prev_values)
    else:
        feat["lag_elev_corr"] = np.nan
        feat["lag_mean"] = np.nan
        feat["field_persistence"] = np.nan

    return feat


def features_humidity(
    values: np.ndarray,
    elev: np.ndarray,
    coords: np.ndarray,
    time: pd.Timestamp,
    meta_extra: Optional[pd.DataFrame] = None,
    prev_values: Optional[np.ndarray] = None,
    prev_elev: Optional[np.ndarray] = None,
    prev_coords: Optional[np.ndarray] = None,
    temp_values: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    hour = time.hour + time.minute / 60.0
    doy = time.dayofyear
    hour_sin, hour_cos = _cyclic_encode(np.array([hour]), 24.0)
    doy_sin, doy_cos = _cyclic_encode(np.array([doy]), 365.25)

    feat = {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "elev_corr": _safe_corr(values, elev),
        "spatial_autocorr": _spatial_autocorr(values, coords),
        "hour_sin": float(hour_sin[0]),
        "hour_cos": float(hour_cos[0]),
        "doy_sin": float(doy_sin[0]),
        "doy_cos": float(doy_cos[0]),
    }
    if temp_values is not None and len(temp_values) == len(values):
        feat["temp_corr"] = _safe_corr(values, temp_values)
    else:
        feat["temp_corr"] = np.nan
    return feat


def features_snow_height(
    values: np.ndarray,
    elev: np.ndarray,
    coords: np.ndarray,
    time: pd.Timestamp,
    meta_extra: Optional[pd.DataFrame] = None,
    prev_values: Optional[np.ndarray] = None,
    prev_elev: Optional[np.ndarray] = None,
    prev_coords: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    meta_extra must contain columns aligned with the stations:
    aspect (degrees), slope.
    """
    doy = time.dayofyear
    doy_sin, doy_cos = _cyclic_encode(np.array([doy]), 365.25)

    snow_mask = values > 0.01
    n_snow = int(snow_mask.sum())

    feat = {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "elev_corr": _safe_corr(values, elev),
        "zero_fraction": float(1.0 - n_snow / max(len(values), 1)),
        "spatial_autocorr": _spatial_autocorr(values, coords),
        "elev_gradient": _elev_gradient(values, elev),
        "doy_sin": float(doy_sin[0]),
        "doy_cos": float(doy_cos[0]),
    }

    if meta_extra is not None and "aspect" in meta_extra.columns and "slope" in meta_extra.columns:
        aspect = meta_extra["aspect"].values
        slope = meta_extra["slope"].values
        # circular aspect correlations
        aspect_rad = np.deg2rad(aspect)
        feat["aspect_cos_corr"] = _safe_corr(values, np.cos(aspect_rad))
        feat["aspect_sin_corr"] = _safe_corr(values, np.sin(aspect_rad))
        if n_snow >= 5:
            feat["mean_slope_snow"] = float(np.nanmean(slope[snow_mask]))
        else:
            feat["mean_slope_snow"] = np.nan
    else:
        feat["aspect_cos_corr"] = np.nan
        feat["aspect_sin_corr"] = np.nan
        feat["mean_slope_snow"] = np.nan

    return feat


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FEATURE_FUNCS = {
    "temperature": features_temperature,
    "precipitation": features_precipitation,
    "wind_speed": features_wind,
    "relative_humidity": features_humidity,
    "snow_height": features_snow_height,
}

# Any column name that may appear in an aggregated parquet → canonical key
NAME_MAP = {
    # temperature
    "temperature": "temperature",
    "temp": "temperature",
    "temp_mean": "temperature",
    "temp_min": "temperature",
    "temp_max": "temperature",
    "TL": "temperature",
    # precipitation
    "precipitation": "precipitation",
    "precip": "precipitation",
    "precip_sum": "precipitation",
    "rain": "precipitation",
    "RR": "precipitation",
    # wind
    "wind_speed": "wind_speed",
    "wind": "wind_speed",
    "wind_mean": "wind_speed",
    "wind_max": "wind_speed",
    "FF": "wind_speed",
    # humidity
    "relative_humidity": "relative_humidity",
    "humidity": "relative_humidity",
    "rh_mean": "relative_humidity",
    "rh": "relative_humidity",
    "RF": "relative_humidity",
    # snow
    "snow_height": "snow_height",
    "snow": "snow_height",
    "snow_mean": "snow_height",
    "snow_min": "snow_height",
    "snow_max": "snow_height",
    "SH": "snow_height",
}

# Preferred column to use when several candidates exist for the same variable
PREFERRED_COLUMNS = {
    "temperature": ["temp_mean", "temperature", "temp"],
    "precipitation": ["precip_sum", "precipitation", "precip"],
    "wind_speed": ["wind_mean", "wind_speed", "wind"],
    "relative_humidity": ["rh_mean", "relative_humidity", "humidity", "rh"],
    "snow_height": ["snow_mean", "snow_height", "snow"],
}


def get_feature_func(var_name: str):
    key = NAME_MAP.get(var_name, var_name)
    if key not in FEATURE_FUNCS:
        raise KeyError(
            f"No feature function registered for variable '{var_name}' "
            f"(mapped to '{key}'). Known: {list(FEATURE_FUNCS)}"
        )
    return FEATURE_FUNCS[key]
