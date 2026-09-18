#!/usr/bin/env python3
"""Contracts for slr / swe / snow_density and the existing two-step fields."""

from __future__ import annotations

import numpy as np

SLR_CLIP = (2.0, 60.0)
DENSITY_CLIP = (50.0, 550.0)

# event / pack presence threshold used by two-step
TRACE = {
    "slr": {"half_hourly": 2.0, "daily": 2.0, "weekly": 2.0, "monthly": 2.0, "seasonal": 2.0},
    "swe": {"half_hourly": 1.0, "daily": 1.0, "weekly": 1.0, "monthly": 1.0, "seasonal": 2.0},
    "snow_density": {"half_hourly": 50.0, "daily": 50.0, "weekly": 50.0, "monthly": 50.0, "seasonal": 50.0},
}


def uses_two_step(var: str) -> bool:
    v = str(var).lower()
    if v in ("slr", "swe", "snow_density"):
        return True
    return v.startswith("precip") or v.startswith("snow")


def needs_temperature(var: str) -> bool:
    v = str(var).lower()
    return v == "slr" or v.startswith("rh")


def extra_keys(var: str, snow_terrain: bool = True) -> list[str]:
    v = str(var).lower()
    keys = []
    if v == "slr":
        keys.append("temp_mean")
    if v in ("swe", "snow_density", "snow_mean", "snow_max", "snow_min") and snow_terrain:
        keys.extend(["slope", "northness"])
    if v.startswith("rh"):
        keys.append("temp_mean")
    return keys


def dry_fill(var: str) -> float:
    """Two-step value when p_event < tau. slr/density stay missing."""
    v = str(var).lower()
    if v in ("slr", "snow_density"):
        return np.nan
    return 0.0


def clip_var(var: str, pred) -> np.ndarray:
    v = str(var).lower()
    out = np.asarray(pred, dtype=np.float64)
    if v == "slr":
        m = np.isfinite(out)
        out = out.copy()
        out[m] = np.clip(out[m], SLR_CLIP[0], SLR_CLIP[1])
        return out
    if v == "snow_density":
        m = np.isfinite(out)
        out = out.copy()
        out[m] = np.clip(out[m], DENSITY_CLIP[0], DENSITY_CLIP[1])
        return out
    if v == "swe" or v.startswith("precip") or v.startswith("snow") or v.startswith("wind"):
        return np.maximum(out, 0.0)
    if v.startswith("rh"):
        return np.clip(out, 0.0, 100.0)
    return out


def derived_regime(var: str) -> str:
    """Cluster-assignment key. Own name until those files exist (fallback params)."""
    v = str(var).lower()
    if v == "slr":
        return "slr"
    if v == "swe":
        return "swe"
    if v == "snow_density":
        return "snow_density"
    return v
