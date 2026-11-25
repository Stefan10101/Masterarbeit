"""Shared evaluation metrics for all interpolation methods.

Single source of truth for the scores reported on the held-out × test set.

Core scores (always):
  n, me (bias), mae, rmse, nse, kge, ccc, r (Pearson)
  rmse_std = rmse / std(held-out × test observations)

Precip / snow extras (when trace is supplied):
  rmse_wet, freq_bias, pod, far, ets

VARIABLE_STD is the 'test' subset of variable_profile.csv
(Alpine Convention held-out stations × test year-quarters).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# (time_resolution, variable) -> std of held-out × test observations
# Source: variable_profile.csv, subset == "test"
VARIABLE_STD: Dict[Tuple[str, str], float] = {
    # monthly
    ("monthly", "precip_sum"): 75.990673,
    ("monthly", "temp_mean"): 7.564727,
    ("monthly", "temp_min"): 8.007600,
    ("monthly", "temp_max"): 8.758920,
    ("monthly", "wind_mean"): 1.087280,
    ("monthly", "wind_max"): 4.262432,
    ("monthly", "rh_mean"): 8.764337,
    ("monthly", "snow_mean"): 40.387076,
    ("monthly", "snow_max"): 52.037970,
    ("monthly", "snow_min"): 33.662175,
    ("monthly", "slr"): 7.746967,
    ("monthly", "slr_min"): 5.559926,
    ("monthly", "slr_max"): 16.334893,
    ("monthly", "swe"): 122.165629,
    ("monthly", "swe_min"): 106.784116,
    ("monthly", "swe_max"): 138.746238,
    # weekly
    ("weekly", "precip_sum"): 28.688116,
    ("weekly", "temp_mean"): 7.922076,
    ("weekly", "temp_min"): 7.677890,
    ("weekly", "temp_max"): 8.867708,
    ("weekly", "wind_mean"): 1.268604,
    ("weekly", "wind_max"): 3.785795,
    ("weekly", "rh_mean"): 13.189455,
    ("weekly", "snow_mean"): 40.839105,
    ("weekly", "snow_max"): 44.679241,
    ("weekly", "snow_min"): 37.497221,
    ("weekly", "slr"): 8.262305,
    ("weekly", "slr_min"): 5.844622,
    ("weekly", "slr_max"): 15.238005,
    ("weekly", "swe"): 122.601491,
    ("weekly", "swe_min"): 118.987323,
    ("weekly", "swe_max"): 126.129536,
    # daily
    ("daily", "precip_sum"): 8.155106,
    ("daily", "temp_mean"): 8.204560,
    ("daily", "temp_min"): 7.454815,
    ("daily", "temp_max"): 9.355584,
    ("daily", "wind_mean"): 1.717758,
    ("daily", "wind_max"): 3.048067,
    ("daily", "rh_mean"): 17.956040,
    ("daily", "snow_mean"): 40.600591,
    ("daily", "snow_max"): 41.456558,
    ("daily", "snow_min"): 39.736769,
    ("daily", "slr"): 8.778677,
    ("daily", "slr_min"): 7.178798,
    ("daily", "slr_max"): 13.711065,
    ("daily", "swe"): 122.319153,
    ("daily", "swe_min"): 122.319153,
    ("daily", "swe_max"): 122.319153,
    # six_hourly
    ("six_hourly", "precip_sum"): 2.985654,
    ("six_hourly", "temp_mean"): 8.604733,
    ("six_hourly", "temp_min"): 8.151051,
    ("six_hourly", "temp_max"): 9.072619,
    ("six_hourly", "wind_mean"): 2.013221,
    ("six_hourly", "wind_max"): 2.688238,
    ("six_hourly", "rh_mean"): 21.040521,
    ("six_hourly", "snow_mean"): 40.112581,
    ("six_hourly", "snow_max"): 40.442554,
    ("six_hourly", "snow_min"): 39.780267,
    ("six_hourly", "slr"): 10.116055,
    ("six_hourly", "slr_min"): 10.116055,
    ("six_hourly", "slr_max"): 10.116055,
    ("six_hourly", "swe"): 122.319153,
    ("six_hourly", "swe_min"): 122.319153,
    ("six_hourly", "swe_max"): 122.319153,
    # half_hourly (not in current profile; kept from previous table)
    ("half_hourly", "temp_mean"): 8.351703,
    ("half_hourly", "precip_sum"): 0.407750,
    ("half_hourly", "wind_mean"): 2.028267,
    ("half_hourly", "rh_mean"): 21.074183,
    ("half_hourly", "snow_mean"): 34.916190,
}

# Wet-day / presence trace (mm or equivalent) used for frequency scores
TRACE: Dict[str, Dict[str, float]] = {
    "precip_sum": {"half_hourly": 0.05, "daily": 0.1, "weekly": 0.7, "monthly": 1.0},
    "snow_mean": {"half_hourly": 0.5, "daily": 0.5, "weekly": 1.0, "monthly": 1.0},
}

_VAR_ALIAS = {
    "temperature": "temp_mean",
    "precipitation": "precip_sum",
    "wind_speed": "wind_mean",
    "relative_humidity": "rh_mean",
    "snow_height": "snow_mean",
}

_CORE_KEYS = ("n", "me", "mae", "rmse", "rmse_std", "nse", "kge", "ccc", "r")
_PRECIP_KEYS = ("rmse_wet", "freq_bias", "pod", "far", "ets")


def _canon(variable: str) -> str:
    v = variable.lower().strip()
    return _VAR_ALIAS.get(v, v)



def _load_profile_std() -> Dict[Tuple[str, str], float]:
    """Test-subset std from variable_profile.csv. Empty if the file is absent."""
    try:
        from paths import get_source_root
        path = get_source_root() / "profile" / "variable_profile.csv"
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        sub = df[df["subset"] == "test"]
        out = {}
        for _, row in sub.iterrows():
            key = (str(row["time_resolution"]), str(row["variable"]))
            val = row.get("std")
            if pd.notna(val):
                out[key] = float(val)
        return out
    except Exception:
        return {}


_PROFILE_STD = _load_profile_std()


def get_std(variable: str, time_resolution: str, obs=None) -> float:
    """Held-out × test standard deviation for a variable and timescale.
    Falls back to the std of the supplied observations when the table has no entry.
    """
    key = (time_resolution, _canon(variable))
    if key in _PROFILE_STD:
        return _PROFILE_STD[key]
    if key in VARIABLE_STD:
        return VARIABLE_STD[key]
    if obs is not None:
        o = np.asarray(obs, dtype=float)
        o = o[np.isfinite(o)]
        if len(o) > 1:
            return float(np.std(o, ddof=0))
    return float("nan")


def get_trace(variable: str, time_resolution: str) -> Optional[float]:
    """Wet-day threshold for precip/snow, or None for continuous variables."""
    return TRACE.get(_canon(variable), {}).get(time_resolution)


def rmse_over_std(rmse: float, variable: str, time_resolution: str, obs=None) -> float:
    """Normalized RMSE: RMSE / std(held-out × test observations)."""
    if rmse is None or not (rmse == rmse):
        return float("nan")
    s = get_std(variable, time_resolution, obs)
    if s is None or not (s == s) or s <= 0:
        return float("nan")
    return float(rmse) / s


def _empty(keys) -> dict:
    return {k: np.nan for k in keys}


def compute_metrics(
    obs,
    pred,
    variable: Optional[str] = None,
    time_resolution: Optional[str] = None,
    trace: Optional[float] = None,
) -> dict:
    """Full evaluation scores on a pair of observation / prediction arrays.

    Always returns the core set. Adds rmse_std when variable and
    time_resolution are given. Adds precip/snow frequency scores when
    trace is given (or can be looked up from variable + time_resolution).
    """
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    n = int(len(o))

    out = _empty(_CORE_KEYS)
    out["n"] = n
    if n < 2:
        return out

    err = p - o
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    me = float(np.mean(err))
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    nse = float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else np.nan

    so = float(np.std(o))
    sp = float(np.std(p))
    if so > 1e-8 and sp > 1e-8:
        r = float(np.corrcoef(o, p)[0, 1])
    else:
        r = np.nan
    alpha = float(sp / so) if so > 0 else np.nan
    beta = float(np.mean(p) / np.mean(o)) if np.mean(o) != 0 else np.nan
    kge = (
        float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
        if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta)
        else np.nan
    )

    mx, my = float(o.mean()), float(p.mean())
    sxx = float(np.mean((o - mx) ** 2))
    syy = float(np.mean((p - my) ** 2))
    sxy = float(np.mean((o - mx) * (p - my)))
    den = sxx + syy + (mx - my) ** 2
    ccc = float(2.0 * sxy / den) if den > 0 else np.nan

    out.update(
        me=me,
        mae=mae,
        rmse=rmse,
        nse=nse,
        kge=kge,
        ccc=ccc,
        r=r,
    )

    if variable is not None and time_resolution is not None:
        out["rmse_std"] = rmse_over_std(rmse, variable, time_resolution, o)
        if trace is None:
            trace = get_trace(variable, time_resolution)

    if trace is not None:
        out.update(precip_scores(o, p, float(trace)))

    return out


def precip_scores(obs, pred, trace: float) -> dict:
    """Wet-day / frequency scores for precip and snow."""
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    n = len(o)
    if n == 0:
        return _empty(_PRECIP_KEYS)

    ow = o >= trace
    pw = p >= trace
    hits = int(np.sum(ow & pw))
    fa = int(np.sum(~ow & pw))
    miss = int(np.sum(ow & ~pw))

    wet = ow
    rmse_wet = float(np.sqrt(np.mean((p[wet] - o[wet]) ** 2))) if wet.any() else np.nan
    freq_bias = float(pw.mean() / ow.mean()) if ow.any() else np.nan
    pod = float(hits / (hits + miss)) if (hits + miss) else np.nan
    far = float(fa / (hits + fa)) if (hits + fa) else np.nan

    ets = np.nan
    denom = hits + fa + miss
    if n and denom:
        hits_rand = (hits + miss) * (hits + fa) / n
        if (denom - hits_rand) != 0:
            ets = float((hits - hits_rand) / (denom - hits_rand))

    return {
        "rmse_wet": rmse_wet,
        "freq_bias": freq_bias,
        "pod": pod,
        "far": far,
        "ets": ets,
    }


def add_rmse_std(metrics: dict, variable: str, time_resolution: str) -> dict:
    """Return a copy of metrics with an extra 'rmse_std' key."""
    out = dict(metrics)
    out["rmse_std"] = rmse_over_std(metrics.get("rmse"), variable, time_resolution)
    return out


def format_eval(metrics: dict, prefix: str = "  ") -> str:
    """One-line summary of the core evaluation scores."""
    def f(key, fmt=".3f"):
        v = metrics.get(key, np.nan)
        if v is None or not (v == v):
            return "nan"
        return format(float(v), fmt)

    line = (
        f"{prefix}n={metrics.get('n', 0):6d}  "
        f"ME={f('me')}  MAE={f('mae')}  RMSE={f('rmse')}  "
        f"RMSE/std={f('rmse_std')}  NSE={f('nse')}  "
        f"KGE={f('kge')}  CCC={f('ccc')}  r={f('r')}"
    )
    if "rmse_wet" in metrics and metrics.get("rmse_wet") == metrics.get("rmse_wet"):
        line += (
            f"  |  wetRMSE={f('rmse_wet')}  freq={f('freq_bias')}  "
            f"POD={f('pod')}  FAR={f('far')}  ETS={f('ets')}"
        )
    return line


def print_eval(metrics: dict, prefix: str = "  ") -> None:
    """Print the standard evaluation line."""
    print(format_eval(metrics, prefix=prefix))
