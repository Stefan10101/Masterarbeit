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

# (time_resolution, variable) -> std of held-out × test observations
# Source: variable_profile.csv, subset == "test"
VARIABLE_STD: Dict[Tuple[str, str], float] = {
    # monthly
    ("monthly", "temp_mean"): 7.080867,
    ("monthly", "precip_sum"): 80.553442,
    ("monthly", "wind_mean"): 1.208947,
    ("monthly", "rh_mean"): 8.861033,
    ("monthly", "snow_mean"): 37.600007,
    # weekly
    ("weekly", "temp_mean"): 7.461013,
    ("weekly", "precip_sum"): 30.415063,
    ("weekly", "wind_mean"): 1.308422,
    ("weekly", "rh_mean"): 11.936912,
    ("weekly", "snow_mean"): 35.574195,
    # daily
    ("daily", "temp_mean"): 7.700932,
    ("daily", "precip_sum"): 8.773573,
    ("daily", "wind_mean"): 1.573367,
    ("daily", "rh_mean"): 15.856649,
    ("daily", "snow_mean"): 34.718104,
    # half_hourly
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


def get_std(variable: str, time_resolution: str) -> float:
    """Held-out × test standard deviation for a variable and timescale."""
    key = (time_resolution, _canon(variable))
    if key not in VARIABLE_STD:
        raise KeyError(
            f"no evaluation std for {variable!r} / {time_resolution!r}. "
            f"Known: {sorted(VARIABLE_STD)}"
        )
    return VARIABLE_STD[key]


def get_trace(variable: str, time_resolution: str) -> Optional[float]:
    """Wet-day threshold for precip/snow, or None for continuous variables."""
    return TRACE.get(_canon(variable), {}).get(time_resolution)


def rmse_over_std(rmse: float, variable: str, time_resolution: str) -> float:
    """Normalized RMSE: RMSE / std(held-out × test observations)."""
    if rmse is None or not (rmse == rmse):
        return float("nan")
    s = get_std(variable, time_resolution)
    if s <= 0:
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
        out["rmse_std"] = rmse_over_std(rmse, variable, time_resolution)
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
