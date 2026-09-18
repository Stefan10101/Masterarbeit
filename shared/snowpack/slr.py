#!/usr/bin/env python3
"""Event SLR from co-located HS and precipitation.

Window: rolling 6 h on the 30-min QC series.
HN(t) = max(HS(t) - HS(t-6h), 0)   [cm]   net pack rise
P(t)  = sum of precip on (t-6h, t] [mm]
Event iff HN >= 2 cm AND P >= 1 mm.
SLR   = (HN cm * 10) / P mm        dimensionless, clip [2, 60].
Missing when not an event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = pd.Timedelta(hours=6)
STEPS_30MIN = 12  # 6 h / 30 min
MIN_HN_CM = 2.0
MIN_P_MM = 1.0
SLR_CLIP = (2.0, 60.0)


def slr_from_half_hourly(ts: pd.Series, hs_cm: pd.Series, precip_mm: pd.Series) -> pd.DataFrame:
    """Return one row per timestamp with hn_cm, p_6h, slr (NA if no event)."""
    out = pd.DataFrame({"time": pd.to_datetime(ts, utc=True, errors="coerce")})
    out["hs_cm"] = pd.to_numeric(hs_cm, errors="coerce").to_numpy()
    out["precip_mm"] = pd.to_numeric(precip_mm, errors="coerce").to_numpy()
    out = out.dropna(subset=["time"]).sort_values("time")
    out = out.drop_duplicates("time", keep="last")
    if out.empty:
        return out.assign(hn_cm=np.nan, p_6h=np.nan, slr=np.nan)

    # exact 6 h lag only when a sample exists 6 h back (no hidden gap)
    out = out.set_index("time")
    hs_lag = out["hs_cm"].shift(freq=WINDOW)
    aligned = out.join(hs_lag.rename("hs_lag"), how="left")
    # precip sum over the closed 6 h window; require 12 steps present
    p_roll = aligned["precip_mm"].rolling(window=WINDOW, min_periods=STEPS_30MIN).sum()
    n_roll = aligned["precip_mm"].rolling(window=WINDOW, min_periods=1).count()
    aligned["p_6h"] = np.where(n_roll >= STEPS_30MIN, p_roll, np.nan)
    aligned["hn_cm"] = (aligned["hs_cm"] - aligned["hs_lag"]).clip(lower=0.0)
    event = (
        aligned["hn_cm"].ge(MIN_HN_CM)
        & aligned["p_6h"].ge(MIN_P_MM)
        & aligned["hn_cm"].notna()
        & aligned["p_6h"].notna()
    )
    slr = (aligned["hn_cm"] * 10.0) / aligned["p_6h"]
    slr = slr.where(event)
    slr = slr.clip(SLR_CLIP[0], SLR_CLIP[1])
    aligned["slr"] = slr
    return aligned.reset_index()


def daily_slr_mean(hh: pd.DataFrame) -> pd.DataFrame:
    """Mean of valid 6 h SLR whose window ends that UTC day."""
    if hh.empty or "slr" not in hh.columns:
        return pd.DataFrame(columns=["date", "slr"])
    x = hh.dropna(subset=["slr"]).copy()
    if x.empty:
        return pd.DataFrame(columns=["date", "slr"])
    x["date"] = pd.to_datetime(x["time"], utc=True).dt.tz_convert("UTC").dt.strftime("%Y-%m-%d")
    g = x.groupby("date", as_index=False)["slr"].mean()
    return g
