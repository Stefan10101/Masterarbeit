#!/usr/bin/env python3
"""30-min grid helpers shared by variable QC scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd

SLOT_MIN = 30


def interpolate_micro_gaps_keep_index(
    df: pd.DataFrame,
    max_gap_hours: float = 2.0,
    col: str = "value",
) -> tuple[pd.DataFrame, int]:
    """Linear-fill gaps of at most max_gap_hours. Keep every index row (NaN stays NaN)."""
    if df.empty or col not in df.columns or len(df) < 2:
        return df, 0
    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        if "timestamp" in work.columns:
            work = work.set_index("timestamp")
        work.index = pd.to_datetime(work.index, utc=True)
    work = work.sort_index()
    before = work[col].isna()
    limit = max(1, int(round(max_gap_hours / (SLOT_MIN / 60.0))) - 1)
    work[col] = work[col].interpolate(method="time", limit=limit, limit_area="inside")
    filled = int((before & work[col].notna()).sum())
    return work, filled


def reindex_full_30min(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    col: str = "value",
) -> pd.DataFrame:
    """Force a closed 30-min UTC grid. Existing values kept; holes stay NaN."""
    work = df.copy()
    if "timestamp" in work.columns and not isinstance(work.index, pd.DatetimeIndex):
        work = work.set_index("timestamp")
    if isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index, utc=True)
        work = work.sort_index()
        work = work[~work.index.duplicated(keep="first")]
    grid = pd.date_range(start, end, freq="30min", tz="UTC")
    work = work.reindex(grid)
    if col in work.columns and "is_missing" in work.columns:
        work["is_missing"] = work[col].isna()
    work.index.name = "timestamp"
    return work


def coverage_mask(values: pd.Series, expected: pd.Series, min_frac: float) -> pd.Series:
    """True where enough non-null slots exist in the resampled bin."""
    expected = expected.replace(0, np.nan)
    return (values / expected) >= min_frac
