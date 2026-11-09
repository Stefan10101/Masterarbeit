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
    """Linear-fill all micro-gaps (original missing timestamps and QC-induced NaNs)
    of at most max_gap_hours. Inserts intermediate 30-min rows where needed.
    """
    if df.empty or col not in df.columns or len(df) < 2:
        return df, 0
    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        if "timestamp" in work.columns:
            work = work.set_index("timestamp")
        work.index = pd.to_datetime(work.index, utc=True)
    work = work.sort_index()
    work = work[~work.index.duplicated(keep="first")]

    valid_mask = work[col].notna()
    if valid_mask.sum() < 2:
        return work, 0
    valid_df = work[valid_mask]
    valid_times = valid_df.index
    new_rows = []
    points_filled = 0

    for i in range(len(valid_times) - 1):
        t1 = valid_times[i]
        t2 = valid_times[i + 1]
        delta_h = (t2 - t1).total_seconds() / 3600.0
        if not (0.5 < delta_h <= max_gap_hours):
            continue
        n_missing = int(round(delta_h / 0.5)) - 1
        if n_missing < 1 or n_missing > 3:
            continue
        v1 = valid_df.loc[t1, col]
        v2 = valid_df.loc[t2, col]
        if pd.isna(v1) or pd.isna(v2):
            continue
        interp_times = pd.date_range(
            start=t1 + pd.Timedelta(minutes=SLOT_MIN),
            periods=n_missing,
            freq=f"{SLOT_MIN}min",
        )
        interp_times = interp_times[interp_times < t2]
        if len(interp_times) == 0:
            continue
        interp_vals = np.linspace(v1, v2, len(interp_times) + 2)[1:-1]
        temp = pd.DataFrame({col: interp_vals}, index=interp_times)
        # copy other columns from the left bounding row
        for c in work.columns:
            if c != col:
                temp[c] = work.loc[t1, c] if c in work.columns else np.nan
        if "is_missing" in temp.columns:
            temp["is_missing"] = False
        new_rows.append(temp)
        points_filled += len(interp_times)

    if new_rows:
        interp_df = pd.concat(new_rows)
        # drop any existing rows at the interp times (e.g. QC NaNs) then concat
        work = work.drop(index=interp_df.index, errors="ignore")
        work = pd.concat([work, interp_df]).sort_index()
        work = work[~work.index.duplicated(keep="first")]
    return work, points_filled


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
